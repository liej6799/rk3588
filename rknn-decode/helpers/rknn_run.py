"""Helper: run inference on a .rknn model on the RK3588 NPU.

Wraps the rknn-toolkit2 runtime flow (load_rknn -> init_runtime -> inference) so a
.rknn produced by `rknn_export` can be executed on the NPU. As with the export, the
toolkit only loads from a file path, so bytes are routed through a tmpfs dir
(RAM-backed) that is removed before returning. rknn-toolkit2 is imported lazily.

    from helpers.rknn_export import onnx_to_rknn_bytes
    from helpers.rknn_run import run_rknn
    out = run_rknn(onnx_to_rknn_bytes(model), [a, b])    # list of output ndarrays
"""
import os, shutil, tempfile
from .rknn_export import _tmpfs_dir

def run_rknn(model, inputs, target: str = "rk3588", core_mask: int = 0, verbose: bool = False) -> list:
  """Run inference on a .rknn model and return the list of output arrays.

  `model` may be raw .rknn bytes or a path to a .rknn file. `inputs` is the list of
  input arrays (one per model input). `target` selects the NPU platform; passing
  `target=None` uses the toolkit's simulator. Raises RuntimeError if the runtime is
  unavailable (e.g. not running on the device).
  """
  from rknn.api import RKNN

  work = None
  if isinstance(model, (bytes, bytearray)):
    work = tempfile.mkdtemp(prefix="rknn_", dir=_tmpfs_dir())   # RAM-backed scratch
    path = os.path.join(work, "model.rknn")
    with open(path, "wb") as f: f.write(model)
    model = path

  rknn = RKNN(verbose=verbose)
  try:
    if rknn.load_rknn(model) != 0: raise RuntimeError(f"rknn.load_rknn failed for {model}")
    if rknn.init_runtime(target=target, core_mask=core_mask) != 0:
      raise RuntimeError(f"rknn.init_runtime failed (target={target}); is the NPU runtime available?")
    return rknn.inference(inputs=list(inputs))
  finally:
    rknn.release()
    if work is not None: shutil.rmtree(work, ignore_errors=True)
