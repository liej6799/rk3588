"""Helper: run inference on a .rknn model on the RK3588 NPU.

Wraps the rknn-toolkit2 runtime flow (load_rknn -> init_runtime -> inference) so a
.rknn produced by `rknn_export` can be executed on the NPU. As with the export, the
toolkit only loads from a file path, so bytes are routed through a tmpfs dir
(RAM-backed) that is removed before returning. rknn-toolkit2 is imported lazily.

    from helpers.rknn_run import run_rknn, RknnSession
    out = run_rknn(model_bytes, [a, b])                  # one-shot: load + init + infer + release
    with RknnSession(model_bytes) as s:                  # reuse: one load + init, many infers
      for ...: out = s.run([a, b])
"""
import os, shutil, tempfile
from .rknn_export import _tmpfs_dir


class RknnSession:
  """Load a .rknn once and run inference many times (a single load + init_runtime).

  Use when the same model is run repeatedly with different inputs (e.g. a matmul's K
  accumulation steps all run the same MULACC model) -- it avoids paying the NPU context
  setup on every call. `model` is .rknn bytes or a path. Use as a context manager.

      with RknnSession(blob) as sess:
        for ...: out = sess.run([a, b, acc])
  """
  def __init__(self, model, target: str = "rk3588", core_mask: int = 0,
               perf_debug: bool = False, verbose: bool = False):
    from rknn.api import RKNN
    self._work = None
    if isinstance(model, (bytes, bytearray)):
      self._work = tempfile.mkdtemp(prefix="rknn_", dir=_tmpfs_dir())
      path = os.path.join(self._work, "model.rknn")
      with open(path, "wb") as f: f.write(bytes(model))
      model = path
    self._rknn = RKNN(verbose=verbose)
    try:
      if self._rknn.load_rknn(model) != 0:
        raise RuntimeError(f"rknn.load_rknn failed for {model}")
      if self._rknn.init_runtime(target=target, core_mask=core_mask, perf_debug=perf_debug) != 0:
        raise RuntimeError(f"rknn.init_runtime failed (target={target}); is the NPU runtime available?")
    except BaseException:
      self.close()                                         # don't leak the runtime / tmpfs dir
      raise

  def run(self, inputs) -> list:
    return self._rknn.inference(inputs=list(inputs))

  def eval_perf(self):
    """Print the runtime's per-layer command/perf table for the last run (needs
    perf_debug=True at construction). This is the NPU command queue as executed."""
    return self._rknn.eval_perf()

  def close(self):
    if self._rknn is not None:
      self._rknn.release(); self._rknn = None
    if self._work is not None:
      shutil.rmtree(self._work, ignore_errors=True); self._work = None

  def __enter__(self): return self
  def __exit__(self, *exc): self.close()


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

def print_rknn_graph(model, target: str = "rk3588", core_mask: int = 0, verbose: bool = False) -> None:
  """Print the compiled NPU layer graph of a .rknn (bytes or path).

  Uses the toolkit's eval_perf() table, which lists every compiled layer with its
  op type, datatype, CPU/NPU target, and shapes -- i.e. the contents that netron
  shows as the un-expandable NNBG node. `target=None` uses the simulator. Raises
  RuntimeError if the runtime is unavailable.
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
    if rknn.init_runtime(target=target, core_mask=core_mask, perf_debug=True) != 0:
      raise RuntimeError(f"rknn.init_runtime failed (target={target}); is the NPU runtime available?")
    rknn.eval_perf()                                           # prints the compiled layer table
  finally:
    rknn.release()
    if work is not None: shutil.rmtree(work, ignore_errors=True)
