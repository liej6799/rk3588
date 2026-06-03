"""Helper: convert an ONNX model into a Rockchip .rknn, in memory.

rknn-toolkit2 is a closed, compiled library with no in-memory API: `load_onnx`
only reads ONNX from a *file path* and `export_rknn` only writes the .rknn to a
*file path*. To avoid leaving artifacts on persistent disk, this helper routes both
through a tmpfs directory (RAM-backed, e.g. /dev/shm): the intermediate .onnx and
the output .rknn exist only in RAM and are removed before returning. The .rknn is
returned as bytes, so callers never have to touch a file.

    from helpers.onnx_export import uops_to_onnx
    from helpers.rknn_export import onnx_to_rknn_bytes
    blob = onnx_to_rknn_bytes(uops_to_onnx(uops))      # bytes, nothing written to disk

rknn-toolkit2 is imported lazily so the core helpers stay dependency-free.
"""
import os, shutil, tempfile

def _tmpfs_dir() -> str:
  """A RAM-backed scratch dir if one is available, else the normal temp dir (disk)."""
  for d in ("/dev/shm", "/run/shm"):
    if os.path.isdir(d) and os.access(d, os.W_OK): return d
  return tempfile.gettempdir()

def onnx_to_rknn_bytes(model, target_platform: str = "rk3588", do_quantization: bool = False,
                       dataset: str | None = None, verbose: bool = False) -> bytes:
  """Build a .rknn from an ONNX model and return it as bytes (no file persists).

  `model` may be an ONNX ModelProto or a path to an .onnx file. Floating-point
  (float16) by default; pass `do_quantization=True` with a `dataset` file (one
  input-sample path per line) for a quantized model. The intermediate .onnx (when
  `model` is a ModelProto) and the .rknn output are materialized only in tmpfs and
  unlinked before returning.
  """
  from rknn.api import RKNN

  if do_quantization and dataset is None:
    raise ValueError("do_quantization=True requires a `dataset` file for calibration")

  work = tempfile.mkdtemp(prefix="rknn_", dir=_tmpfs_dir())   # RAM-backed scratch dir
  rknn_path = os.path.join(work, "model.rknn")
  try:
    if isinstance(model, str):
      onnx_path = model                                       # caller already has a file; read it directly
    else:
      import onnx
      onnx_path = os.path.join(work, "model.onnx")
      onnx.save(model, onnx_path)

    rknn = RKNN(verbose=verbose)
    try:
      rknn.config(target_platform=target_platform)
      if rknn.load_onnx(model=onnx_path) != 0: raise RuntimeError(f"rknn.load_onnx failed for {onnx_path}")
      if rknn.build(do_quantization=do_quantization, dataset=dataset) != 0: raise RuntimeError("rknn.build failed")
      if rknn.export_rknn(rknn_path) != 0: raise RuntimeError("rknn.export_rknn failed")
    finally:
      rknn.release()

    with open(rknn_path, "rb") as f:
      return f.read()
  finally:
    shutil.rmtree(work, ignore_errors=True)                   # drop the tmpfs scratch dir

def onnx_to_rknn(model, export_path: str, **kwargs) -> str:
  """Build a .rknn and write it to `export_path`. Returns `export_path`.

  Thin wrapper over `onnx_to_rknn_bytes`; see it for the keyword arguments. Use the
  bytes variant directly when you want to avoid creating any file at all.
  """
  blob = onnx_to_rknn_bytes(model, **kwargs)            # build first, so a failure leaves no file
  with open(export_path, "wb") as f:
    f.write(blob)
  return export_path
