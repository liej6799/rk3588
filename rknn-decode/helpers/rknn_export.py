"""Helper: convert an ONNX model into a Rockchip .rknn file.

Wraps the `rknn-toolkit2` build flow (config -> load_onnx -> build -> export) so
the rest of the project can turn the ONNX graph from `onnx_export` into an .rknn
artifact runnable on the RK3588 NPU. rknn-toolkit2 is imported lazily so the core
helpers stay dependency-free.

    from helpers.onnx_export import uops_to_onnx
    from helpers.rknn_export import onnx_to_rknn
    onnx_to_rknn(uops_to_onnx(uops), "model.rknn")
"""
import os, tempfile

def onnx_to_rknn(model, export_path: str, target_platform: str = "rk3588",
                 do_quantization: bool = False, dataset: str | None = None,
                 verbose: bool = False) -> str:
  """Build a .rknn file from an ONNX model. `model` may be a ModelProto or a path.

  Floating-point (float16) by default; pass `do_quantization=True` together with a
  `dataset` file (one input-sample path per line) to produce a quantized model.
  Returns the `export_path` the .rknn was written to.
  """
  from rknn.api import RKNN

  # rknn-toolkit2 loads ONNX from disk, so materialize a ModelProto to a temp file
  tmp = None
  if not isinstance(model, str):
    import onnx
    tmp = os.path.join(tempfile.gettempdir(), "rknn_export_in.onnx")
    onnx.save(model, tmp)
    model = tmp

  if do_quantization and dataset is None:
    raise ValueError("do_quantization=True requires a `dataset` file for calibration")

  rknn = RKNN(verbose=verbose)
  try:
    rknn.config(target_platform=target_platform)
    if rknn.load_onnx(model=model) != 0: raise RuntimeError(f"rknn.load_onnx failed for {model}")
    if rknn.build(do_quantization=do_quantization, dataset=dataset) != 0: raise RuntimeError("rknn.build failed")
    if rknn.export_rknn(export_path) != 0: raise RuntimeError(f"rknn.export_rknn failed for {export_path}")
  finally:
    rknn.release()
    if tmp is not None and os.path.exists(tmp): os.remove(tmp)
  return export_path
