"""Helper: host an ONNX model in netron for visual inspection.

Generic over any model; the example-specific graph building lives with each op
(e.g. add_ops/viz.py). netron/onnx are imported lazily so the core helpers stay
dependency-free.
"""
import os, tempfile

def serve(model, address=("0.0.0.0", 8080), browse: bool = False, path: str | None = None):
  """Host an ONNX model in netron. `model` may be a ModelProto or a file path.

  Returns the (host, port) netron is serving on.
  """
  import netron
  if not isinstance(model, str):
    import onnx
    if path is None: path = os.path.join(tempfile.gettempdir(), "model.onnx")
    onnx.save(model, path)
    model = path
  return netron.start(model, address=address, browse=browse)
