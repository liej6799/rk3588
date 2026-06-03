"""Helper: host an ONNX model in netron for visual inspection.

Generic over any model; the example-specific graph building lives with each op
(e.g. add_ops/viz.py). netron is imported lazily so the core helpers stay
dependency-free.
"""

def serve(model, address=("0.0.0.0", 8080), browse: bool = False, name: str = "model.onnx"):
  """Host a model in netron. `model` may be a file path, raw bytes, or an ONNX ModelProto.

  Bytes / a ModelProto are served straight from memory (no file written); a path is
  served from disk. `name` is the filename netron displays and uses to detect the
  format (e.g. "model.onnx" vs "model.rknn"). Returns the (host, port) being served.
  """
  import netron
  if isinstance(model, str):
    return netron.serve(model, address=address, browse=browse)
  data = model if isinstance(model, (bytes, bytearray)) else model.SerializeToString()
  return netron.serve(name, data=bytes(data), address=address, browse=browse)
