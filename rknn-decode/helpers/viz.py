"""Helper: host an ONNX model in netron for visual inspection.

Generic over any model; the example-specific graph building lives with each op
(e.g. add_ops/viz.py). netron is imported lazily so the core helpers stay
dependency-free.
"""

def serve(model, address=("0.0.0.0", 8080), browse: bool = False, name: str = "model.onnx"):
  """Host an ONNX model in netron. `model` may be a ModelProto or a file path.

  A ModelProto is served straight from memory (no file written); a path is served
  from disk. `name` is the filename netron displays and uses to detect the format.
  Returns the (host, port) netron is serving on.
  """
  import netron
  if isinstance(model, str):
    return netron.serve(model, address=address, browse=browse)
  return netron.serve(name, data=model.SerializeToString(), address=address, browse=browse)
