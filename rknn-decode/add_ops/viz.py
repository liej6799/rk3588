"""Visualize the add_ops ONNX graph with netron.

Builds the ONNX model from the fully-unrolled add UOps and hosts it via the
generic `helpers.viz.serve` helper.

    python3 -m add_ops.viz                 # serve on http://0.0.0.0:8080
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

from helpers.unroll import fully_unroll
from helpers.onnx_export import uops_to_onnx
from helpers.viz import serve
from add_ops.uops import load_uops

def build_model():
  """Build the add_ops ONNX model from the unrolled UOps."""
  return uops_to_onnx(fully_unroll(load_uops()), name="add_5x5")

def show(address=("0.0.0.0", 8080), browse: bool = False, path: str | None = None):
  """Host the add_ops ONNX graph in netron. Returns the (host, port) it serves on."""
  return serve(build_model(), address=address, browse=browse, path=path)

if __name__ == "__main__":
  import time
  host, port = show()
  print(f"serving add_ops onnx graph at http://{host}:{port}  (ctrl-c to stop)")
  try:
    while True: time.sleep(3600)
  except KeyboardInterrupt:
    import netron; netron.stop()
