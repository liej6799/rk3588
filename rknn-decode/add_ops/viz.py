"""Visualize the add_ops graph with netron.

Builds the ONNX model (and optionally the exported .rknn) from the fully-unrolled
add UOps and hosts it via the generic `helpers.viz.serve` helper. Everything is
served from memory; no file is written.

    python3 -m add_ops.viz                 # serve the ONNX graph on http://0.0.0.0:8080
    python3 -m add_ops.viz --rknn          # export to .rknn and serve that instead
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

from helpers.unroll import fully_unroll
from helpers.onnx_export import uops_to_onnx
from helpers.rknn_export import onnx_to_rknn_bytes
from helpers.viz import serve
from add_ops.uops import load_uops

def build_model():
  """Build the add_ops ONNX model from the unrolled UOps."""
  return uops_to_onnx(fully_unroll(load_uops()), name="add_5x5")

def build_rknn():
  """Build the add_ops ONNX model and convert it to .rknn bytes (in memory)."""
  return onnx_to_rknn_bytes(build_model())

def show(address=("0.0.0.0", 8080), browse: bool = False):
  """Host the add_ops ONNX graph in netron (served from memory, no file written).

  Returns the (host, port) it serves on."""
  return serve(build_model(), address=address, browse=browse)

def show_rknn(address=("0.0.0.0", 8080), browse: bool = False):
  """Export the add_ops graph to .rknn and host it in netron (served from memory).

  Returns the (host, port) it serves on."""
  return serve(build_rknn(), address=address, browse=browse, name="add_5x5.rknn")

if __name__ == "__main__":
  import time
  rknn = "--rknn" in sys.argv[1:]
  host, port = show_rknn() if rknn else show()
  print(f"serving add_ops {'rknn' if rknn else 'onnx'} graph at http://{host}:{port}  (ctrl-c to stop)")
  try:
    while True: time.sleep(3600)
  except KeyboardInterrupt:
    import netron; netron.stop()
