"""Visualize the mul_ops graph with netron.

Builds the ONNX graph from the unrolled multiply UOps and the toolkit-free synthesized
.rknn, and hosts either in netron. Everything is built from memory (the .rknn is the
toolkit-free synth, not a toolkit build), so only the NPU run elsewhere needs the toolkit.

    python3 -m mul_ops.viz                 # serve the ONNX graph on http://0.0.0.0:8080
    python3 -m mul_ops.viz --rknn          # serve the synthesized .rknn instead
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

from helpers.unroll import fully_unroll
from helpers.onnx_export import uops_to_onnx
from helpers.rknn_synth import uops_to_rknn
from helpers.viz import serve
from mul_ops.uops import load_uops

def build_model():
  """Build the mul_ops ONNX model from the unrolled UOps (Gather/Mul/Concat)."""
  return uops_to_onnx(fully_unroll(load_uops()), name="mul_10x10")

def build_rknn():
  """Build the mul_ops .rknn from the unrolled UOps (toolkit-free synth)."""
  return uops_to_rknn(fully_unroll(load_uops()), 10, 10)

def show(address=("0.0.0.0", 8080), browse: bool = False):
  """Host the mul_ops ONNX graph in netron (served from memory)."""
  return serve(build_model(), address=address, browse=browse)

def show_rknn(address=("0.0.0.0", 8080), browse: bool = False):
  """Host the synthesized mul_ops .rknn in netron (served from memory)."""
  return serve(build_rknn(), address=address, browse=browse, name="mul_10x10.rknn")

if __name__ == "__main__":
  import time
  rknn = "--rknn" in sys.argv[1:]
  host, port = show_rknn() if rknn else show()
  print(f"serving mul_ops {'rknn' if rknn else 'onnx'} graph at http://{host}:{port}  (ctrl-c to stop)")
  try:
    while True: time.sleep(3600)
  except KeyboardInterrupt:
    import netron; netron.stop()
