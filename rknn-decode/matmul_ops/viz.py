"""Visualize the matmul with netron.

Two views (both served from memory, no file written):
  * the ONNX graph of the unrolled matmul (Gather/Mul/Add/Concat) -- the full math; small n
    by default so it's viewable,
  * the toolkit-free MULACC step .rknn (`col*row + acc`, Mul+Add) -- the model that actually
    runs on the NPU, K times, in matmul_ops.run.

    python3 -m matmul_ops.viz                 # ONNX matmul graph (4x4) on http://0.0.0.0:8080
    python3 -m matmul_ops.viz 2               # ONNX matmul graph for n x n
    python3 -m matmul_ops.viz --rknn          # the MULACC step .rknn instead
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

from helpers.unroll import fully_unroll
from helpers.onnx_export import uops_to_onnx
from helpers.rknn_synth import chain_to_rknn
from helpers.viz import serve
from matmul_ops.uops import make_matmul_uops

def build_model(n: int = 4):
  """The unrolled n x n matmul as an ONNX graph (Gather/Mul/Add/Concat)."""
  return uops_to_onnx(fully_unroll(make_matmul_uops(n)), name=f"matmul_{n}x{n}")

def build_rknn(n: int = 8):
  """The toolkit-free MULACC step .rknn (col*row + acc) that matmul_ops.run executes K times."""
  return chain_to_rknn(["Mul", "Add"], n * n)

def show(n: int = 4, address=("0.0.0.0", 8080), browse: bool = False):
  """Host the matmul ONNX graph in netron (served from memory)."""
  return serve(build_model(n), address=address, browse=browse)

def show_rknn(n: int = 8, address=("0.0.0.0", 8080), browse: bool = False):
  """Host the MULACC step .rknn in netron (served from memory)."""
  return serve(build_rknn(n), address=address, browse=browse, name="mulacc.rknn")

if __name__ == "__main__":
  import time
  rknn = "--rknn" in sys.argv[1:]
  nargs = [a for a in sys.argv[1:] if not a.startswith("--")]
  n = int(nargs[0]) if nargs else (8 if rknn else 4)
  host, port = show_rknn(n) if rknn else show(n)
  what = "MULACC step rknn" if rknn else f"{n}x{n} matmul onnx graph"
  print(f"serving {what} at http://{host}:{port}  (ctrl-c to stop)")
  try:
    while True: time.sleep(3600)
  except KeyboardInterrupt:
    import netron; netron.stop()
