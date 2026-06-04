"""matmul_ops example: n x n matmul UOps -> ONNX -> RKNN -> NPU, verified vs numpy.

The reduce expands to a MUL + MULACC (FMA) chain; uops_to_onnx lowers MULACC to Mul + Add,
the toolkit compiles that to the NPU, and we check the result equals numpy's a @ b. Needs
the toolkit (to compile) and the NPU runtime (to run); matmul is NOT toolkit-free here.

    .venv/bin/python3 -m matmul_ops.run          # 8x8
    .venv/bin/python3 -m matmul_ops.run 16       # n x n
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

from collections import Counter
import numpy as np
from helpers.unroll import fully_unroll
from helpers.onnx_export import uops_to_onnx
from helpers.rknn_export import onnx_to_rknn_bytes
from helpers.rknn_decode import decode_rknn
from helpers.rknn_run import run_rknn
from matmul_ops.uops import make_matmul_uops

def main():
  n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
  unr = fully_unroll(make_matmul_uops(n))
  print(f"[1] UOPS  : {dict(Counter(u.op.name for u in unr))}")            # MUL + MULACC (FMA)
  model = uops_to_onnx(unr, name=f"matmul_{n}x{n}")
  print(f"[2] ONNX  : {dict(Counter(nd.op_type for nd in model.graph.node))}")  # MULACC -> Mul+Add

  rng = np.random.default_rng(0)
  A = rng.integers(0, 5, n * n).astype(np.float32)
  B = rng.integers(0, 5, n * n).astype(np.float32)
  ref = (A.reshape(n, n) @ B.reshape(n, n)).reshape(-1)

  blob = onnx_to_rknn_bytes(model)
  d = decode_rknn(blob)
  print(f"[3] RKNN  : {dict(Counter(b['kind'] for b in d['command_queue']))} ({len(d['command_queue'])} blocks)")
  got = np.array(run_rknn(blob, [A, B], target="rk3588")[0]).reshape(-1)[:n * n].astype(np.float32)
  ok = np.allclose(got, ref)
  print(f"[4] NPU == numpy matmul: {ok}")
  return 0 if ok else 1

if __name__ == "__main__":
  sys.exit(main())
