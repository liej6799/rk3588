"""add_ops example: export the 5x5 half add to .rknn and run it on the RK3588 NPU.

Builds the ONNX graph from the unrolled UOps, converts it to .rknn (in memory), runs
it on the NPU, and checks the result equals a + b.

Run from the project root (needs the device runtime + project venv):

    .venv/bin/python3 -m add_ops.run_rknn
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import numpy as np
from helpers.rknn_run import run_rknn
from add_ops.viz import build_rknn

# inputs mirror tests/test_onnx.py (5x5, fp16-exact integer values)
N = 25
A = (np.arange(N) % 100).astype(np.float16)
B = np.full(N, 10, dtype=np.float16)

def main():
  out = run_rknn(build_rknn(), [A, B], target="rk3588")
  got = np.array(out[0]).reshape(-1)[:N]
  print("inputs a :", A.tolist())
  print("inputs b :", B.tolist())
  print("npu out  :", got.tolist())
  print("expected :", (A + B).tolist())
  ok = np.allclose(got, A + B)
  print("MATCH" if ok else "MISMATCH")
  return 0 if ok else 1

if __name__ == "__main__":
  sys.exit(main())
