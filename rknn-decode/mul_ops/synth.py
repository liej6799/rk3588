"""mul_ops example: 2D-loop element-wise multiply UOps -> .rknn (no onnx/toolkit) -> NPU.

    .venv/bin/python3 -m mul_ops.synth          # 10x10 = 100
    .venv/bin/python3 -m mul_ops.synth 16 16    # custom rows cols
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import numpy as np
from helpers.rknn_synth import run_uops_on_npu
from mul_ops.uops import make_mul_uops

def run_one(rows, cols):
  N = rows * cols
  A = (np.arange(N) % 7).astype(np.float16)
  B = (np.arange(N) % 5 + 1).astype(np.float16)
  z = np.asarray(run_uops_on_npu(make_mul_uops(rows, cols), [A, B], rows=rows, cols=cols)).reshape(-1)[:N]
  ok = np.allclose(z, A.astype(np.float32) * B.astype(np.float32))
  print(f"{rows}x{cols} N={N:<5d} NPU {'PASS' if ok else 'FAIL'}  z=a*b  (toolkit-free)")
  return ok

def main():
  rows, cols = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (10, 10)
  return 0 if run_one(rows, cols) else 1

if __name__ == "__main__":
  sys.exit(main())
