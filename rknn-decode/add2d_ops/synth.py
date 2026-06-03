"""add2d_ops example: 2D-loop UOps -> .rknn (no onnx, no toolkit) -> run on the NPU.

Builds the N=rows*cols 2D nested-loop add, unrolls it, synthesizes the .rknn from
scratch, runs it on the NPU, and checks z == a + b. Defaults to 10x10 (N=100).

    .venv/bin/python3 -m add2d_ops.synth          # 10x10 = 100
    .venv/bin/python3 -m add2d_ops.synth 16 16    # custom rows cols
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import numpy as np
from helpers.unroll import fully_unroll
from helpers.rknn_synth import uops_to_rknn
from helpers.rknn_run import run_rknn
from add2d_ops.uops import make_add2d_uops

def run_one(rows, cols):
  N = rows * cols
  blob = uops_to_rknn(fully_unroll(make_add2d_uops(rows, cols)), rows, cols)  # no onnx, no toolkit
  A = (np.arange(N) % 13).astype(np.float16)
  B = (np.arange(N) % 5).astype(np.float16)
  out = np.array(run_rknn(blob, [A, B], target="rk3588")[0]).reshape(-1)[:N].astype(np.float32)
  ok = np.allclose(out, A.astype(np.float32) + B.astype(np.float32))
  print(f"{rows}x{cols} N={N:<5d} rknn={len(blob)}B  NPU {'PASS' if ok else 'FAIL'}  (2D loop, toolkit-free)")
  return ok

def main():
  rows, cols = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (10, 10)
  return 0 if run_one(rows, cols) else 1

if __name__ == "__main__":
  sys.exit(main())
