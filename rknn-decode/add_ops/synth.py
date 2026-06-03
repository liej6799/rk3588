"""add_ops example: UOps -> .rknn with no ONNX and no toolkit, then verify on the NPU.

Builds the unrolled element-wise add UOps for a given size, synthesizes the .rknn
directly (FlatBuffer body + NPU command stream from scratch), runs it on the NPU, and
checks z == a + b.

    .venv/bin/python3 -m add_ops.synth            # 1x1, 3x3, 5x5
    .venv/bin/python3 -m add_ops.synth 8 8        # custom rows cols
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import numpy as np
from helpers.unroll import fully_unroll
from helpers.rknn_synth import uops_to_rknn
from helpers.rknn_run import run_rknn
from add_ops.uops import make_add_uops

def run_one(rows, cols):
  N = rows * cols
  blob = uops_to_rknn(fully_unroll(make_add_uops(N)), rows, cols)   # no onnx, no toolkit
  A = (np.arange(N) % 7).astype(np.float16)
  B = np.full(N, 3, dtype=np.float16)
  out = np.array(run_rknn(blob, [A, B], target="rk3588")[0]).reshape(-1)[:N].astype(np.float32)
  ok = np.allclose(out, A + B)
  print(f"{rows}x{cols} N={N:<5d} rknn={len(blob)}B  NPU {'PASS' if ok else 'FAIL'}  (toolkit-free)")
  return ok

def main():
  if len(sys.argv) == 3:
    sizes = [(int(sys.argv[1]), int(sys.argv[2]))]
  else:
    sizes = [(1, 1), (3, 3), (5, 5)]
  return 0 if all(run_one(r, c) for r, c in sizes) else 1

if __name__ == "__main__":
  sys.exit(main())
