"""matmul_ops example: run an n x n matmul on the NPU fully toolkit-free.

Matmul can't be synthesized as a single .rknn, but it decomposes into element-wise ops we
CAN synthesize toolkit-free. out = A @ B is the sum of K rank-1 terms: for each k, an
element-wise multiply of the broadcast column A[:,k] and row B[k,:], accumulated with
element-wise adds. Every multiply/add is synthesized from UOps via uops_to_rknn (no
compiler) and run on the NPU -- exactly the add_ops/mul_ops synth path. Result == numpy.

    .venv/bin/python3 -m matmul_ops.run            # 8x8
    .venv/bin/python3 -m matmul_ops.run 16         # n x n
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import numpy as np
from helpers.rknn_synth import run_uops_on_npu
from mul_ops.uops import make_mul_uops
from add_ops.uops import make_add_uops

def matmul_npu(A, B, M, K, N):
  """out[i,j] = sum_k A[i,k]*B[k,j] (A: M*K, B: K*N, flat fp16), via K element-wise muls
  + adds on the NPU, each synthesized toolkit-free. Returns the M*N output (fp16)."""
  A2, B2 = A.reshape(M, K), B.reshape(K, N)
  acc = None
  for k in range(K):
    col = np.repeat(A2[:, k], N).astype(np.float16)         # col[i*N+j] = A[i,k]  (broadcast over j)
    row = np.tile(B2[k], M).astype(np.float16)              # row[i*N+j] = B[k,j]  (broadcast over i)
    term = np.asarray(run_uops_on_npu(make_mul_uops(M, N), [col, row])).reshape(-1)[:M*N].astype(np.float16)
    acc = term if acc is None else \
        np.asarray(run_uops_on_npu(make_add_uops(M*N), [acc, term])).reshape(-1)[:M*N].astype(np.float16)
  return acc

def main():
  n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
  rng = np.random.default_rng(0)
  A = rng.integers(0, 5, n*n).astype(np.float16)            # small ints stay fp16-exact
  B = rng.integers(0, 5, n*n).astype(np.float16)
  ref = (A.reshape(n, n).astype(np.float32) @ B.reshape(n, n).astype(np.float32)).reshape(-1)
  t = time.perf_counter()
  out = matmul_npu(A, B, n, n, n)
  dt = time.perf_counter() - t
  ok = np.allclose(out.astype(np.float32), ref)
  print(f"matmul {n}x{n}x{n} = {n} mul + {n-1} add toolkit-free NPU ops in {dt:.2f}s")
  print(f"NPU == numpy matmul: {ok}")
  return 0 if ok else 1

if __name__ == "__main__":
  sys.exit(main())
