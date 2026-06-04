"""matmul_ops example: run an n x n matmul on the NPU fully toolkit-free.

Matmul can't be synthesized as a single .rknn, but it decomposes into element-wise ops we
CAN synthesize toolkit-free. out = A @ B is the sum of K rank-1 terms: for each k, the
broadcast column A[:,k] times row B[k,:], accumulated. Each step is ONE fused multiply-
accumulate (MULACC) .rknn -- a mixed Mul+Add op-chain `col*row + acc` -- synthesized
toolkit-free (no compiler) and run on the NPU. So an n x n matmul = K MULACC NPU ops.

    .venv/bin/python3 -m matmul_ops.run            # 8x8
    .venv/bin/python3 -m matmul_ops.run 16         # n x n
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import numpy as np
from helpers.rknn_synth import chain_to_rknn
from helpers.rknn_run import RknnSession

def matmul_npu(A, B, M, K, N):
  """out[i,j] = sum_k A[i,k]*B[k,j] (A: M*K, B: K*N, flat fp16), via K fused multiply-
  accumulate (MULACC) steps. The MULACC model (col*row + acc, a Mul+Add op-chain) is
  synthesized toolkit-free ONCE and loaded into ONE NPU session; the K accumulation steps
  are just inference calls on it. Returns the M*N output (fp16)."""
  A2, B2 = A.reshape(M, K), B.reshape(K, N)
  acc = np.zeros(M * N, dtype=np.float16)
  blob = chain_to_rknn(["Mul", "Add"], M * N)               # one toolkit-free MULACC .rknn
  with RknnSession(blob) as sess:                           # one load + one init_runtime
    for k in range(K):
      col = np.repeat(A2[:, k], N).astype(np.float16)       # col[i*N+j] = A[i,k]  (broadcast over j)
      row = np.tile(B2[k], M).astype(np.float16)            # row[i*N+j] = B[k,j]  (broadcast over i)
      acc = np.asarray(sess.run([col, row, acc])[0]).reshape(-1)[:M*N].astype(np.float16)
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
  print(f"matmul {n}x{n}x{n} = {n} MULACC toolkit-free NPU ops in {dt:.2f}s")
  print(f"NPU == numpy matmul: {ok}")
  return 0 if ok else 1

if __name__ == "__main__":
  sys.exit(main())
