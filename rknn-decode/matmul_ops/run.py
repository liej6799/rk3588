"""matmul_ops example: run an n x n matmul on the NPU fully toolkit-free.

Matmul can't be synthesized as a single .rknn, but it decomposes into element-wise ops we
CAN synthesize toolkit-free. out = A @ B is the sum of K rank-1 terms: for each k, the
broadcast column A[:,k] times row B[k,:], accumulated. Each step is ONE fused multiply-
accumulate (MULACC) .rknn -- a mixed Mul+Add op-chain `col*row + acc` -- synthesized
toolkit-free (no compiler) and run on the NPU. So an n x n matmul = K MULACC NPU ops.

    .venv/bin/python3 -m matmul_ops.run            # 8x8 (prints the unrolled MULACC UOp summary)
    .venv/bin/python3 -m matmul_ops.run 16         # n x n
    .venv/bin/python3 -m matmul_ops.run --uops     # also dump the full unrolled UOp list
    .venv/bin/python3 -m matmul_ops.run --perf     # also print the NPU command queue (eval_perf)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import numpy as np
from collections import Counter
from helpers.unroll import fully_unroll
from helpers.rknn_synth import chain_to_rknn
from helpers.rknn_run import RknnSession
from matmul_ops.uops import make_matmul_uops

def dump_uops(n, full=False):
  """Print the matmul's original + unrolled UOps (the MULACC reduce). The NPU run uses the
  equivalent element-wise MULACC decomposition, not these UOps directly (a dot-product
  reduce can't be one toolkit-free .rknn)."""
  orig = make_matmul_uops(n)
  unr = fully_unroll(orig)
  print(f"[uops] original {len(orig)} -> unrolled {len(unr)}: "
        f"{dict(Counter(u.op.name for u in unr))}")
  if full:
    for i, u in enumerate(unr):
      print(f"  {i:3d}: {u.op.name:7s} {str(u.dtype):14s} src={[unr.index(s) for s in u.src]}"
            + (f" arg={u.arg}" if u.arg is not None and u.op.name != 'SINK' else ""))

def matmul_npu(A, B, M, K, N, perf=False):
  """out[i,j] = sum_k A[i,k]*B[k,j] (A: M*K, B: K*N, flat fp16), via K fused multiply-
  accumulate (MULACC) steps. The MULACC model (col*row + acc, a Mul+Add op-chain) is
  synthesized toolkit-free ONCE and loaded into ONE NPU session; the K accumulation steps
  are just inference calls on it. Returns the M*N output (fp16).

  perf=True prints the runtime's per-layer command queue (eval_perf) for one step."""
  A2, B2 = A.reshape(M, K), B.reshape(K, N)
  acc = np.zeros(M * N, dtype=np.float16)
  blob = chain_to_rknn(["Mul", "Add"], M * N)               # one toolkit-free MULACC .rknn
  with RknnSession(blob, perf_debug=perf) as sess:          # one load + one init_runtime
    for k in range(K):
      col = np.repeat(A2[:, k], N).astype(np.float16)       # col[i*N+j] = A[i,k]  (broadcast over j)
      row = np.tile(B2[k], M).astype(np.float16)            # row[i*N+j] = B[k,j]  (broadcast over i)
      acc = np.asarray(sess.run([col, row, acc])[0]).reshape(-1)[:M*N].astype(np.float16)
      if perf and k == 0:
        print("--- NPU command queue for one MULACC step (col*row + acc) ---")
        sess.eval_perf()
  return acc

def main():
  args = [a for a in sys.argv[1:] if not a.startswith("--")]
  perf = "--perf" in sys.argv[1:]
  n = int(args[0]) if args else 8
  dump_uops(n, full="--uops" in sys.argv[1:])               # show the unrolled MULACC UOps
  rng = np.random.default_rng(0)
  A = rng.integers(0, 5, n*n).astype(np.float16)            # small ints stay fp16-exact
  B = rng.integers(0, 5, n*n).astype(np.float16)
  ref = (A.reshape(n, n).astype(np.float32) @ B.reshape(n, n).astype(np.float32)).reshape(-1)
  t = time.perf_counter()
  out = matmul_npu(A, B, n, n, n, perf=perf)
  dt = time.perf_counter() - t
  ok = np.allclose(out.astype(np.float32), ref)
  print(f"matmul {n}x{n}x{n} = {n} MULACC toolkit-free NPU ops in {dt:.2f}s")
  print(f"NPU == numpy matmul: {ok}")
  return 0 if ok else 1

if __name__ == "__main__":
  sys.exit(main())
