"""matmul_ops example: an n x n float matmul UOp kernel (out = a @ b).

Three nested loops -- two output loops (i, j) plus a REDUCE loop (k) for the
contraction -- with computed flat indices into row-major [n*n] buffers:

    out[i*n + j] = sum_k  a[i*n + k] * b[k*n + j]

Exercises Ops.REDUCE / .reduce() and the UOp operator syntax. Defaults to 8x8.

Note: this is a REDUCE (matmul), not an element-wise add. It unrolls and runs in the
pure-Python interpreter (helpers.interp), but the toolkit-free NPU synthesizer
(helpers.rknn_synth) is add-only and does NOT emit a matmul command stream -- the NPU
matmul path uses the conv/matmul engine, a separate (un-ported) command format.
"""
from helpers.uop import Ops, UOp, AxisType, KernelInfo, dtypes

def make_matmul_uops(n: int = 8) -> list[UOp]:
  """Build the original (non-unrolled) UOps for an `n` x `n` float matmul."""
  sz = n * n
  c0  = UOp(Ops.PARAM, dtypes.float.ptr(sz), (), 0)          # output
  c2  = UOp.range(n, 1, AxisType.LOOP)                       # i (output row)
  c3  = c2 * UOp.const(dtypes.weakint, n)                    # i*n
  c4  = UOp.range(n, 2, AxisType.LOOP)                       # j (output col)
  c7  = UOp(Ops.PARAM, dtypes.float.ptr(sz), (), 1)          # a
  c8  = UOp.range(n, 0, AxisType.REDUCE)                     # k (contraction)
  c11 = UOp(Ops.PARAM, dtypes.float.ptr(sz), (), 2)          # b
  c15 = c7.index(c3 + c8) * c11.index(c8 * UOp.const(dtypes.weakint, n) + c4)
  c16 = c15.reduce(c8, arg=(Ops.ADD, ()))                    # sum_k a[i,k]*b[k,j]
  c18 = c0.index(c3 + c4).store(c16).end(c2, c4)
  ast = c18.sink(arg=KernelInfo(name="test"))
  return ast.toposort()

def load_uops() -> list[UOp]:
  """The 8x8 matmul example."""
  return make_matmul_uops(8)

if __name__ == "__main__":
  uops = load_uops()
  print(f"loaded {len(uops)} uops ({sum(u.op is Ops.RANGE for u in uops)} ranges, "
        f"{sum(u.op is Ops.REDUCE for u in uops)} reduce, {sum(u.op is Ops.STORE for u in uops)} store)")
