"""mul_ops example: a 2D nested-loop element-wise multiply `z = a * b` (N = rows*cols).

Same 2D-loop shape as add2d_ops but with `*` instead of `+` -- the tinygrad test_mul
kernel. Defaults to 10x10 = 100.

    c0 = UOp(Ops.PARAM, dtypes.half.ptr(N), (), 0)        # output
    c2 = UOp.range(rows, 0, AxisType.LOOP)                # i
    c4 = UOp.range(cols, 1, AxisType.LOOP)                # j
    c5 = c2 * UOp.const(dtypes.weakint, cols) + c4        # flat index i*cols + j
    c7 = UOp(Ops.PARAM, dtypes.half.ptr(N), (), 1)        # a
    c9 = UOp(Ops.PARAM, dtypes.half.ptr(N), (), 2)        # b
    c11 = c7.index(c5) * c9.index(c5)                     # a[idx] * b[idx]
    c13 = c0.index(c5).store(c11).end(c2, c4)
    ast = c13.sink(arg=KernelInfo(name='test'))

Element-wise MUL runs on the NPU toolkit-free: the synthesizer reuses the add tiles with
the EW_CFG ALU op retargeted to the hardware multiply path (EW_OP_TYPE=1).
"""
from helpers.uop import Ops, UOp, AxisType, KernelInfo, dtypes

def make_mul_uops(rows: int = 10, cols: int = 10) -> list[UOp]:
  """Build the original (non-unrolled) UOps for a `rows x cols` 2D-loop element-wise mul."""
  n = rows * cols
  c0  = UOp(Ops.PARAM, dtypes.half.ptr(n), (), 0)
  c2  = UOp.range(rows, 0, AxisType.LOOP)
  c4  = UOp.range(cols, 1, AxisType.LOOP)
  c5  = c2 * UOp.const(dtypes.weakint, cols) + c4
  c7  = UOp(Ops.PARAM, dtypes.half.ptr(n), (), 1)
  c9  = UOp(Ops.PARAM, dtypes.half.ptr(n), (), 2)
  c11 = c7.index(c5) * c9.index(c5)
  c13 = c0.index(c5).store(c11).end(c2, c4)
  ast = c13.sink(arg=KernelInfo(name="test"))
  return ast.toposort()

def load_uops() -> list[UOp]:
  """The N=100 (10x10) element-wise multiply example."""
  return make_mul_uops(10, 10)

if __name__ == "__main__":
  uops = load_uops()
  print(f"loaded {len(uops)} uops ({sum(u.op is Ops.RANGE for u in uops)} ranges, "
        f"{sum(u.op is Ops.MUL for u in uops)} mul, {sum(u.op is Ops.STORE for u in uops)} store)")
  for i, u in enumerate(uops):
    print(f"{i:3d}: {u.op.name:7s} {str(u.dtype):16s} src={[uops.index(s) for s in u.src]} arg={u.arg}")
