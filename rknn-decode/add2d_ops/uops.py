"""add2d_ops example: a 2D nested-loop half `a + b` kernel (N = rows*cols).

Unlike add_ops (a single flat RANGE), this builds the add with TWO nested RANGE loops
and a computed flat index `i*cols + j` -- the tinygrad-style 2D iteration. It exercises
the UOp operator syntax (`*`, `+`, implicit-load INDEX, instance `.sink`). Defaults to
10x10 = 100 elements.

The structure matches the snapshot:

    c0 = UOp(Ops.PARAM, dtypes.half.ptr(N), (), 0)        # output
    c2 = UOp.range(rows, 0, AxisType.LOOP)                # outer loop i
    c4 = UOp.range(cols, 1, AxisType.LOOP)                # inner loop j
    c5 = c2 * UOp.const(dtypes.weakint, cols) + c4        # flat index i*cols + j
    c7 = UOp(Ops.PARAM, dtypes.half.ptr(N), (), 1)        # input a
    c9 = UOp(Ops.PARAM, dtypes.half.ptr(N), (), 2)        # input b
    c11 = c7.index(c5) + c9.index(c5)                     # a[idx] + b[idx]
    c13 = c0.index(c5).store(c11).end(c2, c4)             # out[idx] = ...; close both loops
    ast = c13.sink(arg=KernelInfo(name='test'))
"""
from helpers.uop import Ops, UOp, AxisType, KernelInfo, dtypes

def make_add2d_uops(rows: int = 10, cols: int = 10) -> list[UOp]:
  """Build the original (non-unrolled) UOps for a `rows x cols` 2D-loop half add."""
  n = rows * cols
  c0  = UOp(Ops.PARAM, dtypes.half.ptr(n), (), 0)
  c2  = UOp.range(rows, 0, AxisType.LOOP)
  c4  = UOp.range(cols, 1, AxisType.LOOP)
  c5  = c2 * UOp.const(dtypes.weakint, cols) + c4
  c7  = UOp(Ops.PARAM, dtypes.half.ptr(n), (), 1)
  c9  = UOp(Ops.PARAM, dtypes.half.ptr(n), (), 2)
  c11 = c7.index(c5) + c9.index(c5)
  c13 = c0.index(c5).store(c11).end(c2, c4)
  ast = c13.sink(arg=KernelInfo(name="test"))
  return ast.toposort()

def load_uops() -> list[UOp]:
  """The N=100 (10x10) 2D-loop add example."""
  return make_add2d_uops(10, 10)

if __name__ == "__main__":
  uops = load_uops()
  print(f"loaded {len(uops)} uops ({sum(u.op is Ops.RANGE for u in uops)} ranges, "
        f"{sum(u.op is Ops.STORE for u in uops)} store)")
  for i, u in enumerate(uops):
    print(f"{i:3d}: {u.op.name:7s} {str(u.dtype):16s} src={[uops.index(s) for s in u.src]} arg={u.arg}")
