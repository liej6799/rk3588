"""add_ops example: a copied snapshot of the original UOps for a 5x5 half `a + b`.

`load_uops()` hard-codes the *original* (non-unrolled) linearized UOps captured from
tinygrad (test/backend/test_add.py). Built on the project-local `helpers.uop`
library, so it stays disconnected from tinygrad.
"""
from helpers.uop import Ops, UOp, AxisType, KernelInfo, dtypes

def load_uops() -> list[UOp]:
  """Return the original 14 UOps for the 5x5 half add (copied snapshot)."""
  hp = dtypes.half.ptr(25)
  u0  = UOp(Ops.PARAM, hp, (), 0)                       # output
  u1  = UOp(Ops.PARAM, hp, (), 1)                       # input a
  u2  = UOp(Ops.PARAM, hp, (), 2)                       # input b
  u3  = UOp(Ops.CONST, dtypes.int, (), 25)              # loop bound
  u4  = UOp(Ops.RANGE, dtypes.int, (u3,), (0, AxisType.LOOP))
  u5  = UOp(Ops.INDEX, hp, (u1, u4))
  u6  = UOp(Ops.LOAD, dtypes.half, (u5,))
  u7  = UOp(Ops.INDEX, hp, (u2, u4))
  u8  = UOp(Ops.LOAD, dtypes.half, (u7,))
  u9  = UOp(Ops.INDEX, hp, (u0, u4))
  u10 = UOp(Ops.ADD, dtypes.half, (u6, u8))
  u11 = UOp(Ops.STORE, dtypes.void, (u9, u10))
  u12 = UOp(Ops.END, dtypes.void, (u11, u4))
  u13 = UOp(Ops.SINK, dtypes.void, (u12,), KernelInfo(name="E_25"))
  return [u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13]

if __name__ == "__main__":
  uops = load_uops()
  print(f"loaded {len(uops)} uops")
  for i, u in enumerate(uops):
    print(f"{i:3d}: {u.op.name:8s} dtype={u.dtype} src={[uops.index(s) for s in u.src]} arg={u.arg}")
