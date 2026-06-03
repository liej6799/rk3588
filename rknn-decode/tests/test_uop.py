"""Tests for the standalone UOp helper library."""
from helpers.uop import Ops, UOp, AxisType, KernelInfo, dtypes, PtrDType

def test_dtype_repr():
  assert repr(dtypes.half) == "dtypes.half"
  assert repr(dtypes.int) == "dtypes.int"
  assert repr(dtypes.half.ptr(25)) == "dtypes.half.ptr(25)"
  assert repr(dtypes.half.vec(4)) == "dtypes.half.vec(4)"
  assert repr(dtypes.half.vec(4).ptr(16)) == "dtypes.half.vec(4).ptr(16)"

def test_dtype_properties():
  assert dtypes.half.itemsize == 2
  v = dtypes.half.vec(4)
  assert v.count == 4 and v.itemsize == 8
  assert v.base == dtypes.half          # vec base is the scalar
  p = dtypes.half.ptr(25)
  assert isinstance(p, PtrDType) and p.size == 25 and p.base == dtypes.half

def test_axistype_repr():
  assert repr(AxisType.LOOP) == "AxisType.LOOP"

def test_kernelinfo_repr_matches_tinygrad():
  assert repr(KernelInfo(name="E_25")) == (
    "KernelInfo(name='E_25', axis_types=(), dont_use_locals=False, "
    "applied_opts=(), opts_to_apply=None, estimates=None, beam=0)")

def test_const_and_sink_builders():
  c = UOp.const(dtypes.int, 7)
  assert c.op is Ops.CONST and c.dtype is dtypes.int and c.arg == 7
  s = UOp.sink(c, None, arg="k")          # None sources are dropped
  assert s.op is Ops.SINK and s.src == (c,) and s.arg == "k"

def test_index_load_store_helpers():
  buf = UOp(Ops.PARAM, dtypes.half.ptr(4), (), 0)
  idx = buf.index(UOp.const(dtypes.int, 1))
  assert idx.op is Ops.INDEX and idx.src == (buf, idx.src[1])
  ld = idx.load()
  assert ld.op is Ops.LOAD and ld.dtype is dtypes.half     # load of ptr yields base dtype

def test_toposort_orders_deps_before_uses():
  a = UOp.const(dtypes.int, 1)
  b = UOp.const(dtypes.int, 2)
  add = UOp(Ops.ADD, dtypes.int, (a, b))
  order = add.toposort()
  assert order[-1] is add
  assert order.index(a) < order.index(add) and order.index(b) < order.index(add)

def test_substitute_replaces_node_and_shares_rest():
  p = UOp(Ops.PARAM, dtypes.half.ptr(4), (), 0)
  r = UOp(Ops.RANGE, dtypes.int, (UOp.const(dtypes.int, 4),), (0, AxisType.LOOP))
  idx = p.index(r)
  out = idx.substitute({r: UOp.const(dtypes.int, 3)})
  assert out.op is Ops.INDEX
  assert out.src[0] is p                 # untouched PARAM is reused, not copied
  assert out.src[1].op is Ops.CONST and out.src[1].arg == 3

def test_simplify_folds_constant_int_alu():
  expr = UOp(Ops.ADD, dtypes.int, (UOp(Ops.MUL, dtypes.int, (UOp.const(dtypes.int, 3), UOp.const(dtypes.int, 4))),
                                    UOp.const(dtypes.int, 2)))
  folded = expr.simplify()
  assert folded.op is Ops.CONST and folded.arg == 14     # 3*4 + 2

def test_simplify_leaves_non_constant_alone():
  p = UOp(Ops.PARAM, dtypes.int.ptr(1), (), 0)
  ld = p.index(UOp.const(dtypes.int, 0)).load()
  expr = UOp(Ops.ADD, dtypes.int, (ld, UOp.const(dtypes.int, 5)))
  assert expr.simplify().op is Ops.ADD                   # not all-const -> unchanged op
