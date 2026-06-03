"""Tests for fully_unroll and the add_ops example."""
import pytest
from helpers.uop import Ops, UOp, AxisType, dtypes
from helpers.unroll import fully_unroll
from add_ops.uops import load_uops

def test_load_uops_snapshot():
  uops = load_uops()
  assert len(uops) == 14
  assert [u.op for u in uops[:3]] == [Ops.PARAM, Ops.PARAM, Ops.PARAM]
  assert uops[-1].op is Ops.SINK
  assert sum(u.op is Ops.RANGE for u in uops) == 1
  assert sum(u.op is Ops.STORE for u in uops) == 1

def test_fully_unroll_removes_ranges():
  unrolled = fully_unroll(load_uops())
  assert sum(u.op is Ops.RANGE for u in unrolled) == 0
  assert sum(u.op is Ops.END for u in unrolled) == 0

def test_fully_unroll_replicates_body():
  unrolled = fully_unroll(load_uops())
  assert sum(u.op is Ops.STORE for u in unrolled) == 25   # 25 loop iterations
  assert sum(u.op is Ops.ADD for u in unrolled) == 25
  assert len(unrolled) == 204

def test_fully_unroll_shares_params():
  unrolled = fully_unroll(load_uops())
  # the 3 buffers stay shared across iterations rather than duplicated
  assert sum(u.op is Ops.PARAM for u in unrolled) == 3

def test_fully_unroll_offsets_are_folded_constants():
  unrolled = fully_unroll(load_uops())
  offsets = sorted(u.arg for u in unrolled if u.op is Ops.CONST)
  assert offsets == list(range(25))                       # 0..24, all folded to CONST

def test_fully_unroll_noop_without_ranges():
  uops = fully_unroll(load_uops())          # already unrolled
  assert fully_unroll(uops) is uops         # no RANGE -> returned unchanged

def test_fully_unroll_rejects_non_constant_bound():
  bound = UOp(Ops.LOAD, dtypes.int, (UOp(Ops.PARAM, dtypes.int.ptr(1), (), 0),))   # not a CONST
  r = UOp(Ops.RANGE, dtypes.int, (bound,), (0, AxisType.LOOP))
  out = UOp(Ops.PARAM, dtypes.half.ptr(1), (), 1)
  st = out.index(r).store(UOp.const(dtypes.half, 0))
  sink = UOp.sink(st.end(r))
  with pytest.raises(AssertionError):
    fully_unroll(sink.toposort())
