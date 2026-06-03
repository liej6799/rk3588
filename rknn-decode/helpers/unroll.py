"""Helper: fully unroll constant-bound RANGE loops in a linearized UOp list.

For each constant-bound RANGE it substitutes the loop variable with every concrete
value, constant-folds the resulting index math, and re-flattens the graph.
"""
import itertools
from .uop import Ops, UOp, dtypes

def fully_unroll(uops: list[UOp]) -> list[UOp]:
  ranges = [u for u in uops if u.op is Ops.RANGE]
  if not ranges: return uops
  for r in ranges:
    assert r.src[0].op is Ops.CONST, f"cannot unroll non-constant range bound {r.src[0]}"
  sink = next(u for u in uops if u.op is Ops.SINK)
  stores = [u for u in uops if u.op is Ops.STORE]
  # cartesian product over all ranges -> one (range->const) substitution map per iteration
  per_range = [[(r, UOp.const(dtypes.int, i)) for i in range(int(r.src[0].arg))] for r in ranges]
  new_stores = [st.substitute(dict(combo)).simplify() for combo in itertools.product(*per_range) for st in stores]
  return list(UOp.sink(*new_stores, arg=sink.arg).toposort())
