"""Helper: fully unroll constant-bound RANGE loops in a linearized UOp list.

For each constant-bound loop RANGE it substitutes the loop variable with every concrete
value, constant-folds the resulting index math, and re-flattens the graph. REDUCE ops
(e.g. a matmul contraction) are expanded into the explicit op-fold over their range's
values, so the result is a flat LOAD/MUL/ADD/STORE graph with no RANGE or REDUCE left.
"""
import itertools
from .uop import Ops, UOp, AxisType, dtypes


def _expand_reduce(u: UOp) -> UOp:
  """Replace every REDUCE(src, *rngs, arg=(op, ...)) with the explicit op-fold of src
  over the cartesian product of its ranges' concrete values."""
  def go(x: UOp) -> UOp:
    if x.op is Ops.REDUCE:
      src, rngs, op = x.src[0], x.src[1:], x.arg[0]
      bounds = [int(r.src[0].arg) for r in rngs]
      terms = []
      for combo in itertools.product(*[range(b) for b in bounds]):
        sub = {r: UOp.const(dtypes.int, v) for r, v in zip(rngs, combo)}
        terms.append(go(src.substitute(sub)))           # expand nested reduces too
      acc = terms[0]
      for t in terms[1:]:
        acc = UOp(op, acc.dtype, (acc, t))
      return acc
    return x._rebuild(tuple(go(s) for s in x.src))
  return go(u)


def fully_unroll(uops: list[UOp]) -> list[UOp]:
  ranges = [u for u in uops if u.op is Ops.RANGE]
  if not ranges: return uops
  for r in ranges:
    assert r.src[0].op is Ops.CONST, f"cannot unroll non-constant range bound {r.src[0]}"
  sink = next(u for u in uops if u.op is Ops.SINK)
  stores = [u for u in uops if u.op is Ops.STORE]
  # iterate only the loop axes; REDUCE axes are consumed by _expand_reduce, not the product
  loop_ranges = [r for r in ranges if r.arg[1] is not AxisType.REDUCE]
  per_range = [[(r, UOp.const(dtypes.int, i)) for i in range(int(r.src[0].arg))] for r in loop_ranges]
  combos = list(itertools.product(*per_range)) if per_range else [()]
  new_stores = [_expand_reduce(st.substitute(dict(combo))).simplify()
                for combo in combos for st in stores]
  return list(UOp.sink(*new_stores, arg=sink.arg).toposort())
