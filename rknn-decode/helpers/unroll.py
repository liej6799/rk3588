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


def upcast_elementwise(uops: list[UOp], vec: int = 4) -> list[UOp]:
  """Vectorize a 2-input element-wise op into vec(`vec`) loads/stores.

  Reproduces tinygrad's `OptOps.UPCAST` rendering: instead of N scalar stores, emit N/vec
  groups, each a `CAST` of the base pointer to a vec(`vec`) pointer, a vector `LOAD` per
  input, per-lane `GEP` + scalar op, a `STACK` of the lanes, and one vector `STORE` -- in
  the same emission order tinygrad uses. Requires a fully-unrolled element-wise add/mul
  whose element count N is divisible by `vec`.
  """
  unr = fully_unroll(uops)
  stores = {s.src[0].src[1].arg: s for s in unr if s.op is Ops.STORE}
  N = len(stores)
  if N % vec != 0:
    raise ValueError(f"element count N={N} is not divisible by vec={vec}")
  st0 = stores[0]
  op = st0.src[1].op
  if op not in (Ops.ADD, Ops.MUL) or len(st0.src[1].src) != 2:
    raise ValueError("upcast supports a 2-operand element-wise ADD or MUL")
  out_p = st0.src[0].src[0]
  a_p = st0.src[1].src[0].src[0].src[0]
  b_p = st0.src[1].src[1].src[0].src[0]
  base = out_p.dtype.base                       # half
  sz = out_p.dtype.size                         # N
  vptr, vt = base.vec(vec).ptr(sz), base.vec(vec)
  ki = next(u for u in unr if u.op is Ops.SINK).arg

  out_list, seen = [], set()
  def emit(u):
    if id(u) not in seen:
      seen.add(id(u)); out_list.append(u)
    return u

  store_uops = []
  for g in range(N // vec):
    c = UOp.const(dtypes.int, g * vec)
    oi, oc = out_p.index(c), None
    oc = UOp(Ops.CAST, vptr, (oi,))
    ai = a_p.index(c); ac = UOp(Ops.CAST, vptr, (ai,)); al = UOp(Ops.LOAD, vt, (ac,))
    bi = b_p.index(c); bc = UOp(Ops.CAST, vptr, (bi,)); bl = UOp(Ops.LOAD, vt, (bc,))
    emit(out_p); emit(c); emit(oi); emit(oc)
    emit(a_p); emit(ai); emit(ac); emit(al)
    ga0 = emit(UOp(Ops.GEP, base, (al,), (0,)))
    emit(b_p); emit(bi); emit(bc); emit(bl)
    gb0 = emit(UOp(Ops.GEP, base, (bl,), (0,)))
    muls = [emit(UOp(op, base, (ga0, gb0)))]
    for lane in range(1, vec):
      ga = emit(UOp(Ops.GEP, base, (al,), (lane,)))
      gb = emit(UOp(Ops.GEP, base, (bl,), (lane,)))
      muls.append(emit(UOp(op, base, (ga, gb))))
    stk = emit(UOp(Ops.STACK, vt, tuple(muls)))
    store_uops.append(emit(UOp(Ops.STORE, dtypes.void, (oc, stk))))
  return out_list + [UOp(Ops.SINK, dtypes.void, tuple(store_uops), ki)]
