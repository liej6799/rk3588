"""Helper: fully unroll a linearized UOp list, auto-vectorizing (UPCAST) when it can.

`fully_unroll` first expands every constant-bound loop RANGE (substituting the loop var
with each value and constant-folding the index math) and every REDUCE (op-folding over its
range). Then, mirroring tinygrad's `hand_coded_optimizations`, it AUTO-UPCASTS: if the
result is a contiguous 2-input element-wise add/mul whose element count N is divisible by 4,
it vectorizes 4 elements into a `half.vec(4)` (CAST/vec-LOAD/GEP/op/STACK/vec-STORE) -- the
same `OptOps.UPCAST axis arg=4` tinygrad applies. Otherwise it returns the scalar unroll.
Pass `upcast=1` to force the scalar form, or `upcast=n` to force a specific width.
"""
import itertools
from .uop import Ops, UOp, AxisType, dtypes

AUTO_UPCAST = 4                                   # tinygrad's default UPCAST split


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


def _scalar_unroll(uops: list[UOp]) -> list[UOp]:
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


def _elementwise_layout(flat: list[UOp]):
  """If `flat` is a contiguous, stride-1 2-input element-wise add/mul, return
  (N, op, out_param, a_param, b_param); else None. This is the "can it be upcasted?"
  determination -- the kernel must read/write the same offset k on every buffer so that
  consecutive elements pack into a contiguous vector load/store."""
  stores = [u for u in flat if u.op is Ops.STORE]
  if not stores: return None
  op = out_p = a_p = b_p = None
  offs = []
  for st in stores:
    idx, val = st.src
    if idx.op is not Ops.INDEX or idx.src[0].op is not Ops.PARAM or idx.src[1].op is not Ops.CONST:
      return None
    k = idx.src[1].arg
    if val.op not in (Ops.ADD, Ops.MUL) or len(val.src) != 2:
      return None                                        # reduce tree / non-binary -> not vectorizable here
    if op is None: op = val.op
    elif op is not val.op: return None
    la, lb = val.src
    for ld in (la, lb):
      if (ld.op is not Ops.LOAD or ld.src[0].op is not Ops.INDEX
          or ld.src[0].src[0].op is not Ops.PARAM or ld.src[0].src[1].op is not Ops.CONST
          or ld.src[0].src[1].arg != k):                 # not the same (contiguous) offset
        return None
    out_p = out_p or idx.src[0]; a_p = a_p or la.src[0].src[0]; b_p = b_p or lb.src[0].src[0]
    if idx.src[0] is not out_p or la.src[0].src[0] is not a_p or lb.src[0].src[0] is not b_p:
      return None
    offs.append(k)
  if sorted(offs) != list(range(len(offs))): return None
  return len(offs), op, out_p, a_p, b_p


def _vectorize(flat: list[UOp], vec: int, layout) -> list[UOp]:
  """Vectorize a scalar element-wise unroll into vec(`vec`) loads/stores, in tinygrad's
  UPCAST emission order (CAST base->vec ptr, vec LOAD per input, per-lane GEP + op, STACK,
  vec STORE)."""
  N, op, out_p, a_p, b_p = layout
  base, sz = out_p.dtype.base, out_p.dtype.size
  vptr, vt = base.vec(vec).ptr(sz), base.vec(vec)
  ki = next(u for u in flat if u.op is Ops.SINK).arg
  out_list, seen = [], set()
  def emit(u):
    if id(u) not in seen:
      seen.add(id(u)); out_list.append(u)
    return u
  store_uops = []
  for g in range(N // vec):
    c = UOp.const(dtypes.int, g * vec)
    oc = UOp(Ops.CAST, vptr, (out_p.index(c),))
    al = UOp(Ops.LOAD, vt, (UOp(Ops.CAST, vptr, (a_p.index(c),)),))
    bl = UOp(Ops.LOAD, vt, (UOp(Ops.CAST, vptr, (b_p.index(c),)),))
    emit(out_p); emit(c); emit(oc.src[0]); emit(oc)
    emit(a_p); emit(al.src[0].src[0]); emit(al.src[0]); emit(al)
    ga0 = emit(UOp(Ops.GEP, base, (al,), (0,)))
    emit(b_p); emit(bl.src[0].src[0]); emit(bl.src[0]); emit(bl)
    gb0 = emit(UOp(Ops.GEP, base, (bl,), (0,)))
    muls = [emit(UOp(op, base, (ga0, gb0)))]
    for lane in range(1, vec):
      ga = emit(UOp(Ops.GEP, base, (al,), (lane,)))
      gb = emit(UOp(Ops.GEP, base, (bl,), (lane,)))
      muls.append(emit(UOp(op, base, (ga, gb))))
    stk = emit(UOp(Ops.STACK, vt, tuple(muls)))
    store_uops.append(emit(UOp(Ops.STORE, dtypes.void, (oc, stk))))
  return out_list + [UOp(Ops.SINK, dtypes.void, tuple(store_uops), ki)]


def fully_unroll(uops: list[UOp], upcast="auto") -> list[UOp]:
  """Fully unroll `uops`, auto-vectorizing element-wise kernels (UPCAST) like tinygrad.

  upcast="auto" (default): vectorize by 4 if the unrolled graph is a contiguous element-wise
                           add/mul with N % 4 == 0, else return the scalar unroll.
  upcast=1 / None        : always return the scalar unroll.
  upcast=n (int > 1)     : force vec(n); raises if the graph isn't vectorizable or N % n != 0.
  """
  flat = _scalar_unroll(uops)
  if upcast in (1, None):
    return flat
  layout = _elementwise_layout(flat)
  if upcast == "auto":
    return _vectorize(flat, AUTO_UPCAST, layout) if (layout and layout[0] % AUTO_UPCAST == 0) else flat
  if not isinstance(upcast, int) or upcast < 1:
    raise ValueError(f"upcast must be 'auto', 1/None, or an int >= 1 (got {upcast!r})")
  if layout is None:
    raise ValueError("graph is not a contiguous element-wise add/mul; cannot upcast")
  if layout[0] % upcast != 0:
    raise ValueError(f"element count N={layout[0]} is not divisible by vec={upcast}")
  return _vectorize(flat, upcast, layout)
