"""Helper: a tiny interpreter for a fully-unrolled (range-free) UOp graph.

Enough to execute the kernels in this project and check their numerics, without
tinygrad. Buffers are plain Python lists indexed by PARAM arg; STOREs write in place.
"""
from .uop import Ops

_ALU = {
  Ops.ADD: lambda a, b: a + b,
  Ops.SUB: lambda a, b: a - b,
  Ops.MUL: lambda a, b: a * b,
  Ops.SHL: lambda a, b: a << b,
  Ops.MAX: max,
}

def run_uops(uops: list, buffers: dict[int, list]) -> dict[int, list]:
  """Execute a range-free UOp graph. `buffers` maps PARAM arg -> mutable list.

  STOREs are applied in place; the same `buffers` dict is returned for convenience.
  """
  assert not any(u.op is Ops.RANGE for u in uops), "run_uops needs a fully-unrolled (range-free) graph"
  def ev(u):
    if u.op is Ops.CONST: return u.arg
    if u.op is Ops.PARAM: return buffers[u.arg]
    if u.op is Ops.INDEX: return (ev(u.src[0]), ev(u.src[1]))   # (buffer, offset)
    if u.op is Ops.CAST: return ev(u.src[0])                    # vec-ptr cast: passthrough (buffer, offset)
    if u.op is Ops.LOAD:
      buf, off = ev(u.src[0])
      n = u.dtype.count                                        # vec(n) load reads n contiguous elements
      return buf[off] if n == 1 else [buf[off + i] for i in range(n)]
    if u.op is Ops.GEP: return ev(u.src[0])[u.arg[0]]          # extract one lane of a vector
    if u.op is Ops.STACK: return [ev(s) for s in u.src]         # pack lanes into a vector
    if u.op in _ALU: return _ALU[u.op](*[ev(s) for s in u.src])
    raise NotImplementedError(f"run_uops: unsupported op {u.op}")
  for st in (u for u in uops if u.op is Ops.STORE):
    buf, off = ev(st.src[0])
    val = ev(st.src[1])
    if isinstance(val, list):                                  # vector store: write n contiguous
      buf[off:off + len(val)] = val
    else:
      buf[off] = val
  return buffers
