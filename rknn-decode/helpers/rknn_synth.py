"""Helper: synthesize a runnable .rknn directly from unrolled UOps.

This is the fully toolkit-free path: it does NOT go through ONNX and does NOT call
rknn-toolkit2. It reads a fully-unrolled element-wise add UOp graph, recovers the
element count N, and emits the .rknn (FlatBuffer body + NPU register-command stream +
container) from scratch via the vendored builders (`_rknn_flatbuf` + `_rc_template_gen`,
which depend only on the `flatbuffers` library).

    from helpers.unroll import fully_unroll
    from add_ops.uops import make_add_uops
    from helpers.rknn_synth import uops_to_rknn
    blob = uops_to_rknn(fully_unroll(make_add_uops(25)))   # bytes; no onnx, no toolkit

Scope: 2-input fp16 element-wise add (`z = a + b`) for N = 1..50,231,944. The command
buffer is generated from `_rc_template_gen` (the readable regcmd generator), not copied
from any library output.
"""
import math

from .uop import Ops
from . import _rknn_flatbuf as _fb


def analyze_add(uops) -> int:
  """Validate a fully-unrolled 2-input element-wise add and return its element count N.

  Expects every STORE to be `STORE(INDEX(out,k), ADD(LOAD(INDEX(a,k)), LOAD(INDEX(b,k))))`
  with exactly one output PARAM and two input PARAMs. Raises ValueError otherwise.
  """
  if any(u.op is Ops.RANGE for u in uops):
    raise ValueError("expected a fully-unrolled (range-free) graph; call fully_unroll first")
  stores = [u for u in uops if u.op is Ops.STORE]
  if not stores:
    raise ValueError("no STORE in uops")
  out_params, in_params = set(), set()
  out_offsets = []
  for st in stores:
    idx, val = st.src
    if idx.op is not Ops.INDEX or idx.src[0].op is not Ops.PARAM or idx.src[1].op is not Ops.CONST:
      raise ValueError("STORE target must be INDEX(PARAM, CONST)")
    out_params.add(idx.src[0].arg)
    out_offsets.append(idx.src[1].arg)
    if val.op is not Ops.ADD or len(val.src) != 2:
      raise ValueError("only 2-operand element-wise ADD is supported")
    for ld in val.src:
      if ld.op is not Ops.LOAD or ld.src[0].op is not Ops.INDEX or ld.src[0].src[0].op is not Ops.PARAM:
        raise ValueError("ADD operands must be LOAD(INDEX(PARAM, CONST))")
      in_params.add(ld.src[0].src[0].arg)
  if len(out_params) != 1:
    raise ValueError(f"expected exactly one output PARAM, got {sorted(out_params)}")
  if len(in_params) != 2:
    raise ValueError(f"expected exactly two input PARAMs, got {sorted(in_params)}")
  N = len(stores)
  if sorted(out_offsets) != list(range(N)):
    raise ValueError("STOREs must cover output offsets 0..N-1 exactly once")
  return N


def uops_to_rknn(uops, rows: int | None = None, cols: int | None = None) -> bytes:
  """Build a runnable .rknn (bytes) from a fully-unrolled element-wise add UOp graph.

  No ONNX, no rknn-toolkit2. `rows`/`cols` set the reported shape (cosmetic for the
  runtime); they default to a square if N is a perfect square, else `[1, N]`.
  """
  N = analyze_add(uops)
  if rows is None or cols is None:
    s = math.isqrt(N)
    rows, cols = (s, s) if s * s == N else (1, N)
  body = _fb.build_body(N, 2)                          # FlatBuffer + regcmd, from scratch
  return _fb.assemble_rknn(body, rows, cols, 2)        # + 64-byte header + JSON trailer
