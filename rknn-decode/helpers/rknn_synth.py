"""Helper: synthesize a runnable .rknn directly from unrolled UOps.

This is the fully toolkit-free path: it does NOT go through ONNX and does NOT call
rknn-toolkit2. It reads a fully-unrolled element-wise add UOp graph, recovers the
element count N, and emits the .rknn (FlatBuffer body + NPU register-command stream +
container) from scratch via the vendored builders (`_rknn_flatbuf` + `_rc_template_gen`,
which depend only on the `flatbuffers` library).

    from add_ops.uops import make_add_uops
    from helpers.rknn_synth import uops_to_rknn, run_uops_on_npu
    blob = uops_to_rknn(fully_unroll(make_add_uops(25)))   # bytes; no onnx, no toolkit
    z = run_uops_on_npu(make_add_uops(25), [a, b])         # one call: UOps -> NPU result

Scope: 2-input fp16 element-wise add (`z = a + b`) for N = 1..50,231,944. The command
buffer is generated from `_rc_template_gen` (the readable regcmd generator), not copied
from any library output.
"""
import math

from .uop import Ops
from .unroll import fully_unroll
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
      raise ValueError("only a 2-operand element-wise ADD (z = a + b) is supported")
    for ld in val.src:
      if ld.op is not Ops.LOAD or ld.src[0].op is not Ops.INDEX or ld.src[0].src[0].op is not Ops.PARAM:
        raise ValueError(
          "not a simple element-wise add: each ADD operand must be LOAD(INDEX(PARAM, CONST)). "
          "Kernels with MUL/REDUCE (e.g. matmul) are not supported by the toolkit-free "
          "synthesizer (the NPU matmul/conv engine is a separate, un-ported command stream).")
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


def run_uops_on_npu(uops, inputs, target: str = "rk3588",
                    rows: int | None = None, cols: int | None = None):
  """Run a UOp add graph directly on the NPU. Returns the output as a flat array of N.

  Accepts the original (loop) UOps or an already-unrolled list; any RANGE loops are
  fully unrolled first. `inputs` is the list of input arrays (one per input PARAM, e.g.
  [a, b] as fp16 ndarrays). End to end this is: UOps -> unroll -> synthesize .rknn
  (no onnx, no toolkit) -> run on the NPU. rknn-toolkit2 is used only to submit to the
  NPU, not to build the model.
  """
  from .rknn_run import run_rknn                       # lazy: keeps the build path toolkit-free

  if any(u.op is Ops.RANGE for u in uops):
    uops = fully_unroll(uops)
  N = analyze_add(uops)
  if len(inputs) != 2:
    raise ValueError(f"expected 2 input arrays (a, b), got {len(inputs)}")
  blob = uops_to_rknn(uops, rows, cols)
  out = run_rknn(blob, list(inputs), target=target)
  return out[0].reshape(-1)[:N]
