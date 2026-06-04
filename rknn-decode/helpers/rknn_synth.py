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

Scope: 2-input fp16 element-wise add (`z = a + b`) or multiply (`z = a * b`) for
N = 1..50,231,944. The command buffer is generated from `_rc_template_gen` (the readable
regcmd generator), not copied from any library output; MUL reuses the same tiles with the
EW_CFG ALU op retargeted to the multiply path.
"""
import math

from .uop import Ops
from .unroll import fully_unroll
from . import _rknn_flatbuf as _fb

# DPU EW_CFG (0x4070) value per element-wise op, recovered by diffing toolkit Add vs Mul
# models: ADD uses the ALU sum path (EW_OP_TYPE=0, EW_ALU_ALGO=2); MUL uses the multiply
# path (EW_OP_TYPE=1). Everything else in the tile (geometry, addresses) is identical.
_EW_CFG = {Ops.ADD: 0x108202C0, Ops.MUL: 0x108003C4}
_EW_CFG_REG = ("DPU", 0x4070)


def _peel_index(u):
  """(CAST of) INDEX(PARAM, CONST) -> (param_arg, offset). CAST = the upcast vec-ptr cast."""
  if u.op is Ops.CAST:
    u = u.src[0]
  if u.op is not Ops.INDEX or u.src[0].op is not Ops.PARAM or u.src[1].op is not Ops.CONST:
    raise ValueError("expected (CAST of) INDEX(PARAM, CONST)")
  return u.src[0].arg, u.src[1].arg


def _operand_param(u):
  """ALU operand -> input PARAM arg. Scalar: LOAD(INDEX(p)). Vectorized: GEP(LOAD(CAST(INDEX(p))))."""
  if u.op is Ops.GEP:
    u = u.src[0]
  if u.op is not Ops.LOAD:
    raise ValueError(
      "operand must be LOAD(INDEX(PARAM)) (scalar) or GEP(LOAD(...)) (vectorized); a "
      "reduction (e.g. matmul: sum of products) is not a simple element-wise op")
  return _peel_index(u.src[0])[0]


def analyze_elementwise(uops):
  """Validate a fully-unrolled 2-input element-wise op and return (N, op).

  Accepts both the scalar full-unroll
    `STORE(INDEX(out,k), OP(LOAD(INDEX(a,k)), LOAD(INDEX(b,k))))`
  and the vectorized UPCAST form
    `STORE(CAST(INDEX(out,k)), STACK(OP(GEP(LOAD(CAST(INDEX(a)))), GEP(LOAD(...))), ...))`.
  OP must be the same ADD or MUL across all stores, with one output and two input PARAMs.
  Raises ValueError otherwise (e.g. for REDUCE trees like matmul).
  """
  if any(u.op is Ops.RANGE for u in uops):
    raise ValueError("expected a fully-unrolled (range-free) graph; call fully_unroll first")
  stores = [u for u in uops if u.op is Ops.STORE]
  if not stores:
    raise ValueError("no STORE in uops")
  out_params, in_params, ops, pieces = set(), set(), set(), []
  for st in stores:
    out_param, off = _peel_index(st.src[0])
    out_params.add(out_param)
    val = st.src[1]
    pieces.append((off, val.dtype.count))               # vec store covers `count` offsets
    alu = val.src[0] if val.op is Ops.STACK else val     # peel STACK -> one lane's op
    if alu.op not in (Ops.ADD, Ops.MUL) or len(alu.src) != 2:
      raise ValueError("only a 2-operand element-wise ADD or MUL (z = a+b / a*b) is supported; "
                       "a reduction (e.g. matmul: a MULACC/sum of products) is not")
    ops.add(alu.op)
    for operand in alu.src:
      in_params.add(_operand_param(operand))
  if len(ops) != 1:
    raise ValueError(f"all stores must use the same op, got {sorted(o.name for o in ops)}")
  if len(out_params) != 1:
    raise ValueError(f"expected exactly one output PARAM, got {sorted(out_params)}")
  if len(in_params) != 2:
    raise ValueError(f"expected exactly two input PARAMs, got {sorted(in_params)}")
  pos = 0
  for off, count in sorted(pieces):
    if off != pos:
      raise ValueError("STOREs must tile output offsets contiguously from 0")
    pos += count
  return pos, next(iter(ops))


def analyze_add(uops) -> int:
  """Validate a fully-unrolled 2-input element-wise ADD; return N. (See analyze_elementwise.)"""
  N, op = analyze_elementwise(uops)
  if op is not Ops.ADD:
    raise ValueError(f"expected an element-wise ADD, got {op.name}")
  return N


def _retarget_ew_op(blob: bytes, op) -> bytes:
  """Rewrite every EW_BINARY tile's EW_CFG to select `op` (ADD or MUL). ADD is a no-op."""
  if op is Ops.ADD:
    return blob
  from .rknn_decode import decode_rknn, split_container
  from .rknn_encode import encode_container, reemit_command_stream
  target = _EW_CFG[op]
  version, body, trailer = split_container(blob)
  d = decode_rknn(blob)
  for blk in d["command_queue"]:
    if blk["kind"].startswith("EW_BINARY"):
      blk["regs"] = [(t, off, name, target if (t, off) == _EW_CFG_REG else val)
                     for t, off, name, val in blk["regs"]]
  return encode_container(version, reemit_command_stream(body, d["command_queue"]), trailer)


def uops_to_rknn(uops, rows: int | None = None, cols: int | None = None) -> bytes:
  """Build a runnable .rknn (bytes) from a fully-unrolled element-wise add/mul UOp graph.

  No ONNX, no rknn-toolkit2. `rows`/`cols` set the reported shape (cosmetic for the
  runtime); they default to a square if N is a perfect square, else `[1, N]`.
  """
  N, op = analyze_elementwise(uops)
  if rows is None or cols is None:
    s = math.isqrt(N)
    rows, cols = (s, s) if s * s == N else (1, N)
  body = _fb.build_body(N, 2)                          # FlatBuffer + regcmd, from scratch
  blob = _fb.assemble_rknn(body, rows, cols, 2)        # + 64-byte header + JSON trailer
  return _retarget_ew_op(blob, op)                     # ADD template -> MUL if needed


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
  N, _op = analyze_elementwise(uops)
  if len(inputs) != 2:
    raise ValueError(f"expected 2 input arrays (a, b), got {len(inputs)}")
  blob = uops_to_rknn(uops, rows, cols)
  out = run_rknn(blob, list(inputs), target=target)
  return out[0].reshape(-1)[:N]


_CHAIN_OPS = {"Add", "Sub", "Mul", "Div"}

def chain_to_rknn(ops, N: int, rows: int | None = None, cols: int | None = None) -> bytes:
  """Synthesize a runnable .rknn for an element-wise op-chain (toolkit-free, MIXED ops).

  `ops` is a list like ["Mul", "Add"]; the model computes the left-associated chain
  `(((in0 ops[0] in1) ops[1] in2) ...)` over N elements, with `len(ops)+1` inputs. Each op
  becomes its own EW tiles (Add/Sub/Mul/Div) in one body -- e.g. ["Mul","Add"] is the fused
  multiply-accumulate `a*b + c`. No ONNX, no toolkit.
  """
  bad = [o for o in ops if o not in _CHAIN_OPS]
  if bad:
    raise ValueError(f"unsupported chain ops {bad}; allowed: {sorted(_CHAIN_OPS)}")
  if not 1 <= len(ops) <= 2:
    # the pure-FlatBuffers synth reliably loads for <=3 inputs (<=2 ops, e.g. MULACC a*b+c);
    # longer chains need the reference-body path (a one-time toolkit compile) -- out of scope here.
    raise NotImplementedError(
      f"toolkit-free chains support 1..2 ops (2..3 inputs); got {len(ops)} ops")
  n_inputs = len(ops) + 1
  if rows is None or cols is None:
    s = math.isqrt(N)
    rows, cols = (s, s) if s * s == N else (1, N)
  body = _fb.build_body(N, n_inputs, ops=list(ops))     # FlatBuffer + per-op regcmd tiles
  return _fb.assemble_rknn(body, rows, cols, n_inputs)


def run_chain_on_npu(ops, inputs, target: str = "rk3588",
                     rows: int | None = None, cols: int | None = None):
  """Synthesize an element-wise op-chain .rknn (toolkit-free) and run it on the NPU.

  `inputs` is the list of `len(ops)+1` input arrays (fp16). Returns the N-element output.
  """
  from .rknn_run import run_rknn                         # lazy: keeps the build toolkit-free
  if len(inputs) != len(ops) + 1:
    raise ValueError(f"{len(ops)} ops need {len(ops)+1} inputs, got {len(inputs)}")
  N = len(inputs[0])
  blob = chain_to_rknn(ops, N, rows, cols)
  return run_rknn(blob, list(inputs), target=target)[0].reshape(-1)[:N]
