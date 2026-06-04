"""Simple element-wise MULACC (FMA: z = a*b + c) through interp / ONNX / RKNN.

ONNX has no fused multiply-accumulate, so uops_to_onnx lowers MULACC to Mul + Add. The
toolkit then compiles that to the NPU as separate element-wise MULTIPLY tiles
(EW_CFG=0x8003c4) and ADD tiles (EW_CFG=0x8202c0) -- the FMA is not a single fused tile.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

from collections import Counter
import numpy as np
import pytest
from helpers.uop import Ops, UOp, AxisType, KernelInfo, dtypes
from helpers.unroll import fully_unroll
from helpers.interp import run_uops
from helpers.onnx_export import uops_to_onnx

N = 16

def make_mulacc_uops(n=N):
  """z[k] = a[k]*b[k] + c[k] via the MULACC op (3 inputs)."""
  hp = dtypes.half.ptr(n)
  out = UOp(Ops.PARAM, hp, (), 0)
  a, b, c = (UOp(Ops.PARAM, hp, (), i) for i in (1, 2, 3))
  r = UOp.range(n, 0, AxisType.LOOP)
  val = UOp(Ops.MULACC, dtypes.half, (a.index(r).load(), b.index(r).load(), c.index(r).load()))
  return out.index(r).store(val).end(r).sink(arg=KernelInfo(name="mulacc")).toposort()

def _abc():
  a = (np.arange(N) % 5).astype(np.float16)
  b = (np.arange(N) % 3 + 1).astype(np.float16)
  c = (np.arange(N) % 7).astype(np.float16)
  return a, b, c

def test_unroll_and_interp():
  unr = fully_unroll(make_mulacc_uops())
  assert sum(u.op is Ops.MULACC for u in unr) == N        # one FMA per element (stays scalar)
  st = next(u for u in unr if u.op is Ops.STORE)
  assert st.src[1].op is Ops.MULACC and [s.op for s in st.src[1].src] == [Ops.LOAD, Ops.LOAD, Ops.LOAD]
  a, b, c = _abc()
  out = [0.0] * N
  run_uops(unr, {0: out, 1: list(a), 2: list(b), 3: list(c)})
  assert out == [float(a[i]) * float(b[i]) + float(c[i]) for i in range(N)]

def test_onnx_lowers_mulacc_to_mul_add():
  import onnx
  import onnxruntime as ort
  m = uops_to_onnx(fully_unroll(make_mulacc_uops()), name="mulacc")
  onnx.checker.check_model(m)
  ops = Counter(n.op_type for n in m.graph.node)
  assert ops["Mul"] == N and ops["Add"] == N             # MULACC -> Mul + Add (no ONNX FMA)
  assert [i.name for i in m.graph.input] == ["input1", "input2", "input3"]
  a, b, c = _abc()
  sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
  out = sess.run(None, {"input1": a, "input2": b, "input3": c})[0].reshape(-1)[:N].astype(np.float32)
  np.testing.assert_allclose(out, a.astype(np.float32) * b.astype(np.float32) + c.astype(np.float32))

def test_exported_rknn_is_mul_tiles_plus_add_tiles():
  pytest.importorskip("rknn")
  from helpers.rknn_export import onnx_to_rknn_bytes
  from helpers.rknn_decode import decode_rknn
  m = uops_to_onnx(fully_unroll(make_mulacc_uops()), name="mulacc")
  d = decode_rknn(onnx_to_rknn_bytes(m))
  # the FMA decomposes into element-wise MUL tiles + ADD tiles (not one fused MAC tile):
  # the EW tiles carry exactly two distinct ALU configs (multiply vs add), 16 of each.
  ew_cfgs = Counter(next(v for t, o, _n, v in b["regs"] if (t, o) == ("DPU", 0x4070))
                    for b in d["command_queue"] if b["kind"].startswith("EW_BINARY"))
  assert sorted(ew_cfgs.values()) == [N, N]              # 16 multiply tiles + 16 add tiles
  # the two configs differ only in the EW op-select bits (ALU_ALGO / OP_TYPE / OP_CVT)
  mul_cfg, add_cfg = sorted(ew_cfgs)
  assert (mul_cfg & 0x0F) == 0x4 and (add_cfg & 0x0F) == 0x0   # 0x...c4 (mul) vs 0x...c0 (add)

def test_mulacc_runs_on_npu():
  pytest.importorskip("rknn")
  from helpers.rknn_export import onnx_to_rknn_bytes
  from helpers.rknn_run import run_rknn
  blob = onnx_to_rknn_bytes(uops_to_onnx(fully_unroll(make_mulacc_uops()), name="mulacc"))
  a, b, c = _abc()
  try:
    out = run_rknn(blob, [a, b, c], target="rk3588")
  except RuntimeError as e:
    pytest.skip(f"NPU runtime unavailable: {e}")
  got = np.array(out[0]).reshape(-1)[:N].astype(np.float32)
  np.testing.assert_allclose(got, a.astype(np.float32) * b.astype(np.float32) + c.astype(np.float32))
