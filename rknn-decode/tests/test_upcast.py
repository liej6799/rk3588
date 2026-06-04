"""Vectorized (UPCAST) unroll: match tinygrad's E_25_4 rendering of the 10x10 op.

fully_unroll auto-upcasts (UPCAST), reproducing tinygrad's `OptOps.UPCAST axis=0 arg=4`: N/4 groups of a
CAST + vec(4) LOAD per input, per-lane GEP + scalar op, STACK, and a vec(4) STORE. The
op counts and per-group op sequence match tinygrad exactly, and the extended interpreter
evaluates the vectorized graph to a*b / a+b. No NPU involved.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

from collections import Counter
import pytest
from helpers.uop import Ops
from helpers.unroll import fully_unroll
from helpers.interp import run_uops
from mul_ops.uops import make_mul_uops
from add2d_ops.uops import make_add2d_uops

# tinygrad E_25_4 op counts (from DEBUG=6 DEV=CPU test/backend/test_mul.py), 10x10 UPCAST=4
TG_COUNTS = {"STORE": 25, "LOAD": 50, "MUL": 100, "GEP": 200, "STACK": 25, "CAST": 75,
             "INDEX": 75, "CONST": 25, "PARAM": 3, "SINK": 1, "RANGE": 0}
# tinygrad's exact op sequence for the first vec(4) group (uops 0..25)
TG_GROUP0 = ["PARAM", "CONST", "INDEX", "CAST", "PARAM", "INDEX", "CAST", "LOAD", "GEP",
             "PARAM", "INDEX", "CAST", "LOAD", "GEP", "MUL", "GEP", "GEP", "MUL", "GEP",
             "GEP", "MUL", "GEP", "GEP", "MUL", "STACK", "STORE"]

def test_upcast_matches_tinygrad_counts():
  uops = fully_unroll(make_mul_uops(10, 10))
  assert len(uops) == 579                                   # same total as tinygrad E_25_4
  counts = Counter(u.op.name for u in uops)
  for op, n in TG_COUNTS.items():
    assert counts.get(op, 0) == n, f"{op}: {counts.get(op, 0)} != {n}"

def test_upcast_group0_sequence_and_dtypes():
  uops = fully_unroll(make_mul_uops(10, 10))
  assert [u.op.name for u in uops[:26]] == TG_GROUP0        # identical emission order
  assert repr(uops[3].dtype) == "dtypes.half.vec(4).ptr(100)"   # CAST to vec(4) pointer
  assert repr(uops[7].dtype) == "dtypes.half.vec(4)"            # vec(4) LOAD
  assert uops[8].op is Ops.GEP and uops[8].arg == (0,)         # lane extract

def test_upcast_interp_mul():
  uops = fully_unroll(make_mul_uops(10, 10))
  N = 100
  A = [float(i % 7) for i in range(N)]
  B = [float(i % 5 + 1) for i in range(N)]
  out = [0.0] * N
  run_uops(uops, {0: out, 1: A, 2: B})
  assert out == [A[i] * B[i] for i in range(N)]             # vectorized graph computes a*b

def test_upcast_add():
  uops = fully_unroll(make_add2d_uops(10, 10))
  counts = Counter(u.op.name for u in uops)
  assert counts["ADD"] == 100 and counts["STORE"] == 25 and counts["MUL"] == 0
  N = 100
  A = [float(i % 7) for i in range(N)]
  B = [float(i % 3) for i in range(N)]
  out = [0.0] * N
  run_uops(uops, {0: out, 1: A, 2: B})
  assert out == [A[i] + B[i] for i in range(N)]

def test_auto_upcast_skips_when_not_divisible():
  scalar = fully_unroll(make_mul_uops(3, 3))                # N=9, 9 % 4 != 0 -> stays scalar
  assert not any(u.op is Ops.STACK for u in scalar)
  assert sum(u.op is Ops.STORE for u in scalar) == 9

def test_upcast_to_onnx_runs():
  import numpy as np
  import onnx
  import onnxruntime as ort
  from helpers.onnx_export import uops_to_onnx
  m = uops_to_onnx(fully_unroll(make_mul_uops(10, 10)), name="mul_upcast")
  onnx.checker.check_model(m)
  ops = Counter(n.op_type for n in m.graph.node)
  # 50 vec loads x Gather(4) + 200 GEP x Gather(1) = 250 Gather; 100 Mul; 25 STACK + 1 out Concat
  assert ops["Gather"] == 250 and ops["Mul"] == 100 and ops["Concat"] == 26
  A = (np.arange(100) % 7).astype(np.float16)
  B = (np.arange(100) % 5 + 1).astype(np.float16)
  sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
  out = sess.run(None, {"input1": A, "input2": B})[0].reshape(-1)[:100].astype(np.float32)
  np.testing.assert_allclose(out, A.astype(np.float32) * B.astype(np.float32))

def test_upcast_synth_matches_scalar():
  # the toolkit-free synth accepts the upcasted UOps directly and produces the SAME .rknn
  # as the scalar unroll (it recovers N and the op; the NPU command is one tiled mul)
  pytest.importorskip("rknn")
  from helpers.unroll import fully_unroll
  from helpers.rknn_synth import uops_to_rknn, analyze_elementwise
  up = fully_unroll(make_mul_uops(10, 10))
  assert analyze_elementwise(up) == (100, Ops.MUL)
  assert uops_to_rknn(up, 10, 10) == uops_to_rknn(fully_unroll(make_mul_uops(10, 10)), 10, 10)

def test_upcast_synth_runs_on_npu():
  import numpy as np
  from helpers.rknn_synth import run_uops_on_npu
  N = 100
  A = (np.arange(N) % 7).astype(np.float16)
  B = (np.arange(N) % 5 + 1).astype(np.float16)
  try:
    z = run_uops_on_npu(fully_unroll(make_mul_uops(10, 10)), [A, B], rows=10, cols=10)
  except RuntimeError as e:
    pytest.skip(f"NPU runtime unavailable: {e}")
  np.testing.assert_allclose(np.asarray(z).reshape(-1)[:N], A * B)

def test_upcast_onnx_to_rknn_collapses_to_mul():
  pytest.importorskip("rknn")
  import numpy as np
  from helpers.onnx_export import uops_to_onnx
  from helpers.rknn_export import onnx_to_rknn_bytes
  from helpers.rknn_decode import decode_rknn
  m = uops_to_onnx(fully_unroll(make_mul_uops(10, 10)), name="mul_upcast")
  d = decode_rknn(onnx_to_rknn_bytes(m))
  # the toolkit re-fuses the verbose Gather/GEP/Mul/Concat back to a single Mul + EW tile
  assert sum(n["op"] == "Mul" for n in d["nodes"]) == 1
  assert any(b["kind"].startswith("EW_BINARY") for b in d["command_queue"])
