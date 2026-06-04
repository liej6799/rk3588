"""Matmul UOps (REDUCE): build, unroll the contraction, verify the math in pure Python.

This kernel is a reduction (out = a @ b), not an element-wise add. It exercises
Ops.REDUCE / .reduce() and confirms:
  - fully_unroll expands the REDUCE loop into an explicit sum-of-products,
  - the pure-Python interpreter matches numpy matmul,
  - the toolkit-free .rknn synthesizer correctly REJECTS it (add-only).
No NPU is involved.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import numpy as np
import pytest
from helpers.uop import Ops
from helpers.unroll import fully_unroll
from helpers.interp import run_uops
from helpers.rknn_synth import uops_to_rknn
from matmul_ops.uops import make_matmul_uops

def test_build_has_reduce():
  uops = make_matmul_uops(8)
  assert sum(u.op is Ops.RANGE for u in uops) == 3        # i, j loops + k reduce
  assert sum(u.op is Ops.REDUCE for u in uops) == 1
  assert sum(u.op is Ops.STORE for u in uops) == 1

def test_unroll_expands_reduce_to_mulacc():
  # matches tinygrad's r_8_8_8 (UNROLL axis=0): the reduce fuses into MUL + MULACC (FMA)
  unr = fully_unroll(make_matmul_uops(8))
  assert sum(u.op is Ops.RANGE for u in unr) == 0
  assert sum(u.op is Ops.REDUCE for u in unr) == 0        # the contraction is expanded out
  assert sum(u.op is Ops.STORE for u in unr) == 64        # 8x8 outputs
  assert sum(u.op is Ops.MUL for u in unr) == 64          # one MUL seeds each output's sum
  assert sum(u.op is Ops.MULACC for u in unr) == 64 * 7   # then 7 fused multiply-accumulates
  assert sum(u.op is Ops.ADD for u in unr) == 0
  # CSE dedups the shared operands: a[i,k] and b[k,j] each load once (64 + 64)
  assert sum(u.op is Ops.LOAD for u in unr) == 128

@pytest.mark.parametrize("n", [2, 4, 8])
def test_interp_matches_numpy_matmul(n):
  unr = fully_unroll(make_matmul_uops(n))
  rng = np.random.default_rng(n)
  A = rng.integers(0, 5, size=n * n).astype(np.float64)
  B = rng.integers(0, 5, size=n * n).astype(np.float64)
  out = [0.0] * (n * n)
  run_uops(unr, {0: out, 1: list(A), 2: list(B)})
  ref = (A.reshape(n, n) @ B.reshape(n, n)).reshape(-1)
  assert np.allclose(np.array(out), ref)

def test_synth_rejects_matmul():
  with pytest.raises(ValueError, match="element-wise add|matmul"):
    uops_to_rknn(fully_unroll(make_matmul_uops(8)))        # add-only synthesizer

def _matmul_io(n):
  rng = np.random.default_rng(0)
  A = rng.integers(0, 5, n * n).astype(np.float32)        # float matmul; small ints stay exact
  B = rng.integers(0, 5, n * n).astype(np.float32)
  ref = (A.reshape(n, n) @ B.reshape(n, n)).reshape(-1)
  return A, B, ref

def test_matmul_onnx_runs():
  import onnx
  import onnxruntime as ort
  from helpers.onnx_export import uops_to_onnx
  m = uops_to_onnx(fully_unroll(make_matmul_uops(8)), name="matmul_8x8")
  onnx.checker.check_model(m)
  from collections import Counter
  ops = Counter(nd.op_type for nd in m.graph.node)
  # MULACC -> Mul + Add: 64 seed MUL + 448 from MULACC = 512 Mul, 448 Add; 128 CSE'd Gathers
  assert ops["Mul"] == 512 and ops["Add"] == 448 and ops["Gather"] == 128
  A, B, ref = _matmul_io(8)
  sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
  out = sess.run(None, {"input1": A, "input2": B})[0].reshape(-1)[:64].astype(np.float32)
  np.testing.assert_allclose(out, ref)

def test_matmul_runs_on_npu():
  pytest.importorskip("rknn")
  from helpers.onnx_export import uops_to_onnx
  from helpers.rknn_export import onnx_to_rknn_bytes
  from helpers.rknn_run import run_rknn
  blob = onnx_to_rknn_bytes(uops_to_onnx(fully_unroll(make_matmul_uops(8)), name="matmul_8x8"))
  A, B, ref = _matmul_io(8)
  try:
    out = run_rknn(blob, [A, B], target="rk3588")
  except RuntimeError as e:
    pytest.skip(f"NPU runtime unavailable: {e}")
  np.testing.assert_allclose(np.array(out[0]).reshape(-1)[:64].astype(np.float32), ref)
