"""Synthesize a .rknn straight from unrolled UOps (no ONNX, no toolkit) and run it.

The whole path here is toolkit-free: make_add_uops -> fully_unroll -> uops_to_rknn
(FlatBuffer body + NPU command stream built from scratch) -> run on the NPU. Tested for
1x1, 3x3, 5x5. rknn-toolkit2 is needed only to *run* the result on the NPU, not to build
it (the build asserts no toolkit import is required).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import numpy as np
import pytest
from helpers.unroll import fully_unroll
from helpers.rknn_synth import uops_to_rknn, analyze_add, run_uops_on_npu
from helpers.rknn_decode import decode_rknn, split_container
from add_ops.uops import make_add_uops

SIZES = [(1, 1), (3, 3), (5, 5)]

def _uops(n):
  return fully_unroll(make_add_uops(n))

@pytest.mark.parametrize("rows,cols", SIZES)
def test_analyze_recovers_n(rows, cols):
  assert analyze_add(_uops(rows * cols)) == rows * cols

@pytest.mark.parametrize("rows,cols", SIZES)
def test_build_is_toolkit_free(rows, cols):
  # building the .rknn must not import rknn-toolkit2 (no onnx, no compiler)
  sys.modules.pop("rknn", None)
  blob = uops_to_rknn(_uops(rows * cols), rows, cols)
  assert "rknn" not in sys.modules                     # never touched the toolkit
  assert blob[:4] == b"RKNN"
  version, body, trailer = split_container(blob)
  assert version == 6 and len(body) > 0 and len(trailer) > 0

@pytest.mark.parametrize("rows,cols", SIZES)
def test_synth_decodes_as_add(rows, cols):
  d = decode_rknn(uops_to_rknn(_uops(rows * cols), rows, cols))
  assert d["target_platform"] == "rk3588"
  assert any(b["kind"].startswith("EW_BINARY") for b in d["command_queue"])  # has an add tile

@pytest.mark.parametrize("rows,cols", SIZES)
def test_synth_runs_on_npu(rows, cols):
  from helpers.rknn_run import run_rknn
  N = rows * cols
  blob = uops_to_rknn(_uops(N), rows, cols)
  A = (np.arange(N) % 7).astype(np.float16)
  B = np.full(N, 3, dtype=np.float16)
  try:
    out = run_rknn(blob, [A, B], target="rk3588")
  except RuntimeError as e:
    pytest.skip(f"NPU runtime unavailable: {e}")
  got = np.array(out[0]).reshape(-1)[:N]
  np.testing.assert_allclose(got, A + B)               # toolkit-free model computes a + b

@pytest.mark.parametrize("rows,cols", SIZES)
def test_run_uops_on_npu_one_call(rows, cols):
  # the one-call helper: original (loop) UOps straight to an NPU result
  N = rows * cols
  A = (np.arange(N) % 7).astype(np.float16)
  B = np.full(N, 3, dtype=np.float16)
  try:
    z = run_uops_on_npu(make_add_uops(N), [A, B], rows=rows, cols=cols)
  except RuntimeError as e:
    pytest.skip(f"NPU runtime unavailable: {e}")
  np.testing.assert_allclose(np.asarray(z).reshape(-1)[:N], A + B)

def test_run_uops_on_npu_rejects_wrong_input_count():
  with pytest.raises(ValueError):
    run_uops_on_npu(make_add_uops(4), [np.zeros(4, np.float16)])   # needs exactly 2 inputs
