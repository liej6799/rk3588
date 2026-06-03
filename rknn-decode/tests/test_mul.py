"""Element-wise multiply (z = a * b): build the 2D-loop UOps, unroll, synth, run on NPU.

The same toolkit-free path as the add, with the EW_CFG ALU op retargeted to the NPU
multiply datapath. Verified for N=100 (10x10). rknn-toolkit2 is needed only to run on
the NPU, not to build.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import numpy as np
import pytest
from helpers.uop import Ops
from helpers.unroll import fully_unroll
from helpers.interp import run_uops
from helpers.rknn_synth import analyze_elementwise, uops_to_rknn, run_uops_on_npu, _EW_CFG
from helpers.rknn_decode import decode_rknn
from mul_ops.uops import make_mul_uops

def test_build_and_unroll_is_mul():
  uops = make_mul_uops(10, 10)
  assert sum(u.op is Ops.RANGE for u in uops) == 2
  st = next(u for u in uops if u.op is Ops.STORE)
  assert st.src[1].op is Ops.MUL                          # value is a multiply
  unr = fully_unroll(uops)
  N, op = analyze_elementwise(unr)
  assert N == 100 and op is Ops.MUL
  assert sum(u.op is Ops.STORE for u in unr) == 100
  assert sum(u.op is Ops.MUL for u in unr) == 100         # one product per element
  # every store is MUL(LOAD(INDEX(a,k)), LOAD(INDEX(b,k)))
  for st in (u for u in unr if u.op is Ops.STORE):
    v = st.src[1]
    assert v.op is Ops.MUL and all(s.op is Ops.LOAD for s in v.src)

def test_interp_matches_mul():
  unr = fully_unroll(make_mul_uops(10, 10))
  N = 100
  A = [float(i % 7) for i in range(N)]
  B = [float(i % 5 + 1) for i in range(N)]
  out = [0.0] * N
  run_uops(unr, {0: out, 1: A, 2: B})
  assert out == [A[i] * B[i] for i in range(N)]

def test_synth_selects_mul_ewop():
  # the synthesized .rknn must carry the MUL EW_CFG on its EW tiles (offline check)
  d = decode_rknn(uops_to_rknn(fully_unroll(make_mul_uops(10, 10)), 10, 10))
  ew = [b for b in d["command_queue"] if b["kind"].startswith("EW_BINARY")]
  assert ew
  for b in ew:
    cfg = next(v for t, off, _n, v in b["regs"] if (t, off) == ("DPU", 0x4070))
    assert cfg == _EW_CFG[Ops.MUL]

def test_mul_runs_on_npu():
  N = 100
  A = (np.arange(N) % 7).astype(np.float16)
  B = (np.arange(N) % 5 + 1).astype(np.float16)
  try:
    z = run_uops_on_npu(make_mul_uops(10, 10), [A, B])
  except RuntimeError as e:
    pytest.skip(f"NPU runtime unavailable: {e}")
  np.testing.assert_allclose(np.asarray(z).reshape(-1)[:N], A * B)
