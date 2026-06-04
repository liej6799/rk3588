"""2D nested-loop add (N=100): build the UOps, unroll, synthesize a .rknn, run on NPU.

Exercises the UOp operator syntax (*, +, implicit-load INDEX, instance .sink) and the
toolkit-free synth path on a kernel produced by two nested RANGE loops with a computed
flat index i*cols + j. rknn-toolkit2 is needed only to run on the NPU, not to build.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import numpy as np
import pytest
from helpers.uop import Ops
from helpers.unroll import fully_unroll
from helpers.interp import run_uops
from helpers.rknn_synth import uops_to_rknn, analyze_add
from add2d_ops.uops import make_add2d_uops

def test_build_structure():
  uops = make_add2d_uops(10, 10)
  assert sum(u.op is Ops.RANGE for u in uops) == 2        # two nested loops
  assert sum(u.op is Ops.STORE for u in uops) == 1
  st = next(u for u in uops if u.op is Ops.STORE)
  val = st.src[1]
  assert val.op is Ops.ADD and [s.op for s in val.src] == [Ops.LOAD, Ops.LOAD]

def test_scalar_2d_index_fold():
  unr = fully_unroll(make_add2d_uops(3, 3))               # N=9 (not div 4) -> scalar unroll
  assert sum(u.op is Ops.RANGE for u in unr) == 0
  assert sum(u.op is Ops.STORE for u in unr) == 9         # 3*3 iterations, no vectorization
  assert not any(u.op is Ops.STACK for u in unr)
  assert analyze_add(unr) == 9
  # the computed index i*3 + j folds to constants covering 0..8 exactly once
  assert sorted(u.arg for u in unr if u.op is Ops.CONST) == list(range(9))

def test_auto_upcast_vectorizes():
  unr = fully_unroll(make_add2d_uops(10, 10))             # auto-upcast (N=100 -> vec4)
  assert sum(u.op is Ops.STORE for u in unr) == 25 and any(u.op is Ops.STACK for u in unr)
  assert analyze_add(unr) == 100                          # still recognized as a 100-elem add

def test_interp_matches_add():
  unr = fully_unroll(make_add2d_uops(10, 10))
  N = 100
  A = [float(i % 13) for i in range(N)]
  B = [float(i % 5) for i in range(N)]
  out = [0.0] * N
  run_uops(unr, {0: out, 1: A, 2: B})
  assert out == [A[i] + B[i] for i in range(N)]          # pure-python interp cross-check

def test_synth_runs_on_npu():
  from helpers.rknn_run import run_rknn
  N = 100
  blob = uops_to_rknn(fully_unroll(make_add2d_uops(10, 10)), 10, 10)
  A = (np.arange(N) % 13).astype(np.float16)
  B = (np.arange(N) % 5).astype(np.float16)
  try:
    out = run_rknn(blob, [A, B], target="rk3588")
  except RuntimeError as e:
    pytest.skip(f"NPU runtime unavailable: {e}")
  got = np.array(out[0]).reshape(-1)[:N]
  np.testing.assert_allclose(got, A + B)                  # toolkit-free 2D-loop add on NPU
