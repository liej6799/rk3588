"""End-to-end test of the add_ops example: load -> unroll -> execute -> check a + b.

Replaces eyeballing `python3 -m add_ops.run`. Runnable via pytest or directly
(`python3 tests/test_add.py`).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import pytest
from helpers.unroll import fully_unroll
from helpers.interp import run_uops
from add_ops.uops import load_uops

# inputs mirror test/backend/test_add.py (5x5, fp16-exact integer values)
N = 25
A = [float(i % 100) for i in range(N)]
B = [10.0] * N

def _run():
  uops = fully_unroll(load_uops())
  out = [0.0] * N
  run_uops(uops, {0: out, 1: A, 2: B})
  return out

def test_add_computes_a_plus_b():
  assert _run() == [A[i] + B[i] for i in range(N)]

def test_add_writes_every_output():
  # all outputs were touched (no element left at the sentinel) and inputs unchanged
  out = _run()
  assert out == [A[i] + 10.0 for i in range(N)]
  assert B == [10.0] * N and A == [float(i % 100) for i in range(N)]

def test_run_uops_requires_unrolled_graph():
  with pytest.raises(AssertionError):
    run_uops(load_uops(), {0: [0.0] * N, 1: A, 2: B})   # still has a RANGE
