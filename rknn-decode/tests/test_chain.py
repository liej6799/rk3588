"""Toolkit-free element-wise op-chains with MIXED ops in a single .rknn.

chain_to_rknn builds one body with per-op EW tiles (Add/Sub/Mul/Div), e.g. ["Mul","Add"]
is the fused multiply-accumulate a*b+c. No ONNX, no toolkit. Verified on the NPU.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

from collections import Counter
import numpy as np
import pytest
from helpers.uop import Ops
from helpers.rknn_synth import chain_to_rknn, run_chain_on_npu
from helpers.rknn_decode import decode_rknn

N = 16

def test_chain_validation():
  with pytest.raises(ValueError):
    chain_to_rknn(["Mul", "Pow"], N)                       # unsupported op
  with pytest.raises(ValueError):
    run_chain_on_npu(["Mul", "Add"], [np.zeros(N, np.float16)] * 2)   # needs 3 inputs
  with pytest.raises(NotImplementedError):
    chain_to_rknn(["Mul"] * 7, N)                         # 8 inputs not supported (no rc template)

def test_mulacc_chain_decodes_mixed_tiles():
  d = decode_rknn(chain_to_rknn(["Mul", "Add"], N, 4, 4))   # a*b + c
  assert sum(n["op"] == "Mul" for n in d["nodes"]) == 1
  assert sum(n["op"] == "Add" for n in d["nodes"]) == 1
  cfgs = Counter(next(v for t, o, _n, v in b["regs"] if (t, o) == ("DPU", 0x4070))
                 for b in d["command_queue"] if b["kind"].startswith("EW_BINARY"))
  assert len(cfgs) == 2                                     # two distinct ops (mul + add) in one body
  mul_cfg, add_cfg = sorted(cfgs)
  assert (mul_cfg & 0xF) == 0x4 and (add_cfg & 0xF) == 0x0  # ...c4 (mul) vs ...c0 (add)

import operator
_OP = {"Add": operator.add, "Sub": operator.sub, "Mul": operator.mul}

def _expected(ops, ins):
  acc = ins[0].astype(np.float32)
  for op, x in zip(ops, ins[1:]):
    acc = _OP[op](acc, x.astype(np.float32))             # left-associated chain
  return acc

@pytest.mark.parametrize("ops", [
  ["Mul", "Add"],                # 3 inputs: MULACC a*b+c (mixed mul+add in one rknn)
  ["Add", "Mul"],                # 3 inputs: mixed add+mul
  ["Mul", "Add", "Sub"],         # 4 inputs, mixed
  ["Mul", "Add", "Mul", "Add"],  # 5 inputs, mixed
  ["Add", "Add", "Add", "Add", "Add"],   # 6 inputs (chain)
  ["Mul", "Add"] * 3,            # 7 inputs, mixed
])
def test_chain_runs_on_npu(ops):
  pytest.importorskip("rknn")
  rng = np.random.default_rng(1)
  ins = [rng.integers(0, 4, N).astype(np.float16) for _ in range(len(ops) + 1)]
  try:
    out = run_chain_on_npu(ops, ins)
  except RuntimeError as e:
    pytest.skip(f"NPU runtime unavailable: {e}")
  np.testing.assert_allclose(np.asarray(out).reshape(-1)[:N], _expected(ops, ins))
