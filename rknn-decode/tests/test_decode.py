"""Disassemble the add .rknn and check the decoded container / graph / command queue.

Needs rknn-toolkit2 to build the sample model; the decode itself is pure-Python and
offline (no NPU). Run with the project venv:
    .venv/bin/python3 -m pytest tests/test_decode.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import pytest
from helpers.rknn_decode import decode_rknn, split_container, decode_reg, REG_NAMES

pytest.importorskip("rknn")                            # only needed to build the sample
from add_ops.viz import build_rknn

@pytest.fixture(scope="module")
def model_bytes():
  return build_rknn()

def test_container_splits(model_bytes):
  version, body, trailer = split_container(model_bytes)
  assert version == 6
  assert len(body) > 0 and model_bytes[:4] == b"RKNN"
  assert 0x40 + len(body) + len(trailer) == len(model_bytes)

def test_decode_graph(model_bytes):
  d = decode_rknn(model_bytes)
  assert d["target_platform"] == "rk3588" and d["framework"] == "ONNX"
  ops = [n["op"] for n in d["nodes"]]
  assert ops[0] == "InputOperator" and ops[-1] == "OutputOperator"
  assert ops.count("Add") == 25                        # 25 single-element add nodes
  # tensors carry NC1HWC2 native shapes + a byte size
  named = [t for t in d["tensors"] if t["name"]]
  assert any(t["native"] and len(t["native"]) == 5 for t in named)

def test_command_queue(model_bytes):
  q = decode_rknn(model_bytes)["command_queue"]
  assert len(q) > 0
  kinds = [b["kind"] for b in q]
  assert any(k.startswith("EW_BINARY") for k in kinds)  # the add tiles
  # every block is a run of NPU-target register words, each decodable + named
  for b in q:
    assert b["n_words"] >= 20
    for target, off, name, val in b["regs"]:
      assert isinstance(off, int) and isinstance(val, int) and isinstance(name, str)

def test_reg_names_and_decode():
  assert REG_NAMES[0x4030] == "DPU_DATA_CUBE_WIDTH"     # parsed from refs/rkt_registers.h
  # word = target<<48 | value<<16 | reg
  word = (0x1001 << 48) | (0x1234 << 16) | 0x4030
  tname, off, name, val = decode_reg(word)
  assert tname == "DPU" and off == 0x4030 and name == "DPU_DATA_CUBE_WIDTH" and val == 0x1234
