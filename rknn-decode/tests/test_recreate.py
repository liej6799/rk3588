"""Recreate a .rknn from its decoded representation and compare to the library output.

The container header is rebuilt from scratch and the NPU command stream is re-emitted
from the decoded command queue; the result must be byte-identical to the library model,
and must still run correctly on the NPU.

Run with the project venv:  .venv/bin/python3 -m pytest tests/test_recreate.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import hashlib
import numpy as np
import pytest
from helpers.rknn_encode import encode_container, encode_reg, recreate_rknn
from helpers.rknn_decode import decode_rknn, split_container, decode_reg

pytest.importorskip("rknn")                            # needed to build the library model
from helpers.kernel import compile_rknn
from add_ops.uops import load_uops
def build_rknn():
  return compile_rknn(load_uops(), name="add_5x5")     # the library-generated .rknn

@pytest.fixture(scope="module")
def lib_model():
  return build_rknn()                                  # the library-generated .rknn

def test_encode_reg_roundtrips():
  word = (0x1001 << 48) | (0x1234 << 16) | 0x4030
  tname, off, _name, val = decode_reg(word)
  assert encode_reg(tname, off, val) == word           # decode then encode is identity

def test_container_rebuilds_from_scratch(lib_model):
  version, body, trailer = split_container(lib_model)
  assert encode_container(version, body, trailer) == lib_model   # header rebuilt from scratch

def test_recreate_is_byte_identical(lib_model):
  recreated = recreate_rknn(lib_model)
  assert recreated == lib_model
  assert hashlib.sha256(recreated).hexdigest() == hashlib.sha256(lib_model).hexdigest()

def test_recreated_decodes_the_same(lib_model):
  a, b = decode_rknn(recreate_rknn(lib_model)), decode_rknn(lib_model)
  assert [n["op"] for n in a["nodes"]] == [n["op"] for n in b["nodes"]]
  assert [x["kind"] for x in a["command_queue"]] == [x["kind"] for x in b["command_queue"]]

def test_recreated_runs_on_npu(lib_model):
  from helpers.rknn_run import run_rknn
  N = 25
  A = (np.arange(N) % 100).astype(np.float16)
  B = np.full(N, 10, dtype=np.float16)
  try:
    out = run_rknn(recreate_rknn(lib_model), [A, B], target="rk3588")
  except RuntimeError as e:
    pytest.skip(f"NPU runtime unavailable: {e}")
  got = np.array(out[0]).reshape(-1)[:N]
  np.testing.assert_allclose(got, A + B)               # recreated model computes a + b
