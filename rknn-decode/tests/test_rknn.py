"""Build a .rknn from the unrolled add UOps and check the exported artifact.

Needs rknn-toolkit2 (and its onnx/onnxruntime deps); skipped if not installed.
Run with the project venv:  .venv/bin/python3 -m pytest tests/test_rknn.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import glob
import pytest
pytest.importorskip("rknn")                            # rknn-toolkit2
from helpers.rknn_export import onnx_to_rknn, onnx_to_rknn_bytes
from add_ops.viz import build_model

def test_export_bytes_in_memory():
  before = set(glob.glob("/dev/shm/rknn_*"))
  blob = onnx_to_rknn_bytes(build_model())
  assert isinstance(blob, bytes) and blob[:4] == b"RKNN"   # got the .rknn back as bytes
  assert set(glob.glob("/dev/shm/rknn_*")) == before       # tmpfs scratch dir cleaned up, nothing persists

def test_export_writes_rknn_file(tmp_path):
  out = str(tmp_path / "add_5x5.rknn")
  assert onnx_to_rknn(build_model(), out) == out
  assert os.path.exists(out)
  with open(out, "rb") as f:
    assert f.read(4) == b"RKNN"                         # exported file carries the RKNN magic

def test_quantization_without_dataset_raises(tmp_path):
  out = str(tmp_path / "unused.rknn")
  with pytest.raises(ValueError):
    onnx_to_rknn(build_model(), out, do_quantization=True)
  assert not os.path.exists(out)                        # bails before writing anything
