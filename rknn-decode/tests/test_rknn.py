"""Build a .rknn from the unrolled add UOps and check the exported artifact.

Needs rknn-toolkit2 (and its onnx/onnxruntime deps); skipped if not installed.
Run with the project venv:  .venv/bin/python3 -m pytest tests/test_rknn.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import pytest
pytest.importorskip("rknn")                            # rknn-toolkit2
from helpers.rknn_export import onnx_to_rknn
from add_ops.viz import build_model

def test_export_writes_rknn_file(tmp_path):
  out = str(tmp_path / "add_5x5.rknn")
  assert onnx_to_rknn(build_model(), out) == out
  assert os.path.exists(out)
  with open(out, "rb") as f:
    assert f.read(4) == b"RKNN"                         # exported file carries the RKNN magic

def test_quantization_without_dataset_raises():
  with pytest.raises(ValueError):
    onnx_to_rknn(build_model(), "unused.rknn", do_quantization=True)
