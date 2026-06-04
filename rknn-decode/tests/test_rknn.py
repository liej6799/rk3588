"""Build a .rknn from the unrolled add UOps and check the exported artifact.

Needs rknn-toolkit2 (and its onnx/onnxruntime deps); skipped if not installed.
Run with the project venv:  .venv/bin/python3 -m pytest tests/test_rknn.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import glob, time, urllib.request
import numpy as np
import pytest
pytest.importorskip("rknn")                            # rknn-toolkit2
from helpers.rknn_export import onnx_to_rknn, onnx_to_rknn_bytes
from helpers.rknn_run import run_rknn, print_rknn_graph
from helpers.kernel import to_onnx, compile_rknn
from helpers.viz import serve
from add_ops.uops import load_uops

def build_model():
  return to_onnx(load_uops(), name="add_5x5")
def build_rknn():
  return compile_rknn(load_uops(), name="add_5x5")
def show_rknn(address=("0.0.0.0", 8080)):
  return serve(build_rknn(), address=address, name="add_5x5.rknn")

def test_export_bytes_in_memory():
  before = set(glob.glob("/dev/shm/rknn_*"))
  blob = onnx_to_rknn_bytes(build_model())
  assert isinstance(blob, bytes) and blob[:4] == b"RKNN"   # got the .rknn back as bytes
  assert set(glob.glob("/dev/shm/rknn_*")) == before       # tmpfs scratch dir cleaned up, nothing persists

def test_show_rknn_hosts_from_memory():
  netron = pytest.importorskip("netron")
  host, port = show_rknn(address=("127.0.0.1", 8098))      # export -> serve the .rknn from bytes, no file
  try:
    time.sleep(1)
    assert urllib.request.urlopen(f"http://{host}:{port}/", timeout=5).getcode() == 200
  finally:
    netron.stop()

def test_export_writes_rknn_file(tmp_path):
  out = str(tmp_path / "add_5x5.rknn")
  assert onnx_to_rknn(build_model(), out) == out
  assert os.path.exists(out)
  with open(out, "rb") as f:
    assert f.read(4) == b"RKNN"                         # exported file carries the RKNN magic

def test_run_on_npu_matches_add():
  N = 25
  A = (np.arange(N) % 100).astype(np.float16)
  B = np.full(N, 10, dtype=np.float16)
  try:
    out = run_rknn(build_rknn(), [A, B], target="rk3588")
  except RuntimeError as e:
    pytest.skip(f"NPU runtime unavailable: {e}")        # not running on the device
  got = np.array(out[0]).reshape(-1)[:N]
  np.testing.assert_allclose(got, A + B)

def test_npu_matches_onnxruntime():
  import onnxruntime as ort
  N = 25
  A = (np.arange(N) % 100).astype(np.float16)
  B = np.full(N, 10, dtype=np.float16)
  try:
    npu = np.array(run_rknn(build_rknn(), [A, B], target="rk3588")[0]).reshape(-1)[:N].astype(np.float32)
  except RuntimeError as e:
    pytest.skip(f"NPU runtime unavailable: {e}")        # not running on the device
  sess = ort.InferenceSession(build_model().SerializeToString(), providers=["CPUExecutionProvider"])
  onnx = sess.run(None, {"input1": A, "input2": B})[0].reshape(-1)[:N].astype(np.float32)
  np.testing.assert_array_equal(npu, onnx)              # NPU bit-matches onnxruntime on this kernel

def test_print_rknn_graph_runs():
  try:
    print_rknn_graph(build_rknn(), target="rk3588")     # prints the compiled layer table via eval_perf
  except RuntimeError as e:
    pytest.skip(f"NPU runtime unavailable: {e}")         # not running on the device

def test_quantization_without_dataset_raises(tmp_path):
  out = str(tmp_path / "unused.rknn")
  with pytest.raises(ValueError):
    onnx_to_rknn(build_model(), out, do_quantization=True)
  assert not os.path.exists(out)                        # bails before writing anything
