"""Tests for the netron visualization helper and the add_ops viz."""
import os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import onnx
import pytest
from helpers.viz import serve
from add_ops.viz import build_model

def test_build_model_is_valid():
  onnx.checker.check_model(build_model())            # the built graph is a valid model

def test_serve_saves_and_hosts(tmp_path):
  netron = pytest.importorskip("netron")
  path = str(tmp_path / "add.onnx")
  host, port = serve(build_model(), address=("127.0.0.1", 8096), path=path)
  try:
    assert os.path.exists(path)                      # serve() wrote the ModelProto to disk
    time.sleep(1)
    assert urllib.request.urlopen(f"http://{host}:{port}/", timeout=5).getcode() == 200
  finally:
    netron.stop()

if __name__ == "__main__":
  import traceback, tempfile, pathlib
  failed = 0
  for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
    try:
      fn(pathlib.Path(tempfile.mkdtemp())) if fn.__code__.co_argcount else fn()
      print(f"PASS {name}")
    except Exception:
      failed += 1; print(f"FAIL {name}"); traceback.print_exc()
  sys.exit(1 if failed else 0)
