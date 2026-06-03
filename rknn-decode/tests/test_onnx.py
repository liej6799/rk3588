"""Build an ONNX graph from the unrolled add UOps and check it computes a + b."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import numpy as np
import onnx
import onnxruntime as ort
from helpers.unroll import fully_unroll
from helpers.onnx_export import uops_to_onnx
from add_ops.uops import load_uops

N = 25
A = (np.arange(N) % 100).astype(np.float16)
B = np.full(N, 10, dtype=np.float16)

def _model():
  return uops_to_onnx(fully_unroll(load_uops()), name="add_5x5")

def test_onnx_model_is_valid():
  model = _model()
  onnx.checker.check_model(model)
  # one Gather per loaded element (2 inputs x 25), and a single Concat producing the output
  ops = [n.op_type for n in model.graph.node]
  assert ops.count("Gather") == 2 * N
  assert ops.count("Add") == N
  assert ops.count("Concat") == 1
  assert [o.name for o in model.graph.output] == ["output"]

def test_onnx_runs_and_matches_add():
  sess = ort.InferenceSession(_model().SerializeToString(), providers=["CPUExecutionProvider"])
  out = sess.run(None, {"input1": A, "input2": B})[0]
  np.testing.assert_array_equal(out, A + B)

if __name__ == "__main__":
  import traceback
  tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
  failed = 0
  for fn in tests:
    try:
      fn(); print(f"PASS {fn.__name__}")
    except Exception:
      failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
  print(f"\n{len(tests) - failed}/{len(tests)} passed")
  sys.exit(1 if failed else 0)
