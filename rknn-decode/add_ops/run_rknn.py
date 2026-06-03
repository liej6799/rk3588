"""add_ops example: run the 5x5 half add on the NPU and compare against ONNX.

Builds the ONNX graph from the unrolled UOps, runs it with onnxruntime (CPU), then
converts it to .rknn (in memory) and runs that on the RK3588 NPU, and checks both
agree with each other and with a + b.

Run from the project root (needs the device runtime + project venv):

    .venv/bin/python3 -m add_ops.run_rknn            # run + compare ONNX vs NPU
    .venv/bin/python3 -m add_ops.run_rknn --graph    # print the ONNX and compiled NPU graphs
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import numpy as np
import onnxruntime as ort
from helpers.onnx_export import print_onnx_graph
from helpers.rknn_run import run_rknn, print_rknn_graph
from add_ops.viz import build_model, build_rknn

# inputs mirror tests/test_onnx.py (5x5, fp16-exact integer values)
N = 25
A = (np.arange(N) % 100).astype(np.float16)
B = np.full(N, 10, dtype=np.float16)

def print_graphs():
  print_onnx_graph(build_model())
  print()
  print("=== compiled NPU graph (rknn) ===")
  print_rknn_graph(build_rknn(), target="rk3588")

def main():
  if "--graph" in sys.argv[1:]:
    print_graphs()
    return 0

  expected = (A + B).astype(np.float32)

  sess = ort.InferenceSession(build_model().SerializeToString(), providers=["CPUExecutionProvider"])
  onnx_out = sess.run(None, {"input1": A, "input2": B})[0].reshape(-1)[:N].astype(np.float32)

  npu_out = np.array(run_rknn(build_rknn(), [A, B], target="rk3588")[0]).reshape(-1)[:N].astype(np.float32)

  print("inputs a :", A.tolist())
  print("inputs b :", B.tolist())
  print("expected :", expected.tolist())
  print("onnx out :", onnx_out.tolist())
  print("npu out  :", npu_out.tolist())
  print("onnx vs npu max abs diff:", float(np.abs(onnx_out - npu_out).max()))
  ok = np.array_equal(onnx_out, npu_out) and np.allclose(npu_out, expected)
  print("MATCH" if ok else "MISMATCH")
  return 0 if ok else 1

if __name__ == "__main__":
  sys.exit(main())
