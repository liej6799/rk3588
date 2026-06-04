"""Full view: upcasted UOps -> ONNX -> rknn -> NPU, decoded at each stage.

Takes the vectorized (UPCAST=4) multiply UOps, lowers them to ONNX (CAST/vec-LOAD/GEP ->
Gather, STACK -> Concat), checks them with onnxruntime, compiles to .rknn with the
toolkit, decodes the raw NPU command queue, and runs it on the NPU. Shows how the
upcasted UOp graph maps all the way down to the raw rknn command.

    .venv/bin/python3 -m mul_ops.pipeline          # 10x10 = 100
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

from collections import Counter
import numpy as np
from helpers.unroll import upcast_elementwise
from helpers.onnx_export import uops_to_onnx
from helpers.rknn_export import onnx_to_rknn_bytes
from helpers.rknn_decode import decode_rknn
from helpers.rknn_run import run_rknn
from mul_ops.uops import make_mul_uops

def main():
  N = 100
  up = upcast_elementwise(make_mul_uops(10, 10), 4)
  print(f"[1] UPCAST UOPS  ({len(up)}): {dict(Counter(u.op.name for u in up))}")

  model = uops_to_onnx(up, name="mul_upcast_10x10")
  print(f"[2] ONNX         : {dict(Counter(n.op_type for n in model.graph.node))}")

  import onnxruntime as ort
  A = (np.arange(N) % 7).astype(np.float16)
  B = (np.arange(N) % 5 + 1).astype(np.float16)
  sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
  ort_out = sess.run(None, {"input1": A, "input2": B})[0].reshape(-1)[:N].astype(np.float32)
  print(f"    onnxruntime == a*b: {np.allclose(ort_out, A.astype(np.float32) * B.astype(np.float32))}")

  blob = onnx_to_rknn_bytes(model)
  d = decode_rknn(blob)
  print(f"[3] RKNN nodes   : {dict(Counter(n['op'] for n in d['nodes']))}")
  print(f"    RKNN command : {dict(Counter(b['kind'] for b in d['command_queue']))} "
        f"({len(d['command_queue'])} blocks)")

  npu = np.array(run_rknn(blob, [A, B], target="rk3588")[0]).reshape(-1)[:N].astype(np.float32)
  ok = np.allclose(npu, A.astype(np.float32) * B.astype(np.float32))
  print(f"[4] NPU == a*b   : {ok}")
  return 0 if ok else 1

if __name__ == "__main__":
  sys.exit(main())
