#!/usr/bin/env python3
"""Build a 10x10 N-input chained-Add reference .rknn with tensor names matching
rknn_flatbuf._io / _memory_plan_for (inputs a,b,c..., intermediates t1,t2...,
output = letter after last input).  Used to mint embedded template bodies.
Usage: gen_ref_addn.py N_INPUTS
"""
import sys, string
from pathlib import Path
import onnx
from onnx import TensorProto, helper
from rknn.api import RKNN

ROOT = Path(__file__).resolve().parent
TARGET = "rk3588"
R = C = 10


def checked(ret, step):
    if ret != 0:
        raise RuntimeError(f"{step} failed: {ret}")


def main():
    ni = int(sys.argv[1])
    shape = [R, C]
    ins = list(string.ascii_lowercase[:ni])         # a,b,c,...
    outp = string.ascii_lowercase[ni]               # letter after last input
    n_inter = ni - 2
    inter = ["t"] if n_inter == 1 else [f"t{k}" for k in range(1, n_inter + 1)]

    inputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT16, shape) for n in ins]
    out = helper.make_tensor_value_info(outp, TensorProto.FLOAT16, shape)

    nodes, prev = [], ins[0]
    for i in range(1, ni):
        o = outp if i == ni - 1 else inter[i - 1]
        nodes.append(helper.make_node("Add", [prev, ins[i]], [o], name=f"add{i}"))
        prev = o

    graph = helper.make_graph(nodes, f"add{ni}", inputs, [out])
    model = helper.make_model(graph, producer_name="rknn_ref",
                              opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx_path = ROOT / f"add{ni}_10x10.onnx"
    rknn_path = ROOT / f"add{ni}_10x10.rknn"
    onnx.save(model, onnx_path)

    rknn = RKNN(verbose=False)
    try:
        checked(rknn.config(target_platform=TARGET, float_dtype="float16"), "config")
        checked(rknn.load_onnx(model=str(onnx_path), inputs=ins,
                               input_size_list=[shape] * ni), "load_onnx")
        checked(rknn.build(do_quantization=False), "build")
        checked(rknn.export_rknn(str(rknn_path)), "export_rknn")
    finally:
        rknn.release()
    print(f"wrote {rknn_path} ({rknn_path.stat().st_size} bytes) inputs={ni} "
          f"names={ins}->{outp} inter={inter}")


if __name__ == "__main__":
    main()
