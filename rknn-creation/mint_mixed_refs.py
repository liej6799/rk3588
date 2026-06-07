#!/usr/bin/env python3
"""Mint N-input parallel mixed CPU(And)+NPU(Add) references via the toolkit.

Builds, per topology, a parallel graph with a single N-input CPU AND branch and a
single M-input NPU ADD branch, exports the .rknn, so we can decode how the mixed
RC scales with per-branch arity (the piece the 2-input-only references can't show).
"""
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from rknn.api import RKNN

ROOT = Path(__file__).resolve().parent
SHAPE = [1, 4]


def chain(op, ins, out, dtype, prefix):
    """Left-associated chain: out = ((i0 op i1) op i2) ... ; returns node list."""
    nodes = []
    acc = ins[0]
    for k in range(1, len(ins)):
        res = out if k == len(ins) - 1 else f"{prefix}_t{k}"
        nodes.append(helper.make_node(op, [acc, ins[k]], [res], name=f"{op}:{prefix}{k}"))
        acc = res
    return nodes


def build(n_cpu, n_npu, out_path):
    cpu_ins = [f"a{i}" for i in range(n_cpu)]
    npu_ins = [f"x{i}" for i in range(n_npu)]
    vis = [helper.make_tensor_value_info(nm, TensorProto.BOOL, SHAPE) for nm in cpu_ins]
    vis += [helper.make_tensor_value_info(nm, TensorProto.FLOAT, SHAPE) for nm in npu_ins]
    out1 = helper.make_tensor_value_info("out1", TensorProto.BOOL, SHAPE)
    out2 = helper.make_tensor_value_info("out2", TensorProto.FLOAT, SHAPE)
    nodes = chain("And", cpu_ins, "out1", TensorProto.BOOL, "and")
    nodes += chain("Add", npu_ins, "out2", TensorProto.FLOAT, "add")
    graph = helper.make_graph(nodes, f"mix_{n_cpu}and_{n_npu}add", vis, [out1, out2])
    model = helper.make_model(graph, producer_name="mint_mixed_refs",
                              opset_imports=[helper.make_opsetid("", 15)])
    onnx_path = str(ROOT / f"_mint_{n_cpu}and_{n_npu}add.onnx")
    onnx.save(model, onnx_path)

    rk = RKNN(verbose=False)
    try:
        if rk.config(target_platform="rk3588") != 0:
            print(f"[{n_cpu}and+{n_npu}add] config FAIL"); return False
        if rk.load_onnx(model=onnx_path) != 0:
            print(f"[{n_cpu}and+{n_npu}add] load FAIL"); return False
        if rk.build(do_quantization=False) != 0:
            print(f"[{n_cpu}and+{n_npu}add] build FAIL"); return False
        if rk.export_rknn(out_path) != 0:
            print(f"[{n_cpu}and+{n_npu}add] export FAIL"); return False
    finally:
        rk.release()
    print(f"[{n_cpu}and+{n_npu}add] OK -> {out_path}")
    return True


if __name__ == "__main__":
    topos = [(3, 2), (2, 3), (4, 2), (2, 4), (3, 3)]
    if len(sys.argv) > 1:
        topos = [tuple(int(x) for x in a.split(",")) for a in sys.argv[1:]]
    for nc, nn in topos:
        build(nc, nn, str(ROOT / f"_ref_{nc}and_{nn}add.rknn"))
