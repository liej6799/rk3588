#!/usr/bin/env python3
"""Build mixed CPU(And)/NPU(Add) reference .rknn models with the toolkit, to
study how the vendor schedules a graph that contains both a CPU-fallback op and
an NPU element-wise op.

We try several graph shapes and report which the toolkit accepts/builds.
"""
import sys
import onnx
from onnx import TensorProto, helper
from rknn.api import RKNN


def build(name, graph_fn, dtypes):
    onnx_path = f"/tmp/{name}.onnx"
    out_path = f"/tmp/{name}_ref.rknn"
    inputs, outputs, nodes, initializers = graph_fn()
    graph = helper.make_graph(nodes, name, inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, producer_name=name,
                              opset_imports=[helper.make_opsetid("", 15)])
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"[{name}] onnx check WARN: {e}")
    onnx.save(model, onnx_path)
    rknn = RKNN(verbose=True)
    try:
        r = rknn.config(target_platform="rk3588")
        print(f"[{name}] config:", r)
        r = rknn.load_onnx(model=onnx_path)
        print(f"[{name}] load_onnx:", r)
        if r != 0:
            return False
        r = rknn.build(do_quantization=False)
        print(f"[{name}] build:", r)
        if r != 0:
            return False
        r = rknn.export_rknn(out_path)
        print(f"[{name}] export:", r, out_path)
        return r == 0
    finally:
        rknn.release()


def g_andadd_bool_then_float():
    # a,b bool -> t = a AND b ; cast t -> float ; out = tf + e (float)
    a = helper.make_tensor_value_info("a", TensorProto.BOOL, [1, 4])
    b = helper.make_tensor_value_info("b", TensorProto.BOOL, [1, 4])
    e = helper.make_tensor_value_info("e", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])
    nodes = [
        helper.make_node("And", ["a", "b"], ["t"], name="And:and1"),
        helper.make_node("Cast", ["t"], ["tf"], to=TensorProto.FLOAT, name="Cast:cast1"),
        helper.make_node("Add", ["tf", "e"], ["out"], name="Add:add1"),
    ]
    return [a, b, e], [out], nodes, []


def g_add_then_and():
    # x,y float -> s = x + y ; greater than 0 -> bool ; out = sb AND c
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    c = helper.make_tensor_value_info("c", TensorProto.BOOL, [1, 4])
    out = helper.make_tensor_value_info("out", TensorProto.BOOL, [1, 4])
    zero = helper.make_tensor("zero", TensorProto.FLOAT, [1, 4], [0, 0, 0, 0])
    nodes = [
        helper.make_node("Add", ["x", "y"], ["s"], name="Add:add1"),
        helper.make_node("Greater", ["s", "zero"], ["sb"], name="Greater:gt1"),
        helper.make_node("And", ["sb", "c"], ["out"], name="And:and1"),
    ]
    return [x, y, c], [out], nodes, [zero]


def g_parallel():
    # two independent ops: out1 = a AND b (bool); out2 = x + y (float)
    a = helper.make_tensor_value_info("a", TensorProto.BOOL, [1, 4])
    b = helper.make_tensor_value_info("b", TensorProto.BOOL, [1, 4])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    out1 = helper.make_tensor_value_info("out1", TensorProto.BOOL, [1, 4])
    out2 = helper.make_tensor_value_info("out2", TensorProto.FLOAT, [1, 4])
    nodes = [
        helper.make_node("And", ["a", "b"], ["out1"], name="And:and1"),
        helper.make_node("Add", ["x", "y"], ["out2"], name="Add:add1"),
    ]
    return [a, b, x, y], [out1, out2], nodes, []


def g_and_add_and_chain():
    # Full chain: And -> Cast -> Add -> Greater -> And
    # a,b bool -> t1 = a AND b; cast to float; + x (float); > 0.5 -> bool; AND c -> out
    a = helper.make_tensor_value_info("a", TensorProto.BOOL, [1, 4])
    b = helper.make_tensor_value_info("b", TensorProto.BOOL, [1, 4])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    c = helper.make_tensor_value_info("c", TensorProto.BOOL, [1, 4])
    out = helper.make_tensor_value_info("out", TensorProto.BOOL, [1, 4])
    thresh = helper.make_tensor("thresh", TensorProto.FLOAT, [1, 4], [0.5, 0.5, 0.5, 0.5])
    nodes = [
        helper.make_node("And", ["a", "b"], ["t1"], name="And:and1"),
        helper.make_node("Cast", ["t1"], ["t1f"], to=TensorProto.FLOAT, name="Cast:cast1"),
        helper.make_node("Add", ["t1f", "x"], ["t2"], name="Add:add1"),
        helper.make_node("Greater", ["t2", "thresh"], ["t2b"], name="Greater:gt1"),
        helper.make_node("And", ["t2b", "c"], ["out"], name="And:and2"),
    ]
    return [a, b, x, c], [out], nodes, [thresh]


def g_parallel_2and_1add():
    # Three independent branches: out1=a&b, out2=c&d, out3=x+y
    a = helper.make_tensor_value_info("a", TensorProto.BOOL, [1, 4])
    b = helper.make_tensor_value_info("b", TensorProto.BOOL, [1, 4])
    c = helper.make_tensor_value_info("c", TensorProto.BOOL, [1, 4])
    d = helper.make_tensor_value_info("d", TensorProto.BOOL, [1, 4])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    out1 = helper.make_tensor_value_info("out1", TensorProto.BOOL, [1, 4])
    out2 = helper.make_tensor_value_info("out2", TensorProto.BOOL, [1, 4])
    out3 = helper.make_tensor_value_info("out3", TensorProto.FLOAT, [1, 4])
    nodes = [
        helper.make_node("And", ["a", "b"], ["out1"], name="And:and1"),
        helper.make_node("And", ["c", "d"], ["out2"], name="And:and2"),
        helper.make_node("Add", ["x", "y"], ["out3"], name="Add:add1"),
    ]
    return [a, b, c, d, x, y], [out1, out2, out3], nodes, []


CASES = {
    "andadd_b2f": g_andadd_bool_then_float,
    "add2and":    g_add_then_and,
    "parallel":   g_parallel,
    "and_add_and_chain": g_and_add_and_chain,
    "parallel_2and_1add": g_parallel_2and_1add,
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in CASES.items():
        if which not in ("all", name):
            continue
        print("=" * 60)
        ok = build(name, fn, None)
        print(f"[{name}] RESULT: {'OK' if ok else 'FAIL'}")
