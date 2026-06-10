"""ONNX graph -> RKNN container, node-per-op (no NPU compute blocks).

The whole ONNX graph is replicated into the RKNN FlatBuffer node table:
one RKNN node per ONNX node (op name preserved), one tensor per ONNX value.
Initializers and op attributes are carried in the RKNN trailer JSON.

This produces a valid RKNN container for our toolkit-free runtime; loading it
shows which op types the NPU EW path could service (fp16 Add/Sub/Mul/Div) and
which must fall back to CPU.

Usage: python3 onnx_to_rknn.py model.onnx model.rknn
"""
import json
import struct
import sys

import flatbuffers
import numpy as np
import onnx
from onnx import numpy_helper

HEADER_SIZE = 0x40


def _str(b, s):
    return b.CreateString(s)


def _vec(b, vs):
    b.StartVector(4, len(vs), 4)
    for v in reversed(vs):
        b.PrependUint32(v)
    return b.EndVector()


def _ovec(b, offsets):
    b.StartVector(4, len(offsets), 4)
    for o in reversed(offsets):
        b.PrependUOffsetTRelative(o)
    return b.EndVector()


def _tensor(b, name, shape):
    nm = _str(b, name)
    f3 = _vec(b, shape)
    f4 = _vec(b, shape)
    b.StartObject(18)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, f4, 0)
    b.PrependUOffsetTRelativeSlot(3, f3, 0)
    return b.EndObject()


def _node(b, op, name, ins, outs):
    op_s = _str(b, op)
    nm_s = _str(b, name)
    f4 = _vec(b, ins)
    f5 = _vec(b, outs)
    b.StartObject(13)
    b.PrependUOffsetTRelativeSlot(5, f5, 0)
    b.PrependUOffsetTRelativeSlot(4, f4, 0)
    b.PrependUOffsetTRelativeSlot(2, nm_s, 0)
    b.PrependUOffsetTRelativeSlot(1, op_s, 0)
    return b.EndObject()


def convert(onnx_path, rknn_path, npu_ew_model=None):
    m = onnx.load(onnx_path)
    g = m.graph

    tname2idx, shapes = {}, {}
    consts = {}

    def add_value(name, shape=None):
        if name in tname2idx:
            return tname2idx[name]
        tname2idx[name] = len(tname2idx)
        shapes[name] = shape or [1]
        return tname2idx[name]

    for vi in list(g.input) + list(g.output):
        add_value(vi.name, [d.dim_value or 1 for d in vi.type.tensor_type.shape.dim] or [1])
    for init in g.initializer:
        idx = add_value(init.name, list(init.dims) or [1])
        arr = numpy_helper.to_array(init)
        consts[idx] = {"dtype": str(arr.dtype), "data": arr.ravel().tolist()}
    for n in g.node:
        for v in list(n.input) + list(n.output):
            add_value(v)

    node_specs, node_attrs = [], {}
    for i, vi in enumerate(g.input):
        node_specs.append(("InputOperator", f"InputOperator:{vi.name}", [], [tname2idx[vi.name]]))
    for n in g.node:
        ni = len(node_specs)
        attrs = {}
        for a in n.attribute:
            if a.type == onnx.AttributeProto.INT:
                attrs[a.name] = int(a.i)
            elif a.type == onnx.AttributeProto.STRING:
                attrs[a.name] = a.s.decode()
            elif a.type == onnx.AttributeProto.TENSOR:
                arr = numpy_helper.to_array(a.t)
                attrs[a.name] = {"dtype": str(arr.dtype), "data": arr.ravel().tolist()}
        if attrs:
            node_attrs[ni] = attrs
        node_specs.append((n.op_type, n.name or f"{n.op_type}_{ni}",
                           [tname2idx[v] for v in n.input], [tname2idx[v] for v in n.output]))
    for vi in g.output:
        node_specs.append(("OutputOperator", f"OutputOperator:{vi.name}", [tname2idx[vi.name]], []))

    b = flatbuffers.Builder(1 << 20)
    toffs = [_tensor(b, nm, shapes[nm]) for nm in tname2idx]
    noffs = [_node(b, *spec) for spec in node_specs]
    tvec, nvec = _ovec(b, toffs), _ovec(b, noffs)
    b.StartObject(16)
    b.PrependUOffsetTRelativeSlot(1, nvec, 0)
    b.PrependUOffsetTRelativeSlot(0, tvec, 0)
    sg = b.EndObject()
    sgs = _ovec(b, [sg])
    b.StartObject(22)
    b.PrependUOffsetTRelativeSlot(2, sgs, 0)
    root = b.EndObject()
    b.Finish(root)
    body = bytes(b.Output())

    trailer = {
        "uop_graph": 1,
        "npu_ew_model": npu_ew_model,
        "consts": consts,
        "node_attrs": node_attrs,
        "inputs": [vi.name for vi in g.input],
        "outputs": [vi.name for vi in g.output],
        "output_shapes": {vi.name: shapes[vi.name] for vi in g.output},
    }
    tj = json.dumps(trailer).encode()
    hdr = bytearray(HEADER_SIZE)
    hdr[0:4] = b"RKNN"
    struct.pack_into("<Q", hdr, 0x08, 6)
    struct.pack_into("<Q", hdr, 0x10, len(body))
    blob = bytes(hdr) + body + struct.pack("<Q", len(tj)) + tj
    with open(rknn_path, "wb") as f:
        f.write(blob)
    ops = sorted({s[0] for s in node_specs})
    print(f"{rknn_path}: {len(node_specs)} nodes, {len(tname2idx)} tensors, {len(consts)} consts")
    print("op types:", ops)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
