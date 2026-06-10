"""tinygrad lowered-uop list -> ONNX converter (loop-free / unrolled kernels).

Reads /tmp/rand_kernels.json (op, dtype, src, arg per uop, kernels already
unrolled — no RANGE/END). Emits one ONNX model per kernel.

dtype strategy: ONNX has no uint32 bitcast and no fp exponent tricks, so the
BITCAST(uint->float) of (mantissa | 0x3f800000) is rewritten as
  (x & 0x7fffff) * 2^-23 + 1.0
which is bit-exact for the [1,2) trick used by tinygrad's rand.
"""
import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

UI, FL, BO = TensorProto.UINT32, TensorProto.FLOAT, TensorProto.BOOL

def convert_kernel(uops, name):
    nodes, inits, values = [], [], {}
    inputs, outputs = [], []
    params = {}
    indexes = {}
    stores = {}
    nid = 0

    def cu32(v):
        nonlocal nid
        n = f"c_{nid}"; nid += 1
        inits.append(numpy_helper.from_array(np.array([v & 0xffffffff], dtype=np.uint32), n))
        return n

    def cf32(v):
        nonlocal nid
        n = f"f_{nid}"; nid += 1
        inits.append(numpy_helper.from_array(np.array([v], dtype=np.float32), n))
        return n

    def emit(op, srcs, **attrs):
        nonlocal nid
        n = f"v_{nid}"; nid += 1
        nodes.append(helper.make_node(op, srcs, [n], **attrs))
        return n

    for i, u in enumerate(uops):
        op, dt, src, arg = u['op'], u['dtype'], u['src'], u['arg']
        if op == 'PARAM':
            pi = int(arg)
            size = int(dt.split('(')[1].rstrip(')'))
            params[i] = (f"data{pi}", size, FL if 'float' in dt else UI)
            if 'float' not in dt:
                inputs.append(helper.make_tensor_value_info(f"data{pi}", UI, [size]))
        elif op == 'CONST':
            v = arg
            if dt == 'dtypes.float':
                f = float(v.split('(')[1].rstrip(')')) if 'ConstFloat' in v else float(v)
                values[i] = cf32(f)
            elif dt == 'dtypes.bool':
                n = f"b_{nid}"; nid += 1
                inits.append(numpy_helper.from_array(np.array([v == 'True']), n))
                values[i] = n
            else:
                values[i] = cu32(int(v))
        elif op == 'INDEX':
            indexes[i] = (src[0], src[1])
        elif op == 'CAST' and 'ptr' in dt:
            indexes[i] = indexes[src[0]]
        elif op == 'LOAD':
            p, idx_uop = indexes[src[0]]
            pname = params[p][0]
            iname = f"i_{nid}"; nid += 1
            inits.append(numpy_helper.from_array(np.array([int(uops[idx_uop]['arg'])], dtype=np.int64), iname))
            values[i] = emit('Gather', [pname, iname], axis=0)
        elif op == 'ADD':
            values[i] = emit('Add', [values[src[0]], values[src[1]]])
        elif op == 'SUB':
            values[i] = emit('Sub', [values[src[0]], values[src[1]]])
        elif op == 'MUL':
            values[i] = emit('Mul', [values[src[0]], values[src[1]]])
        elif op == 'MULACC':
            m = emit('Mul', [values[src[0]], values[src[1]]])
            values[i] = emit('Add', [m, values[src[2]]])
        elif op == 'SHR':
            values[i] = emit('BitShift', [values[src[0]], values[src[1]]], direction='RIGHT')
        elif op == 'SHL':
            values[i] = emit('BitShift', [values[src[0]], values[src[1]]], direction='LEFT')
        elif op == 'XOR':
            values[i] = emit('BitwiseXor', [values[src[0]], values[src[1]]])
        elif op == 'OR':
            values[i] = emit('BitwiseOr', [values[src[0]], values[src[1]]])
        elif op == 'AND':
            if dt == 'dtypes.bool':
                values[i] = emit('And', [values[src[0]], values[src[1]]])
            else:
                values[i] = emit('BitwiseAnd', [values[src[0]], values[src[1]]])
        elif op == 'CMOD':
            values[i] = emit('Mod', [values[src[0]], values[src[1]]], fmod=0)
        elif op == 'CMPLT':
            values[i] = emit('Less', [values[src[0]], values[src[1]]])
        elif op == 'CMPNE':
            eq = emit('Equal', [values[src[0]], values[src[1]]])
            values[i] = emit('Not', [eq])
        elif op == 'CAST':
            values[i] = emit('Cast', [values[src[0]]], to=FL if dt == 'dtypes.float' else UI)
        elif op == 'BITCAST':
            mant = emit('BitwiseAnd', [values[src[0]], cu32(0x7fffff)])
            mantf = emit('Cast', [mant], to=FL)
            scaled = emit('Mul', [mantf, cf32(1.0 / (1 << 23))])
            values[i] = emit('Add', [scaled, cf32(1.0)])
        elif op == 'STACK':
            values[i] = emit('Concat', [values[s] for s in src], axis=0)
        elif op == 'STORE':
            p, idx_uop = indexes[src[0]]
            stores[int(uops[idx_uop]['arg'])] = (p, values[src[1]])
        elif op in ('GROUP', 'SINK'):
            pass
        else:
            raise NotImplementedError(f"uop {op} at {i}")

    pieces = [stores[k][1] for k in sorted(stores)]
    p = stores[sorted(stores)[0]][0]
    pname, size, ptype = params[p]
    out_name = f"{pname}_out"
    nodes.append(helper.make_node('Concat', pieces, [out_name], axis=0))
    outputs.append(helper.make_tensor_value_info(out_name, ptype, [size]))

    graph = helper.make_graph(nodes, name, inputs, outputs, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
    onnx.checker.check_model(model)
    return model


if __name__ == '__main__':
    data = json.load(open('/tmp/rand_kernels.json'))
    for ki, k in enumerate(data['kernels']):
        m = convert_kernel(k, f'rand_k{ki + 1}')
        onnx.save(m, f'/tmp/rand_k{ki + 1}.onnx')
        print(f'rand_k{ki + 1}.onnx: {len(m.graph.node)} nodes')
