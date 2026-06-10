#!/usr/bin/env python3
"""
Create ONNX int32 ADD model, convert to RKNN, run via librknnrt ctypes, verify.

Steps:
1. Create simple ONNX Add(int32) model
2. Convert ONNX -> RKNN (need rknn-toolkit2; fallback to existing .rknn)
3. Run the RKNN model via rknn_* ctypes API
4. Verify output matches expected int32 ADD result
"""
import ctypes, os, sys
import numpy as np
import onnx
from onnx import helper, TensorProto

# ── Constants from rknn_api.h ────────────────────────────────────────────────
RKNN_QUERY_IN_OUT_NUM    = 0
RKNN_QUERY_INPUT_ATTR    = 1
RKNN_QUERY_OUTPUT_ATTR   = 2

RKNN_TENSOR_NHWC         = 0
RKNN_TENSOR_NCHW         = 1
RKNN_TENSOR_INT32        = 4

# ── Step 1: Create ONNX int32 ADD model ─────────────────────────────────────
def create_int32_add_onnx(path, shape=(4,)):
    x = helper.make_tensor_value_info('x', TensorProto.INT32, list(shape))
    y = helper.make_tensor_value_info('y', TensorProto.INT32, list(shape))
    z = helper.make_tensor_value_info('z', TensorProto.INT32, list(shape))
    node = helper.make_node('Add', inputs=['x', 'y'], outputs=['z'])
    graph = helper.make_graph([node], 'add_graph', [x, y], [z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"Created ONNX: {path}")
    return path

# ── Step 2 & 3: Run RKNN model via librknnrt ctypes ─────────────────────────
class rknn_input_output_num(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]

class rknn_tensor_attr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 16),
        ("name", ctypes.c_char * 256),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("qnt_type", ctypes.c_uint32),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]

class rknn_input(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_uint32),
        ("fmt", ctypes.c_uint32),
    ]

class rknn_output(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]

def run_int32_add_rknn(rknn_path):
    lib = ctypes.CDLL('/data/rk3588/rknn-header/librknnrt.so')

    ctx = ctypes.c_void_p()
    ret = lib.rknn_init(ctypes.byref(ctx), rknn_path.encode(), 0, 0, None)
    print(f"rknn_init: ret={ret}")
    if ret != 0:
        print("rknn_init FAILED"); return False

    # Query I/O count
    io_num = rknn_input_output_num()
    lib.rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, ctypes.byref(io_num), ctypes.sizeof(io_num))
    n_input = io_num.n_input
    n_output = io_num.n_output
    print(f"n_inputs={n_input}  n_outputs={n_output}")

    if n_input == 0 or n_output == 0:
        print("Model has 0 inputs or 0 outputs — cannot run")
        lib.rknn_destroy(ctx)
        return False

    # Query input attrs
    input_attrs = []
    total_elems = 1
    for i in range(n_input):
        attr = rknn_tensor_attr(index=i)
        lib.rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, ctypes.byref(attr), ctypes.sizeof(attr))
        dims = [attr.dims[j] for j in range(attr.n_dims)]
        n_elems = attr.n_elems
        total_elems = n_elems
        print(f"  input[{i}]: name={attr.name.decode()} dims={dims} n_elems={n_elems} "
              f"type={attr.type} fmt={attr.fmt} size={attr.size}")
        input_attrs.append(attr)

    # Query output attrs
    output_attrs = []
    for i in range(n_output):
        attr = rknn_tensor_attr(index=i)
        lib.rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, ctypes.byref(attr), ctypes.sizeof(attr))
        dims = [attr.dims[j] for j in range(attr.n_dims)]
        print(f"  output[{i}]: name={attr.name.decode()} dims={dims} n_elems={attr.n_elems} "
              f"type={attr.type} fmt={attr.fmt} size={attr.size}")
        output_attrs.append(attr)

    N = total_elems
    a_data = np.arange(N, dtype=np.int32) * 3
    b_data = np.arange(N, dtype=np.int32) * 7
    expected = a_data + b_data
    print(f"  N={N}, a[:5]={a_data[:min(5,N)]}, b[:5]={b_data[:min(5,N)]}")

    # Set inputs (pass_through=0 lets runtime convert int32→internal format)
    inputs = (rknn_input * n_input)()
    for i, data in enumerate([a_data, b_data][:n_input]):
        inputs[i].index = i
        inputs[i].buf = data.ctypes.data
        inputs[i].size = data.nbytes
        inputs[i].pass_through = 0
        inputs[i].type = RKNN_TENSOR_INT32
        inputs[i].fmt = RKNN_TENSOR_NHWC

    ret = lib.rknn_inputs_set(ctx, n_input, inputs)
    print(f"rknn_inputs_set: ret={ret}")

    # Run
    ret = lib.rknn_run(ctx, None)
    print(f"rknn_run: ret={ret}")
    if ret != 0:
        print("rknn_run FAILED"); lib.rknn_destroy(ctx); return False

    # Get outputs — float mode to see what the runtime actually computed
    out_buf_f = np.zeros(N, dtype=np.float32)
    outputs = (rknn_output * n_output)()
    outputs[0].want_float = 1
    outputs[0].is_prealloc = 1
    outputs[0].index = 0
    outputs[0].buf = out_buf_f.ctypes.data
    outputs[0].size = out_buf_f.nbytes

    ret = lib.rknn_outputs_get(ctx, n_output, outputs, None)
    print(f"rknn_outputs_get (float): ret={ret}")

    result = out_buf_f[:N]
    print(f"  result[:10]:   {result[:min(10,N)]}")
    print(f"  expected[:10]: {expected[:min(10,N)].astype(np.float32)}")

    match = np.allclose(result, expected.astype(np.float32), atol=0.5)
    print(f"  MATCH: {match}")
    if not match:
        mismatches = np.where(np.abs(result - expected.astype(np.float32)) > 0.5)[0]
        n_mis = len(mismatches)
        print(f"  {n_mis} mismatches out of {N}")
        for idx in mismatches[:10]:
            print(f"    [{idx}] got={result[idx]} expected={expected[idx]}")

    lib.rknn_outputs_release(ctx, n_output, outputs)
    lib.rknn_destroy(ctx)
    return match


# ── ONNX → RKNN conversion ──────────────────────────────────────────────────
def convert_onnx_to_rknn(onnx_path, rknn_path):
    """Convert ONNX → RKNN using /data/.venv rknn-toolkit2."""
    import subprocess
    script = f'''
import sys
sys.path.insert(0, "/data/.venv/lib/python3.12/site-packages")
from rknn.api import RKNN
rknn = RKNN()
rknn.config(mean_values=[[0]*10, [0]*10], std_values=[[1]*10, [1]*10], target_platform="rk3588")
ret = rknn.load_onnx(model="{onnx_path}")
if ret != 0: print(f"load_onnx failed: {{ret}}"); sys.exit(1)
ret = rknn.build(do_quantization=False)
if ret != 0: print(f"build failed: {{ret}}"); sys.exit(1)
ret = rknn.export_rknn("{rknn_path}")
if ret != 0: print(f"export_rknn failed: {{ret}}"); sys.exit(1)
rknn.release()
print(f"Converted: {onnx_path} -> {rknn_path}")
'''
    result = subprocess.run(["/data/.venv/bin/python3", "-c", script],
                            capture_output=True, text=True)
    print(result.stdout, end="")
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            if "E RKNN" in line or "error" in line.lower():
                print(f"  stderr: {line}")
    return result.returncode == 0


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    onnx_path = "/tmp/int32_add.onnx"
    rknn_path = "/tmp/int32_add.rknn"

    create_int32_add_onnx(onnx_path, shape=(1, 10))

    # Convert ONNX → RKNN via /data/.venv rknn-toolkit2
    converted = convert_onnx_to_rknn(onnx_path, rknn_path)

    if converted:
        run_int32_add_rknn(rknn_path)
    else:
        print("Conversion failed, trying existing RKNN...")
        existing = "/data/rk3588/rknn-creation/_int32_add.rknn"
        if os.path.exists(existing):
            run_int32_add_rknn(existing)
