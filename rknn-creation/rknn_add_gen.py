#!/usr/bin/env python3
"""Generate fp16 element-wise Add/Sub/Mul/Div RKNN models for RK3588.

The normal path is fully toolkit-free for 2..16 inputs and
N = rows*cols <= 50,231,944.  It builds the FlatBuffer body from scratch via
rknn_flatbuf.build_body(), emits the RKNN container/trailer, and never calls
onnx or rknn-toolkit2 for supported sizes.

`--allow-toolkit` is only a large-model escape hatch for N > MAX_N.

Usage
-----
  # 2-input (default): z = x + y
  rknn_add_gen.py --rows 4096 --cols 4096 -o big.rknn --verify
  # element-wise Mul: z = x * y
  rknn_add_gen.py --op Mul --rows 10 --cols 10 -o mul.rknn --verify
  # 3-input mixed: d = (a * b) - c
  rknn_add_gen.py --inputs 3 --ops Mul,Sub --rows 10 --cols 10 -o mix.rknn --verify
  # 4-input: e = ((a / b) + c) * d
  rknn_add_gen.py --inputs 4 --ops Div,Add,Mul --rows 10 --cols 10 -o mix4.rknn
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rknn_flatbuf

ROOT = Path(__file__).resolve().parent
SURF_W = rknn_flatbuf.SURF_W
MAX_CH = rknn_flatbuf.MAX_CH
MAX_C1_PER_TILE = rknn_flatbuf.MAX_C1_PER_TILE
MAX_TILES = rknn_flatbuf.MAX_TILES
MAX_SURF = MAX_C1_PER_TILE * MAX_TILES
MAX_N = SURF_W * MAX_SURF

VALID_OPS = ("Add", "Sub", "Mul", "Div")
MAX_INPUTS = rknn_flatbuf.MAX_INPUTS


def _parse_ops(n_inputs, op=None, ops_text=None):
    min_ops = max(1, n_inputs - 1)
    max_ops = MAX_INPUTS - 1
    if ops_text:
        ops = [o.strip() for o in ops_text.split(",") if o.strip()]
        bad = [o for o in ops if o not in VALID_OPS]
        if bad:
            raise SystemExit(f"unknown ops: {bad} (valid: {list(VALID_OPS)})")
        if len(ops) < min_ops:
            raise SystemExit(f"need at least {min_ops} ops for {n_inputs} inputs, got {len(ops)}")
        if len(ops) > max_ops:
            raise SystemExit(f"at most {max_ops} ops supported, got {len(ops)}")
        return ops
    if op:
        return [op] * min_ops
    return ["Add"] * min_ops


def _model_stats(body, N, n_inputs, n_ops=None):
    C1, W = rknn_flatbuf.surface_split(N)
    n_tiles = len(rknn_flatbuf.tile_split(C1))
    spans, _ = rknn_flatbuf._regcmd_spans(body)
    expected = 6 * (n_ops or max(1, n_inputs - 1))
    if len(spans) != expected:
        raise SystemExit(f"internal error: generated {len(spans)} regcmd blocks, expected {expected}")
    return C1, W, n_tiles, len(spans)


def generate(rows, cols, out, n_inputs, emit_parts=False, ops=None, dtype="float16"):
    N = rows * cols
    if not (1 <= N <= MAX_N):
        raise SystemExit(f"N={N} out of range 1..{MAX_N} (max {MAX_SURF} surfaces x {SURF_W})")
    ops = ops or ["Add"] * max(1, n_inputs - 1)
    if len(ops) < max(1, n_inputs - 1):
        raise SystemExit(f"need at least {n_inputs - 1} ops for {n_inputs} inputs, got {len(ops)}")

    body = rknn_flatbuf.build_body(N, n_inputs, rows=rows, cols=cols, ops=ops, dtype=dtype)
    data = rknn_flatbuf.assemble_rknn(body, rows, cols, n_inputs)
    Path(out).write_bytes(data)

    C1, W, n_tiles, n_blocks = _model_stats(body, N, n_inputs, len(ops))
    tile_info = f", {n_tiles} tiles" if n_tiles > 1 else ""
    dtype_info = "" if dtype == "float16" else f", dtype={dtype}"
    print(f"generated {out}: {rows}x{cols} N={N} ({n_inputs}-input {'+'.join(ops)}, "
          f"C1={C1} x W={W}, {n_blocks} blocks{tile_info}{dtype_info}, toolkit-free)")

    if emit_parts:
        trailer = data[0x40 + len(body):]
        Path(out + ".fb").write_bytes(body)
        Path(out + ".trailer").write_bytes(trailer)
        print(f"  parts: {out}.fb ({len(body)} B), {out}.trailer ({len(trailer)} B)")


def verify(out, ops):
    runner = ROOT / "rknn_verify_n"
    if not runner.exists():
        print("VERIFY: skipped (build rknn_verify_n first)")
        return None
    cmd = [str(runner), out, ",".join(ops)]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env={"LD_LIBRARY_PATH": "/usr/lib64", "PATH": "/usr/bin"})
    text = (r.stdout + r.stderr).strip()
    line = next((l for l in text.splitlines() if "PASS" in l or "FAIL" in l),
                text.splitlines()[-1] if text else "")
    print("VERIFY:", line)
    return r.returncode == 0


def input_output_names(n_inputs):
    if n_inputs == 2:
        return ["x", "y"], "z"
    letters = "abcdefghijklmnopqrstuvwxyz"
    return list(letters[:n_inputs]), letters[n_inputs]


def generate_via_toolkit(rows, cols, out, n_inputs, ops=None):
    import onnx
    from onnx import TensorProto, helper
    from rknn.api import RKNN

    ops = ops or ["Add"] * (n_inputs - 1)
    shape = [rows, cols]
    ins, outp = input_output_names(n_inputs)
    onnx_path = str(ROOT / f"_gen_{rows}x{cols}_{n_inputs}.onnx")
    vis = [helper.make_tensor_value_info(nm, TensorProto.FLOAT16, shape) for nm in ins]
    vout = helper.make_tensor_value_info(outp, TensorProto.FLOAT16, shape)
    nodes, acc = [], ins[0]
    for i in range(1, n_inputs):
        res = outp if i == n_inputs - 1 else ("t" if n_inputs == 3 else f"t{i}")
        op_name = ops[i - 1]
        nodes.append(helper.make_node(op_name, [acc, ins[i]], [res],
                                      name=f"{op_name.lower()}{i}"))
        acc = res
    graph = helper.make_graph(nodes, "ew_gen", vis, [vout])
    model = helper.make_model(graph, producer_name="rknn_add_gen",
                              opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, onnx_path)

    rk = RKNN(verbose=False)
    try:
        rk.config(target_platform="rk3588", float_dtype="float16")
        rk.load_onnx(model=onnx_path, inputs=ins, input_size_list=[shape] * n_inputs)
        rk.build(do_quantization=False)
        rk.export_rknn(out)
    finally:
        rk.release()
    print(f"generated {out}: {rows}x{cols} N={rows*cols} "
          f"({n_inputs}-input {'+'.join(ops)}, via toolkit)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--inputs", type=int, default=2,
                    choices=range(2, MAX_INPUTS + 1), metavar=f"2..{MAX_INPUTS}",
                    help="number of inputs: 2 for z=x+y, 3 for d=a+b+c (default: 2)")
    ap.add_argument("--op", choices=VALID_OPS, default=None,
                    help="element-wise operation applied uniformly")
    ap.add_argument("--ops", type=str, default=None,
                    help="comma-separated per-op list, e.g. Mul,Add,Sub")
    ap.add_argument("--dtype", choices=sorted(rknn_flatbuf.DTYPES), default="float16",
                    help="tensor data type: float16 (NPU fp16 compute, default), "
                         "float32 (converted to/from fp16), int8 (exact integer "
                         "element-wise), or bool (RKNN_TENSOR_BOOL, int8-style: "
                         "Add~OR but Mul is not boolean AND; use a CPU-fallback And "
                         "model for true logical ops)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--emit-parts", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--allow-toolkit", action="store_true",
                    help=f"for N>{MAX_N}, fall back to the on-device toolkit build")
    args = ap.parse_args()

    ops = _parse_ops(args.inputs, args.op, args.ops)
    N = args.rows * args.cols
    if N > MAX_N:
        if not args.allow_toolkit:
            raise SystemExit(f"N={N} > {MAX_N} (max {MAX_TILES} tiles x "
                             f"{MAX_C1_PER_TILE} surfaces x {SURF_W}). "
                             "Re-run with --allow-toolkit.")
        generate_via_toolkit(args.rows, args.cols, args.out, args.inputs, ops)
    else:
        generate(args.rows, args.cols, args.out, args.inputs, args.emit_parts, ops,
                 args.dtype)

    if args.verify and verify(args.out, ops) is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
