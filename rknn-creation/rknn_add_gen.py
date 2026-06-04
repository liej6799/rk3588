#!/usr/bin/env python3
"""rknn_add_gen.py - self-contained generator for fp16 element-wise Add/Mul .rknn models.

Generates a runnable .rknn for ANY shape (rows x cols, N=rows*cols <= 50,231,944)
WITHOUT the rknn-toolkit2 compiler or any external template files, by embedding the
known-good FlatBuffer+regcmd bodies for 2/3/4-input Add and patching tensor shapes,
memory plans and register-command streams in-place. Reverse-engineered for RKNPU2
v2.3.2 / rk3588.

For 5+ inputs, falls back to make_template() which builds the ONNX and compiles it
once with the on-device toolkit, caching the result.

The work is laid into C1=ceil(N/8176) NC1HWC2 channel surfaces (W=ceil(N/C1) each).
When C1 <= 1024, a single add tile covers all surfaces (CHANNEL = C1*8-1, fits in the
13-bit hardware field). When C1 > 1024, the template regcmd blocks are split into
multiple tiles, each covering up to 1024 surfaces with byte-offset base addresses.
Ceiling: 6 tiles x 1024 surfaces x 8176 width = 50,231,944 elements per Add operation.

`--allow-toolkit` falls back to the on-device rknn-toolkit2 build for N > MAX_N.

Usage
-----
  # 2-input (default): z = x + y
  rknn_add_gen.py --rows 4096 --cols 4096 -o big.rknn --verify
  # 3-input: d = a + b + c
  rknn_add_gen.py --inputs 3 --rows 4096 --cols 4096 -o big3.rknn --verify
  # 4-input: e = a + b + c + d
  rknn_add_gen.py --inputs 4 --rows 4096 --cols 4096 -o big4.rknn --verify
  # element-wise Mul: z = x * y
  rknn_add_gen.py --op Mul --rows 10 --cols 10 -o mul.rknn --verify
"""
import argparse, json, math, struct, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_rknn_build_queue import FB
import rc_template_gen

ROOT = Path(__file__).resolve().parent
HEADER_SIZE = 0x40
SURF_W = 8176
CH_BITS = 13
MAX_CH = (1 << CH_BITS) - 1
MAX_C1_PER_TILE = (MAX_CH + 1) // 8
BLOCKS_PER_ADD = 6
MAX_TILES = BLOCKS_PER_ADD
MAX_SURF = MAX_C1_PER_TILE * MAX_TILES
MAX_N = SURF_W * MAX_SURF

_EMBEDDED = {}
TEMPLATES = {
    2: "add_10x10.rknn",
    3: "add3_10x10.rknn",
    4: "add4_10x10.rknn",
}


def _get_body(n_inputs, ops=None):
    raw_body = ROOT / f"_body_add{n_inputs}_10x10.body"
    if raw_body.exists():
        return bytearray(raw_body.read_bytes())
    if n_inputs not in _EMBEDDED:
        try:
            from _embedded_bodies import _EMBEDDED as _emb
        except ModuleNotFoundError:
            _emb = {}
        _EMBEDDED.update(_emb)
    if n_inputs in _EMBEDDED:
        return bytearray(_EMBEDDED[n_inputs])
    template = make_template(n_inputs, ops=ops)
    body, _ = split(Path(template).read_bytes())
    return body
VERIFY_BINS = {
    2: "rknn_verify",
    3: "rknn_verify3",
    4: "rknn_verify4",
}

TARGETS = {0x0101, 0x0201, 0x0801, 0x1001, 0x2001, 0x4001,
           0x8001, 0x10001, 0x20001, 0x40001}
DPU, RDMA = 0x1001, 0x2001


def surface_split(N):
    C1 = max(1, math.ceil(N / SURF_W))
    return C1, math.ceil(N / C1)


def tile_split(C1):
    if C1 <= MAX_C1_PER_TILE:
        return [(C1, 0)]
    n_tiles = min(MAX_TILES, math.ceil(C1 / MAX_C1_PER_TILE))
    c1_per = math.ceil(C1 / n_tiles)
    tiles, off, rem = [], 0, C1
    for _ in range(n_tiles):
        c1 = min(c1_per, rem)
        tiles.append((c1, off))
        off += c1; rem -= c1
    return tiles


def split(buf):
    bs = struct.unpack_from("<Q", buf, 0x10)[0]
    return bytearray(buf[HEADER_SIZE:HEADER_SIZE + bs]), bytes(buf[HEADER_SIZE + bs:])


def assemble(body, trailer, version=6):
    h = bytearray(HEADER_SIZE); h[0:4] = b"RKNN"
    struct.pack_into("<Q", h, 0x08, version); struct.pack_into("<Q", h, 0x10, len(body))
    return bytes(h) + bytes(body) + bytes(trailer)


def fb_for(body):
    h = bytearray(HEADER_SIZE); h[0:4] = b"RKNN"
    struct.pack_into("<Q", h, 0x08, 6); struct.pack_into("<Q", h, 0x10, len(body))
    return FB(bytes(h) + bytes(body))


def tensors_of(fb):
    root = HEADER_SIZE + fb.u32(HEADER_SIZE)
    sg = fb.vector_table_positions(root, 2)[0]
    return fb.vector_table_positions(sg, 0)


def set_vec(fb, body, p, fidx, vals):
    ab = fb.field_abs(p, fidx)
    if ab is None: return False
    tgt = ab + fb.u32(ab); n = fb.u32(tgt)
    if n != len(vals): return False
    for k, v in enumerate(vals):
        struct.pack_into("<I", body, tgt + 4 + 4*k - HEADER_SIZE, v)
    return True


def set_scalar(fb, body, p, fidx, val):
    ab = fb.field_abs(p, fidx)
    if ab is None: return
    struct.pack_into("<I", body, ab - HEADER_SIZE, val)


def regcmd_spans(body):
    n = len(body)//8; w = list(struct.unpack_from("<%dQ" % n, body, 0)); out = []; i = 0
    while i < n:
        if ((w[i] >> 48) & 0xFFFF) in TARGETS:
            j = i
            while j < n and ((w[j] >> 48) & 0xFFFF) in TARGETS: j += 1
            if j - i >= 20: out.append((i, j - i))
            i = j
        else: i += 1
    return out, w


def _set(w, b, c, t, reg, val):
    for k in range(c):
        r = w[b+k]
        if ((r >> 48) & 0xFFFF) == t and (r & 0xFFFF) == reg:
            w[b+k] = (r & ~(0xFFFFFFFF << 16)) | ((val & 0xFFFFFFFF) << 16); return


def _get_reg(w, b, c, t, reg):
    for k in range(c):
        r = w[b+k]
        if ((r >> 48) & 0xFFFF) == t and (r & 0xFFFF) == reg:
            return (r >> 16) & 0xFFFFFFFF
    return None


def _patch_block(w, i, c, W, ch, base, st, op="Add"):
    for reg in (0x4030, 0x405c):
        _set(w, i, c, DPU, reg, W - 1)
    _set(w, i, c, RDMA, 0x500c, W - 1)
    ch_val = ((W - 1) << 16 | ch) if op == "Mul" else (ch << 16 | ch)
    _set(w, i, c, DPU, 0x403c, ch_val)
    _set(w, i, c, DPU, 0x4058, ch)
    _set(w, i, c, RDMA, 0x5014, ch)
    _set(w, i, c, DPU, 0x4034, 0)
    _set(w, i, c, RDMA, 0x5010, 0)
    _set(w, i, c, DPU, 0x4020, base)
    _set(w, i, c, RDMA, 0x5018, base)
    _set(w, i, c, RDMA, 0x5038, base)
    _set(w, i, c, DPU, 0x4024, st)
    _set(w, i, c, DPU, 0x40c0, st)
    _set(w, i, c, RDMA, 0x5040, st)
    _set(w, i, c, RDMA, 0x504c, 0)
    _set(w, i, c, RDMA, 0x506c, 0)


def _patch_ew_ops(body, ops):
    n_adds = len(ops)
    if all(o == "Add" for o in ops):
        return
    total_groups = ADD_GROUPS.get(n_adds + 1, n_adds)
    if total_groups == 0:
        total_groups = n_adds
    spans, w = regcmd_spans(body)
    spans_per_group = len(spans) // total_groups if total_groups > 0 else len(spans)
    for g in range(n_adds):
        op = ops[g]
        ew_cfg_val = rc_template_gen._ew_cfg(
            rc_template_gen.EW_OP_MUL if op == "Mul" else rc_template_gen.EW_OP_ADD
        )
        group = spans[g * spans_per_group : (g + 1) * spans_per_group]
        for start, count in group:
            _set(w, start, count, DPU, 0x4070, ew_cfg_val)
            if op == "Mul":
                ch_val = _get_reg(w, start, count, DPU, 0x403c)
                if ch_val is not None:
                    ch_lower = ch_val & 0xFFFF
                    w_val = _get_reg(w, start, count, DPU, 0x4030)
                    w_minus_1 = w_val if w_val is not None else ch_lower
                    _set(w, start, count, DPU, 0x403c, (w_minus_1 << 16) | ch_lower)
    for idx, val in enumerate(w):
        struct.pack_into("<Q", body, idx * 8, val)


MEMORY_PLANS = {
    2: {
        "x":   (0, False), "y":   (1, False), "z":   (2, False),
        "z-rs": (0, True),  "x_rs": (1, True), "y_rs": (2, True),
    },
    3: {
        "a":    (0, False), "b":    (1, False), "c":    (2, False), "d":    (3, False),
        "d-rs": (0, True),  "a_rs": (1, True),  "b_rs": (2, True),
        "c_rs": (3, True),  "t-rs": (None, True),
    },
    4: {
        "a":    (0, False), "b":    (1, False), "c":    (2, False), "d":    (3, False),
        "e":    (4, False),
        "e-rs": (0, True),  "a_rs": (1, True),  "b_rs": (2, True),
        "c_rs": (3, True),  "d_rs": (4, True),
        "t1-rs": (None, True), "t2-rs": (5, True),
    },
}


def memory_plan_for(n_inputs):
    """Derive the {tensor_name: (slot, is_work)} plan for any n_inputs.
    Verified to reproduce the hand-tuned MEMORY_PLANS[2..4] exactly, and extends
    to arbitrary n: inputs (slots 0..n-1) + output (slot n) external; output-rs
    (slot 0) + each input_rs (slots 1..n) + intermediates working. The first
    intermediate aliases the output region (slot None), later ones get fresh slots."""
    if n_inputs in MEMORY_PLANS:
        return MEMORY_PLANS[n_inputs]
    ins, outp = input_output_names(n_inputs)
    plan = {nm: (i, False) for i, nm in enumerate(ins)}
    plan[outp] = (n_inputs, False)
    plan[f"{outp}-rs"] = (0, True)
    for i, nm in enumerate(ins):
        plan[f"{nm}_rs"] = (i + 1, True)
    n_inter = n_inputs - 2
    for k in range(1, n_inter + 1):
        nm = "t-rs" if n_inter == 1 else f"t{k}-rs"
        plan[nm] = (None if k == 1 else n_inputs + k - 1, True)
    return plan


def plan_memory(body, N, n_inputs):
    C1, W = surface_split(N)
    Npad = C1 * W
    fb = fb_for(body)
    al = lambda s, a=256: ((s + a - 1)//a)*a
    ext, work = al(N*2), al(Npad*16)
    plan = {}
    for nm, (slot, is_work) in memory_plan_for(n_inputs).items():
        if slot is None:
            f3, f4, f12 = [1, C1, 1, W, 8], [1, C1, 1, W], Npad * 16
            f13 = None
        elif is_work:
            f3, f4, f12 = [1, C1, 1, W, 8], [1, C1, 1, W], Npad * 16
            f13 = slot * work
        else:
            f3, f4, f12 = [1, N], [1, N], N * 2
            f13 = slot * ext
        plan[nm] = (f3, f4, f12, f13)
    for p in tensors_of(fb):
        nm = fb.string(p, 5)
        if nm in plan:
            f3, f4, f12, f13 = plan[nm]
            set_vec(fb, body, p, 3, f3)
            set_vec(fb, body, p, 4, f4)
            set_scalar(fb, body, p, 12, f12)
            if f13 is not None:
                set_scalar(fb, body, p, 13, f13)


ADD_GROUPS = {2: 1, 3: 2, 4: 3}


def make_tiles(body, N, n_inputs, ops=None):
    C1, W = surface_split(N)
    tiles = tile_split(C1)
    n_tiles = len(tiles)
    spans, w = regcmd_spans(body)
    n_adds = ADD_GROUPS.get(n_inputs, n_inputs - 1)
    if ops is None:
        ops = ["Add"] * n_adds
    spans_per_add = len(spans) // n_adds
    st = W * 16
    if n_tiles > spans_per_add:
        raise SystemExit(f"need {n_tiles} tiles but only {spans_per_add} blocks per Add "
                         f"(max N = {MAX_N:,}; use --allow-toolkit for larger models)")
    for g in range(n_adds):
        group = spans[g * spans_per_add : (g + 1) * spans_per_add]
        op = ops[g] if g < len(ops) else "Add"
        for tile_idx, (i, c) in enumerate(group):
            tidx = min(tile_idx, n_tiles - 1)
            C1_tile, surf_offset = tiles[tidx]
            ch = C1_tile * 8 - 1
            base = surf_offset * st
            _patch_block(w, i, c, W, ch, base, st, op)
    for idx, val in enumerate(w):
        struct.pack_into("<Q", body, idx * 8, val)
    return body, len(spans), C1, W, n_tiles


def make_trailer(rows, cols, n_inputs):
    ins, outp = input_output_names(n_inputs)
    n_tensors = n_inputs + 1
    dtype = {"qnt_method": "", "qnt_type": "", "vx_type": ""}
    norm_tensor = []
    for i, nm in enumerate(ins + [outp]):
        norm_tensor.append({"dim_num": 2, "dtype": dtype, "size": [rows, cols],
                            "tensor_id": i, "url": nm})
    connection = []
    for i in range(n_inputs):
        connection.append({"left": "input", "left_tensor_id": i, "node_id": 0,
                           "right_tensor": {"tensor_id": i, "type": "norm_tensor"}})
    connection.append({"left": "output", "left_tensor_id": 0, "node_id": 0,
                       "right_tensor": {"tensor_id": n_inputs, "type": "norm_tensor"}})
    graph = []
    for i in range(n_inputs):
        graph.append({"left": "input", "left_tensor_id": i,
                       "right": "norm_tensor", "right_tensor_id": i})
    graph.append({"left": "output", "left_tensor_id": 0,
                   "right": "norm_tensor", "right_tensor_id": n_inputs})
    js = {
        "connection": connection, "const_tensor": [], "graph": graph,
        "input_num": n_inputs, "name": "rknn model", "network_platform": "ONNX",
        "node_num": 1,
        "nodes": [{"input_num": n_inputs, "lid": "npu_network_bin_graph", "name": "nnbg",
                    "nn": {"nbg": {"type": "RKNN_OP_NNBG"}}, "op": "RKNN_OP_NNBG",
                    "output_num": 1, "uid": 0}],
        "norm_tensor": norm_tensor, "norm_tensor_num": n_tensors,
        "ori_network_platform": "ONNX", "output_num": 1,
        "target_platform": ["rk3588"], "version": "2.3.2", "virtual_tensor": [],
    }
    nj = json.dumps(js, separators=(",", ":")).encode()
    return struct.pack("<Q", len(nj)) + nj


def input_output_names(n_inputs):
    """Tensor names the template uses (matches the original hand-built templates):
    2-input -> (['x','y'], 'z'); n>=3 -> first n letters as inputs, next letter as output."""
    if n_inputs == 2:
        return ["x", "y"], "z"
    letters = "abcdefghijklmnopqrstuvwxyz"
    return list(letters[:n_inputs]), letters[n_inputs]


def template_path(n_inputs, ops=None):
    if ops and any(o != "Add" for o in ops):
        op_suffix = "_" + "_".join(o.lower() for o in ops)
        return ROOT / f"ew{n_inputs}{op_suffix}_10x10.rknn"
    return ROOT / TEMPLATES.get(n_inputs, f"add{n_inputs}_10x10.rknn")


def make_template(n_inputs, shape=(10, 10), ops=None):
    """Create the n_inputs-input chained-EW template .rknn from the program itself.

    Builds out = in0 op1 in1 op2 ... opN inN as an ONNX graph of (n_inputs-1)
    EW nodes (Add or Mul per ops list) and compiles it with the on-device toolkit
    (a ONE-TIME bootstrap; the result is cached and then patched TOOLKIT-FREE by
    generate() for every real model).
    """
    n_adds = n_inputs - 1
    if ops is None:
        ops = ["Add"] * n_adds
    path = template_path(n_inputs, ops)
    if path.exists():
        return path
    import onnx
    from onnx import TensorProto, helper
    from rknn.api import RKNN
    shp = list(shape)
    ins, outp = input_output_names(n_inputs)
    vis = [helper.make_tensor_value_info(nm, TensorProto.FLOAT16, shp) for nm in ins]
    vout = helper.make_tensor_value_info(outp, TensorProto.FLOAT16, shp)
    n_inter = n_inputs - 2
    nodes, acc = [], ins[0]
    for i in range(1, n_inputs):
        if i == n_inputs - 1:
            res = outp
        elif n_inter == 1:
            res = "t"
        else:
            res = f"t{i}"
        op_name = ops[i - 1]
        nodes.append(helper.make_node(op_name, [acc, ins[i]], [res],
                                      name=f"{op_name.lower()}{i}"))
        acc = res
    g_name = "ew" + "".join(o[0].lower() for o in ops) + str(n_inputs)
    g = helper.make_graph(nodes, g_name, vis, [vout])
    m = helper.make_model(g, producer_name="rknn_add_gen",
                          opset_imports=[helper.make_opsetid("", 13)])
    onnx_path = str(ROOT / f"_tmpl_ew{n_inputs}_{shp[0]}x{shp[1]}.onnx")
    onnx.save(m, onnx_path)
    rk = RKNN(verbose=False)
    try:
        rk.config(target_platform="rk3588", float_dtype="float16")
        rk.load_onnx(model=onnx_path, inputs=ins, input_size_list=[shp] * n_inputs)
        rk.build(do_quantization=False)
        rk.export_rknn(str(path))
    finally:
        rk.release()
    print(f"created template {path.name} ({n_inputs}-input) via toolkit (cached)")
    return path


def generate(rows, cols, out, n_inputs, emit_parts=False, ops=None):
    N = rows * cols
    if not (1 <= N <= MAX_N):
        raise SystemExit(f"N={N} out of range 1..{MAX_N} (max {MAX_SURF} surfaces x {SURF_W})")
    n_adds = n_inputs - 1
    if ops is None:
        ops = ["Add"] * n_adds
    if len(ops) != n_adds:
        raise SystemExit(f"expected {n_adds} ops for {n_inputs} inputs, got {len(ops)}")
    body = _get_body(n_inputs, ops=ops)
    _patch_ew_ops(body, ops)
    plan_memory(body, N, n_inputs)
    body, nblk, C1, W, ntiles = make_tiles(body, N, n_inputs, ops)
    trailer = make_trailer(rows, cols, n_inputs)
    Path(out).write_bytes(assemble(body, trailer))
    tile_info = f", {ntiles} tiles" if ntiles > 1 else ""
    op_desc = "+".join(ops)
    print(f"generated {out}: {rows}x{cols} N={N} ({n_inputs}-input {op_desc}, C1={C1} x W={W}, "
          f"{nblk} blocks{tile_info}, toolkit-free)")
    if emit_parts:
        Path(out + ".fb").write_bytes(body)
        Path(out + ".trailer").write_bytes(trailer)
        print(f"  parts: {out}.fb ({len(body)} B), {out}.trailer ({len(trailer)} B)")


def verify(out, n_inputs):
    # rknn_verify_n is generic (queries the input count); use the per-n harness if present,
    # else fall back to the generic one so any n_inputs can be verified.
    bin_name = VERIFY_BINS.get(n_inputs, "rknn_verify_n")
    runner = ROOT / bin_name
    if not runner.exists():
        runner = ROOT / "rknn_verify_n"
    if not runner.exists():
        print(f"VERIFY: skipped (build {runner.name} first)"); return
    r = subprocess.run([str(runner), out], capture_output=True, text=True,
                       env={"LD_LIBRARY_PATH": "/usr/lib64", "PATH": "/usr/bin"})
    line = next((l for l in r.stdout.splitlines() if "mismatches" in l), r.stdout.strip())
    print("VERIFY:", line or r.stderr.strip())
    return "PASS" in r.stdout


def generate_via_toolkit(rows, cols, out, n_inputs, ops=None):
    import onnx
    from onnx import TensorProto, helper
    from rknn.api import RKNN
    n_adds = n_inputs - 1
    if ops is None:
        ops = ["Add"] * n_adds
    shape = [rows, cols]
    ins, outp = input_output_names(n_inputs)
    onnx_path = str(ROOT / f"_gen_{rows}x{cols}.onnx")
    vis = [helper.make_tensor_value_info(nm, TensorProto.FLOAT16, shape) for nm in ins]
    vout = helper.make_tensor_value_info(outp, TensorProto.FLOAT16, shape)
    n_inter = n_inputs - 2
    nodes, acc = [], ins[0]
    for i in range(1, n_inputs):
        if i == n_inputs - 1:
            res = outp
        elif n_inter == 1:
            res = "t"
        else:
            res = f"t{i}"
        op_name = ops[i - 1]
        nodes.append(helper.make_node(op_name, [acc, ins[i]], [res],
                                      name=f"{op_name.lower()}{i}"))
        acc = res
    g = helper.make_graph(nodes, "ew_gen", vis, [vout])
    m = helper.make_model(g, producer_name="rknn_add_gen",
                          opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(m, onnx_path)
    rk = RKNN(verbose=False)
    try:
        rk.config(target_platform="rk3588", float_dtype="float16")
        rk.load_onnx(model=onnx_path, inputs=ins, input_size_list=[shape] * n_inputs)
        rk.build(do_quantization=False)
        rk.export_rknn(out)
    finally:
        rk.release()
    op_desc = "+".join(ops)
    print(f"generated {out}: {rows}x{cols} N={rows*cols} ({n_inputs}-input {op_desc}, via toolkit)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--inputs", type=int, default=2, choices=range(2, 9), metavar="2..8",
                    help="number of inputs: 2 for z=x+y, 3 for d=a+b+c (default: 2)")
    ap.add_argument("--op", choices=["Add", "Mul"], default=None,
                    help="element-wise operation applied uniformly (Add or Mul)")
    ap.add_argument("--ops", type=str, default=None,
                    help="comma-separated per-op list, e.g. Mul,Add for d=(a*b)+c")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--emit-parts", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--allow-toolkit", action="store_true",
                    help="for N>%d, fall back to the on-device toolkit build" % MAX_N)
    a = ap.parse_args()
    n_adds = a.inputs - 1
    if a.ops:
        ops = [o.strip() for o in a.ops.split(",")]
        bad = [o for o in ops if o not in ("Add", "Mul")]
        if bad:
            raise SystemExit(f"unknown ops: {bad} (valid: Add, Mul)")
        if len(ops) != n_adds:
            raise SystemExit(f"expected {n_adds} ops for {a.inputs} inputs, got {len(ops)}")
    elif a.op:
        ops = [a.op] * n_adds
    else:
        ops = ["Add"] * n_adds
    N = a.rows * a.cols
    if N > MAX_N:
        if not a.allow_toolkit:
            raise SystemExit(f"N={N} > {MAX_N} (max {MAX_TILES} tiles x "
                             f"{MAX_C1_PER_TILE} surfaces x {SURF_W}). "
                             f"Re-run with --allow-toolkit.")
        generate_via_toolkit(a.rows, a.cols, a.out, a.inputs, ops)
    else:
        generate(a.rows, a.cols, a.out, a.inputs, a.emit_parts, ops)
    if a.verify and verify(a.out, a.inputs) is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
