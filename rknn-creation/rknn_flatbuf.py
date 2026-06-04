"""Build RKNN FlatBuffer bodies for fp16 element-wise Add/Sub/Mul/Div.

build_body() is the toolkit-free production path for 2..64 inputs.  It builds the
FlatBuffer metadata, register-command stream, and task descriptor data from
decoded formulas/templates.  build_body_scratch() keeps the old reference-body
splice path for reverse-engineering comparisons only.
"""
import math
import struct
from pathlib import Path

import flatbuffers

import rc_template_gen

SURF_W = 8176
MAX_CH = (1 << 13) - 1
MAX_C1_PER_TILE = (MAX_CH + 1) // 8
BLOCKS_PER_ADD = 6
MAX_TILES = BLOCKS_PER_ADD
HEADER_SIZE = 0x40
MAX_INPUTS = 64
SUPPORTED_INPUTS = range(2, MAX_INPUTS + 1)

_DPU = 0x1001
_RDMA = 0x2001
_TARGETS = {0x0101, 0x0201, 0x0801, _DPU, _RDMA, 0x4001,
            0x8001, 0x10001, 0x20001, 0x40001}

_RC_TEMPLATES = rc_template_gen.all_templates(max_n=MAX_INPUTS)

_EMBEDDED = {}

MEMORY_PLANS = {
    2: {
        "x": (0, False), "y": (1, False), "z": (2, False),
        "z-rs": (0, True), "x_rs": (1, True), "y_rs": (2, True),
    },
    3: {
        "a": (0, False), "b": (1, False), "c": (2, False), "d": (3, False),
        "d-rs": (0, True), "a_rs": (1, True), "b_rs": (2, True),
        "c_rs": (3, True), "t-rs": (None, True),
    },
    4: {
        "a": (0, False), "b": (1, False), "c": (2, False), "d": (3, False),
        "e": (4, False),
        "e-rs": (0, True), "a_rs": (1, True), "b_rs": (2, True),
        "c_rs": (3, True), "d_rs": (4, True),
        "t1-rs": (None, True), "t2-rs": (5, True),
    },
}


def _memory_plan_for(n_inputs):
    if n_inputs in MEMORY_PLANS:
        return MEMORY_PLANS[n_inputs]
    ins, outp = _io(n_inputs)
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


def _get_template_body(n_inputs):
    # Reference 10x10 bodies are stored as raw binary .body files (no base64),
    # the same convention as _template_8192x8192.body.
    if n_inputs not in _EMBEDDED:
        path = Path(__file__).resolve().parent / f"_body_add{n_inputs}_10x10.body"
        if not path.exists():
            raise ValueError(f"no embedded body for {n_inputs} inputs ({path.name})")
        _EMBEDDED[n_inputs] = path.read_bytes()
    return bytearray(_EMBEDDED[n_inputs])


_LARGE_TEMPLATE = None

def _get_large_template_body():
    global _LARGE_TEMPLATE
    if _LARGE_TEMPLATE is None:
        path = Path(__file__).resolve().parent / "_template_8192x8192.body"
        _LARGE_TEMPLATE = bytearray(path.read_bytes())
    return bytearray(_LARGE_TEMPLATE)


_LARGE_MEMORY_PLAN_2D = {
    "x": (0, False), "y": (1, False), "z": (2, False),
    "z-rs": (0, True), "x_rs": (1, True), "y_rs": (2, True),
}


def _plan_memory_2d(body, rows, cols, n_inputs):
    C1 = math.ceil(rows / 8)
    W = cols
    Npad = C1 * W
    al = lambda s, a=256: ((s + a - 1) // a) * a
    ext = al(rows * cols * 2)
    work = al(Npad * 16)
    plan = {}
    for nm, (slot, is_work) in _memory_plan_for(n_inputs).items():
        if slot is None:
            f3, f4, f12 = [1, C1, 1, W, 8], [1, C1, 1, W], Npad * 16
            f13 = None
        elif is_work:
            f3, f4, f12 = [1, C1, 1, W, 8], [1, C1, 1, W], Npad * 16
            f13 = slot * work
        else:
            f3, f4, f12 = [rows, cols], [rows, cols], rows * cols * 2
            f13 = slot * ext
        plan[nm] = (f3, f4, f12, f13)
    for tp in _fb_tensor_positions(body):
        nm = _fb_string(body, tp, 5)
        if nm in plan:
            f3, f4, f12, f13 = plan[nm]
            _fb_set_vec(body, tp, 3, f3)
            _fb_set_vec(body, tp, 4, f4)
            _fb_set_scalar(body, tp, 12, f12)
            if f13 is not None:
                _fb_set_scalar(body, tp, 13, f13)


def _patch_large_add_tiles(body, rows, cols, n_inputs):
    DPU_M = 0x1001
    RDMA_M = 0x2001
    MODE_ADD = 0x48000002
    spans, w = _regcmd_spans(body)
    C1 = math.ceil(rows / 8)
    W = cols
    st = W * 16
    ch_full = C1 * 8 - 1

    for idx, (start, count) in enumerate(spans):
        mode = None
        for k in range(count):
            x = w[start + k]
            tgt = (x >> 48) & 0xFFFF
            reg = x & 0xFFFF
            if tgt == DPU_M and reg == 0x4010:
                mode = (x >> 16) & 0xFFFFFFFF
                break
        if mode != MODE_ADD:
            continue
        for reg in (0x4030, 0x405c):
            _rc_set(w, start, count, DPU_M, reg, W - 1)
        _rc_set(w, start, count, RDMA_M, 0x500c, W - 1)
        _rc_set(w, start, count, DPU_M, 0x403c, (ch_full << 16) | ch_full)
        _rc_set(w, start, count, DPU_M, 0x4058, ch_full)
        _rc_set(w, start, count, RDMA_M, 0x5014, ch_full)
        _rc_set(w, start, count, DPU_M, 0x4034, 0)
        _rc_set(w, start, count, RDMA_M, 0x5010, 0)
        _rc_set(w, start, count, DPU_M, 0x4020, 0)
        _rc_set(w, start, count, RDMA_M, 0x5018, 0)
        _rc_set(w, start, count, RDMA_M, 0x5038, 0)
        _rc_set(w, start, count, DPU_M, 0x4024, st)
        _rc_set(w, start, count, DPU_M, 0x40c0, st)
        _rc_set(w, start, count, RDMA_M, 0x5040, st)
        _rc_set(w, start, count, RDMA_M, 0x504c, 0)
        _rc_set(w, start, count, RDMA_M, 0x506c, 0)

    for idx, val in enumerate(w):
        struct.pack_into("<Q", body, idx * 8, val)


MAX_N_6TILE = MAX_TILES * MAX_C1_PER_TILE * SURF_W


def _fb_u16(data, off):
    return struct.unpack_from("<H", data, off)[0]

def _fb_u32(data, off):
    return struct.unpack_from("<I", data, off)[0]

def _fb_i32(data, off):
    return struct.unpack_from("<i", data, off)[0]

def _fb_field_offset(data, pos, field):
    vt = pos - _fb_i32(data, pos)
    vts = _fb_u16(data, vt)
    entry = vt + 4 + field * 2
    if entry + 2 > vt + vts:
        return 0
    return _fb_u16(data, entry)

def _fb_field_abs(data, pos, field):
    off = _fb_field_offset(data, pos, field)
    return pos + off if off else None

def _fb_string(data, pos, field):
    ab = _fb_field_abs(data, pos, field)
    if ab is None:
        return None
    tgt = ab + _fb_u32(data, ab)
    n = _fb_u32(data, tgt)
    return data[tgt + 4:tgt + 4 + n].decode("ascii", errors="replace")

def _fb_vec_target(data, pos, field):
    ab = _fb_field_abs(data, pos, field)
    if ab is None:
        return None
    return ab + _fb_u32(data, ab)

def _fb_vec_u32(data, off):
    n = _fb_u32(data, off)
    return [_fb_u32(data, off + 4 + k * 4) for k in range(n)]

def _fb_tensor_positions(data):
    root = _fb_u32(data, 0)
    sg_abs = _fb_field_abs(data, root, 2)
    if sg_abs is None:
        return []
    sg_vec = sg_abs + _fb_u32(data, sg_abs)
    n_sg = _fb_u32(data, sg_vec)
    if n_sg < 1:
        return []
    sg = (sg_vec + 4) + _fb_u32(data, sg_vec + 4)
    tvec_abs = _fb_field_abs(data, sg, 0)
    if tvec_abs is None:
        return []
    tvec = tvec_abs + _fb_u32(data, tvec_abs)
    n = _fb_u32(data, tvec)
    out = []
    for i in range(n):
        entry = tvec + 4 + i * 4
        out.append(entry + _fb_u32(data, entry))
    return out

def _fb_set_vec(data, pos, field, vals):
    ab = _fb_field_abs(data, pos, field)
    if ab is None:
        return False
    tgt = ab + _fb_u32(data, ab)
    n = _fb_u32(data, tgt)
    if n != len(vals):
        return False
    for k, v in enumerate(vals):
        struct.pack_into("<I", data, tgt + 4 + 4 * k, v)
    return True

def _fb_set_scalar(data, pos, field, val):
    ab = _fb_field_abs(data, pos, field)
    if ab is None:
        return
    struct.pack_into("<I", data, ab, val)


def _regcmd_spans(body):
    n = len(body) // 8
    w = list(struct.unpack_from(f"<{n}Q", body, 0))
    out = []
    i = 0
    while i < n:
        if ((w[i] >> 48) & 0xFFFF) in _TARGETS:
            j = i
            while j < n and ((w[j] >> 48) & 0xFFFF) in _TARGETS:
                j += 1
            if j - i >= 20:
                out.append((i, j - i))
            i = j
        else:
            i += 1
    return out, w


def _plan_memory(body, N, n_inputs):
    C1, W = surface_split(N)
    Npad = C1 * W
    al = lambda s, a=256: ((s + a - 1) // a) * a
    ext, work = al(N * 2), al(Npad * 16)
    plan = {}
    for nm, (slot, is_work) in _memory_plan_for(n_inputs).items():
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
    for tp in _fb_tensor_positions(body):
        nm = _fb_string(body, tp, 5)
        if nm in plan:
            f3, f4, f12, f13 = plan[nm]
            _fb_set_vec(body, tp, 3, f3)
            _fb_set_vec(body, tp, 4, f4)
            _fb_set_scalar(body, tp, 12, f12)
            if f13 is not None:
                _fb_set_scalar(body, tp, 13, f13)


def _patch_tiles(body, N, n_inputs):
    C1, W = surface_split(N)
    tiles = tile_split(C1)
    n_tiles = len(tiles)
    spans, w = _regcmd_spans(body)
    n_adds = n_inputs - 1
    spans_per_add = len(spans) // n_adds
    st = W * 16
    for g in range(n_adds):
        group = spans[g * spans_per_add:(g + 1) * spans_per_add]
        for tile_idx, (i, c) in enumerate(group):
            tidx = min(tile_idx, n_tiles - 1)
            C1_tile, surf_offset = tiles[tidx]
            ch = C1_tile * 8 - 1
            base = surf_offset * st
            _rc_patch_block(w, i, c, W, ch, base, st)
    for idx, val in enumerate(w):
        struct.pack_into("<Q", body, idx * 8, val)


def build_body(N, n_inputs, rows=None, cols=None, ops=None):
    return _build_body_scratch_flatbuffers(N, n_inputs, ops)


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


def _io(n):
    if n == 2:
        return ["x", "y"], "z"
    names = _io_names(n + 1)
    return names[:n], names[n]


def _io_names(count):
    # First 26 names are a..z (keeps byte-exact parity with toolkit reference
    # bodies for n<=25). Beyond that, spreadsheet-style names aa, ab, ... so the
    # input/output count is effectively unbounded. None collide with the
    # intermediate "t<k>-rs" tensors (those always start with 't' + digits).
    L = "abcdefghijklmnopqrstuvwxyz"
    out = []
    i = 0
    while len(out) < count:
        if i < 26:
            out.append(L[i])
        else:
            j = i - 26
            out.append(L[j // 26] + L[j % 26])
        i += 1
    return out


def _mem(n):
    ins, outp = _io(n)
    p = {}
    for i, nm in enumerate(ins):
        p[nm] = (i, False)
    p[outp] = (n, False)
    p[f"{outp}-rs"] = (0, True)
    for i, nm in enumerate(ins):
        p[f"{nm}_rs"] = (i + 1, True)
    ni = n - 2
    for k in range(1, ni + 1):
        nm = "t-rs" if ni == 1 else f"t{k}-rs"
        p[nm] = (None if k == 1 else n + k - 1, True)
    return p


def _ev(b):
    b.StartVector(4, 0, 4)
    return b.EndVector()


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


def _vec_u8(b, data):
    b.StartVector(1, len(data), 1)
    for v in reversed(data):
        b.PrependByte(v)
    return b.EndVector()


def _vec_pairs(b, pairs):
    b.StartVector(8, len(pairs), 4)
    for a, c in reversed(pairs):
        b.PrependUint32(c)
        b.PrependUint32(a)
    return b.EndVector()


def _str(b, s):
    return b.CreateString(s)


def _align(s, a=256):
    return ((s + a - 1) // a) * a


def _u32_table2(b, a, c):
    b.StartObject(2)
    b.PrependUint32Slot(1, c, 0)
    b.PrependUint32Slot(0, a, 0)
    return b.EndObject()


def _u32_table3(b, a, c, d):
    b.StartObject(3)
    b.PrependUint32Slot(2, d, 0)
    b.PrependUint32Slot(1, c, 0)
    b.PrependUint32Slot(0, a, 0)
    return b.EndObject()


def _vec_table3(b, a, c, d):
    va, vc, vd = _vec(b, a), _vec(b, c), _vec(b, d)
    b.StartObject(3)
    b.PrependUOffsetTRelativeSlot(2, vd, 0)
    b.PrependUOffsetTRelativeSlot(1, vc, 0)
    b.PrependUOffsetTRelativeSlot(0, va, 0)
    return b.EndObject()


def _str_vec_table3(b, name, c, d):
    vn, vc, vd = _str(b, name), _vec_pairs(b, c), _vec(b, d)
    b.StartObject(3)
    b.PrependUOffsetTRelativeSlot(2, vd, 0)
    b.PrependUOffsetTRelativeSlot(1, vc, 0)
    b.PrependUOffsetTRelativeSlot(0, vn, 0)
    return b.EndObject()


def _str_scalar_table2(b, name, val):
    vn = _str(b, name)
    b.StartObject(2)
    b.PrependUint32Slot(1, val, 0)
    b.PrependUOffsetTRelativeSlot(0, vn, 0)
    return b.EndObject()


def _u64_table2(b, a, c):
    b.StartObject(2)
    b.PrependUint64Slot(1, c, 0)
    b.PrependUint64Slot(0, a, 0)
    return b.EndObject()


def _root_attrs(n_inputs, N):
    ins, outp = _io(n_inputs)
    side = math.isqrt(N)
    shape = [side, side] if side * side == N else [1, N]
    attrs = {}
    quant = {}
    for i, nm in enumerate(ins):
        attrs[nm] = {
            "idx": i, "shape": shape, "layout": "nchw", "layout_ori": "nchw",
            "is_output": False, "range": [0, 1], "origin_dynamic": False,
            "dtype": "float32", "mean": [0] * 10, "std": [1] * 10,
            "rgb2bgr": False,
        }
        quant[nm] = {
            "dtype": "float32", "qmethod": "", "qtype": "", "min": [],
            "max": [], "scale": [], "zero_point": [], "name": nm,
            "shape": shape,
        }
    attrs[outp] = {
        "is_output": True, "idx": 0, "shape": shape, "dtype": "float32",
        "layout": "nchw",
    }
    quant[outp] = {
        "dtype": "float16", "qmethod": "", "qtype": "", "min": [],
        "max": [], "scale": [], "zero_point": [], "name": outp,
        "shape": shape,
    }
    return str({"attrs": attrs, "quant_tab": quant, "dynamic_shapes": {}})


def _generate_sg_f7_specs(n_inputs):
    ins, outp = _io(n_inputs)
    n_adds = n_inputs - 1
    n_inter = n_inputs - 2
    step = 80 * n_adds
    _IO_OFF = [0, 0, 192, 0, 64, 112]
    _IO_MASK = [1, 0, 1, 0, 1, 1]
    _INT_MASK = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
    specs = []
    for k, nm in enumerate(ins):
        base = 55 if k == 0 else (61 if k == 1 else 141 + (k - 2) * 80)
        specs.append((f"{nm}_rs",
                      [(_IO_OFF[i], base + i * step) for i in range(6)],
                      list(_IO_MASK)))
    out_base = 5 + (n_inputs - 2) * 80
    specs.append((f"{outp}-rs",
                  [(_IO_OFF[i], out_base + i * step) for i in range(6)],
                  list(_IO_MASK)))
    for k in range(1, n_inter + 1):
        t_name = "t-rs" if n_inter == 1 else f"t{k}-rs"
        g1 = 5 + (k - 1) * 80
        g2 = 135 + (k - 1) * 80
        pairs = []
        for i in range(6):
            pairs.append((_IO_OFF[i], g1 + i * step))
            pairs.append((_IO_OFF[i], g2 + i * step))
        specs.append((t_name, pairs, list(_INT_MASK)))
    return specs


def _generate_exsec_f13(n):
    _CYCLE = [384, 768, 1152, None]
    ins, outp = _io(n)
    result = {}
    for i, nm in enumerate(ins):
        if i <= 1:
            result[nm] = 320 * n
        else:
            result[nm] = _CYCLE[(i - 2) % 4]
    result[f"{outp}-rs"] = 384 if n % 4 == 2 else None
    return result



def _rc_set(w, b, c, t, reg, val):
    for k in range(c):
        r = w[b + k]
        if ((r >> 48) & 0xFFFF) == t and (r & 0xFFFF) == reg:
            w[b + k] = (r & ~(0xFFFFFFFF << 16)) | ((val & 0xFFFFFFFF) << 16)
            return


def _rc_patch_block(w, i, c, W, ch, base, st, op="Add"):
    for reg in (0x4030, 0x405c):
        _rc_set(w, i, c, _DPU, reg, W - 1)
    _rc_set(w, i, c, _RDMA, 0x500c, W - 1)
    ch_val = ((W - 1) << 16 | ch) if op != "Add" else (ch << 16 | ch)
    _rc_set(w, i, c, _DPU, 0x403c, ch_val)
    _rc_set(w, i, c, _DPU, 0x4058, ch)
    _rc_set(w, i, c, _RDMA, 0x5014, ch)
    _rc_set(w, i, c, _DPU, 0x4034, 0)
    _rc_set(w, i, c, _RDMA, 0x5010, 0)
    _rc_set(w, i, c, _DPU, 0x4020, base)
    _rc_set(w, i, c, _RDMA, 0x5018, base)
    _rc_set(w, i, c, _RDMA, 0x5038, base)
    _rc_set(w, i, c, _DPU, 0x4024, st)
    _rc_set(w, i, c, _DPU, 0x40c0, st)
    _rc_set(w, i, c, _RDMA, 0x5040, st)
    _rc_set(w, i, c, _RDMA, 0x504c, 0)
    _rc_set(w, i, c, _RDMA, 0x506c, 0)


def _tensor_indices(n_inputs, ins, outp, n_adds):
    idx = {}
    t = 0
    idx["empty"] = t; t += 1
    for nm in ins:
        idx[f"{nm}_rsi1"] = t; t += 1
    idx[f"{outp}_rsi1"] = t; t += 1
    idx[ins[0]] = t; t += 1
    for nm in ins[1:]:
        idx[nm] = t; t += 1
    for i, nm in enumerate(ins):
        idx[f"{nm}_exsec"] = t; t += 1
        idx[f"{nm}_rs"] = t; t += 1
        if 1 <= i <= n_inputs - 2:
            inter = "t-rs" if n_inputs == 3 else f"t{i}-rs"
            idx[inter] = t; t += 1
    idx[f"{outp}_rs"] = t; t += 1
    idx[f"{outp}_rs_exsec"] = t; t += 1
    idx[outp] = t; t += 1
    idx["regcmd"] = t; t += 1
    idx["task"] = t; t += 1
    return idx


def _build_tensors(b, n_inputs, ins, outp, n_adds, plan, idx,
                   s2, s4, s5, N, Npad, ext, work, C1, W):
    toffs = [None] * (idx["task"] + 1)
    sec2 = [10, 10]
    sec4 = [1, 10, 1, 10]

    toffs[idx["task"]] = _cmd_tensor(b, "task", f2=10, f18=n_adds + 3 + 1)
    toffs[idx["regcmd"]] = _cmd_tensor(b, "regcmd", f2=9, f18=n_adds + 3)

    toffs[idx[outp]] = _ext_tensor(b, outp, s2, N * 2,
                                    plan[outp][0] * ext, f2=2)
    out_exsec_f13 = _generate_exsec_f13(n_inputs)[f"{outp}-rs"]
    toffs[idx[f"{outp}_rs_exsec"]] = _exsec_tensor(
        b, f"{outp}-rs_exSecondary", sec4, 1, out_exsec_f13, f1=2)
    toffs[idx[f"{outp}_rs"]] = _rs_tensor(
        b, f"{outp}-rs", s5, s4, Npad * 16,
        plan[f"{outp}-rs"][0] * work, has_f13=True,
        emit_zero_f13=n_inputs > 2)

    for name, (slot, _) in plan.items():
        if name.startswith("t") and name.endswith("-rs") and name in idx:
            offset = 0 if slot is None else slot * work
            toffs[idx[name]] = _rs_tensor(
                b, name, s5, s4, Npad * 16, offset, has_f13=slot is not None)

    for nm in reversed(ins):
        rs_name = f"{nm}_rs"
        toffs[idx[f"{nm}_rs"]] = _rs_tensor(
            b, rs_name, s5, s4, Npad * 16,
            plan[rs_name][0] * work, has_f13=True)
        toffs[idx[f"{nm}_exsec"]] = _exsec_tensor(
            b, f"{nm}_exSecondary", sec2, 1, _generate_exsec_f13(n_inputs)[nm], f1=None)

    for nm in reversed(ins[1:]):
        toffs[idx[nm]] = _ext_tensor(b, nm, s2, N * 2, plan[nm][0] * ext)

    toffs[idx[ins[0]]] = _ext_tensor(b, ins[0], s2, N * 2, f13=0)

    for i in range(n_inputs, -1, -1):
        all_names = ins + [outp]
        nm = all_names[i]
        name_str = f"{nm}_rs_i1" if nm in ins else f"{outp}-rs_i1"
        f12 = 16 if i == n_inputs else 32
        toffs[idx[f"{all_names[i]}_rsi1"]] = _rsi1_tensor(b, name_str, i + 1, f12)

    toffs[0] = _empty_tensor(b)
    return toffs


def _build_nodes(b, n_inputs, ins, outp, n_adds, idx, ops=None):
    noffs = []

    for nm in ins:
        noffs.append(_input_node(b, nm, idx[nm]))

    for nm in ins[:2]:
        noffs.append(_reshape_node(
            b, f"{nm}_rs", nm,
            [idx[nm], idx[f"{nm}_rsi1"], idx[f"{nm}_exsec"]],
            [idx[f"{nm}_rs"]]))

    for k in range(n_adds):
        if k == 0:
            rs_a = f"{ins[0]}_rs"
        else:
            rs_a = "t-rs" if n_inputs == 3 else f"t{k}-rs"
        rs_b = f"{ins[k + 1]}_rs"
        if k == n_adds - 1:
            rs_out = f"{outp}_rs"
        else:
            rs_out = "t-rs" if n_inputs == 3 else f"t{k + 1}-rs"
        op_name = (ops[k] if ops and k < len(ops) else None) or "Add"
        noffs.append(_add_node(b, k + 1, idx[rs_a], idx[rs_b], idx[rs_out], op_name))

        next_input = k + 2
        if next_input < n_inputs:
            nm = ins[next_input]
            noffs.append(_reshape_node(
                b, f"{nm}_rs", nm,
                [idx[nm], idx[f"{nm}_rsi1"], idx[f"{nm}_exsec"]],
                [idx[f"{nm}_rs"]]))

    noffs.append(_reshape_node(b, f"{outp}-rs", outp,
                               [idx[f"{outp}_rs"], idx[f"{outp}_rsi1"],
                                idx[f"{outp}_rs_exsec"]],
                               [idx[outp]]))
    noffs.append(_output_node(b, outp, idx[outp]))
    return noffs


def _build_subgraph(b, toffs, noffs, n_adds, idx, ins, outp, n_external=None):
    tvec = _ovec(b, toffs)
    nvec = _ovec(b, noffs)
    n_inputs = len(ins)
    n_ext = n_external if n_external is not None else n_inputs
    sg_f4 = _ovec(b, [
        _vec_table3(b, [0] * 10, [0x3f800000] * 10, list(range(10)))
        for _ in ins
    ])
    sg_f7_tables = [
        _str_vec_table3(b, name, vals, mask)
        for name, vals, mask in _generate_sg_f7_specs(n_inputs)
    ]
    sg_f7 = _ovec(b, sg_f7_tables)
    sg_f10 = _vec(b, [0, 0, 0, n_adds, n_adds * 3])
    sg_f12 = _ovec(b, [_str_scalar_table2(b, outp, n_adds)])
    sg_f2 = _vec(b, [idx[ins[i]] for i in range(n_ext)])
    sg_f3 = _vec(b, [idx[outp]])
    evs = [_ev(b) for _ in range(7)]

    b.StartObject(17)
    b.PrependUOffsetTRelativeSlot(16, evs[0], 0)
    b.PrependUOffsetTRelativeSlot(15, evs[1], 0)
    b.PrependUOffsetTRelativeSlot(14, evs[2], 0)
    b.PrependUOffsetTRelativeSlot(13, evs[3], 0)
    b.PrependUOffsetTRelativeSlot(12, sg_f12, 0)
    b.PrependUOffsetTRelativeSlot(10, sg_f10, 0)
    b.PrependUOffsetTRelativeSlot(9, evs[4], 0)
    b.PrependUOffsetTRelativeSlot(8, evs[5], 0)
    b.PrependUOffsetTRelativeSlot(7, sg_f7, 0)
    b.PrependUOffsetTRelativeSlot(6, evs[6], 0)
    b.PrependUOffsetTRelativeSlot(4, sg_f4, 0)
    b.PrependUOffsetTRelativeSlot(3, sg_f3, 0)
    b.PrependUOffsetTRelativeSlot(2, sg_f2, 0)
    b.PrependUOffsetTRelativeSlot(1, nvec, 0)
    b.PrependUOffsetTRelativeSlot(0, tvec, 0)
    sg = b.EndObject()

    b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(sg)
    return b.EndVector()

def _build_root(b, sgvec, n_adds, C1, W, tiles, N, n_inputs, ops=None, n_rc_inputs=None):
    rc_n = n_rc_inputs if n_rc_inputs is not None else n_inputs
    ins, outp = _io(rc_n)
    s_target = _str(b, "RKNPU v2")
    s_toolkit = _str(b, "2.3.2(compiler version: 2.3.2 (@2025-04-03T08:26:16))")
    s_platform = _str(b, "rk3588")
    s_framework = _str(b, "ONNX")
    dtype_in = {nm: {"dtype": "float16", "layout": "UNDEFINED"} for nm in ins}
    dtype_out = {outp: {"dtype": "float16", "layout": "NCHW"}}
    import json
    root_f12 = _str(b, json.dumps(dtype_in, separators=(", ", ": ")))
    root_f13 = _str(b, json.dumps(dtype_out, separators=(", ", ": ")))
    root_f14 = _vec_u8(b, b"0")
    root_f15 = _ev(b)
    root_f16 = _str(b, "static_shape")
    root_f17 = _ev(b)
    root_f18 = _ev(b)
    root_f19 = _u64_table2(b, 192 + (rc_n - 2) * 64, 1472 + (rc_n - 2) * 320)
    root_ev3 = _str(b, _root_attrs(rc_n, N))
    root_ev11 = _str(b, "")
    b.StartObject(22)
    b.PrependUint32Slot(21, 1, 0)
    b.PrependUint32Slot(20, 1, 0)
    b.PrependUOffsetTRelativeSlot(19, root_f19, 0)
    b.PrependUOffsetTRelativeSlot(18, root_f18, 0)
    b.PrependUOffsetTRelativeSlot(17, root_f17, 0)
    b.PrependUOffsetTRelativeSlot(16, root_f16, 0)
    b.PrependUOffsetTRelativeSlot(15, root_f15, 0)
    b.PrependUOffsetTRelativeSlot(14, root_f14, 0)
    b.PrependUOffsetTRelativeSlot(13, root_f13, 0)
    b.PrependUOffsetTRelativeSlot(12, root_f12, 0)
    b.PrependUOffsetTRelativeSlot(11, root_ev11, 0)
    b.PrependUint8Slot(10, 2, 0)
    b.PrependUOffsetTRelativeSlot(9, s_framework, 0)
    b.PrependUOffsetTRelativeSlot(8, s_platform, 0)
    b.PrependUOffsetTRelativeSlot(7, s_toolkit, 0)
    b.PrependUint32Slot(6, 20302, 0)
    b.PrependUOffsetTRelativeSlot(3, root_ev3, 0)
    b.PrependUOffsetTRelativeSlot(2, sgvec, 0)
    b.PrependUOffsetTRelativeSlot(1, s_target, 0)
    b.PrependUint32Slot(0, 6, 0)
    root = b.EndObject()

    b.Finish(root, b"RKNN")
    fb = bytearray(b.Output())
    ew_ops = None
    if ops:
        ew_ops = [rc_template_gen.ew_op_id(o) for o in ops]
    rc_n = n_rc_inputs if n_rc_inputs is not None else n_inputs
    rc_raw = rc_template_gen.build_template(rc_n, ew_ops)
    taskdesc = _taskdesc(rc_n)

    full = fb + rc_raw + taskdesc
    rc_len = len(rc_raw)
    fb_len = len(fb)
    root_off = struct.unpack_from("<I", full, 0)[0]
    if root_off == 60 and full[8:12] == b"\x00\x00\x00\x00":
        if rc_n == 2:
            del full[8:12]
            struct.pack_into("<I", full, 0, 56)
            fb_len -= 4
        else:
            full[8:8] = b"\x00\x00\x00\x00"
            struct.pack_into("<I", full, 0, 64)
            fb_len += 4
    elif root_off == 56 and rc_n in (3, 4):
        full[8:8] = b"\x00" * 8
        struct.pack_into("<I", full, 0, 64)
        fb_len += 8

    required_mod = 4 if rc_n % 2 == 0 else 0
    pad = (required_mod - fb_len % 8) % 8
    if pad:
        full[fb_len:fb_len] = b"\x00" * pad
        fb_len += pad

    _patch_regcmd(full, fb_len, n_adds, C1, W, tiles, ops)

    rc_target = _rc_target_offset(rc_raw, rc_n)
    root_rt = struct.unpack_from("<I", full, 0)[0]
    for field_idx, offset in [(20, rc_target), (21, rc_len + rc_target + 4)]:
        ab = _fb_field_abs(full, root_rt, field_idx)
        struct.pack_into("<I", full, ab, fb_len + offset - ab)

    return bytes(full[:fb_len]), bytes(full[fb_len:])


def _rc_target_offset(rc, n_inputs):
    target_val = n_inputs + 4
    n_u32 = len(rc) // 4
    zero_run = 0
    for i in range(n_u32):
        v = struct.unpack_from("<I", rc, i * 4)[0]
        if v == 0:
            zero_run += 1
        else:
            if zero_run >= 8 and v == target_val:
                return i * 4
            zero_run = 0
    raise ValueError(f"RC target not found for n_inputs={n_inputs}")


def _patch_root_f20_f21(fb, f20, f21):
    rt = struct.unpack_from("<I", fb, 0)[0]
    vt = rt - struct.unpack_from("<i", fb, rt)[0]
    vts = struct.unpack_from("<H", fb, vt)[0]
    nf = (vts - 4) // 2
    if nf < 22:
        return
    for field_idx, val in [(20, f20), (21, f21)]:
        entry = vt + 4 + field_idx * 2
        off = struct.unpack_from("<H", fb, entry)[0]
        if off:
            struct.pack_into("<I", fb, rt + off, val)


# Task command-tensor data (the embedded "task" tensor). A sequence of 64-byte
# (8-word) reshape descriptors: word0 hi-32 = the reshape-info tensor's f12 byte
# size (0x10 output / 0x20 input), following words' lo-32 = its dims, zero-padded;
# bracketed by leading/trailing zero words. Shape-independent (never patched), so
# it is generated rather than stored as a literal.
_TD_F12_OUT, _TD_F12_IN = 0x10, 0x20    # f12 of output / input reshape-info tensors
_TD_DIM = 0x0a                          # template dim (cosmetic, from add_10x10)
_TD_REC_WORDS = 8                       # 64 bytes per descriptor


def _td_rec(f12, dims):
    return [f12 << 32] + list(dims) + [0] * (_TD_REC_WORDS - 1 - len(dims))


def _taskdesc(n_inputs):
    if n_inputs < 2:
        raise ValueError(f"taskdesc not available for {n_inputs} inputs")
    rec_out = _td_rec(_TD_F12_OUT, [_TD_DIM, _TD_DIM])
    rec_in = _td_rec(_TD_F12_IN, [1, _TD_DIM, 1, _TD_DIM])
    if n_inputs == 2:
        words = [0] + rec_out + rec_in + rec_in + [0]
    else:
        leading_zeros = n_inputs // 3 + 1
        # Taskdesc lengths of 288 + 64*k bytes (leading_zeros 11, 19, 27, ...
        # with the 3 reshape records + bracket zeros) are rejected by the
        # runtime's ModelBuffer verifier; every neighbouring length loads. The
        # records are cosmetic/never-patched, so nudge the zero-pad off any bad
        # length. Affects n_inputs in {30,31,32}, {54,55,56}, ... ; n<=25 is
        # unaffected (leading_zeros < 11 there).
        if leading_zeros >= 11 and (leading_zeros - 11) % 8 == 0:
            leading_zeros += 1
        words = [0] * leading_zeros + rec_in * 3 + [0]
    return struct.pack(f"<{len(words)}Q", *words)


def _patch_regcmd(full, rc_offset, n_adds, C1, W, tiles, ops=None):
    n = len(full) // 8
    spans, w = _regcmd_spans(bytes(full))
    n_tiles = len(tiles)
    spans_per_add = len(spans) // n_adds
    st = W * 16

    for g in range(n_adds):
        group = spans[g::n_adds]
        op_name = (ops[g] if ops and g < len(ops) else None) or "Add"
        op_id = rc_template_gen.ew_op_id(op_name)
        ew_cfg_val = rc_template_gen._ew_cfg(op_id)
        out_res = rc_template_gen._DPU_OUT_RES.get(op_id, 0x00010001)
        bn_mul = rc_template_gen._RDMA_BN_MUL.get(op_id, 0x00017849)
        for tile_idx, (i, c) in enumerate(group):
            tidx = min(tile_idx, n_tiles - 1)
            C1_tile, surf_offset = tiles[tidx]
            ch = C1_tile * 8 - 1
            base = surf_offset * st
            _rc_patch_block(w, i, c, W, ch, base, st, op_name)
            _rc_set(w, i, c, _DPU, 0x4070, ew_cfg_val)
            _rc_set(w, i, c, _DPU, 0x4084, out_res)
            _rc_set(w, i, c, _RDMA, 0x5044, bn_mul)

    for idx, val in enumerate(w):
        struct.pack_into("<Q", full, idx * 8, val)


def _empty_tensor(b):
    nm = _str(b, "")
    evs = [_ev(b) for _ in range(11)]
    b.StartObject(18)
    for i, e in [(17, evs[0]), (16, evs[1]), (15, evs[2]), (11, evs[3]),
                 (10, evs[4]), (9, evs[5]), (8, evs[6]), (7, evs[7]), (6, evs[8])]:
        b.PrependUOffsetTRelativeSlot(i, e, 0)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, evs[9], 0)
    b.PrependUOffsetTRelativeSlot(3, evs[10], 0)
    return b.EndObject()


def _rsi1_tensor(b, name, f18, f12):
    nm = _str(b, name)
    sh = _vec(b, [f12 // 8])
    evs = [_ev(b) for _ in range(9)]
    b.StartObject(19)
    b.PrependUint32Slot(18, f18, 0)
    b.PrependUint32Slot(12, f12, 0)
    for i, e in [(17, evs[0]), (16, evs[1]), (15, evs[2]), (11, evs[3]),
                 (10, evs[4]), (9, evs[5]), (8, evs[6]), (7, evs[7]), (6, evs[8])]:
        b.PrependUOffsetTRelativeSlot(i, e, 0)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, sh, 0)
    b.PrependUOffsetTRelativeSlot(3, sh, 0)
    b.PrependUint8Slot(2, 5, 0)
    b.PrependUint8Slot(0, 7, 0)
    return b.EndObject()


def _ext_tensor(b, name, shape, size, f13=0, f2=1):
    nm = _str(b, name)
    sh = _vec(b, shape)
    evs = [_ev(b) for _ in range(9)]
    b.StartObject(18)
    b.PrependUint32Slot(13, f13, 0)
    b.PrependUint32Slot(12, size, 0)
    for i, e in [(17, evs[0]), (16, evs[1]), (15, evs[2]), (11, evs[3]),
                 (10, evs[4]), (9, evs[5]), (8, evs[6]), (7, evs[7]), (6, evs[8])]:
        b.PrependUOffsetTRelativeSlot(i, e, 0)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, sh, 0)
    b.PrependUOffsetTRelativeSlot(3, sh, 0)
    b.PrependUint8Slot(2, f2, 0)
    b.PrependUint8Slot(0, 10, 0)
    return b.EndObject()


def _exsec_tensor(b, name, shape, size, offset, f1=0):
    nm = _str(b, name)
    sh = _vec(b, shape)
    evs = [_ev(b) for _ in range(9)]
    b.StartObject(18)
    if offset is not None:
        b.PrependUint32Slot(13, offset, 0xFFFFFFFF)
    b.PrependUint32Slot(12, size, 0)
    for i, e in [(17, evs[0]), (16, evs[1]), (15, evs[2]), (11, evs[3]),
                 (10, evs[4]), (9, evs[5]), (8, evs[6]), (7, evs[7]), (6, evs[8])]:
        b.PrependUOffsetTRelativeSlot(i, e, 0)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, sh, 0)
    b.PrependUOffsetTRelativeSlot(3, sh, 0)
    b.PrependUint8Slot(2, 3, 0)
    if f1 is not None:
        b.PrependUint8Slot(1, f1, 1)
    b.PrependUint8Slot(0, 10, 0)
    return b.EndObject()


def _rs_tensor(b, name, s5, s4, size, offset, has_f13=True, emit_zero_f13=False):
    nm = _str(b, name)
    v5, v4 = _vec(b, s5), _vec(b, s4)
    evs = [_ev(b) for _ in range(9)]
    b.StartObject(20)
    b.PrependUint32Slot(19, 4, 0)
    if has_f13 and emit_zero_f13 and offset == 0:
        b.PrependUint32(0)
        b.Slot(13)
    elif has_f13 and offset != 0:
        b.PrependUint32Slot(13, offset, 0)
    b.PrependUint32Slot(12, size, 0)
    for i, e in [(17, evs[0]), (16, evs[1]), (15, evs[2]), (11, evs[3]),
                 (10, evs[4]), (9, evs[5]), (8, evs[6]), (7, evs[7]), (6, evs[8])]:
        b.PrependUOffsetTRelativeSlot(i, e, 0)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, v4, 0)
    b.PrependUOffsetTRelativeSlot(3, v5, 0)
    b.PrependUint8Slot(2, 3, 0)
    b.PrependUint8Slot(1, 64, 0)
    b.PrependUint8Slot(0, 10, 0)
    return b.EndObject()


def _cmd_tensor(b, name, f2, f18):
    nm = _str(b, name)
    evs = [_ev(b) for _ in range(11)]
    b.StartObject(19)
    b.PrependUint32Slot(18, f18, 0)
    for i, e in [(17, evs[0]), (16, evs[1]), (15, evs[2]), (11, evs[3]),
                 (10, evs[4]), (9, evs[5]), (8, evs[6]), (7, evs[7]), (6, evs[8])]:
        b.PrependUOffsetTRelativeSlot(i, e, 0)
    b.PrependUOffsetTRelativeSlot(5, nm, 0)
    b.PrependUOffsetTRelativeSlot(4, evs[9], 0)
    b.PrependUOffsetTRelativeSlot(3, evs[10], 0)
    b.PrependUint8Slot(2, f2, 0)
    b.PrependUint8Slot(0, 13, 0)
    return b.EndObject()


def _input_node(b, name, tensor_idx):
    op_s = _str(b, "InputOperator")
    nm_s = _str(b, f"InputOperator:{name}")
    f5 = _vec(b, [tensor_idx])
    f9 = _ev(b)
    f4 = _ev(b)
    f10 = _vec(b, [0, 0, 0, 0, 0, 0])
    f11 = _vec(b, [0, 0, 0, 0, 0, 0])
    f12 = _vec(b, [0, 0, 0, 0, 0, 0, 0, 0, 0])
    b.StartObject(13)
    b.PrependUOffsetTRelativeSlot(12, f12, 0)
    b.PrependUOffsetTRelativeSlot(11, f11, 0)
    b.PrependUOffsetTRelativeSlot(10, f10, 0)
    b.PrependUOffsetTRelativeSlot(9, f9, 0)
    b.PrependUOffsetTRelativeSlot(5, f5, 0)
    b.PrependUOffsetTRelativeSlot(4, f4, 0)
    b.PrependUOffsetTRelativeSlot(2, nm_s, 0)
    b.PrependUOffsetTRelativeSlot(1, op_s, 0)
    return b.EndObject()


def _reshape_node(b, rs_name, src_name, input_indices, output_indices):
    op_s = _str(b, "Reshape")
    nm_s = _str(b, f"Reshape:{rs_name}")
    f4 = _vec(b, input_indices)
    f5 = _vec(b, output_indices)
    f9 = _ev(b)
    f10 = _vec(b, [0, 0, 0, 0, 0, 0])
    f11 = _vec(b, [0, 0, 0, 0, 0, 0])
    f12 = _vec(b, [0, 0, 0, 0, 0, 0, 0, 0, 0])
    b.StartObject(13)
    b.PrependUOffsetTRelativeSlot(12, f12, 0)
    b.PrependUOffsetTRelativeSlot(11, f11, 0)
    b.PrependUOffsetTRelativeSlot(10, f10, 0)
    b.PrependUOffsetTRelativeSlot(9, f9, 0)
    b.PrependUOffsetTRelativeSlot(5, f5, 0)
    b.PrependUOffsetTRelativeSlot(4, f4, 0)
    b.PrependUOffsetTRelativeSlot(2, nm_s, 0)
    b.PrependUOffsetTRelativeSlot(1, op_s, 0)
    return b.EndObject()


def _add_node(b, add_num, in_a_idx, in_b_idx, out_idx, op="Add"):
    op_s = _str(b, op)
    nm_s = _str(b, f"{op}:{op.lower()}{add_num}")
    f4 = _vec(b, [in_a_idx, in_b_idx])
    f5 = _vec(b, [out_idx])
    f9 = _ev(b)
    f10 = _vec(b, [1, 1, 1, 1, 1, 1])
    f11 = _vec(b, [0, 0, 0, 0, 0, 0])
    f12 = _vec(b, [160, 0, 0, 80, 80, 0, 64, 48, 48])
    b.StartObject(13)
    b.PrependUOffsetTRelativeSlot(12, f12, 0)
    b.PrependUOffsetTRelativeSlot(11, f11, 0)
    b.PrependUOffsetTRelativeSlot(10, f10, 0)
    b.PrependUOffsetTRelativeSlot(9, f9, 0)
    b.PrependUOffsetTRelativeSlot(5, f5, 0)
    b.PrependUOffsetTRelativeSlot(4, f4, 0)
    b.PrependUint8Slot(3, 2, 0)
    b.PrependUOffsetTRelativeSlot(2, nm_s, 0)
    b.PrependUOffsetTRelativeSlot(1, op_s, 0)
    return b.EndObject()


def _output_node(b, outp, tensor_idx):
    op_s = _str(b, "OutputOperator")
    nm_s = _str(b, f"OutputOperator:{outp}")
    full_name = f"OutputOperator:{outp}"
    name_bytes = full_name.encode("ascii")
    op_bytes = b"OutputOperator"
    f9_data = [0, 0, 0, 0, 0, 1, tensor_idx, len(name_bytes)]
    name_aligned = name_bytes[:len(name_bytes) - len(name_bytes) % 4] if len(name_bytes) % 4 else name_bytes
    f9_data += list(struct.unpack(f"<{len(name_aligned)//4}I", name_aligned))
    f9_data += [0, len(op_bytes)]
    op_space = max(0, 16 - len(f9_data))
    op_trunc = op_bytes[:op_space * 4]
    op_padded = op_trunc + b'\x00' * ((4 - len(op_trunc) % 4) % 4)
    if op_padded:
        f9_data += list(struct.unpack(f"<{len(op_padded)//4}I", op_padded))
    f9_data = (f9_data + [0] * 16)[:16]
    f4 = _vec(b, [tensor_idx])
    f5 = _ev(b)
    f9 = _vec(b, f9_data[:16])
    f10 = _vec(b, [0, 0, 0, 0, 0, 0])
    f11 = _vec(b, [0, 0, 0, 0, 0, 0])
    f12 = _vec(b, [0, 0, 0, 0, 0, 0, 0, 0, 0])
    b.StartObject(13)
    b.PrependUOffsetTRelativeSlot(12, f12, 0)
    b.PrependUOffsetTRelativeSlot(11, f11, 0)
    b.PrependUOffsetTRelativeSlot(10, f10, 0)
    b.PrependUOffsetTRelativeSlot(9, f9, 0)
    b.PrependUOffsetTRelativeSlot(5, f5, 0)
    b.PrependUOffsetTRelativeSlot(4, f4, 0)
    b.PrependUOffsetTRelativeSlot(2, nm_s, 0)
    b.PrependUOffsetTRelativeSlot(1, op_s, 0)
    return b.EndObject()


def _make_trailer(rows, cols, n_inputs):
    import json
    ins, outp = _io(n_inputs)
    norm_tensor = []
    for i, nm in enumerate(ins + [outp]):
        norm_tensor.append({
            "dim_num": 2, "dtype": {"qnt_method": "", "qnt_type": "", "vx_type": ""},
            "size": [rows, cols], "tensor_id": i, "url": nm
        })
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
        "norm_tensor": norm_tensor, "norm_tensor_num": n_inputs + 1,
        "ori_network_platform": "ONNX", "output_num": 1,
        "target_platform": ["rk3588"], "version": "2.3.2", "virtual_tensor": [],
    }
    nj = json.dumps(js, separators=(",", ":")).encode()
    return struct.pack("<Q", len(nj)) + nj


def assemble_rknn(body, rows, cols, n_inputs):
    trailer = _make_trailer(rows, cols, n_inputs)
    h = bytearray(HEADER_SIZE)
    h[0:4] = b"RKNN"
    struct.pack_into("<Q", h, 0x08, 6)
    struct.pack_into("<Q", h, 0x10, len(body))
    return bytes(h) + body + trailer


def _root_command_offsets(body):
    root = _fb_u32(body, 0)
    out = []
    for field in (20, 21):
        ab = _fb_field_abs(body, root, field)
        if ab is None:
            raise ValueError(f"template body is missing root field {field}")
        val = _fb_u32(body, ab)
        if val < HEADER_SIZE:
            raise ValueError(f"root field {field} has invalid absolute offset {val}")
        out.append(val - HEADER_SIZE)
    return tuple(out)


def build_body_scratch(N, n_inputs):
    """Rebuild a spec-conformant body from decoded components.

    The RKNN runtime's schema verifier rejects the generic FlatBuffers builder
    layout below even though it is structurally valid FlatBuffers. The accepted
    layout keeps the toolkit-produced FlatBuffer skeleton, regenerates the
    command/template bytes from readable specs, then patches shapes/memory/regcmd.
    """
    if n_inputs not in _RC_TEMPLATES:
        raise ValueError(f"scratch regcmd template not available for {n_inputs} inputs")

    template = _get_template_body(n_inputs)
    fb_end, _ = _root_command_offsets(template)
    rc_raw = _RC_TEMPLATES[n_inputs]
    body = bytearray(template[:fb_end] + rc_raw + _taskdesc(n_inputs))
    _patch_root_f20_f21(body, HEADER_SIZE + fb_end, HEADER_SIZE + fb_end + len(rc_raw))
    _plan_memory(body, N, n_inputs)
    _patch_tiles(body, N, n_inputs)
    return bytes(body)


def _build_body_scratch_flatbuffers(N, n_inputs, ops=None):
    min_ops = max(1, n_inputs - 1)
    n_ops = len(ops) if ops and len(ops) >= min_ops else min_ops
    is_multiop = n_ops > n_inputs - 1
    n_external = n_inputs
    n_virtual = n_ops + 1 if is_multiop else n_inputs
    if n_virtual not in _RC_TEMPLATES:
        raise NotImplementedError(
            f"pure FlatBuffers generation not available for {n_inputs} inputs "
            f"with {n_ops} ops (internal n={n_virtual}); "
            f"available: {sorted(_RC_TEMPLATES)}"
        )
    C1, W = surface_split(N)
    Npad = C1 * W
    n_adds = n_virtual - 1
    tiles = tile_split(C1)
    ins, outp = _io(n_virtual)
    ext = _align(N * 2)
    work = _align(Npad * 16)
    plan = _mem(n_virtual)

    idx = _tensor_indices(n_virtual, ins, outp, n_adds)

    if is_multiop:
        for k in range(n_external, n_virtual):
            tgt = 1 + (k - 1) % (n_external - 1)
            plan[ins[k]] = plan[ins[tgt]]
            plan[f"{ins[k]}_rs"] = plan[f"{ins[tgt]}_rs"]

    s5 = [1, C1, 1, W, 8]
    s4 = [1, C1, 1, W]
    s2 = [1, N]

    idx_nodes = dict(idx)
    if is_multiop:
        for k in range(n_external, n_virtual):
            tgt = 1 + (k - 1) % (n_external - 1)
            for suffix in ["", "_rs", "_exsec", "_rsi1"]:
                src_key = f"{ins[tgt]}{suffix}"
                dst_key = f"{ins[k]}{suffix}"
                if src_key in idx_nodes:
                    idx_nodes[dst_key] = idx_nodes[src_key]

    b = flatbuffers.Builder(65536)

    noffs = _build_nodes(b, n_virtual, ins, outp, n_adds, idx_nodes, ops)
    toffs = _build_tensors(b, n_virtual, ins, outp, n_adds, plan, idx,
                           s2, s4, s5, N, Npad, ext, work, C1, W)
    n_sg2 = n_external if is_multiop else n_virtual
    sg = _build_subgraph(b, toffs, noffs, n_adds, idx, ins, outp, n_sg2)
    fb_bytes, rc_bytes = _build_root(b, sg, n_adds, C1, W, tiles, N, n_inputs, ops,
                                     n_rc_inputs=n_virtual)
    return fb_bytes + rc_bytes
