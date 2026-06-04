"""rc_template_gen.py - Algorithmic generator for rknn_flatbuf._RC_TEMPLATES.

Each RC template (n_inputs = 2..7+) is:

    [OFF[n] zero bytes]                      alignment lead
    PREFIX[n] words                          header + cascade + chains + descriptors
    6 tiles x ( for j in 0..n_adds-1:        CANON register blocks
        block(71 if j<n_adds-1 else 69) + gap )
    TRAILING[n] words                        tail task descriptors / zero pad

PREFIX is generated algorithmically by _build_prefix(n).  CANON contains 52 DPU +
17 DPU_RDMA register writes.  A 71-block adds a 2-word PC preamble.  Patched
fields (those _rc_patch_block overwrites at build time) are stored as 0.
"""
import struct

_DPU, _RDMA, _PC = 0x1001, 0x2001, 0x0101

EW_OP_ADD = 0
EW_OP_SUB = 1
EW_OP_MUL = 2
EW_OP_DIV = 3

_EW_CFG = {
    EW_OP_ADD: 0x108202c0,
    EW_OP_SUB: 0x108402c0,
    EW_OP_MUL: 0x108003c4,
    EW_OP_DIV: 0x108303c0,
}

_DPU_OUT_RES = {
    EW_OP_ADD: 0x00010001,
    EW_OP_SUB: 0x00010001,
    EW_OP_MUL: 0x00010001,
    EW_OP_DIV: 0x00000001,
}

_RDMA_BN_MUL = {
    EW_OP_ADD: 0x00017849,
    EW_OP_SUB: 0x00017849,
    EW_OP_MUL: 0x00017849,
    EW_OP_DIV: 0x00017841,
}

_OP_NAMES = {"Add": EW_OP_ADD, "Sub": EW_OP_SUB, "Mul": EW_OP_MUL, "Div": EW_OP_DIV}


def ew_op_id(name):
    return _OP_NAMES.get(name, EW_OP_ADD)


def _ew_cfg(op):
    return _EW_CFG.get(op, _EW_CFG[EW_OP_ADD])


def _w(tgt, reg, val):
    return (tgt << 48) | ((val & 0xFFFFFFFF) << 16) | reg


def _canon(op=EW_OP_ADD):
    return [
    (0x1001, 0x4004, 0x0000000e, False),
    (0x2001, 0x5004, 0x0000000e, False),
    (0x1001, 0x400c, 0x000001e5, False),
    (0x1001, 0x4010, 0x48000002, False),
    (0x1001, 0x4014, 0x00000000, False),
    (0x1001, 0x4020, 0x00000000, True),
    (0x1001, 0x4024, 0x00000000, True),
    (0x1001, 0x4030, 0x00000000, True),
    (0x1001, 0x4034, 0x00000000, True),
    (0x1001, 0x4038, 0x00000000, False),
    (0x1001, 0x403c, 0x00000000, True),
    (0x1001, 0x4040, 0x00000053, False),
    (0x1001, 0x4044, 0x00000000, False),
    (0x1001, 0x4048, 0x00000000, False),
    (0x1001, 0x404c, 0x00000000, False),
    (0x1001, 0x4050, 0x00000002, False),
    (0x1001, 0x4054, 0x00000000, False),
    (0x1001, 0x4058, 0x00000000, True),
    (0x1001, 0x405c, 0x00000000, True),
    (0x1001, 0x4060, 0x00000053, False),
    (0x1001, 0x4064, 0x00000000, False),
    (0x1001, 0x4068, 0x00000000, False),
    (0x1001, 0x406c, 0x00000000, False),
    (0x1001, 0x4070, _ew_cfg(op), False),
    (0x1001, 0x4074, 0x00000000, False),
    (0x1001, 0x4078, 0x00000001, False),
    (0x1001, 0x407c, 0x00000000, False),
    (0x1001, 0x4080, 0x00000000, False),
    (0x1001, 0x4084, _DPU_OUT_RES.get(op, 0x00010001), False),
    (0x1001, 0x4088, 0x00000000, False),
    (0x1001, 0x4090, 0x00000000, False),
    (0x1001, 0x4094, 0x00000000, False),
    (0x1001, 0x4098, 0x00000000, False),
    (0x1001, 0x409c, 0x00000000, False),
    (0x1001, 0x40a0, 0x00000000, False),
    (0x1001, 0x40a4, 0x00000000, False),
    (0x1001, 0x40a8, 0x00000000, False),
    (0x1001, 0x40ac, 0x00000000, False),
    (0x1001, 0x40c0, 0x00000000, True),
    (0x1001, 0x40c4, 0x00000000, False),
    (0x1001, 0x4100, 0x00000000, False),
    (0x1001, 0x4104, 0x00000000, False),
    (0x1001, 0x4108, 0x00000000, False),
    (0x1001, 0x410c, 0x00000000, False),
    (0x1001, 0x4110, 0x00000000, False),
    (0x1001, 0x4114, 0x00000000, False),
    (0x1001, 0x4118, 0x00000000, False),
    (0x1001, 0x411c, 0x00000000, False),
    (0x1001, 0x4120, 0x00000000, False),
    (0x1001, 0x4124, 0x00000000, False),
    (0x1001, 0x4128, 0x00000000, False),
    (0x1001, 0x412c, 0x00000000, False),
    (0x2001, 0x500c, 0x00000000, True),
    (0x2001, 0x5010, 0x00000000, True),
    (0x2001, 0x5014, 0x00000000, True),
    (0x2001, 0x5018, 0x00000000, True),
    (0x2001, 0x501c, 0x00000000, False),
    (0x2001, 0x5020, 0x00000000, False),
    (0x2001, 0x5028, 0x00000000, False),
    (0x2001, 0x502c, 0x00000000, False),
    (0x2001, 0x5034, 0x40000008, False),
    (0x2001, 0x5038, 0x00000000, True),
    (0x2001, 0x5040, 0x00000000, True),
    (0x2001, 0x5044, _RDMA_BN_MUL.get(op, 0x00017849), False),
    (0x2001, 0x5048, 0x00000000, False),
    (0x2001, 0x504c, 0x00000000, True),
    (0x2001, 0x5064, 0x00000000, False),
    (0x2001, 0x5068, 0x01010101, False),
    (0x2001, 0x506c, 0x00000000, True),
]

GAP69 = [
    0x0000000000000000, 0x0101000000000014,
    0x0041000000000000, 0x0081000000180008,
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000,
]

GAP71 = [
    0x0041000000000000, 0x0081000000180008,
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000,
    0x0000000000000000,
]

PC14 = 0x0101000000240014

_EVEN_CORE = [
    0x0000000400000001, 0x0000000400000001, 0x0000000000340028,
    0x000c000800040000, 0x001c001800140010, 0x0000000000240020,
    0x0030002c00280000, 0x0000006800000028, 0x0000005400000060,
    0x0000003c00000048, 0x0000002c00000034, 0x0000001c00000024,
    0x0000000c00000014, 0x0000000000000004,
]

_ODD_CORE = [
    0x0000000100000004, 0x0034002800000004, 0x0004000000000000,
    0x00140010000c0008, 0x00240020001c0018, 0x0028000000000000,
    0x000000280030002c, 0x0000006000000068, 0x0000004800000054,
    0x000000340000003c, 0x000000240000002c, 0x000000140000001c,
    0x000000040000000c,
]


def _build_prefix(n):
    n_adds = n - 1
    even = (n % 2 == 0)

    if even:
        n_lead = (n - 2) // 2 + 1
        str_w2 = 0x0031695f73725f78 if n == 2 else 0x0031695f73725f61
        header = [0] * n_lead + [0x0000000700000000, str_w2] + list(_EVEN_CORE)
    else:
        n_lead = 3 + (1 if n >= 7 else 0)
        header = [0] * n_lead + [0x73725f6100000007, 0x000000010031695f] + list(_ODD_CORE)

    words = list(header) + [0] * 7

    n_inter = (n - 1) // 2
    topmost_inter = n_inter + 1

    if even:
        words.append(((148 + 28 * (n - 2)) << 32) | (n + 4))
    else:
        words.append((n + 4) << 32)

    for k in range(topmost_inter, 1, -1):
        sz = 40 + 56 * k
        off = 68 + 56 * k
        if not even and k == topmost_inter:
            off -= 4
        words.append((sz << 32) | off)

    words += [0x000000600000007c, 0x0000002800000044]

    anomaly = 64 if n >= 6 else 0
    chain0 = 0x74 + 24 * (n - 2)
    chain1 = 0x15c + 280 * (n - 2) - anomaly
    canon_base = 0x1084 + 4120 * (n - 2) - anomaly

    words += [0x000600000000000c, 0x0000000600040012, chain0]
    words += [0x0006000000000000, 0x0000000600040012, chain1]
    for k in range(n + 1):
        addr = canon_base + k * 0x28
        marker = 0x0000000600040010 if k == n else 0x0000000600040012
        words += [0x0006000000000000, marker, addr]

    words += [0x0004000400000000, (240 * n_adds) << 32 | 4]

    for tile in range(6):
        for add in range(n_adds):
            counter = (n + 2 + 2 * add) << 32
            addr = (tile * n_adds + add) * 0x280
            words += [counter, 0x0000030000000018, 0x000000000001ffff,
                      0x0000000000000045, addr]

    n_zeros = 2 * ((n - 2) % 4) + 1
    words += [0] * n_zeros
    words.append((3840 * n_adds) << 32)

    return words


OFF = {n: (4 if n % 2 == 0 else 0) for n in range(2, 99)}
PREFIX = {n: _build_prefix(n) for n in range(2, 99)}


def _build_trailing(n):
    ng = max(0, n - 2)
    if ng == 0:
        return [0] * 6
    lead = 6 if n % 6 == 2 else 7
    w = [0] * lead
    w += [0x0000001000000000, 0x000000000000000a, 0x000000000000000a]
    if ng > 1:
        w += [0] * 5
    for k in range(1, ng):
        w += [0x0000002000000000, 0x0000000000000001, 0x000000000000000a,
              0x0000000000000001, 0x000000000000000a]
        if k < ng - 1:
            w += [0] * 3
    trail = max(0, 5 - 2 * ng)
    if ng == 3:
        trail = 1
    w += [0] * trail
    return w


def _canon_words(op=EW_OP_ADD):
    return [_w(t, r, 0 if p else v) for (t, r, v, p) in _canon(op)]


def build_template(n_inputs, ops=None):
    n_adds = n_inputs - 1
    if ops is None:
        ops = [EW_OP_ADD] * n_adds
    elif isinstance(ops, int):
        ops = [ops] * n_adds
    canon_per = [_canon_words(op) for op in ops]
    words = list(PREFIX[n_inputs])
    gbi = 0
    for _tile in range(6):
        for j in range(n_adds):
            if j < n_adds - 1:
                base = _w(_PC, 0x0010, (gbi + 1) * 0x280)
                words += canon_per[j] + [base, PC14] + GAP71
            else:
                words += canon_per[j] + GAP69
            gbi += 1
    words += _build_trailing(n_inputs)
    return b"\x00" * OFF[n_inputs] + struct.pack(f"<{len(words)}Q", *words)


def all_templates(max_n=None):
    upper = (max_n or 7) + 1
    return {n: build_template(n) for n in range(2, upper + 1)}
