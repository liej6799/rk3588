"""Algorithmic regcmd template generator for chained element-wise RKNN models.

Each supported RC template (n_inputs = 2..64) is:

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

MAX_INPUTS = 64

_DPU, _RDMA, _PC = 0x1001, 0x2001, 0x0101

EW_OP_ADD = 0
EW_OP_SUB = 1
EW_OP_MUL = 2
EW_OP_DIV = 3
# Pure copy/reshape mode: the NPU side of a CPU-fallback op runs the same 69
# register block as the element-wise ops, just configured to copy rather than
# compute. It is a regular member of the op-dispatch tables below.
EW_OP_COPY = 4

_EW_CFG = {
    EW_OP_ADD: 0x108202c0,
    EW_OP_SUB: 0x108402c0,
    EW_OP_MUL: 0x108003c4,
    EW_OP_DIV: 0x108303c0,
    EW_OP_COPY: 0x00000383,
}

_DPU_OUT_RES = {
    EW_OP_ADD: 0x00010001,
    EW_OP_SUB: 0x00010001,
    EW_OP_MUL: 0x00010001,
    EW_OP_DIV: 0x00000001,
    EW_OP_COPY: 0x00000001,
}

_RDMA_BN_MUL = {
    EW_OP_ADD: 0x00017849,
    EW_OP_SUB: 0x00017849,
    EW_OP_MUL: 0x00017849,
    EW_OP_DIV: 0x00017841,
    EW_OP_COPY: 0x00007801,
}

# Copy mode overrides two more cfg registers vs the element-wise ops, and bakes
# the fixed [1,4] bool reshape geometry into the fields the element-wise path
# leaves zero and patches per-block at build time (copy geometry is constant).
_COPY_DPU_MODE = 0x00000000      # 0x4010 (element-wise: 0x48000002)
_COPY_RDMA_MODE = 0x00000001     # 0x5034 (element-wise: 0x40000008)
_COPY_SHAPE_FILL = {
    0x4024: 0x00000040, 0x4030: 0x00000001, 0x4034: 0x00000001,
    0x403c: 0x000f000f, 0x4058: 0x0000000f, 0x405c: 0x00010001,
    0x40c0: 0x00000040, 0x500c: 0x00000001, 0x5010: 0x00000001,
    0x5014: 0x0000000f,
}

_OP_NAMES = {"Add": EW_OP_ADD, "Sub": EW_OP_SUB, "Mul": EW_OP_MUL, "Div": EW_OP_DIV}

# CPU-fallback ops execute on a CPU kernel (node field f7 = the enum below), with
# the NPU running only reshape/copy blocks (no DPU compute). The enum values are
# the runtime's internal CPU op-type codes, decoded byte-exact from references.
CPU_OP_ENUMS = {
    "And": 85,
}

# Single modular op classification used by both the FlatBuffer node builder and
# the regcmd generator. NPU ops dispatch to element-wise _canon compute blocks;
# CPU ops dispatch to reshape/copy-only blocks.
def is_cpu_op(name):
    return name in CPU_OP_ENUMS


def is_npu_op(name):
    return name in _OP_NAMES


def cpu_op_id(name):
    """Runtime CPU op-type enum (node field f7) for a CPU-fallback op."""
    return CPU_OP_ENUMS[name]


def ew_op_id(name):
    return _OP_NAMES.get(name, EW_OP_ADD)


def _ew_cfg(op):
    return _EW_CFG.get(op, _EW_CFG[EW_OP_ADD])


def _w(tgt, reg, val):
    return (tgt << 48) | ((val & 0xFFFFFFFF) << 16) | reg


def _canon(op=EW_OP_ADD):
    copy = (op == EW_OP_COPY)
    return [
    (0x1001, 0x4004, 0x0000000e, False),
    (0x2001, 0x5004, 0x0000000e, False),
    (0x1001, 0x400c, 0x000001e5, False),
    (0x1001, 0x4010, _COPY_DPU_MODE if copy else 0x48000002, False),
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
    (0x2001, 0x5034, _COPY_RDMA_MODE if copy else 0x40000008, False),
    (0x2001, 0x5038, 0x00000000, True),
    (0x2001, 0x5040, 0x00000000, True),
    (0x2001, 0x5044, _RDMA_BN_MUL.get(op, 0x00017849), False),
    (0x2001, 0x5048, 0x00000000, False),
    (0x2001, 0x504c, 0x00000000, True),
    (0x2001, 0x5064, 0x00000000, False),
    (0x2001, 0x5068, 0x01010101, False),
    (0x2001, 0x506c, 0x00000000, True),
]

def reshape_copy_canon_words():
    """The 69-word NPU reshape/copy canon (CPU-op block) for the [1,4] bool shape.

    This is the same _canon register block as the element-wise ops, dispatched
    through EW_OP_COPY: copy cfg values plus the constant [1,4] reshape geometry
    baked into the otherwise-patched fields. Byte-exact vs the chained-And refs.
    """
    return _canon_words(EW_OP_COPY)


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
        n_lead = min((n - 2) // 2 + 1, 3)
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

    # chain1/canon_base shrink by 64 for every 4 inputs (matches the toolkit's
    # surface-tiling step). Verified byte-exact against the n=4,5,10..16 reference
    # bodies; n=2..9 keep their original (n>=6 -> 64) value since (n-2)//4 == 1 there.
    anomaly = 64 * ((n - 2) // 4)
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
    w = [0] * 7
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
    if op == EW_OP_COPY:
        # Copy mode bakes the constant [1,4] reshape geometry into the fields the
        # element-wise path patches per-block, so they are not zeroed here.
        return [_w(t, r, _COPY_SHAPE_FILL.get(r, 0) if p else v)
                for (t, r, v, p) in _canon(op)]
    return [_w(t, r, 0 if p else v) for (t, r, v, p) in _canon(op)]


def _normalize_ops(n_adds, ops):
    """Return a list of op-name strings, one per binary op in the chain.

    Accepts None (all Add), a single int EW id, a single op-name string, a list
    of int EW ids, or a list of op-name strings. This is the single modular entry
    used to decide NPU- vs CPU-dispatch per op.
    """
    _EW_ID_TO_NAME = {v: k for k, v in _OP_NAMES.items()}
    if ops is None:
        return ["Add"] * n_adds
    if isinstance(ops, int):
        return [_EW_ID_TO_NAME.get(ops, "Add")] * n_adds
    if isinstance(ops, str):
        return [ops] * n_adds
    out = []
    for o in ops:
        if isinstance(o, int):
            out.append(_EW_ID_TO_NAME.get(o, "Add"))
        else:
            out.append(o)
    if len(out) < n_adds:
        out += ["Add"] * (n_adds - len(out))
    return out[:n_adds]


def build_template(n_inputs, ops=None):
    """Modular regcmd generator.

    Dispatches on op type: a chain of NPU element-wise ops (Add/Sub/Mul/Div)
    produces compute _canon blocks; a chain of CPU-fallback ops (And/...)
    produces reshape/copy canon blocks with a CPU descriptor prefix. Mixed
    NPU/CPU chains are not yet supported by the RC generator (the FlatBuffer
    node builder already handles them modularly).
    """
    if not 2 <= n_inputs <= MAX_INPUTS:
        raise NotImplementedError(
            f"regcmd template generation supports 2..{MAX_INPUTS} inputs, got {n_inputs}")
    n_adds = n_inputs - 1
    op_names = _normalize_ops(n_adds, ops)

    cpu_flags = [is_cpu_op(o) for o in op_names]
    if any(cpu_flags):
        if not all(cpu_flags):
            raise NotImplementedError(
                "mixed NPU/CPU op chains are not yet supported by the RC "
                f"generator (got ops={op_names})")
        return build_cpu_template(n_inputs, op_names)

    ew_ids = [ew_op_id(o) for o in op_names]
    canon_per = [_canon_words(op) for op in ew_ids]
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


# ── CPU-fallback op regcmd (reshape/copy NPU side; CPU does the compute) ──
#
# A CPU-fallback op (And/...) runs no NPU compute. The NPU only reshapes/copies
# data; the CPU kernel does the logic. The regcmd is built as a uint32 stream:
#
#   CPU descriptor prefix (u32)   header + copy-descriptor table + DMA cmd list
#   (n+1) x reshape/copy canon block (u32) + PC/GAP framing
#   CPU trailing (u32)            cosmetic task descriptors
#
# The reshape/copy canon (reshape_copy_canon_words) is shape-invariant for the
# [1,4] bool shape. The framing (PC chain address + PC14 + GAP) is identical to
# the NPU element-wise path. The CPU prefix is generated from closed-form
# descriptor formulas; only the short lead/alignment schedule is table-driven.
#
# Working on a uint32 basis (rather than uint64) handles the half-word
# alignment of even-n streams naturally; the generated prefix/trailing schedules
# produce the correct half-word alignment.

def _u64_list_to_u32(words64):
    out = []
    for w in words64:
        out.append(w & 0xFFFFFFFF)
        out.append((w >> 32) & 0xFFFFFFFF)
    return out


def _reshape_copy_canon_u32():
    return _u64_list_to_u32(reshape_copy_canon_words())


def _gap71_u32():
    return _u64_list_to_u32(GAP71)


def _gap69_u32():
    return _u64_list_to_u32(GAP69)


def _pc14_u32():
    return _u64_list_to_u32([PC14])


def _pc_chain_u32(addr):
    # PC(0x0010, addr) as one u64 -> two u32
    return _u64_list_to_u32([_w(_PC, 0x0010, addr)])


def _cpu_fixed_header():
    """The constant CPU descriptor header (was the flat _CPU_FIXED30_U32 blob).

    Structure (30 u32): a length-prefixed reshape tensor name, a few small
    constants, an interleaved 16-bit ascending offset stream, and a descending
    copy-descriptor size table. It is identical for every n.
    """
    out = [7]                                            # name length
    out += list(struct.unpack("<2I", b"a_rs_i1\x00"))    # "a_rs_i1" name
    out += [1, 4, 1, 4]                                  # bool [1,4] shape pair
    out.append((0x34 << 16) | 0x28)                      # packed (0x28, 0x34)
    # interleaved 16-bit offsets: 3 zeros, 4..36 step 4, 3 zeros, 40,44,48
    stream = [0, 0, 0] + list(range(4, 40, 4)) + [0, 0, 0] + [40, 44, 48]
    for i in range(0, len(stream), 2):
        out.append((stream[i + 1] << 16) | stream[i])
    # descending offset table: a 40 lead, then the values 104,96,...,4 (an
    # ascending run 4..60 step 8, then 72..104 step 12, reversed).
    asc = [4 + 8 * k for k in range(8)] + [72 + 12 * k for k in range(3)] + [104]
    out += [40] + asc[::-1]
    return out


# Offset-lead suffix sequence: an offset lead of length L (odd, 1..7) is the last
# L entries of this list, followed by 11 zeros.
_CPU_OFFSET_LEAD_SEQ = [0x2c, 0x24, 0x20, 0x18, 0x10, 0x08, 0x01]


def _cpu_nonlead_words(n):
    """u32 word count of the CPU prefix excluding the alignment lead."""
    header = 30 + 15 + 1 + (n + 4)         # fixed header + zeros + count + addr list
    copydesc = (n + 3) * 6
    preamble = 4
    dma = 3 * (n + 1) * 10 + ((2 * n) % 16) + 1
    return header + copydesc + preamble + dma


def _cpu_lead_u32(n, rc_word_off):
    """Alignment lead so the reshape/copy canon lands on a 64-byte boundary.

    rc_word_off is the RC section's u32 offset from the start of the file; the
    section base header (0x40 bytes) contributes nothing mod 16, so this is
    equivalently (FlatBuffer-body length // 4). The lead length is the pad
    (mod 16 u32 = 64 bytes) needed to align the canon, bumped by a full 16-word
    block when the raw pad is < 4 words. Leads of >= 12 words carry the
    descending offset suffix; shorter leads are all zeros.
    """
    residue = rc_word_off % 16
    pad = (-(residue + _cpu_nonlead_words(n))) % 16
    length = pad if pad >= 4 else pad + 16
    if length >= 12:
        return _CPU_OFFSET_LEAD_SEQ[-(length - 11):] + [0] * 11
    return [0] * length


def _cpu_prefix_u32(n, rc_word_off):
    """Closed-form CPU descriptor prefix for a bool [1,4] CPU-op chain."""
    words = _cpu_lead_u32(n, rc_word_off)
    words += _cpu_fixed_header()
    words += [0] * 15

    words.append(n + 4)
    words.append(0x0c + 28 * (n + 3) - 4)
    for i in range(1, n + 4):
        words.append(0x0c + 28 * (n + 3 - i))

    anomaly = 64 * (n // 8)
    chain0 = 0x74 + 24 * (n - 2)
    chain1 = 0x1dc + 152 * (n - 2) - anomaly
    canon_base = 0x984 + 792 * (n - 2) - anomaly
    copy_addrs = [chain0, chain1]
    copy_addrs += [canon_base + 40 * k for k in range(n + 1)]
    for i, addr in enumerate(copy_addrs):
        marker = 0x00040010 if i == len(copy_addrs) - 1 else 0x00040012
        words += [0x00060000, marker, 0x00000006, addr, 0, 0]

    words += [0x00040004, 0x00000004, 0x168 + 120 * (n - 2), 0]
    for _group in range(3):
        for k in range(n + 1):
            count = n if k == 0 else n + 2 * k - 1
            words += [
                count, 0x00000018, 0x00000300, 0x0001ffff, 0,
                0x00000045, 0, 0x280 * k, 0, 0,
            ]

    words += [0] * ((2 * n) % 16)
    words.append(0x280 * (n + 1))
    return words


def _cpu_trailing(n, prefix_words):
    """CPU-op regcmd tail as u64 words (cosmetic task descriptors; never patched).

    Record family parallel to _build_trailing, for the [1,4] bool shape: the
    output reshape record carries dim 4 (count 1); intermediate records carry
    [1,1,1,4]. The content is a prefix of the repeated intermediate-record
    stream; the target length follows the toolkit's total RC section size, so
    it absorbs whatever the alignment lead added to the prefix.
    """
    if n == 2:
        return [0]

    total_u32 = 677 + 215 * (n - 2) - 16 * (n // 8)
    target_u64 = (total_u32 - prefix_words - 160 * (n + 1)) // 2

    w = [0] * 7 + [0x0000001000000000, 0x1, 0x4]
    w += [0] * 5
    record = [0x0000002000000000, 0x1, 0x1, 0x1, 0x4, 0, 0, 0]
    while len(w) < target_u64:
        w += record
    return w[:target_u64]


def _cpu_trailing_u32(n, prefix_words):
    return _u64_list_to_u32(_cpu_trailing(n, prefix_words))


def build_cpu_template(n_inputs, ops=None, rc_word_off=0):
    """Build the reshape-only regcmd for an n-input CPU-fallback op chain.

    rc_word_off is the RC section's u32 offset from the file start (equivalently
    the FlatBuffer-body length // 4). The alignment lead is computed from it so
    the reshape/copy canon lands on a 64-byte boundary, matching the toolkit's
    placement. There is no per-n lead table; the layout offset is the only input.
    """
    if not 2 <= n_inputs <= MAX_INPUTS:
        raise NotImplementedError(
            f"CPU regcmd template generation supports 2..{MAX_INPUTS} inputs, "
            f"got {n_inputs}")
    n_blocks = n_inputs + 1
    u = _cpu_prefix_u32(n_inputs, rc_word_off)
    prefix_words = len(u)
    canon = _reshape_copy_canon_u32()
    for blk in range(n_blocks):
        if blk < n_blocks - 1:
            u += canon + _pc_chain_u32((blk + 1) * 0x280) + _pc14_u32() + _gap71_u32()
        else:
            u += canon + _gap69_u32()
    u += _cpu_trailing_u32(n_inputs, prefix_words)
    return struct.pack(f"<{len(u)}I", *u)


def verify_cpu_rc(refs_dir="/tmp"):
    """Prove build_cpu_template(n) == the chained-op reference RC, byte-for-byte.

    Returns {n: bool}. Reference files are and_chain{n}_ref.rknn.
    """
    import os
    results = {}
    for n in range(2, MAX_INPUTS + 1):
        path = os.path.join(refs_dir, f"and_chain{n}_ref.rknn")
        if not os.path.exists(path):
            continue
        d = open(path, "rb").read()
        hdr = 0x40
        root = hdr + struct.unpack_from("<I", d, hdr)[0]

        def u32(o):
            return struct.unpack_from("<I", d, o)[0]

        def i32(o):
            return struct.unpack_from("<i", d, o)[0]

        def u16(o):
            return struct.unpack_from("<H", d, o)[0]

        def fld(p, f):
            vt = p - i32(p)
            vts = u16(vt)
            e = vt + 4 + f * 2
            return u16(e) if e + 2 <= vt + vts else 0

        rc_off = u32(root + fld(root, 20))
        ref_rc = d[rc_off:u32(root + fld(root, 21))]
        results[n] = build_cpu_template(n, rc_word_off=rc_off // 4) == ref_rc
    return results



def all_templates(max_n=None):
    upper = max_n or MAX_INPUTS
    if upper > MAX_INPUTS:
        raise NotImplementedError(
            f"regcmd template generation supports up to {MAX_INPUTS} inputs, got {upper}")
    return {n: build_template(n) for n in range(2, upper + 1)}
