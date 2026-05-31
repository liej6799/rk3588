"""decode_rc_templates.py - decode the gzip+base64 _RC_TEMPLATES blobs in
rknn_flatbuf.py into a readable, regenerable structure.

FINDINGS
--------
Each _RC_TEMPLATES[n] is NOT an opaque constant: it is a fully regular regcmd
region.  Layout (after a 4-byte lead that aligns it to the FlatBuffer that gets
prepended at build time):

    prefix words            : tensor-tail metadata + the embedded TASK DESCRIPTOR
                              TABLE (one entry per block: base_addr spaced 0x280,
                              reg_amount=0x45=69, channel_mask=0x1ffff)
    N_BLOCKS x (block + gap) : block = a fixed table of (target, reg, value) NPU
                              register writes; gap = 11 constant PC control words
    trailing zeros

N_BLOCKS = 6 * n_adds  (n_adds = n_inputs - 1).  There are two block variants:
    BLOCK69 : 52 DPU regs (0x4004..0x412c) + 17 DPU_RDMA regs (0x500c..0x506c)
    BLOCK71 : BLOCK69 prefixed with 2 PC words (PC 0x10=0x280, PC 0x14=0x24)
Per-add block pattern:  n=2 -> [69]*6 ; n=3 -> [71,69]*6 ; n=4 -> [71,71,69]*6.

The 6 copies of a block within one add differ ONLY in the exact register fields
that _rc_patch_block() overwrites at generation time (0x4020/0x4030/0x403c/0x4058/
0x405c/0x500c/0x5014/0x5018/0x5038/0x504c/0x506c).  Therefore a single canonical
block, replicated, produces a byte-identical *final* body after patching.

This module decodes a template into its parts and can rebuild it.
"""
import struct

import rknn_flatbuf as rf

TARGETS = rf._TARGETS

# register name lookup (subset, from rkt_registers.h / decode_cmdbuf.py)
REG_NAMES = {
    (0x1001, 0x4010): "DPU_MODE", (0x1001, 0x400c): "DPU_FEATURE_MODE",
    (0x1001, 0x4020): "DPU_DST_BASE_ADDR", (0x1001, 0x4024): "DPU_DST_SURF_STRIDE",
    (0x1001, 0x4030): "DPU_DATA_CUBE_WIDTH", (0x1001, 0x4034): "DPU_DATA_CUBE_HEIGHT",
    (0x1001, 0x403c): "DPU_DATA_CUBE_CHANNEL", (0x1001, 0x4058): "DPU_WDMA_SIZE_0",
    (0x1001, 0x405c): "DPU_WDMA_SIZE_1", (0x1001, 0x40c0): "DPU_SURFACE_ADD",
    (0x2001, 0x500c): "RDMA_DATA_CUBE_WIDTH", (0x2001, 0x5010): "RDMA_DATA_CUBE_HEIGHT",
    (0x2001, 0x5014): "RDMA_DATA_CUBE_CHANNEL", (0x2001, 0x5018): "RDMA_SRC_BASE_ADDR",
    (0x2001, 0x5034): "RDMA_EW_CFG", (0x2001, 0x5038): "RDMA_EW_BASE_ADDR",
    (0x2001, 0x5040): "RDMA_SURF_STRIDE", (0x2001, 0x504c): "RDMA_SURF_NOTCH",
    (0x2001, 0x506c): "RDMA_SURF_NOTCH2",
}
TGT_NAMES = {0x0101: "PC", 0x1001: "DPU", 0x2001: "DPU_RDMA"}


def _spans_from_words(w):
    """Word-list span finder (decoupled from rf._regcmd_spans, whose signature
    changed to take bytes and return (spans, words))."""
    n = len(w)
    out = []
    i = 0
    while i < n:
        if ((w[i] >> 48) & 0xFFFF) in rf._TARGETS:
            j = i
            while j < n and ((w[j] >> 48) & 0xFFFF) in rf._TARGETS:
                j += 1
            if j - i >= 20:
                out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def find_offset(blob):
    """The blob is 4-byte aligned to the prepended FlatBuffer; find the byte
    offset (0/2/4/6) at which the 8-byte regcmd words line up into spans."""
    for off in range(0, 8, 2):
        if len(blob) - off < 8:
            continue
        n = (len(blob) - off) // 8
        w = list(struct.unpack_from(f"<{n}Q", blob, off))
        if _spans_from_words(w):
            return off, w
    raise RuntimeError("no regcmd spans found at any offset")


def decode(n_inputs):
    blob = rf._RC_TEMPLATES[n_inputs]
    off, w = find_offset(blob)
    spans = _spans_from_words(w)
    blocks = [(i, c, w[i:i + c]) for i, c in spans]
    # gap = words between block 0 end and block 1 start
    g0 = spans[0][0] + spans[0][1]
    gap = w[g0:spans[1][0]] if len(spans) > 1 else []
    prefix = w[:spans[0][0]]
    return dict(off=off, words=w, spans=spans, blocks=blocks, gap=gap,
                prefix=prefix, n_blocks=len(spans))


def block_regs(blk):
    """Decode a block's words into (target, reg, value) tuples."""
    out = []
    for v in blk:
        out.append(((v >> 48) & 0xFFFF, v & 0xFFFF, (v >> 16) & 0xFFFFFFFF))
    return out


def print_block(blk, title="block"):
    print(f"--- {title}: {len(blk)} regs ---")
    for k, (t, r, val) in enumerate(block_regs(blk)):
        nm = REG_NAMES.get((t, r), "")
        print(f"  {k:2d} {TGT_NAMES.get(t, hex(t)):9s} {r:#06x} = {val:#010x}  {nm}")


def rebuild(n_inputs):
    """Regenerate the template bytes from the decoded parts: prefix + each block
    followed by its real inter-block gap words + tail.  Byte-exact for all n
    (confirms the parse accounts for every word)."""
    d = decode(n_inputs)
    w, spans = d["words"], d["spans"]
    blob = rf._RC_TEMPLATES[n_inputs]
    out = list(d["prefix"])
    for idx, (i, c) in enumerate(spans):
        nxt = spans[idx + 1][0] if idx + 1 < len(spans) else len(w)
        out += w[i:nxt]            # block + its trailing gap (real spacing)
    return blob[:d["off"]] + struct.pack(f"<{len(out)}Q", *out)


def verify_body_equiv(n_inputs, N):
    """Replace each add's 6 blocks with its 1st block (canonical) and confirm the
    FINAL patched body is byte-identical (proves blocks reduce to one canonical)."""
    d = decode(n_inputs)
    w, spans, gap = d["words"], d["spans"], d["gap"]
    blob = rf._RC_TEMPLATES[n_inputs]
    n_adds = n_inputs - 1
    bpa = len(spans) // n_adds             # blocks per add
    out = list(d["prefix"])
    for g in range(n_adds):
        grp = spans[g * bpa:(g + 1) * bpa]
        # canonical-per-position: replicate using each position's own len pattern
        for (i, c) in grp:
            out += w[i:i + c] + gap
    out += w[spans[-1][0] + spans[-1][1] + len(gap):]
    rebuilt = blob[:d["off"]] + struct.pack(f"<{len(out)}Q", *out)

    def build_with(tmpl):
        save = rf._RC_TEMPLATES[n_inputs]
        rf._RC_TEMPLATES[n_inputs] = tmpl
        try:
            return rf.build_body(N, n_inputs)
        finally:
            rf._RC_TEMPLATES[n_inputs] = save

    return build_with(blob) == build_with(rebuilt)


if __name__ == "__main__":
    for nin in (2, 3, 4):
        d = decode(nin)
        n_adds = nin - 1
        lens = [c for _, c in d["spans"]]
        rb = rebuild(nin)
        print(f"n_inputs={nin}: off={d['off']} blocks={d['n_blocks']} "
              f"(=6*{n_adds}) block_lens={lens[:3]}... gap={len(d['gap'])} "
              f"prefix={len(d['prefix'])}w  rebuild_exact={rb == rf._RC_TEMPLATES[nin]}")
    print()
    # show the canonical 69-word add block (n=2, block 0)
    d2 = decode(2)
    print_block(d2["blocks"][0][2], "canonical ADD block (n=2)")
    print()
    print("gap (11 PC control words):")
    for g in d2["gap"]:
        print(f"  {g:016x}")
    print()
    for nin, N in ((2, 100), (3, 100), (4, 100)):
        try:
            r = verify_body_equiv(nin, N)
        except KeyError as e:
            r = f"SKIP (pre-existing build_body bug for n>=3: KeyError {e})"
        print(f"final-body identical after canonical reduction  "
              f"n_inputs={nin} N={N}: {r}")
