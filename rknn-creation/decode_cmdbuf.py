#!/usr/bin/env python3
import struct, sys, os

TARGET_MAP = {
    0x0101: "PC", 0x0201: "CNA", 0x0801: "CORE", 0x1001: "DPU",
    0x2001: "DPU_RDMA", 0x4001: "PPU", 0x8001: "PPU_RDMA",
    0x10001: "DDMA", 0x20001: "SDMA", 0x40001: "GLOBAL",
}

def decode_reg(raw_val):
    target = (raw_val >> 48) & 0xFFFF
    value = (raw_val >> 16) & 0xFFFFFFFF
    reg = raw_val & 0xFFFF
    tname = TARGET_MAP.get(target, f"UNK({target:#x})")
    return tname, reg, value

# DPU/SDP register meanings derived by diffing op buffers (see report/README.txt).
# NOTE: the task's op_idx is a per-graph compiler index, NOT a fixed hardware
# function (op4 is a real conv in the matmul graph but a plain copy in the add
# graph), so classify by register signature rather than trusting the op_idx name.
DPU_MODE   = ("DPU", 0x4010)       # 0x48000002=normal, 0x24000001=reformat/transpose
DPU_WIDTH  = ("DPU", 0x4030)       # element count - 1
DPU_OUT    = ("DPU", 0x4020)       # output DMA address
RDMA_SRC   = ("DPU_RDMA", 0x5018)  # primary source DMA address
RDMA_EW    = ("DPU_RDMA", 0x5034)  # bit30 set => second operand enabled (elementwise)
RDMA_SRC2  = ("DPU_RDMA", 0x5038)  # second-operand source address
_MODE_NAMES = {0x48000002: "", 0x24000001: "/reformat"}

def classify_op(decoded):
    """Functional label for a DPU/SDP task from its register signature.
    Returns (kind, detail) where kind is one of EW_BINARY / COPY / OTHER."""
    m = {(t, r): v for t, r, v in decoded}
    mode = m.get(DPU_MODE)
    if mode is None:  # not a DPU/SDP task (e.g. a CNA/CORE conv task)
        tgts = {}
        for t, _, _ in decoded:
            tgts[t] = tgts.get(t, 0) + 1
        return "OTHER", " ".join(f"{t}:{c}" for t, c in sorted(tgts.items()))
    width = m.get(DPU_WIDTH, 0) + 1
    binary = (m.get(RDMA_EW, 0) & 0x40000000) != 0
    kind = "EW_BINARY" if binary else "COPY"
    flav = _MODE_NAMES.get(mode, f"/mode={mode:#x}")
    out, src = m.get(DPU_OUT, 0), m.get(RDMA_SRC, 0)
    detail = f"w={width} out={out:#010x} in={src:#010x}"
    if binary:
        detail += f" in2={m.get(RDMA_SRC2, 0):#010x}  (reads 2 operands -> add/mul)"
    return kind + flav, detail

def dump_decoded(path, show_diff_only=None):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 8
    vals = struct.unpack(f"<{n}Q", data)
    name = os.path.basename(path)
    decoded = []
    for i, v in enumerate(vals):
        tname, reg, value = decode_reg(v)
        decoded.append((tname, reg, value))
    
    if show_diff_only is not None:
        ref_name, ref_decoded = show_diff_only
        diffs = []
        for i, (tname, reg, value) in enumerate(decoded):
            if i < len(ref_decoded):
                rtname, rreg, rvalue = ref_decoded[i]
                if value != rvalue:
                    diffs.append(f"  reg[{i:3d}] {tname}+0x{reg:04X} = 0x{value:08X}  (ref: 0x{rvalue:08X}, delta={value-rvalue:+d})")
            else:
                diffs.append(f"  reg[{i:3d}] {tname}+0x{reg:04X} = 0x{value:08X}  (extra)")
        if diffs:
            print(f"\n{name} vs {ref_name} ({len(diffs)} differences):")
            for d in diffs:
                print(d)
        else:
            print(f"\n{name} vs {ref_name}: IDENTICAL")
        return decoded
    
    print(f"\n{name} ({n} regs):")
    for i, (tname, reg, value) in enumerate(decoded):
        print(f"  [{i:3d}] {tname:10s}+0x{reg:04X} = 0x{value:08X} ({value})")
    return decoded

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: decode_cmdbuf.py [dump|diff|summary|classify] <files...>")
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "classify":
        print(f"{'file':36s} {'op':18s} detail")
        for path in sorted(sys.argv[2:]):
            with open(path, "rb") as f:
                data = f.read()
            decoded = [decode_reg(v) for v in struct.unpack(f"<{len(data)//8}Q", data)]
            kind, detail = classify_op(decoded)
            print(f"{os.path.basename(path):36s} {kind:18s} {detail}")
    elif mode == "diff" and len(sys.argv) >= 4:
        ref_path = sys.argv[2]
        ref_decoded = dump_decoded(ref_path)
        for path in sys.argv[3:]:
            dump_decoded(path, show_diff_only=(os.path.basename(ref_path), ref_decoded))
    elif mode == "dump":
        for path in sys.argv[2:]:
            dump_decoded(path)
    elif mode == "summary":
        for path in sys.argv[2:]:
            with open(path, "rb") as f:
                data = f.read()
            n = len(data) // 8
            vals = struct.unpack(f"<{n}Q", data)
            name = os.path.basename(path)
            targets = {}
            for v in vals:
                t = (v >> 48) & 0xFFFF
                targets[t] = targets.get(t, 0) + 1
            tstr = " ".join(f"{TARGET_MAP.get(t, f'?{t:#x}')}: {c}" for t, c in sorted(targets.items()))
            print(f"{name}: {n} regs [{tstr}]")
