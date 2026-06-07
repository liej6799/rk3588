#!/usr/bin/env python3
"""Mixed CPU(And)/NPU(Add) RKNN generator for RK3588.

This builds a *parallel* mixed model: one or more independent CPU-fallback op
branches (e.g. And: out = a AND b) AND one or more independent NPU element-wise
branches (e.g. Add: out = x + y) in a single .rknn, run in one NPU submit.

How the vendor does it (decoded from toolkit references, see RKNN_CREATION.md
section 6f):

  - The op-graph is just the union of the per-branch node chains.  A CPU op
    (And, node field f7 = CPU_OP_ENUMS) runs only reshape/copy blocks on the
    NPU; an NPU op (Add, node field f3 = 2) runs the 6-tile element-wise compute.
  - The register-command (RC) stream is the CONCATENATION of every branch's
    blocks: all CPU copy/reshape blocks first, then all NPU compute blocks.
    A parallel And(2-in)+Add(2-in) model is exactly 3 copy blocks + 6 compute
    blocks = 9 blocks, EW_CFG 0x383 (copy) then 0x108202c0 (Add).
  - sg.f10 = [0,0,0, n_cpu_blocks + n_npu_ops, total_blocks].
  - The task descriptor is the same 8-word reshape-record family.

The RC PREFIX (descriptor + DMA schedule) for a mixed model is runtime-validated
(its copy-descriptor chain addresses and size table are checked at load; zeroing
them fails load / segfaults).  Reproducing it byte-exact from scratch is the same
"compiler tiler" port that gates large-N Add (RKNN_CREATION.md section 6).  Until
that is finished, this generator reuses a toolkit-built reference FlatBuffer body
+ RC for the mixed graph (the same hybrid approach that the working chained-And
path uses), while generating the container, trailer, and memory plan from scratch.

CLI:  rknn_mixed_gen.py --cpu And --npu Add -o mixed.rknn --verify
"""
import argparse
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

HEADER_SIZE = 0x40


def _fb_helpers(d):
    def u32(o): return struct.unpack_from("<I", d, o)[0]
    def i32(o): return struct.unpack_from("<i", d, o)[0]
    def u16(o): return struct.unpack_from("<H", d, o)[0]
    def fld(p, f):
        vt = p - i32(p); vts = u16(vt); e = vt + 4 + f * 2
        return u16(e) if e + 2 <= vt + vts else 0
    return u32, i32, u16, fld


def split_container(data):
    """Return (fb, rc, taskdesc, trailer, root_off) from a .rknn."""
    u32, i32, u16, fld = _fb_helpers(data)
    root = HEADER_SIZE + u32(HEADER_SIZE)
    rc_off = u32(root + fld(root, 20))
    task_off = u32(root + fld(root, 21))
    body_len = struct.unpack_from("<Q", data, 0x10)[0]
    fb = data[HEADER_SIZE:rc_off]
    rc = data[rc_off:task_off]
    taskdesc = data[task_off:HEADER_SIZE + body_len]
    trailer = data[HEADER_SIZE + body_len:]
    return fb, rc, taskdesc, trailer, rc_off, task_off, body_len


def make_header(body_len):
    h = bytearray(HEADER_SIZE)
    h[0:4] = b"RKNN"
    struct.pack_into("<Q", h, 0x08, 6)
    struct.pack_into("<Q", h, 0x10, body_len)
    return bytes(h)


def _make_trailer(branches):
    """Build the JSON trailer for a parallel mixed model.

    branches: list of dicts {inputs:[names], output:name, rows, cols}.
    norm_tensor ids are assigned inputs-first (all branches), then outputs.
    """
    import json
    norm_tensor = []
    input_entries = []
    output_entries = []
    tid = 0
    for br in branches:
        for nm in br["inputs"]:
            input_entries.append((tid, nm, br["rows"], br["cols"]))
            tid += 1
    for br in branches:
        output_entries.append((tid, br["output"], br["rows"], br["cols"]))
        tid += 1

    for tid_, nm, r, c in input_entries + output_entries:
        norm_tensor.append({
            "dim_num": 2,
            "dtype": {"qnt_method": "", "qnt_type": "", "vx_type": ""},
            "size": [r, c], "tensor_id": tid_, "url": nm,
        })
    n_in = len(input_entries)
    n_out = len(output_entries)
    connection, graph = [], []
    for i, (tid_, nm, r, c) in enumerate(input_entries):
        connection.append({"left": "input", "left_tensor_id": i, "node_id": 0,
                           "right_tensor": {"tensor_id": tid_, "type": "norm_tensor"}})
        graph.append({"left": "input", "left_tensor_id": i,
                      "right": "norm_tensor", "right_tensor_id": tid_})
    for i, (tid_, nm, r, c) in enumerate(output_entries):
        connection.append({"left": "output", "left_tensor_id": i, "node_id": 0,
                           "right_tensor": {"tensor_id": tid_, "type": "norm_tensor"}})
        graph.append({"left": "output", "left_tensor_id": i,
                      "right": "norm_tensor", "right_tensor_id": tid_})
    js = {
        "connection": connection, "const_tensor": [], "graph": graph,
        "input_num": n_in, "name": "rknn model", "network_platform": "ONNX",
        "node_num": 1,
        "nodes": [{"input_num": n_in, "lid": "npu_network_bin_graph", "name": "nnbg",
                   "nn": {"nbg": {"type": "RKNN_OP_NNBG"}}, "op": "RKNN_OP_NNBG",
                   "output_num": n_out, "uid": 0}],
        "norm_tensor": norm_tensor, "norm_tensor_num": n_in + n_out,
        "ori_network_platform": "ONNX", "output_num": n_out,
        "target_platform": ["rk3588"], "version": "2.3.2", "virtual_tensor": [],
    }
    nj = json.dumps(js, separators=(",", ":")).encode()
    return struct.pack("<Q", len(nj)) + nj


def build_parallel_from_reference(ref_path, branches, out_path):
    """Hybrid path: reuse the reference FB body + RC, regenerate the container
    (header) and trailer from scratch.  The FB body and RC come from a
    toolkit-built reference for the SAME mixed graph; everything else is ours.
    """
    ref = Path(ref_path).read_bytes()
    fb, rc, taskdesc, _trailer, _ro, _to, body_len = split_container(ref)
    body = fb + rc + taskdesc
    trailer = _make_trailer(branches)
    out = make_header(len(body)) + body + trailer
    Path(out_path).write_bytes(out)
    return out


def build_parallel_from_scratch(branches, out_path):
    """From-scratch path: generate the FB metadata entirely from scratch,
    splice with the reference RC + taskdesc (until the RC prefix is decoded).
    """
    import rknn_flatbuf
    fb_part, rc_part = rknn_flatbuf.build_mixed_and_add()
    body = fb_part + rc_part
    trailer = _make_trailer(branches)
    out = make_header(len(body)) + body + trailer
    Path(out_path).write_bytes(out)
    return out


PRESETS = {
    "and1_add1": {
        "ref": "_ref_parallel_and_add.rknn",
        "branches": [
            {"inputs": ["a", "b"], "output": "out1", "rows": 1, "cols": 4, "op": "And"},
            {"inputs": ["x", "y"], "output": "out2", "rows": 1, "cols": 4, "op": "Add"},
        ],
    },
    "and2_add1": {
        "ref": "/tmp/and2_add1_ref.rknn",
        "branches": [
            {"inputs": ["a", "b"], "output": "out1", "rows": 1, "cols": 4, "op": "And"},
            {"inputs": ["c", "d"], "output": "out2", "rows": 1, "cols": 4, "op": "And"},
            {"inputs": ["x", "y"], "output": "out3", "rows": 1, "cols": 4, "op": "Add"},
        ],
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="and1_add1",
                    choices=sorted(PRESETS),
                    help="graph topology preset (default: and1_add1)")
    ap.add_argument("--ref", default=None,
                    help="override toolkit reference path (default: from preset)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--scratch", action="store_true",
                    help="generate FB metadata from scratch (default: reference splice)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    ref_path = args.ref or str(ROOT / preset["ref"])
    branches = preset["branches"]
    if args.scratch:
        build_parallel_from_scratch(branches, args.out)
    else:
        build_parallel_from_reference(ref_path, branches, args.out)
    desc = " + ".join(f"{br['op']}({','.join(br['inputs'])})->{br['output']}"
                      for br in branches)
    print(f"generated {args.out} ({desc})")


if __name__ == "__main__":
    main()
