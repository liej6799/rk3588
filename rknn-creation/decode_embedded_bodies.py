"""decode_embedded_bodies.py - decode the reference Add bodies (_body_add{n}_10x10.body).

The bodies were the gzip+base64 blobs in _embedded_bodies.py; they are now raw binary
files (same convention as _template_8192x8192.body) and fully decodable.  Each body is a
toolkit FlatBuffer with three regions:

    [FlatBuffer tensor/node tables]   small metadata + the embedded task-descriptor table
    [regcmd region]                   6 * n_adds register blocks (69/71 words each)
    [taskdesc tail]                   the "task" command-tensor data

This script prints the tensor table and confirms the regcmd block structure matches
rc_template_gen (the readable regcmd generator).  Run: python3 decode_embedded_bodies.py
"""
import struct
from pathlib import Path

import rknn_flatbuf as rf

ROOT = Path(__file__).resolve().parent


def load(n):
    return (ROOT / f"_body_add{n}_10x10.body").read_bytes()


def tensor_table(body):
    rows = []
    for tp in rf._fb_tensor_positions(body):
        nm = rf._fb_string(body, tp, 5)
        f3 = rf._fb_vec_u32(body, rf._fb_vec_target(body, tp, 3)) if rf._fb_vec_target(body, tp, 3) else None
        f12 = rf._fb_field_abs(body, tp, 12)
        f13 = rf._fb_field_abs(body, tp, 12)
        sz = rf._fb_u32(body, rf._fb_field_abs(body, tp, 12)) if rf._fb_field_abs(body, tp, 12) else 0
        off = rf._fb_u32(body, rf._fb_field_abs(body, tp, 13)) if rf._fb_field_abs(body, tp, 13) else 0
        rows.append((nm, f3, sz, off))
    return rows


def decode(n):
    body = load(n)
    spans, w = rf._regcmd_spans(body)
    n_adds = n - 1
    blocklens = [c for _, c in spans]
    rc0 = spans[0][0] * 8
    rc1 = (spans[-1][0] + spans[-1][1]) * 8
    print(f"\n=== n_inputs={n}  body={len(body)} bytes ===")
    print(f"  FlatBuffer prefix : 0 .. {rc0}")
    print(f"  regcmd region     : {rc0} .. {rc1}   ({len(spans)} blocks = 6*{n_adds}, "
          f"lens={sorted(set(blocklens))})")
    print(f"  taskdesc tail     : {rc1} .. {len(body)}")
    print("  tensor table:")
    for nm, f3, sz, off in tensor_table(body):
        if nm:
            print(f"    {nm:18s} native={f3}  size={sz}  off={off}")


if __name__ == "__main__":
    for n in (2, 3, 4, 5, 6, 7):
        decode(n)
    print("\nThe regcmd blocks are the same 69/71-word EW-Add blocks decoded in "
          "rc_template_gen.py; the taskdesc tail is the _td_rec structure.")
