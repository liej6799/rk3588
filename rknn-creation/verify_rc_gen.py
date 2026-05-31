"""Verify rc_template_gen.build_template() is functionally identical to the
original gzip+base64 _RC_TEMPLATES: after rknn_flatbuf._rc_patch_block runs
(which is ALWAYS called in build_body), the regcmd regions must match byte-for-
byte for every shape.  This is the real correctness criterion, since the patched
fields are overwritten and the unpatched fields/prefix/gaps must be reproduced.
"""
import struct

import rknn_flatbuf as rf
import rc_template_gen as g
from decode_rc_templates import find_offset


def patch(blob, N, n_inputs):
    """Run the real _rc_patch_block path over a standalone template blob and
    return the patched bytes (aligned back to blob coordinates)."""
    off, _ = find_offset(blob)
    pad = (8 - off) % 8                       # shift blocks onto 8-byte words
    full = bytearray(b"\x00" * pad + blob)
    C1, W = rf.surface_split(N)
    n_adds = n_inputs - 1
    tiles = rf.tile_split(C1)
    rf._patch_regcmd(full, 0, n_adds, C1, W, tiles)
    return bytes(full[pad:])


def main():
    shapes = [1, 2, 7, 100, 400, 1024, 8176, 10000, 65408, 1_000_000, 16_000_000]
    all_ok = True
    for n in (2, 3, 4):
        orig = rf._RC_TEMPLATES[n]
        gen = g.build_template(n)
        same_len = len(orig) == len(gen)
        # functional check: identical after patching, for each shape
        fn_ok = True
        for N in shapes:
            if patch(orig, N, n) != patch(gen, N, n):
                fn_ok = False
                break
        all_ok &= same_len and fn_ok
        print(f"n_inputs={n}: len(orig)={len(orig)} len(gen)={len(gen)} "
              f"same_len={same_len}  identical_after_patch(all shapes)={fn_ok}")
    print()
    print("ALL TEMPLATES FUNCTIONALLY IDENTICAL:", all_ok)

    # end-to-end for the working n=2 path: full body byte-identical
    def build_with(tmpl, N, n):
        save = rf._RC_TEMPLATES[n]
        rf._RC_TEMPLATES[n] = tmpl
        try:
            return rf.build_body(N, n)
        finally:
            rf._RC_TEMPLATES[n] = save
    e2e = all(build_with(rf._RC_TEMPLATES[2], N, 2) == build_with(g.build_template(2), N, 2)
              for N in shapes)
    print("n=2 full build_body() body identical (all shapes):", e2e)


if __name__ == "__main__":
    main()
