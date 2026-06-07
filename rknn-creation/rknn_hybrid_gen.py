#!/usr/bin/env python3
"""Generate a parallel hybrid RKNN: CPU logical op + NPU arithmetic op.

Supported (verified on RK3588):
  CPU op: And | Or | Xor     (Xor is a fused And/Or/Not DAG)
  NPU op: Add | Sub | Div

Shape note: this mixed path is currently fixed to bool/fp16 [1,4].  The 2+2 path
uses the decoded mixed body/RC path and supports fused CPU Xor.  N-input branches
use toolkit-minted And+Add references for the same (CPU inputs, NPU inputs)
topology, then patch CPU And->Or and NPU Add->Sub/Div in place.  N-input mixed Xor
needs a fused-DAG reference and is intentionally rejected here.
"""
import argparse
import subprocess
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import rknn_flatbuf
import rknn_mixed_gen

CPU_OPS = ("And", "Or", "Xor")
NPU_OPS = ("Add", "Sub", "Div")


def _patch_exact_fb_string(data, old, new):
    """Patch a FlatBuffer string in place when the padded allocation still fits."""
    old_b = old.encode("ascii")
    new_b = new.encode("ascii")
    if len(new_b) > len(old_b):
        raise ValueError(f"cannot patch {old!r} to longer string {new!r} in place")
    needle = struct.pack("<I", len(old_b)) + old_b
    repl = struct.pack("<I", len(new_b)) + new_b + b"\x00" * (len(old_b) - len(new_b))
    buf = bytearray(data)
    n = 0
    pos = 0
    while True:
        i = buf.find(needle, pos)
        if i < 0:
            break
        buf[i:i + len(needle)] = repl
        n += 1
        pos = i + len(needle)
    return bytes(buf), n


def _indexed_names(prefix, n):
    return [f"{prefix}{i}" for i in range(n)]


def _reference_path(n_cpu, n_npu, ref=None):
    if ref:
        return Path(ref)
    path = ROOT / f"_ref_{n_cpu}and_{n_npu}add.rknn"
    if path.exists():
        return path
    raise FileNotFoundError(
        f"missing N-input mixed reference {path.name}; mint it with "
        f"`.venv/bin/python mint_mixed_refs.py {n_cpu},{n_npu}` or pass --ref")


def _generate_from_reference(cpu_op, npu_op, n_cpu, n_npu, out, ref=None):
    if cpu_op == "Xor":
        raise NotImplementedError(
            "N-input mixed Xor needs a fused-DAG reference; the 2+2 path supports Xor")

    ref_path = _reference_path(n_cpu, n_npu, ref)
    ref_data = ref_path.read_bytes()
    fb, rc, taskdesc, _trailer, _ro, _to, _body_len = rknn_mixed_gen.split_container(ref_data)

    if cpu_op == "Or":
        fb, patched = _patch_exact_fb_string(fb, "And", "Or")
        if patched == 0:
            raise ValueError(f"no And op strings found in {ref_path}")
    elif cpu_op != "And":
        raise NotImplementedError(f"unsupported N-input CPU op: {cpu_op}")

    if npu_op != "Add":
        fb, _ = _patch_exact_fb_string(fb, "Add", npu_op)
        rc = rknn_flatbuf._patch_mixed_npu_op(rc, npu_op)

    body = fb + rc + taskdesc
    branches = [
        {"inputs": _indexed_names("a", n_cpu), "output": "out1", "rows": 1,
         "cols": 4, "op": cpu_op},
        {"inputs": _indexed_names("x", n_npu), "output": "out2", "rows": 1,
         "cols": 4, "op": npu_op},
    ]
    data = rknn_mixed_gen.make_header(len(body)) + body + rknn_mixed_gen._make_trailer(branches)
    Path(out).write_bytes(data)
    print(f"generated {out}: CPU {cpu_op}({n_cpu} inputs)->out1 + "
          f"NPU {npu_op}({n_npu} inputs)->out2, shape=[1,4], {len(data)} B "
          f"from {ref_path.name}")


def generate(cpu_op, npu_op, out, n_cpu=2, n_npu=2, ref=None):
    if n_cpu != 2 or n_npu != 2:
        _generate_from_reference(cpu_op, npu_op, n_cpu, n_npu, out, ref)
        return

    fb_part, rc_part = rknn_flatbuf.build_mixed_and_add(cpu_op, npu_op)
    body = fb_part + rc_part
    branches = [
        {"inputs": ["a", "b"], "output": "out1", "rows": 1, "cols": 4, "op": cpu_op},
        {"inputs": ["x", "y"], "output": "out2", "rows": 1, "cols": 4, "op": npu_op},
    ]
    data = rknn_mixed_gen.make_header(len(body)) + body + rknn_mixed_gen._make_trailer(branches)
    Path(out).write_bytes(data)
    print(f"generated {out}: CPU {cpu_op}(a,b)->out1 + NPU {npu_op}(x,y)->out2, "
          f"shape=[1,4], {len(data)} B")


def verify(out, cpu_op, npu_op, n_cpu=2, n_npu=2):
    runner = ROOT / ("verify_hybrid" if (n_cpu, n_npu) == (2, 2) else "verify_hybrid_n")
    if not runner.exists():
        print(f"VERIFY: skipped (build {runner.name} first)")
        return None
    cmd = ([str(runner), out, cpu_op, npu_op] if (n_cpu, n_npu) == (2, 2)
           else [str(runner), out, str(n_cpu), cpu_op, str(n_npu), npu_op])
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env={"LD_LIBRARY_PATH": "/usr/lib64", "PATH": "/usr/bin"})
    text = (r.stdout + r.stderr).strip()
    line = next((l for l in text.splitlines() if "PASS" in l or "FAIL" in l),
                text.splitlines()[-1] if text else "")
    print("VERIFY:", line)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu", choices=CPU_OPS, default="And")
    ap.add_argument("--npu", choices=NPU_OPS, default="Add")
    ap.add_argument("--cols", type=int, default=4,
                     help="element count N. Currently only 4 is supported by mixed CPU+NPU RC")
    ap.add_argument("--cpu-inputs", type=int, default=2,
                    help="number of bool inputs for the CPU logical branch")
    ap.add_argument("--npu-inputs", type=int, default=2,
                    help="number of fp16 inputs for the NPU arithmetic branch")
    ap.add_argument("--ref", default=None,
                    help="override N-input And+Add reference RKNN path")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.cols != 4:
        raise SystemExit("hybrid CPU+NPU generation currently supports only --cols 4 "
                         "(the CPU/mixed RC descriptor path is fixed to bool/fp16 [1,4])")
    if args.cpu_inputs < 2 or args.npu_inputs < 2:
        raise SystemExit("hybrid CPU+NPU generation requires --cpu-inputs >=2 and --npu-inputs >=2")
    generate(args.cpu, args.npu, args.out, args.cpu_inputs, args.npu_inputs, args.ref)
    if args.verify and verify(args.out, args.cpu, args.npu,
                              args.cpu_inputs, args.npu_inputs) is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
