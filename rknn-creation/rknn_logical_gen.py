#!/usr/bin/env python3
"""Generate CPU-fallback logical-op RKNN models (And / Or / Not / Xor) for RK3588.

Logical ops on bool tensors run on a CPU kernel, not the NPU DPU: the NPU side
only reshapes/copies data and the CPU kernel does the boolean logic.  The whole
model is generated from scratch (no toolkit) by the same modular path that the
chained-And builder uses -- the FlatBuffer body, register-command (RC) stream,
and task descriptor are all emitted by rknn_flatbuf / rc_template_gen.

The NPU-side machinery is COMPLETELY op-agnostic: And and Or models of the same
arity are byte-identical except for the op-name strings (and the node field f7).
Verified on-device, the runtime selects the CPU kernel by the op-NAME string, not
by f7.  Supported ops live in rc_template_gen.CPU_OP_SPECS:

    And = 85  binary  (verified on-device: out = a & b)
    Or  = 86  binary  (verified on-device: out = a | b)
    Not = 78  UNARY   (verified on-device: out = ~a)  -- NOT gate, 1 input

Xor is NOT a native CPU kernel on this runtime: librknnrt.so 2.3.2 rejects a node
named "Xor" at load ("Unsupport CPU op: Xor in this librknnrt.so").  The generator
therefore emits Xor as a fused And/Or/Not DAG: (a OR b) AND NOT(a AND b), chained
left-associated for n-input parity.

A binary n-input op chains left-associated:  out = (((a OP b) OP c) ... OP <n-th>),
shape [1, N] bool, for n = 2 .. 64.  Not is unary (always 1 input).

Usage
-----
  # 2-input:  out = a OR b
  rknn_logical_gen.py --op Or -o or2.rknn --verify
  # 3-input:  out = (a AND b) AND c
  rknn_logical_gen.py --op And --inputs 3 -o and3.rknn --verify
  # NOT gate:  out = ~a   (unary; --inputs is ignored)
  rknn_logical_gen.py --op Not -o not.rknn --verify
  # probe a candidate op's f7 enum on-device (sweep a range; for adding new ops)
  rknn_logical_gen.py --op Or --probe-enum 84-90 -o /tmp/or_probe
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import rknn_flatbuf
import rc_template_gen

# And/Or/Not are native CPU kernels; Xor is a fused 4-op DAG (no native kernel).
VALID_OPS = tuple(rc_template_gen.CPU_OP_SPECS) + ("Xor",)   # And, Or, Not, Xor
MAX_INPUTS = rc_template_gen.MAX_INPUTS                  # 64
COLS = 4                                                 # bool [1,4] shape


def generate(op, n_inputs, out, cols=COLS, f7_override=None):
    """Build a bool [1,cols] CPU logical-op model.

    Binary ops (And/Or) use n_inputs inputs (2..64); unary ops (Not) ignore
    n_inputs and always build a single-input model.

    f7_override temporarily forces the node f7 enum (used by --probe-enum to test
    candidate op-type codes on-device without editing the source table).
    """
    unary = rc_template_gen.is_unary_cpu_op(op)
    fused = (op == "Xor")
    if unary:
        n_inputs = 1
    elif not 2 <= n_inputs <= MAX_INPUTS:
        raise SystemExit(f"inputs must be 2..{MAX_INPUTS}, got {n_inputs}")
    # XOR is a fused DAG; for n>2 it chains parity stages (a0^a1^...).
    ops = [op] if unary else [op] * (n_inputs - 1)

    saved = None
    if f7_override is not None and not fused:
        saved = rc_template_gen.CPU_OP_ENUMS.get(op)
        rc_template_gen.CPU_OP_ENUMS[op] = f7_override
    try:
        body = rknn_flatbuf.build_body(cols, n_inputs, rows=1, cols=cols, ops=ops)
        data = rknn_flatbuf.assemble_rknn(body, 1, cols, n_inputs)
    finally:
        if f7_override is not None and not fused:
            if saved is None:
                rc_template_gen.CPU_OP_ENUMS.pop(op, None)
            else:
                rc_template_gen.CPU_OP_ENUMS[op] = saved

    Path(out).write_bytes(data)
    if fused:
        desc = ("fused parity DAG: a0^a1^...^a%d" % (n_inputs - 1)) if n_inputs > 2 \
            else "fused DAG: (a OR b) AND NOT(a AND b)"
    else:
        f7 = f7_override if f7_override is not None else rc_template_gen.cpu_op_id(op)
        verified = "" if rc_template_gen.cpu_op_verified(op) and f7_override is None \
            else "  [f7 UNVERIFIED -- confirm with --verify]"
        desc = f"f7={f7}{verified}"
    print(f"generated {out}: {n_inputs}-input {op} bool[1,{cols}], "
          f"{desc}, {len(data)} B, toolkit-free")
    return data


def verify(out, op):
    """Run the matching on-device harness and report PASS/FAIL for the op."""
    env = {"LD_LIBRARY_PATH": "/usr/lib64", "PATH": "/usr/bin"}
    # Unary Not has a dedicated harness; And/Or use verify_bool's semantic check.
    if op == "Not":
        runner = ROOT / "verify_not"
        if not runner.exists():
            print("VERIFY: skipped (build verify_not first)")
            return None
        r = subprocess.run([str(runner), out], capture_output=True, text=True, env=env)
        text = (r.stdout + r.stderr).strip()
        line = next((l for l in text.splitlines() if l in ("PASS", "FAIL")), "")
        outline = next((l for l in text.splitlines() if l.startswith("NOT output")), "")
        print("VERIFY:", outline)
        print(f"VERIFY: Not semantics {line or 'UNKNOWN'}")
        return line == "PASS"

    runner = ROOT / "verify_bool"
    if not runner.exists():
        print("VERIFY: skipped (build verify_bool first: make verify_bool)")
        return None
    r = subprocess.run([str(runner), out], capture_output=True, text=True, env=env)
    text = (r.stdout + r.stderr).strip()
    mism = next((l for l in text.splitlines() if "mismatches" in l), "")
    print("VERIFY:", mism or (text.splitlines()[-1] if text else "(no output)"))
    # 0 mismatches for the requested op == correct (verify_bool checks AND/OR/XOR)
    key = {"And": "AND=", "Or": "OR=", "Xor": "XOR="}[op]
    ok = f"{key}0 " in (mism + " ")
    print(f"VERIFY: {op} semantics {'PASS' if ok else 'FAIL'}")
    return ok


def probe_enum(op, lo, hi, out_prefix):
    """Generate one model per candidate f7 in [lo,hi] and verify each on-device.

    Prints the f7 value(s) whose output matches `op` semantics -- that is the
    runtime's true CPU op-type enum for this op on this device.
    """
    matches = []
    for f7 in range(lo, hi + 1):
        out = f"{out_prefix}_f7_{f7}.rknn"
        generate(op, 2, out, f7_override=f7)
        if verify(out, op):
            matches.append(f7)
        print()
    print("=" * 50)
    if matches:
        print(f"{op}: f7 enum(s) matching {op} semantics: {matches}")
        print(f"  -> set CPU_OP_SPECS['{op}'] = ({matches[0]}, True) in rc_template_gen.py")
    else:
        print(f"{op}: NO candidate f7 in [{lo},{hi}] produced {op} semantics "
              f"(this runtime may not have a {op} CPU kernel)")
    return matches


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", choices=VALID_OPS, default="Or",
                    help="logical op (default: Or)")
    ap.add_argument("--inputs", type=int, default=2,
                    metavar=f"2..{MAX_INPUTS}",
                    help="number of inputs: 2 for a OP b, 3 for (a OP b) OP c (default: 2)")
    ap.add_argument("--cols", type=int, default=COLS,
                    help=f"element count N for the bool [1,N] shape (default: {COLS})")
    ap.add_argument("-o", "--out", required=True,
                    help="output .rknn path (or prefix when --probe-enum is used)")
    ap.add_argument("--verify", action="store_true",
                    help="run verify_bool on-device and check the op semantics")
    ap.add_argument("--probe-enum", metavar="LO-HI", default=None,
                    help="sweep candidate f7 enums LO..HI on-device to discover the "
                         "true op-type code (e.g. 84-90); implies --verify")
    args = ap.parse_args()

    if args.probe_enum:
        lo, _, hi = args.probe_enum.partition("-")
        if not hi:
            raise SystemExit("--probe-enum needs a range like 84-90")
        probe_enum(args.op, int(lo), int(hi), args.out)
        return

    generate(args.op, args.inputs, args.out, cols=args.cols)
    if args.verify and verify(args.out, args.op) is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
