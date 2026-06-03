"""mul_ops example: disassemble the toolkit-free synthesized multiply .rknn (offline).

Builds the .rknn from the unrolled multiply UOps (no toolkit) and statically decodes the
container + FlatBuffer graph + NPU command queue. The EW tiles carry the MUL EW_CFG.

    .venv/bin/python3 -m mul_ops.disasm            # graph + command-queue summary
    .venv/bin/python3 -m mul_ops.disasm --regs     # also dump every decoded register
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

from helpers.rknn_decode import decode_rknn, print_rknn_disasm
from mul_ops.viz import build_rknn

def main():
  print_rknn_disasm(decode_rknn(build_rknn()), regs="--regs" in sys.argv[1:])
  return 0

if __name__ == "__main__":
  sys.exit(main())
