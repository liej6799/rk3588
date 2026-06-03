"""mul_ops entrypoint: load the 2D-loop multiply UOps and fully unroll them.

Run from the project root:  python3 -m mul_ops.run            # 10x10 = 100
                            python3 -m mul_ops.run 3 3        # custom rows cols
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

from helpers.unroll import fully_unroll
from mul_ops.uops import load_uops, make_mul_uops

def _dump(title: str, uops):
  print(f"=== {title} ({len(uops)} uops) ===")
  for i, u in enumerate(uops):
    print(f"{i:3d}: {u.op.name:7s} dtype={u.dtype} src={[uops.index(s) for s in u.src]} arg={u.arg}")

if __name__ == "__main__":
  uops = make_mul_uops(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else load_uops()
  _dump("original", uops)
  unrolled = fully_unroll(uops)
  print(f"\n{len(uops)} uops -> {len(unrolled)} unrolled "
        f"(ranges left: {sum(u.op.name == 'RANGE' for u in unrolled)}, "
        f"MUL: {sum(u.op.name == 'MUL' for u in unrolled)}, "
        f"STORE: {sum(u.op.name == 'STORE' for u in unrolled)})")
  _dump("unrolled", unrolled)
