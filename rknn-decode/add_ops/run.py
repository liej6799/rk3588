"""add_ops example entrypoint: load the original UOps, then fully unroll them.

Run from the project root:  python3 -m add_ops.run
"""
from helpers.unroll import fully_unroll
from add_ops.uops import load_uops

def _dump(title: str, uops):
  print(f"=== {title} ({len(uops)} uops) ===")
  for i, u in enumerate(uops):
    print(f"{i:3d}: {u.op.name:8s} dtype={u.dtype} src={[uops.index(s) for s in u.src]} arg={u.arg}")

if __name__ == "__main__":
  uops = load_uops()                                  # load the original uops first
  _dump("original", uops)
  unrolled = fully_unroll(uops)                       # then unroll the constant-bound ranges
  print(f"\n{len(uops)} uops -> {len(unrolled)} unrolled (ranges left: {sum(u.op.name == 'RANGE' for u in unrolled)})")
  _dump("unrolled", unrolled)
