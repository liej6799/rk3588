"""add_ops example: recreate the .rknn from its decoded form and compare to the library.

Builds the 5x5 add .rknn with the toolkit, then rebuilds it from the decoded
representation (header from scratch + command stream re-emitted from the decoded queue)
and checks the two are byte-identical.

    .venv/bin/python3 -m add_ops.recreate
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import hashlib
from helpers.rknn_encode import recreate_rknn
from add_ops.viz import build_rknn

def main():
  lib = build_rknn()
  mine = recreate_rknn(lib)
  print(f"library   : {len(lib):7d} bytes  sha256={hashlib.sha256(lib).hexdigest()[:16]}")
  print(f"recreated : {len(mine):7d} bytes  sha256={hashlib.sha256(mine).hexdigest()[:16]}")
  same = mine == lib
  print("BYTE-IDENTICAL" if same else f"DIFFER (first at byte {next(i for i in range(min(len(lib),len(mine))) if lib[i]!=mine[i])})")
  return 0 if same else 1

if __name__ == "__main__":
  sys.exit(main())
