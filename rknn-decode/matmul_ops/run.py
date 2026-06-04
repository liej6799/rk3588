"""matmul_ops example: run the n x n matmul on the NPU without re-invoking the compiler.

Matmul can't be synthesized toolkit-free (no matmul/conv command-stream emitter), so the
.rknn must be compiled by the toolkit ONCE. That compile is the slow part (~30s for 8x8,
which becomes ~1391 element-wise tiles). After that we cache it, then on every run we
*recreate* the .rknn from its decoded form (toolkit-free, ~0.3s) and run it on the NPU
(~0.1s) -- no compiler involved. Result is checked against numpy's a @ b.

    .venv/bin/python3 -m matmul_ops.run            # 8x8 (compiles once, caches, then recreate+run)
    .venv/bin/python3 -m matmul_ops.run 16         # n x n
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # make project root importable

import numpy as np
from helpers.unroll import fully_unroll
from helpers.rknn_encode import recreate_rknn
from helpers.rknn_run import run_rknn
from matmul_ops.uops import make_matmul_uops

ROOT = Path(__file__).resolve().parent

def compiled_rknn(n):
  """The toolkit-compiled .rknn for n x n matmul, cached on disk. Returns (bytes, cache_hit).
  Only the first call (cache miss) imports/uses the toolkit compiler."""
  cache = ROOT / f"_matmul_{n}x{n}.rknn"                  # gitignored (*.rknn)
  if cache.exists():
    return cache.read_bytes(), True
  from helpers.onnx_export import uops_to_onnx           # lazy: a cache hit never touches the toolkit
  from helpers.rknn_export import onnx_to_rknn_bytes
  blob = onnx_to_rknn_bytes(uops_to_onnx(fully_unroll(make_matmul_uops(n)), name=f"matmul_{n}x{n}"))
  cache.write_bytes(blob)
  return blob, False

def main():
  n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
  rng = np.random.default_rng(0)
  A = rng.integers(0, 5, n * n).astype(np.float32)
  B = rng.integers(0, 5, n * n).astype(np.float32)
  ref = (A.reshape(n, n) @ B.reshape(n, n)).reshape(-1)

  t = time.perf_counter(); blob, hit = compiled_rknn(n); t_c = time.perf_counter() - t
  print(f"[compile]  {'cache hit' if hit else 'toolkit (one-time)':18s} {t_c:6.2f}s  ({len(blob)} bytes)")

  t = time.perf_counter(); rec = recreate_rknn(blob); t_r = time.perf_counter() - t   # toolkit-free
  print(f"[recreate] toolkit-free       {t_r:6.2f}s  (byte-identical: {rec == blob})")

  t = time.perf_counter()
  got = np.array(run_rknn(rec, [A, B], target="rk3588")[0]).reshape(-1)[:n * n].astype(np.float32)
  t_run = time.perf_counter() - t
  ok = np.allclose(got, ref)
  print(f"[run]      NPU                {t_run:6.2f}s  (== numpy matmul: {ok})")
  return 0 if ok else 1

if __name__ == "__main__":
  sys.exit(main())
