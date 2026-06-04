"""Generic driver for UOp kernels — the one place the debuggable features live.

An example op only needs a `uops.py` exposing `load_uops()`. Everything else (dump the
unrolled UOps, visualize the ONNX / .rknn graph in netron, disassemble the command queue,
run on the NPU and verify) is provided here and works for any such kernel, so adding a
custom op is just writing its UOps.

    from helpers.kernel import load, dump, run_verify
    uops = load("add_ops")            # imports add_ops.uops.load_uops()
    dump(uops); ok, npu, ref = run_verify(uops)

CLI (same features for any op):

    python3 -m helpers.kernel add_ops dump [--full]
    python3 -m helpers.kernel mul_ops onnx | rknn | disasm [--regs]
    python3 -m helpers.kernel add2d_ops run            # synth -> NPU, verify vs interp
    python3 -m helpers.kernel matmul_ops run [--perf]   # reductions run via decomposition

Verification ground truth is the pure-Python interpreter (`helpers.interp`), so no
hand-written reference per kernel is needed.
"""
import importlib

from .uop import Ops
from .unroll import fully_unroll
from .interp import run_uops
from .onnx_export import uops_to_onnx
from .rknn_synth import uops_to_rknn, run_uops_on_npu, chain_to_rknn
from .rknn_run import RknnSession


# --------------------------------------------------------------------------- #
# kernel loading + introspection
# --------------------------------------------------------------------------- #
def load(pkg: str):
  """Import `<pkg>.uops` and return its kernel UOps via load_uops()."""
  return importlib.import_module(f"{pkg}.uops").load_uops()


def _params(uops):
  """{param_arg: PARAM uop} over the (unrolled) graph."""
  return {u.arg: u for u in fully_unroll(uops) if u.op is Ops.PARAM}


def shape(uops):
  """(N, np_dtype) of the kernel from its output PARAM (arg 0)."""
  import numpy as np
  out = _params(uops)[0]
  N = out.dtype.size
  np_dt = {"half": np.float16, "float": np.float32}.get(out.dtype.base.name, np.float32)
  return N, np_dt


def n_inputs(uops):
  return sum(1 for a in _params(uops) if a != 0)


def random_inputs(uops, seed: int = 0):
  """One small-int array per input PARAM (fp16/fp32 per the kernel dtype), exact-safe."""
  import numpy as np
  N, np_dt = shape(uops)
  rng = np.random.default_rng(seed)
  ins = sorted(a for a in _params(uops) if a != 0)
  return [rng.integers(0, 5, N).astype(np_dt) for _ in ins]


# --------------------------------------------------------------------------- #
# build / dump
# --------------------------------------------------------------------------- #
def to_onnx(uops, name: str = "kernel"):
  """ONNX model of the unrolled kernel (Gather / op / Concat)."""
  return uops_to_onnx(fully_unroll(uops), name=name)

def synth_rknn(uops):
  """Toolkit-free .rknn synthesized from the unrolled kernel (element-wise add/mul)."""
  return uops_to_rknn(fully_unroll(uops))

def compile_rknn(uops, name: str = "kernel"):
  """Toolkit-compiled .rknn (ONNX -> rknn-toolkit2). Works for any kernel ONNX exports."""
  from .rknn_export import onnx_to_rknn_bytes
  return onnx_to_rknn_bytes(to_onnx(uops, name))

def dump(uops, full: bool = False):
  """Print the original and unrolled UOps (summary, or the full list with full=True)."""
  from collections import Counter
  unr = fully_unroll(uops)
  print(f"original {len(uops)} uops -> unrolled {len(unr)}: {dict(Counter(u.op.name for u in unr))}")
  if full:
    for i, u in enumerate(unr):
      print(f"  {i:3d}: {u.op.name:7s} {str(u.dtype):16s} src={[unr.index(s) for s in u.src]}"
            + (f" arg={u.arg}" if u.arg is not None and u.op is not Ops.SINK else ""))


# --------------------------------------------------------------------------- #
# execute + verify  (ground truth = interp)
# --------------------------------------------------------------------------- #
def interp_run(uops, inputs):
  """Pure-Python ground truth: run the unrolled kernel and return the output list."""
  N, _ = shape(uops)
  in_args = sorted(a for a in _params(uops) if a != 0)
  buffers = {0: [0.0] * N}
  buffers.update({a: list(x) for a, x in zip(in_args, inputs)})
  run_uops(fully_unroll(uops), buffers)
  return buffers[0]


def _is_reduction(uops):
  return any(u.op is Ops.MULACC for u in fully_unroll(uops))


def run_verify(uops, inputs=None, target: str = "rk3588", seed: int = 0, perf: bool = False):
  """Run the kernel on the NPU and compare to the interpreter. Returns (ok, npu, ref).

  Element-wise kernels are synthesized toolkit-free and run directly; a reduction (matmul)
  runs via the element-wise MULACC decomposition (`matmul_npu`)."""
  import numpy as np
  if inputs is None:
    inputs = random_inputs(uops, seed)
  ref = np.asarray(interp_run(uops, inputs), dtype=np.float64)
  N, _ = shape(uops)
  if _is_reduction(uops):
    n = int(round(N ** 0.5))
    if n * n != N or len(inputs) != 2:
      raise NotImplementedError("only a square 2-input matmul reduction is supported here")
    npu = np.asarray(matmul_npu(inputs[0], inputs[1], n, n, n, perf=perf), dtype=np.float64)
  else:
    npu = np.asarray(run_uops_on_npu(uops, inputs, target=target)).reshape(-1)[:N].astype(np.float64)
  return bool(np.allclose(npu, ref)), npu, ref


# --------------------------------------------------------------------------- #
# matmul: run a reduction as element-wise MULACC NPU ops (one session)
# --------------------------------------------------------------------------- #
def matmul_npu(A, B, M, K, N, target: str = "rk3588", perf: bool = False):
  """out[i,j] = sum_k A[i,k]*B[k,j] via K fused multiply-accumulate (MULACC) NPU ops.

  The MULACC model (col*row + acc) is synthesized toolkit-free once and loaded into one
  RknnSession; the K accumulation steps are inference calls. Returns the M*N output (fp16)."""
  import numpy as np
  A2, B2 = np.asarray(A).reshape(M, K), np.asarray(B).reshape(K, N)
  acc = np.zeros(M * N, dtype=np.float16)
  blob = chain_to_rknn(["Mul", "Add"], M * N)
  with RknnSession(blob, target=target, perf_debug=perf) as sess:
    for k in range(K):
      col = np.repeat(A2[:, k], N).astype(np.float16)
      row = np.tile(B2[k], M).astype(np.float16)
      acc = np.asarray(sess.run([col, row, acc])[0]).reshape(-1)[:M*N].astype(np.float16)
      if perf and k == 0:
        print("--- NPU command queue for one MULACC step (col*row + acc) ---")
        sess.eval_perf()
  return acc


# --------------------------------------------------------------------------- #
# netron
# --------------------------------------------------------------------------- #
def serve_onnx(uops, address=("0.0.0.0", 8080), browse: bool = False):
  from .viz import serve
  return serve(to_onnx(uops), address=address, browse=browse)

def serve_rknn(uops, address=("0.0.0.0", 8080), browse: bool = False):
  from .viz import serve
  return serve(synth_rknn(uops), address=address, browse=browse, name="kernel.rknn")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
  import sys
  argv = list(sys.argv[1:] if argv is None else argv)
  flags = {a for a in argv if a.startswith("--")}
  pos = [a for a in argv if not a.startswith("--")]
  if len(pos) < 2:
    print("usage: python3 -m helpers.kernel <pkg> {dump|onnx|rknn|disasm|run} [--full|--regs|--perf]")
    return 2
  pkg, cmd = pos[0], pos[1]
  uops = load(pkg)

  if cmd in ("dump", "uops"):
    dump(uops, full="--full" in flags)
  elif cmd == "onnx":
    host, port = serve_onnx(uops)
    _serve_loop(host, port, f"{pkg} onnx graph")
  elif cmd == "rknn":
    host, port = serve_rknn(uops)
    _serve_loop(host, port, f"{pkg} synthesized .rknn")
  elif cmd == "disasm":
    from .rknn_decode import decode_rknn, print_rknn_disasm
    print_rknn_disasm(decode_rknn(synth_rknn(uops)), regs="--regs" in flags)
  elif cmd == "run":
    dump(uops)
    ok, npu, ref = run_verify(uops, perf="--perf" in flags)
    print(f"NPU == interp: {ok}")
    print(f"  npu[:8]: {npu[:8].tolist()}")
    print(f"  ref[:8]: {ref[:8].tolist()}")
    return 0 if ok else 1
  else:
    print(f"unknown command {cmd!r}")
    return 2
  return 0


def _serve_loop(host, port, what):
  import time
  print(f"serving {what} at http://{host}:{port}  (ctrl-c to stop)")
  try:
    while True: time.sleep(3600)
  except KeyboardInterrupt:
    import netron; netron.stop()


if __name__ == "__main__":
  import sys
  sys.exit(main())
