# rknn-decode

Standalone tooling for working with tinygrad-style linearized **UOps** without
depending on tinygrad. UOp graphs are captured (copied) from tinygrad and then
manipulated here — e.g. fully unrolling their loops — as a step toward RKNN decoding.

## Layout

```
rknn-decode/
├── helpers/            # reusable, tinygrad-free helper library
│   ├── uop.py          #   minimal UOp / Ops / dtypes / AxisType / KernelInfo + simplify/substitute/toposort
│   ├── unroll.py       #   fully_unroll(uops): expand constant-bound RANGE loops
│   ├── interp.py       #   run_uops(uops, buffers): execute a range-free graph
│   ├── onnx_export.py  #   uops_to_onnx(uops) / print_onnx_graph(model): build + print an ONNX graph
│   ├── rknn_export.py  #   onnx_to_rknn_bytes(model): build a .rknn for the RK3588 NPU (in RAM)
│   ├── rknn_run.py     #   run_rknn(model, inputs) / print_rknn_graph(model): run / print a .rknn on the NPU
│   ├── rknn_decode.py  #   decode_rknn(model): disassemble a .rknn (FlatBuffer graph + NPU command queue)
│   ├── rknn_encode.py  #   recreate_rknn(model): rebuild a .rknn from its decoded form (byte-exact)
│   ├── rknn_synth.py   #   uops_to_rknn(uops): synthesize a .rknn from UOps (no onnx, no toolkit)
│   ├── _rknn_flatbuf.py#   vendored: build the FlatBuffer body + container from scratch
│   ├── _rc_template_gen.py # vendored: generate the NPU register-command stream from scratch
│   └── viz.py          #   serve(model): host an ONNX/.rknn model in netron
├── refs/               # vendored references (no external deps)
│   └── rkt_registers.h #   rk3588 NPU register names, used to label the decoded command queue
├── add2d_ops/          # example: 2D nested-loop half add (N=100), index i*cols+j
│   ├── uops.py         #   make_add2d_uops(rows, cols): two RANGE loops + computed index
│   └── synth.py        #   2D-loop UOps -> .rknn (no onnx/toolkit) -> run on NPU
├── matmul_ops/         # example: n x n float matmul UOps (REDUCE contraction)
│   └── uops.py         #   make_matmul_uops(n): out[i,j] = sum_k a[i,k]*b[k,j]
├── add_ops/            # example: the 5x5 half `a + b` kernel
│   ├── uops.py         #   load_uops(): hard-coded snapshot of the original UOps
│   ├── run.py          #   load the uops, then fully unroll and print
│   ├── run_rknn.py     #   run on the NPU and compare against onnxruntime + a + b
│   ├── disasm.py       #   build the .rknn and statically disassemble it (offline)
│   ├── recreate.py     #   rebuild the .rknn from its decoded form, compare to the library
│   ├── synth.py        #   UOps -> .rknn with no onnx/toolkit, verify on the NPU
│   └── viz.py          #   host the ONNX graph (or exported .rknn) in netron
├── tests/              # pytest suite for helpers + the example
│   ├── test_uop.py
│   ├── test_unroll.py
│   ├── test_add.py     #   load -> unroll -> execute -> check a + b
│   ├── test_onnx.py    #   load -> unroll -> ONNX -> onnxruntime -> check a + b
│   ├── test_viz.py     #   build the ONNX file + netron serves it
│   ├── test_rknn.py    #   load -> unroll -> ONNX -> .rknn export
│   ├── test_decode.py  #   disassemble the .rknn: container + graph + command queue
│   ├── test_recreate.py#   rebuild the .rknn from decode and assert byte-identity + NPU run
│   ├── test_synth.py   #   UOps -> .rknn (no onnx/toolkit) for 1x1/3x3/5x5, run on NPU
│   ├── test_add2d.py   #   2D nested-loop add (N=100): unroll -> synth -> run on NPU
│   └── test_matmul.py  #   matmul (REDUCE): unroll -> interp == numpy; synth rejects it
├── conftest.py         # marks the project root for pytest imports
└── README.MD
```

`helpers/` is the shared library. Each example op lives in its own folder (e.g.
`add_ops/`) holding a copied UOp snapshot plus a runner.

## Run the example

From the project root:

```sh
python3 -m add_ops.run
```

This loads the original 14 UOps for the 5x5 half add, then fully unrolls the loop
(14 → 204 UOps, no RANGE left).

### Visualize the ONNX graph

Build the ONNX graph from the unrolled UOps and host it in [netron](https://github.com/lutzroeder/netron):

```sh
.venv/bin/python3 -m add_ops.viz       # serve on http://0.0.0.0:8080
```

Then open the printed URL in a browser. Or from code: `from add_ops.viz import show; show()`.
The graph is streamed to netron from memory — no `.onnx` file is written to disk.

### Export to .rknn

Build the ONNX graph and convert it to a Rockchip `.rknn` for the RK3588 NPU with
[rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2). The conversion runs
**in memory** — `onnx_to_rknn_bytes` returns the `.rknn` as bytes:

```python
from add_ops.viz import build_model
from helpers.rknn_export import onnx_to_rknn_bytes
blob = onnx_to_rknn_bytes(build_model())               # bytes; float16; target_platform="rk3588"
```

Want a file instead? `onnx_to_rknn(model, "add_5x5.rknn")` wraps it and writes the
bytes out.

Visualize the exported `.rknn` in netron (also served from memory):

```sh
python3 -m add_ops.viz --rknn          # export to .rknn and serve on http://0.0.0.0:8080
```

Or from code: `from add_ops.viz import show_rknn; show_rknn()`.

### Run the .rknn on the NPU

Run the model on the RK3588 NPU and compare against onnxruntime (and `a + b`):

```sh
.venv/bin/python3 -m add_ops.run_rknn
```

On this kernel the NPU output bit-matches onnxruntime exactly (max abs diff 0.0).

### Print the graphs

Dump the logical ONNX graph and the compiled NPU layer table side by side:

```sh
.venv/bin/python3 -m add_ops.run_rknn --graph
```

The ONNX listing shows the op graph (`Gather`/`Add`/`Concat`); the rknn side prints
the toolkit's `eval_perf` layer table — the lowered NPU ops that netron hides inside
the un-expandable NNBG node. From code: `print_onnx_graph(model)` and
`print_rknn_graph(model)`. (rknn-toolkit2 has no rknn->onnx export, so the .rknn graph
is only available as this text table, not as an ONNX file.)

Or from code:

```python
from add_ops.viz import build_rknn
from helpers.rknn_run import run_rknn
out = run_rknn(build_rknn(), [a, b], target="rk3588")  # list of output arrays
```

This needs the on-device NPU runtime (`librknnrt.so`); pass `target=None` to use the
toolkit's simulator instead.

Exports float16 by default; pass `do_quantization=True` with a `dataset` file (one
input-sample path per line) for a quantized model.

rknn-toolkit2 only loads/exports via file *paths* (it has no in-memory API), so the
intermediate `.onnx` and the `.rknn` are routed through a tmpfs dir (`/dev/shm`, RAM)
and removed before returning — nothing is written to persistent disk. Needs
`rknn-toolkit2` in the venv (which in turn needs `setuptools<81` for `pkg_resources`).

### Disassemble the .rknn (offline)

The `.rknn` graph that netron hides inside the NNBG node can be recovered statically —
no runtime, no NPU. `decode_rknn` parses the container, the FlatBuffer body (graph
nodes + the NC1HWC2 tensor table with byte sizes/offsets), and decodes the embedded
NPU **register-command stream** into a command queue, labelling each 64-bit word from
`refs/rkt_registers.h`:

```sh
.venv/bin/python3 -m add_ops.disasm           # graph + command-queue summary
.venv/bin/python3 -m add_ops.disasm --regs    # also dump every decoded register
```

From code:

```python
from helpers.rknn_decode import decode_rknn, print_rknn_disasm
m = decode_rknn("model.rknn")     # bytes or path
print_rknn_disasm(m)
# m["nodes"], m["tensors"], m["command_queue"] (each block: kind/detail + decoded regs)
```

Each `.rknn` is `[64-byte header][FlatBuffer body][u64 len + JSON trailer]`; the body's
`regcmd` tensor is a flat array of `target<<48 | value<<16 | reg` words grouped into
per-tile blocks. Blocks are classified `EW_BINARY` (the Add tiles), `COPY/reformat`
(reshape/convert), `COPY/alu` (split/slice data movement), or `OTHER`. The DMA
base-address fields read 0 — the runtime patches those per tile at submit time; the
geometry (width/height/channel/stride) is already baked in the file. This is the
disassembly direction of the decode work; it has no external dependencies (the register
header is vendored under `refs/`).

**[MAPPING.md](MAPPING.md)** traces how the unrolled UOps line up with this command
stream: the 25 scalar UOp adds map 1:1 to 25 compiled `Add` nodes and 25 `EW_BINARY`
regcmd tiles (each width-1), while the 50 `LOAD`s collapse into reformat/slice tiles and
the 25 `STORE`s into `Concat` + output reshape.

### Recreate the .rknn from its decoded form (byte-exact)

`recreate_rknn` is the inverse of the decoder: it rebuilds the container header **from
scratch** and re-emits the NPU command stream from the decoded command queue. For a
library-generated model the result is **byte-identical** (same SHA-256) and still runs
correctly on the NPU:

```sh
.venv/bin/python3 -m add_ops.recreate        # build with toolkit, recreate, compare bytes
```

```python
from helpers.rknn_encode import recreate_rknn
assert recreate_rknn(open("model.rknn","rb").read()) == open("model.rknn","rb").read()
```

This proves the container + command-stream codec is lossless. The FlatBuffer metadata
(vtables, tensor table, JSON attrs) and the trailing task tensor are carried through
verbatim — those are the compiler-owned parts.

### Synthesize a .rknn from UOps — no ONNX, no toolkit

The end goal: turn an unrolled UOp graph straight into a runnable `.rknn` without the
ONNX conversion *and* without rknn-toolkit2. `uops_to_rknn` validates a fully-unrolled
2-input element-wise add, recovers the element count, and builds the whole file from
scratch — the FlatBuffer body, the **NPU register-command stream** (generated by
`_rc_template_gen`, not copied from any library output), and the container:

```sh
.venv/bin/python3 -m add_ops.synth            # 1x1, 3x3, 5x5 -> build -> run on NPU
.venv/bin/python3 -m add_ops.synth 8 8        # custom size
```

```python
from helpers.unroll import fully_unroll
from helpers.rknn_synth import uops_to_rknn, run_uops_on_npu
from add_ops.uops import make_add_uops
blob = uops_to_rknn(fully_unroll(make_add_uops(25)))   # bytes; no onnx, no toolkit
z = run_uops_on_npu(make_add_uops(25), [a, b])         # one call: UOps -> NPU result array
```

`run_uops_on_npu(uops, [a, b])` is the whole pipeline in one call: it unrolls any RANGE
loops, synthesizes the `.rknn` from scratch, submits it to the NPU, and returns the
output array. It accepts the original loop UOps or an already-unrolled list.

Verified on the NPU for 1×1, 3×3, 5×5 (`z == a + b`, see `tests/test_synth.py`). The
build never imports `rknn` — only the `flatbuffers` library.

The same path handles a **2D nested-loop** kernel (`add2d_ops`): two `RANGE` loops with a
computed flat index `i*cols + j`, using the UOp operator syntax (`*`, `+`, an `INDEX` used
as a value is implicitly loaded, instance `.sink`). After `fully_unroll` (cartesian
product over both loops) the graph is the same flat per-element add, so `uops_to_rknn`
treats it identically — verified on the NPU for N=100 (10×10):

```sh
.venv/bin/python3 -m add2d_ops.synth          # 10x10 = 100, toolkit-free, run on NPU
```

**Reductions / matmul (`matmul_ops`).** The UOp layer also handles `Ops.REDUCE`: a matmul
`out[i,j] = Σ_k a[i,k]·b[k,j]` (three loops, one of them a `REDUCE`) builds, and
`fully_unroll` expands the contraction into an explicit sum-of-products — verified against
numpy by the pure-Python interpreter (`tests/test_matmul.py`). But matmul is **not** an
element-wise add, so `uops_to_rknn` deliberately rejects it: the NPU matmul/conv engine
(the `CNA`/`CORE` blocks) is a separate command stream that the toolkit-free synthesizer
does not emit. So matmul is reproducible/verifiable at the UOp level, but not yet runnable
on the NPU toolkit-free. This is **functional**
equivalence: the synthesized file computes the same result, but it is not byte-identical
to a toolkit build (it uses a compact single-tile / multi-surface memory plan instead of
the compiler's per-element `Split`/`Concat` layout). The builders are vendored under
`helpers/_rknn_flatbuf.py` + `helpers/_rc_template_gen.py` (reverse-engineered for RKNPU2
2.3.2 / rk3588); scope is the fp16 element-wise add (`z = a + b`).

## Tests

From the project root:

```sh
python3 -m pytest
```

The `uop`/`unroll`/`interp` helpers and their tests are pure-Python (no deps).
The ONNX export + its test need `onnx`/`onnxruntime`/`numpy`, so run those with the
project venv:

```sh
.venv/bin/python3 -m pytest
```

## Notes

- No tinygrad dependency: `helpers/uop.py` reimplements just the UOp pieces we need.
- `fully_unroll` only handles **constant-bound** ranges (it asserts otherwise).
- UOp snapshots are copied by hand for now; later this can read real UOps from tinygrad.
