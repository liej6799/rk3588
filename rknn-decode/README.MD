# rknn-decode

Standalone tooling for working with tinygrad-style linearized **UOps** without
depending on tinygrad. UOp graphs are captured (copied) from tinygrad and then
manipulated here — e.g. fully unrolling their loops — as a step toward RKNN decoding.

## Layout

```
rknn-decode/
├── helpers/            # reusable, tinygrad-free helper library
│   ├── uop.py          #   minimal UOp / Ops / dtypes / AxisType / KernelInfo + simplify/substitute/toposort
│   ├── unroll.py       #   fully_unroll(uops): expand RANGE/REDUCE, then auto-upcast (vectorize) like tinygrad
│   ├── interp.py       #   run_uops(uops, buffers): execute a range-free graph
│   ├── onnx_export.py  #   uops_to_onnx(uops) / print_onnx_graph(model): build + print an ONNX graph
│   ├── rknn_export.py  #   onnx_to_rknn_bytes(model): build a .rknn for the RK3588 NPU (in RAM)
│   ├── rknn_run.py     #   run_rknn / RknnSession (load once, run many) / print_rknn_graph
│   ├── rknn_decode.py  #   decode_rknn(model): disassemble a .rknn (FlatBuffer graph + NPU command queue)
│   ├── rknn_encode.py  #   recreate_rknn(model): rebuild a .rknn from its decoded form (byte-exact)
│   ├── rknn_synth.py   #   uops_to_rknn / chain_to_rknn: synthesize a .rknn from UOps / op-chains (no onnx, no toolkit)
│   ├── kernel.py       #   GENERIC DRIVER + CLI: dump / onnx / rknn / disasm / run+verify for any UOp kernel
│   ├── _rknn_flatbuf.py#   vendored: build the FlatBuffer body + container from scratch
│   ├── _rc_template_gen.py # vendored: generate the NPU register-command stream from scratch
│   └── viz.py          #   serve(model): host an ONNX/.rknn model in netron
├── refs/               # vendored references (no external deps)
│   └── rkt_registers.h #   rk3588 NPU register names, used to label the decoded command queue
│
│   # Each example op is JUST a uops.py. All debug features come from helpers/kernel.py.
├── add_ops/uops.py     # load_uops(): the 5x5 half `a + b` kernel (flat RANGE loop)
├── add2d_ops/uops.py   # make_add2d_uops(rows, cols): 2D nested loops, index i*cols+j (N=100)
├── mul_ops/uops.py     # make_mul_uops(rows, cols): element-wise multiply (tinygrad test_mul)
├── matmul_ops/uops.py  # make_matmul_uops(n): out[i,j] = sum_k a[i,k]*b[k,j]  (REDUCE)
├── tests/              # pytest suite for helpers + the examples
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
│   ├── test_mul.py     #   element-wise multiply (N=100): unroll -> synth (MUL EW op) -> NPU
│   ├── test_upcast.py  #   fully_unroll auto-upcast == tinygrad E_25_4 (vec(4)) + interp check
│   ├── test_chain.py   #   toolkit-free mixed-op chains (MULACC a*b+c) in one .rknn on NPU
│   └── test_matmul.py  #   matmul (REDUCE): unroll -> interp == numpy; synth rejects it
├── conftest.py         # marks the project root for pytest imports
└── README.MD
```

`helpers/` is the shared library. **An example op is just a `uops.py`** — every debuggable
feature is provided generically by `helpers/kernel.py`, so adding a custom kernel means
writing only its UOps.

## The driver — one CLI for every kernel

```sh
python3 -m helpers.kernel <pkg> dump [--full]   # original -> unrolled UOps (summary or full)
python3 -m helpers.kernel <pkg> onnx            # serve the ONNX graph in netron (from memory)
python3 -m helpers.kernel <pkg> rknn            # serve the toolkit-free synthesized .rknn in netron
python3 -m helpers.kernel <pkg> disasm [--regs] # decode + print the NPU command queue
python3 -m helpers.kernel <pkg> run [--perf]    # synthesize -> run on NPU -> verify vs interpreter
```

`<pkg>` is any example folder (`add_ops`, `add2d_ops`, `mul_ops`, `matmul_ops`) or your own
— it just needs `uops.py` with `load_uops()`. Examples:

```sh
python3 -m helpers.kernel add_ops dump          # 14 -> 204 unrolled UOps (5x5 add)
python3 -m helpers.kernel mul_ops onnx          # the multiply graph in netron
python3 -m helpers.kernel matmul_ops run        # 8x8 matmul on the NPU == interpreter
```

Verification ground truth is the pure-Python interpreter (`helpers.interp`), so there's no
hand-written reference per kernel. Element-wise kernels are synthesized toolkit-free and run
directly; a reduction (matmul) runs via the element-wise MULACC decomposition
(`helpers.kernel.matmul_npu`). From code: `from helpers.kernel import load, dump, run_verify;
run_verify(load("add_ops"))`.

### Adding a custom kernel

Drop a `mykernel/uops.py` with a `load_uops()` returning your UOp graph (using
`helpers.uop` — `UOp.range`, `.index`, `+`, `*`, `.reduce`, `.store`, `.sink`). Then every
command above works on it: `python3 -m helpers.kernel mykernel run`.

## Internals (what the driver is built on)

### Export to .rknn

Build the ONNX graph and convert it to a Rockchip `.rknn` for the RK3588 NPU with
[rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2). The conversion runs
**in memory** — `onnx_to_rknn_bytes` returns the `.rknn` as bytes:

```python
from helpers.kernel import load, to_onnx
from helpers.rknn_export import onnx_to_rknn_bytes
blob = onnx_to_rknn_bytes(to_onnx(load("add_ops")))    # bytes; float16; target_platform="rk3588"
```

Want a file instead? `onnx_to_rknn(model, "add_5x5.rknn")` wraps it and writes the
bytes out.

Visualize the exported `.rknn` in netron (also served from memory):

```sh
python3 -m helpers.kernel add_ops rknn   # serve the synthesized .rknn in netron
```

Or from code: `from helpers.kernel import load, serve_rknn; serve_rknn(load("add_ops"))`.

### Run the .rknn on the NPU

Run the model on the RK3588 NPU and compare against onnxruntime (and `a + b`):

```sh
.venv/bin/python3 -m helpers.kernel add_ops run
```

On this kernel the NPU output bit-matches onnxruntime exactly (max abs diff 0.0).

### Print the graphs

Dump the logical ONNX graph and the compiled NPU layer table side by side:

```sh
.venv/bin/python3 -m helpers.kernel add_ops disasm
```

The ONNX listing shows the op graph (`Gather`/`Add`/`Concat`); the rknn side prints
the toolkit's `eval_perf` layer table — the lowered NPU ops that netron hides inside
the un-expandable NNBG node. From code: `print_onnx_graph(model)` and
`print_rknn_graph(model)`. (rknn-toolkit2 has no rknn->onnx export, so the .rknn graph
is only available as this text table, not as an ONNX file.)

Or from code:

```python
from helpers.kernel import load, compile_rknn
from helpers.rknn_run import run_rknn
out = run_rknn(compile_rknn(load("add_ops")), [a, b], target="rk3588")  # list of output arrays
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
.venv/bin/python3 -m helpers.kernel add_ops disasm          # graph + command-queue summary
.venv/bin/python3 -m helpers.kernel add_ops disasm --regs   # also dump every decoded register
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
python3 -c 'from helpers.kernel import load, compile_rknn; from helpers.rknn_encode import recreate_rknn; b=compile_rknn(load("add_ops")); print(recreate_rknn(b)==b)'
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
.venv/bin/python3 -m helpers.kernel add_ops run   # synth -> NPU -> verify vs interp
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
.venv/bin/python3 -m helpers.kernel add2d_ops run   # 10x10 = 100, toolkit-free, run on NPU
```

**Element-wise multiply (`mul_ops`).** `z = a * b` runs on the NPU toolkit-free too — same
tiles as the add, with the EW datapath retargeted from the ALU-sum op to the multiply op
(`EW_CFG` `0x108202c0` → `0x108003c4`, i.e. `EW_OP_TYPE=1`; recovered by diffing a toolkit
Add vs Mul model). `uops_to_rknn` picks the op automatically from the UOp graph
(`analyze_elementwise` returns `(N, ADD|MUL)`). Verified on the NPU for N=100:

```sh
.venv/bin/python3 -m helpers.kernel mul_ops run   # 10x10 = 100, z = a*b, toolkit-free
```

**Auto-vectorization (UPCAST), like tinygrad.** `fully_unroll` doesn't just scalar-expand —
it then **auto-upcasts**, mirroring tinygrad's `hand_coded_optimizations` rule *"if nothing
is upcasted and the last axis % 4 == 0, apply `OptOps.UPCAST` by 4"*. So a contiguous
element-wise add/mul with `N % 4 == 0` is vectorized into `half.vec(4)` (`CAST` base→vec
pointer, a vec(4) `LOAD` per input, per-lane `GEP` + scalar op, a `STACK`, a vec(4) `STORE`),
and anything else stays scalar (N not divisible by 4, or a reduction like matmul). The
determination is automatic — there is no separate upcast method.

For the 10×10 mul this is **byte-for-byte identical** to tinygrad's `E_25_4` (579 UOps; same
op/dtype/src on every line — `tests/test_upcast.py`); a 5×5 (N=25) stays scalar (204 UOps).

```sh
python3 -m helpers.kernel mul_ops dump          # 10x10 -> auto-upcast vec(4) listing (== E_25_4)
python3 -m helpers.kernel mul_ops dump --full   # the full vectorized UOp list
```

It's a host-codegen detail: scalar and vectorized forms compute the same thing and
synthesize the **same** NPU `.rknn` — `uops_to_rknn`/`analyze_elementwise` accept either and
key off the element count and op, not the vectorization (byte-identical output, verified).

The upcasted UOps can also be lowered the **other** way — through ONNX and the toolkit — to
see the full path down to the raw NPU command. `uops_to_onnx` handles the vectorized ops
(`CAST`/vec `LOAD` → `Gather` of n contiguous indices, `GEP` → `Gather` of one lane,
`STACK` → `Concat`), so the upcast graph becomes a verbose ONNX (250 Gather / 100 Mul / 26
Concat). Compiling that with the toolkit **re-fuses it back to a single `Mul`** → one
`EW_BINARY` tile on the NPU:

```python
from helpers.kernel import load, to_onnx, compile_rknn          # upcast UOps -> ONNX -> rknn
from helpers.rknn_decode import decode_rknn
m = to_onnx(load("mul_ops"))                                    # {Gather:250, Mul:100, Concat:26}
d = decode_rknn(compile_rknn(load("mul_ops")))                  # toolkit re-fuses -> {Mul:1} node
```

So whether the multiply enters as scalar UOps, vectorized UOps, or a verbose ONNX, it ends
up as the same single element-wise tile on the NPU.

`uops_to_rknn` accepts the **upcasted UOps directly** too (toolkit-free): `analyze_elementwise`
recovers `(N, op)` from either the scalar or the vectorized graph (peeling `CAST`/`GEP`/`STACK`),
so synthesizing from the upcast form yields a `.rknn` **byte-identical** to the scalar one
(same SHA-256) — verified on the NPU. The vectorization is host-side bookkeeping; the NPU
command is one tiled op regardless of how the UOps are shaped.

**Reductions / matmul (`matmul_ops`).** The UOp layer also handles `Ops.REDUCE`: a matmul
`out[i,j] = Σ_k a[i,k]·b[k,j]` (three loops, one of them a `REDUCE`) builds, and
`fully_unroll` expands the contraction into a **fused multiply-accumulate** chain —
`MUL` to seed each output's sum then `MULACC` (FMA), exactly like tinygrad's `r_8_8_8`
kernel (`UNROLL axis=0`). For 8×8 this matches tinygrad's op counts (64 `STORE`, 64 `MUL`,
448 `MULACC`, 0 `RANGE`); CSE (hash-consing) dedups the shared `a[i,k]`/`b[k,j]` loads to
128 (tinygrad further vec4-groups the contiguous `a` loads to 80, and uses a
scheduler-specific op order). Verified against numpy by the interpreter
(`tests/test_matmul.py`).

A single matmul `.rknn` can't be synthesized toolkit-free (no matmul/conv command-stream
emitter — `uops_to_rknn` rejects it). But matmul **decomposes into element-wise ops we can**.
The synth now supports **mixed-op chains in one `.rknn`** — `chain_to_rknn(ops, N)` builds a
body with per-op EW tiles (Add/Sub/Mul/Div), e.g. `["Mul","Add"]` is a fused
multiply-accumulate `a*b + c` (both Mul tiles `EW_CFG …c4` and Add tiles `…c0` in one body).
So `A @ B` runs as K **MULACC** NPU ops: for each k, `acc = A[:,k]*B[k,:] + acc`, each a
single toolkit-free `.rknn`:

```sh
.venv/bin/python3 -m helpers.kernel matmul_ops run   # 8x8 matmul on NPU == interp (8 MULACC ops)
# matmul 8x8x8 = 8 MULACC toolkit-free NPU ops in ~2.6s
# NPU == numpy matmul: True
```

The K MULACC steps all run the **same** model (only the inputs change), so `matmul_npu`
synthesizes the MULACC `.rknn` once and loads it into one `RknnSession` (a single
`load_rknn` + `init_runtime`); the K steps are just `inference` calls. That setup dominates,
so 8×8 and 16×16 both run in ~2s (the per-step cost is tiny). It can't be fused to a
*single* inference — the dot-product `Σ a·b` isn't a left-associated op-chain — but it's one
NPU context, not K.

(Broadcasts are host-side numpy; the arithmetic — multiply + accumulate — runs on the NPU
via toolkit-free element-wise tiles. The chain synth handles **2–7 inputs / 1–6 mixed ops**
in one `.rknn` (verified on the NPU); 8+ inputs need an extracted regcmd template. No
compiler, no caching.) But matmul is **not** an
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
