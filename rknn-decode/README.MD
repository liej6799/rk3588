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
│   └── viz.py          #   serve(model): host an ONNX/.rknn model in netron
├── refs/               # vendored references (no external deps)
│   └── rkt_registers.h #   rk3588 NPU register names, used to label the decoded command queue
├── add_ops/            # example: the 5x5 half `a + b` kernel
│   ├── uops.py         #   load_uops(): hard-coded snapshot of the original UOps
│   ├── run.py          #   load the uops, then fully unroll and print
│   ├── run_rknn.py     #   run on the NPU and compare against onnxruntime + a + b
│   ├── disasm.py       #   build the .rknn and statically disassemble it (offline)
│   └── viz.py          #   host the ONNX graph (or exported .rknn) in netron
├── tests/              # pytest suite for helpers + the example
│   ├── test_uop.py
│   ├── test_unroll.py
│   ├── test_add.py     #   load -> unroll -> execute -> check a + b
│   ├── test_onnx.py    #   load -> unroll -> ONNX -> onnxruntime -> check a + b
│   ├── test_viz.py     #   build the ONNX file + netron serves it
│   ├── test_rknn.py    #   load -> unroll -> ONNX -> .rknn export
│   └── test_decode.py  #   disassemble the .rknn: container + graph + command queue
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
(reshape/convert), or `OTHER`. The DMA base-address fields read 0 — the runtime patches
those per tile at submit time; the geometry (width/height/channel/stride) is already
baked in the file. This is the disassembly direction of the decode work; it has no
external dependencies (the register header is vendored under `refs/`).

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
