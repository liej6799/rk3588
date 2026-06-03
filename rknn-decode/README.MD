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
│   └── onnx_export.py  #   uops_to_onnx(uops): build an ONNX graph from a range-free graph
├── add_ops/            # example: the 5x5 half `a + b` kernel
│   ├── uops.py         #   load_uops(): hard-coded snapshot of the original UOps
│   ├── run.py          #   load the uops, then fully unroll and print
│   └── viz.py          #   build the ONNX graph and host it in netron
├── tests/              # pytest suite for helpers + the example
│   ├── test_uop.py
│   ├── test_unroll.py
│   ├── test_add.py     #   load -> unroll -> execute -> check a + b
│   ├── test_onnx.py    #   load -> unroll -> ONNX -> onnxruntime -> check a + b
│   └── test_viz.py     #   build the ONNX file + netron serves it
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
