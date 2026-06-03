# Mapping: unrolled UOps ↔ RKNN raw command stream

How the logical UOp dataflow becomes NPU register-commands, traced end-to-end on the
5×5 half `a + b` example. Pipeline:

```
UOps --(fully_unroll)--> ONNX --(rknn-toolkit2 compile)--> .rknn --(decode_rknn)--> command queue
```

## Layer-by-layer

| UOp (per element k = 0..24) | ONNX | compiled node | regcmd block |
|---|---|---|---|
| `LOAD(INDEX(a,k))`, `LOAD(INDEX(b,k))` — 50 LOADs | `Gather` ×50 | `Reshape` (input1_rs/input2_rs) + `Split`/`Slice` → per-element `ldN_2sl` tensors | 2 × `COPY/reformat` + 10 × `COPY/alu` (data movement, no compute) |
| `ADD` — 25 | `Add` ×25 | `Add` node ×25 | **`EW_BINARY/alu` ×25** (each width = 1) |
| `STORE(INDEX(out,k))` — 25 | `Concat` | `Concat` + `Reshape` (output-rs) | folded into reformat / OTHER |
| `PARAM` out / a / b | model inputs+output | `InputOperator` ×2 / `OutputOperator` + working tensors `*_rs` | none (compute-free; supply DMA addresses) |

## The one invariant that lines up exactly

```
25 scalar UOp ADDs  ==  25 ONNX Add  ==  25 compiled Add nodes  ==  25 EW_BINARY regcmd tiles
```

Each add tile has `DATA_CUBE_WIDTH = 0` (width = 1) — the direct image of the scalar,
1-element UOp add. (Asserted in `tests/test_decode.py`.)

## What does *not* map 1:1

- **The 50 LOADs are not 50 NPU loads.** The toolkit reformats the whole contiguous
  `[25]` input into NC1HWC2 once (`Reshape`), then carves per-element slices
  (`Split`/`Slice` → the `ldN_2sl` tensors). The "loads" become data-movement (`COPY`)
  tiles plus runtime address patching, not compute instructions.
- **The 25 STOREs collapse** into `Concat` + the output `Reshape` (NC1HWC2 → contiguous).
- **Total blocks ≠ total nodes**: 42 regcmd blocks for 35 nodes. The extra blocks are
  ping/pong template copies and the CNA/CORE (`OTHER`) helper tiles; the runtime's
  `task` descriptor selects which blocks actually execute.

## How an add tile is identified (the EW config bits)

The decoder distinguishes a real 2-operand add from a 1-operand copy by the **baked**
(not runtime-patched) elementwise-config bits — see `helpers/rknn_decode.py::_classify`:

| | `EW_CFG` (0x4070) | `ERDMA_CFG` (0x5034) | meaning |
|---|---|---|---|
| **add tile** | `0x8202c0` → `EW_OP_SRC=1`, `EW_ALU_ALGO=2`, not bypassed | `0x8` → `ERDMA_DATA_SIZE=2` (fp16), reader enabled | 2nd operand read from memory ⇒ binary |
| **copy/split tile** | `0x383` → `EW_BYPASS=1` | `0x1` → `ERDMA_DISABLE=1` | EW engine off, 1 operand ⇒ copy |

So `EW_BINARY = (DATA_FORMAT == alu) and EW_OP_SRC and not EW_BYPASS`.

## The addressing gap (why this is structural, not per-element)

The operand/destination DMA base fields in the file read **0**
(`DPU_DST_BASE_ADDR`, `DPU_RDMA_RDMA_SRC_BASE`, `DPU_RDMA_RDMA_EW_BASE`). The runtime
patches them per tile at submit time from the tensor **memory plan**:

```
src0(0x5018) -> a-slice base ,  src1(0x5038) -> b-slice base ,  dst(0x4020) -> z-slice base
```

where the bases come from each tensor's `f13` offset (decoded in `m["tensors"]`). So a
static decode shows each tile's *geometry* and *op type*, but the per-element operand
binding (which `a[k]`/`b[k]` a given tile reads) lives in the memory plan + runtime
relocation, not in the regcmd words themselves.

## Summary

A UOp graph is a flat per-element dataflow (`LOAD`/`ADD`/`STORE`). Its NPU image is a
small set of **data-reformat tiles** (contiguous ↔ NC1HWC2) + **N scalar EW-add tiles** +
**reassembly**, with per-element addressing supplied by the memory plan rather than by
explicit load instructions. The compute (the 25 adds) maps 1:1; the memory layout is the
part the compiler owns.
