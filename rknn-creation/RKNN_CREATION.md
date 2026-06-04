# How an RKNN model is created (fp16 element-wise Add, RKNPU2 2.3.2 / rk3588)

This documents the full path from ONNX to a runnable `.rknn`, what each part of the
file means, and exactly which parts we can reproduce **without** the rknn-toolkit2
compiler. Verified against the real NPU (`/usr/lib64/librknnrt.so`, `/dev/dri/renderD128`)
using `rknn_run_generic` (x[i]=i, y=10, expects z=i+10).

---

## 1. The two-layer toolkit pipeline

`gen_add_rknn.py` calls `rknn.api.RKNN`: `config -> load_onnx -> build -> export_rknn`.
Internally (see `RKNN_EXPORT_pipeline.txt`):

* **Layer 1 (Python/Cython)** rewrites the ONNX graph: `base_optimize`, `fold_constant`,
  `fuse_ops`. For Add this inserts `unsqueeze_to_4d_add` -> the `x_rs / y_rs / z-rs`
  reshape nodes you see in the model.
* **Layer 2 (`librknnc.so`, 34 passes)** is the real compiler. The passes that matter
  for reproduction are `RKNNMultiSurfacePass` (tiles work across cores),
  `RKNNSubGraphMemoryPlanPass` (assigns every tensor a padded size + an offset in the
  shared feature DMA buffer), `OpEmit` / `RKNNModelRegCmdbuildPass` (emit the 64-bit NPU
  register-command stream), and `RKNNModelExportPass` (serialise to the container).

So "creating an RKNN" = **graph -> tiled register-command stream + a memory plan**, then
serialise. The memory plan and the tiling are the hard, compiler-owned parts.

---

## 2. Container format  (`flatbuffer_to_rknn.py`, `assemble_rknn.py`)

```
[0x00] 64-byte header
       0x00  "RKNN" magic (+4 zero)
       0x08  u64 version (= 6)
       0x10  u64 bodySize (length of the FlatBuffer body)
       0x18..0x40 zero
[0x40] FlatBuffer body        (bodySize bytes)  -- the compiled model
[end]  u64 jsonLen + JSON     -- the trailer ("connection"/graph/io metadata)
```

The header rebuilds from scratch; round-trips byte-exact. **The trailer JSON `size`
dims are cosmetic for the runtime** — it reports tensor dims from the FlatBuffer body,
not the trailer (proven: patching the trailer to `2x50` still reports `[10,10]`).

---

## 3. FlatBuffer body schema (the tensor table)

Root.f2 -> subgraph[0].f0 -> vector of 15 tensor tables. For the 10x10 Add:

```
 0 (header)   4 x          5 y          6 x_exSecondary   7 x_rs
 8 y_exSecondary  9 y_rs  10 z-rs  11 z-rs_exSecondary  12 z  13 regcmd  14 task
```

Per-tensor fields that matter (recovered with `extract_rknn_build_queue.FB`):

| field | meaning | example (x, 10x10) |
|------|---------|--------------------|
| f3 | **native** NPU shape (NC1HWC2) | `x_rs = [1,2,1,10,8]` = [N,C1,H,W,C2] |
| f4 | **logical** shape | `x_rs = [1,10,1,10]` |
| f5 | name string | `"x"` |
| f12 | **buffer byte size** (padded) | `320` |
| f13 | **offset** in feature DMA buffer | `x_rs @ 704` |

`C2 = 8` (a 16-byte fp16 atom); `C1 = ceil(C/8)`. The runtime allocates each tensor
`f12` bytes at offset `f13`, copies the user input into it, then patches the regcmd
base-address fields to `feature_base + f13`.

### Memory-plan size rule (DERIVED BY DYNAMIC SWEEP — the toolkit build runs on-device)

Running the real compiler (`gen_add_rknn.py`) over a sweep of shapes and reading back
the emitted `TensorMemory` (f12 size / f13 offset) per tensor gives the exact allocator math
for the working buffers `x_rs/y_rs/z-rs`:

```
native shape = [1, C1, 1, C, 8]          C1 = ceil(R/8)   (R folds into channel surfaces)
byte size    = C1 * pad4(C) * 16         pad4(x) = round up to multiple of 4 ; 16 = C2(8)*2B
```
Confirmed: 7x7 -> W 7->8 (1*8*16=128); 6x9 -> 9->12 (1*12*16=192); 9x8 -> C1=2 (256);
17x8 -> C1=3 (384); 1x32 -> 1*32*16=512. External x,y are contiguous-ish and the output z
is separately optimised; offsets f13 are packed per-region with reuse (allocator output).

### Memory-plan size rule (also visible across 10x10 / 16x16 / 20x20)

* external `x,y`:  `f12 = ceil(R/8) * C * 16`
* working `x_rs,y_rs,z-rs`: `f12 = ceil(R/8) * pad4(C) * 16`  (W rounded up to mult-of-4:
  10->12 gives 384, but 16->16 and 20->20 stay)
* output `z`: irregular/optimised (256 for 10x10, 832 for 20x20 — not a clean formula)
* `f13` offsets: packed across regions **with reuse** (e.g. 20x20 has `z-rs@0` and
  `y_rs@0` in different regions). This is allocator output.

These irregularities (pad-to-4, optimised output sizing, aliased offsets) are why a
brand-new element count cannot be produced by formula alone — it needs the compiler's
`RKNNSubGraphMemoryPlanPass`.

---

## 4. The register-command stream (regcmd)

A flat array of 8-byte words. Decode:

```
target = (word >> 48) & 0xFFFF     # DPU=0x1001, DPU_RDMA=0x2001, ...
value  = (word >> 16) & 0xFFFFFFFF
reg    =  word        & 0xFFFF     # names in rkt_registers.h
```

The compiler emits one ~69-word block per tile. 10x10 and 20x20 use **6 blocks**
(3 tasks x ping/pong); 16x16 uses **9**. A real Add block (`DPU_RDMA_RDMA_ERDMA_CFG`
0x5034 bit30 set = second operand enabled) reads two operands and writes one:

```
DPU_DATA_CUBE_WIDTH      0x4030 = W-1
DPU_DATA_CUBE_HEIGHT     0x4034 = H-1
DPU_DATA_CUBE_CHANNEL    0x403c = 0x00070007   (fp16 8-channel pack, constant)
DPU_DST_BASE_ADDR        0x4020 = output  (patched at runtime)
DPU_WDMA_SIZE_1          0x405c = W-1
DPU_RDMA_RDMA_SRC_BASE   0x5018 = input0  (patched)
DPU_RDMA_RDMA_EW_BASE    0x5038 = input1  (patched)
DPU_RDMA_RDMA_SURF_NOTCH 0x504c / 0x506c     (tiling remainder)
```

**Crucially the runtime patches only the three DMA base addresses, never geometry** —
so the cube geometry (W/H/channel/stride/notch) must already be correct in the file.

### 4b. The regcmd template is fully decodable — NOT an opaque blob (`rc_template_gen.py`)

`rknn_flatbuf.py` used to carry the regcmd region as three gzip+base64 constants
(`_RC_TEMPLATES[2/3/4]`). These are **completely regular** and have been decoded into a
readable generator (`rc_template_gen.py`); `rknn_flatbuf.py` now does
`_RC_TEMPLATES = rc_template_gen.all_templates()`.

Each template (n_inputs = 2..4) is:

```
[OFF[n] zero bytes]              FlatBuffer alignment lead (4,0,4 for n=2,3,4)
PREFIX[n] words                  header + embedded TASK DESCRIPTOR TABLE
                                 (6*n_adds entries: base spaced 0x280, reg_amount=0x45=69,
                                  channel_mask=0x1ffff)  -- the table from §6 UPDATE 4
6 tiles x ( for j in 0..n_adds-1: block + gap )   ping/pong x 3 tasks = 6 tiles
TRAILING[n] words                tail task descriptors (n=2 zero; n=3/4 non-zero)
```

* **block** = 52 DPU regs (`0x4004..0x412c`) + 17 DPU_RDMA regs (`0x500c..0x506c`) = the
  canonical EW-Add config (`CANON`). The patched fields (`0x4020/4030/403c/4058/405c/4034/
  4024/40c0` and RDMA `500c/5014/5010/5018/5038/5040/504c/506c`) are stored as 0 because
  `_rc_patch_block` fills them per-tile at build time.
* **two block variants**: 69-word, and 71-word = `CANON` + a 2-word PC preamble at the END
  (`PC 0x10` base + `PC 0x14 = 0x24`). The 71-block is used for chained-add intermediates.
* **block count = 6 * n_adds**; per tile the first `n_adds-1` blocks are 71-word, the last
  is 69-word. The slot (block+gap) is always **80 words** (69+11 or 71+9).
* the **only** non-patch word that varies across blocks is the 71-block's `PC 0x10` base,
  which follows a clean formula: `base = (global_block_index + 1) * 0x280`.
* the 6 copies of a block differ ONLY in the fields `_rc_patch_block` overwrites, so they
  reduce to ONE canonical block.

**Verified** (`verify_rc_gen.py`): `build_template(n)` is byte-identical to the original
gzip blob *after* `_rc_patch_block` runs, for n=2,3,4 across all shapes (1 .. 16M); the n=2
full `build_body()` body is byte-identical end-to-end. Decode/inspect tooling:
`decode_rc_templates.py` (parser, register-name annotation, exact round-trip rebuild).

Note: the prefix's task-descriptor table is exactly the 6-entry structure §6 UPDATE 4
identified as runtime-validated — it sits in `PREFIX[n]`, untouched by patching, and is why
the 6-block cap holds.

---

## 5. The single-tile recipe (what we author ourselves)

Instead of reproducing the compiler's multi-surface tiling, we replace every block with
**one full-width tile** that processes all N elements in a single DPU pass. The official
`fp16_add_1024` model already contains exactly such a block (W=1023), which proves the
hardware accepts it. Per N:

```
0x4030 = 0x405c = 0x500c = N-1     # all width regs
0x4034 = 0x5010 = 0                # height = 1
0x504c = 0x506c = 0                # NO notch (nonzero notch caused tail corruption)
0x4020 = 0x5018 = 0x5038 = 0       # base offsets (runtime patches)
0x4024 = 0x40c0 = 0x5040 = N*16    # surface strides
keep the constant EW-Add config regs (0x400c, 0x4010, 0x403c, 0x4070, 0x5034, ...)
```

Verified `correct true` on the NPU for N=100 (10x10 body) and N=400 (20x20 body).
The two failure modes encountered and fixed:
* keeping the original surface stride (192B < 200B) -> tail wrapped at element 96;
* leaving `SURF_NOTCH = 32` -> tiling remainder corrupted the last elements.

---

## 6. What `rknn_add_gen.py` produces, and the boundary

Status note: the first part of this section records the original single-surface
implementation and its 8176-element ceiling. The current implementation extends
that by using C1 surface tiling and up to 6 add tiles; see **TOOLKIT-FREE
MULTI-SURFACE TILING** and **UPDATE 3** below for the active limit.

`rknn_add_gen.py` (toolkit-free) SYNTHESISES the whole model for an arbitrary new element
count, using a clean 1D-contiguous plan (C1=1) instead of reproducing the compiler's
irregular padded/aliased 2D plan, on the overwrite-tolerant 6-block add_10x10 body:
  - external x,y,z : native [1,N], size N*2
  - working x_rs/y_rs/z-rs : native [1,1,1,N,8], size N*16
  - feature offsets : 256-byte aligned, non-overlapping
  - single-tile regcmd, forcing the 8-channel pack (DATA_CUBE_CHANNEL=0x00070007,
    WDMA_SIZE_0=7, RDMA chan=7) so it doesn't depend on the copied template block.

Verified PASS on the NPU (rknn_verify, fp16-safe inputs, 0 mismatches) for arbitrary shapes:
1x1, 2x3, 7x7, 13x11, 10x10, 20x20, 32x32, 37x41, 50x50, 64x64, 70x100, 1x2048, 1x8176,
8176x1. **Range: N = rows*cols in 1..8176** (one working buffer N*16 must stay < 128KB).
The runtime reports the contiguous [1,N] shape; rows*cols is recorded in the trailer.
(Forcing QUERY dims to [R,C] makes rknn_inputs_set expect a C-channel padded layout that
conflicts with the contiguous plan and breaks submission for larger N, so we keep [1,N].)

**Reproducible without the compiler:** container, trailer, element-wise *math*, AND the
memory plan + layout for any N<=8176 via the 1D-contiguous scheme.

**Still compiler-only:** the native *2D NC1HWC2 padded* plan with multi-surface tiling and
aliased offsets (only needed if you want the padded [R,C] layout rather than the equivalent
contiguous one), and N>8176 multi-tile splitting (cf. the official 8192/16384 models).

### Why 8176 is a hard single-tile ceiling
The DPU element-wise path requires the 8-channel fp16 pack (DATA_CUBE_CHANNEL=0x00070007),
so the working buffer is inherently N*16 bytes (8 lanes * 2 B, 7 lanes are padding). Setting
CH=0 (1 lane, N*2 buffer) was tested and FAILS (87/100 mismatches at N=100) — the hardware
does not read a dense contiguous buffer. The H (height) dimension doesn't help either: a
W x H tile still needs W*H*16 bytes. The runtime rejects a working buffer >= 128KB
(N*16 >= 131072 => N >= 8192; with 256B alignment the safe max is 8176). Going past it requires
TRUE multi-tile: several <=8176-element chunks, each its own regcmd block + task-array entry,
processed in sequence. That needs reconstructing the embedded "task" command tensor (the
runtime expands it into rknpu_task[]); bodies with >6 blocks / >3 tasks (16x16, the 1D 1024
model) reject our 6-block-tolerant overwrite, confirming the task count is what gates it.

### Multi-tile attempt (decisive negative result)
ioctl-sniffer confirms the runtime submits exactly **3 tasks = regcmd blocks 0,1,2** (0x280
spacing), core_mask, flags=0x5 (PC|PINGPONG). The per-tile width limit is **W<=8175** (one tile
= 8176 elems); the official 16384 model uses TWO such add-tiles, and its 256KB working buffer
allocates fine here, so the cap is the tile width, NOT buffer size.

I tried assigning the 3 active blocks to different chunks (block k -> chunk k%nchunks, regcmd
base offset = chunk_start*16). Result: **chunk 0 correct, everything from element 8176 on is
zero** — `first bad @8176, got 0.0` for N=10000/12000/13000 alike, independent of the chunk
split. Cause: the **input reshape `x->x_rs` is itself a single 8176-capped NPU tile**, and our
single-tile recipe OVERWROTE those reshape blocks, so `x_rs[8176..] = 0` and every later
chunk adds zeros. The output reshape `z-rs->z` is likewise a capped tile.

**Why per-block chunk offsets don't work (key discovery):** the runtime OVERWRITES the regcmd
base-address registers (DST 0x4020, SRC 0x5018, EW 0x5038) with each tensor's planned DMA base
at rknn_run — it does not add to whatever offset we wrote. Proof: a model where all blocks were
set to process elements [4000,8000) (dst/src offset 64000) produced correct output at z[0..3999]
and zero at z[4000..] — i.e. our offset was discarded and every block wrote [0,W). Since all
blocks bind to the same x_rs/y_rs/z-rs (offset 0), every tile reads/writes the same region.
=> Per-chunk addressing must come from DISTINCT per-chunk TENSORS (each with its own f13 offset),
which means growing the tensor vector AND reconstructing the regcmd->tensor relocation/binding
table the runtime uses to choose what to patch. That binding table is the compiler's, not in
our reach by template reuse.

**Why the H (height) dimension doesn't rescue it:** a single tile with HEIGHT>0 and row stride
DST_SURF_STRIDE=TW*16 DOES read a TW x TH cube contiguously — verified for tiny cases (100x2,
100x3 PASS, 0 mismatches). But it breaks for TH>=4 and for any large TW*TH (8176x2 fails at
@12865; 2000x3 @2592). The multi-row NC1HWC2 layout needs the surface/notch tiling the compiler
emits, which we don't reproduce. So H buys at most ~3 tiny rows, not real scaling.

Conclusion: lifting the cap is not possible by reusing the 6-block body. It needs the full
tiled PIPELINE — per chunk: reshape-in(x), reshape-in(y), add, reshape-out(z), each <=8176 —
DISTINCT per-chunk tensors, the regcmd->tensor relocation table, and an expanded `task` array.
That is exactly `RKNNMultiSurfacePass` + memory planner + scheduler, i.e. reimplementing the
compiler's tiler. **8176 stands as the hard ceiling for the template-reuse approach.**

### How the toolkit ACTUALLY implements N>8176 (captured via ioctl_sniff)

Built `add_128x128` (N=16384) and `add_256x256` (N=65536) with the toolkit and captured the
command queue (`capture_cmdqueue.py`). Both verify PASS (0 mismatches, fp16-safe inputs). The
single ONNX `Add` becomes a **pipelined, multi-tile** queue in ONE submit:

```
add_10x10  (100)   : 1 submit, 3 tasks   = 3 add tiles (single-pass, no convert)
add_128x128(16384) : 1 submit, 12 tasks  = 3 transpose + 3 reshape + 4 ADD + 2 store
add_256x256(65536) : 1 submit, 12 tasks  = same 4-ADD-tile pipeline
fp16_add_16384(1D) : 1 submit, 48 tasks  = 18 transpose + 18 reshape + 9 add + 3 store
```

So the user's hypothesis is correct: **the task is separated**. For N>8176 the work is a 4-stage
pipeline — transpose+reshape (load/convert inputs contiguous->NC1HWC2), several ADD tiles, store
(convert output back) — instead of the single-pass 3-tile form used for small N. The 2D `[R,C]`
form is far more compact (12 tasks for 65536) than the 1D form (48 tasks for 16384), because
`[R,C]` maps to NC1HWC2 with C1=ceil(R/8) surfaces processed in few wide tiles.

**THE ADDRESSING MECHANISM (proved by dumping the runtime-patched regcmd live).** The sniffer
already reads each task's regcmd from the mmap'd buffer at submit time (i.e. AFTER patching) and
writes per-task .bin files. Decoding those for add_10x10 and add_128x128 gives the authoritative
rule:

```
patched_addr(field)  =  tensor_f13_base(bound_tensor)  +  per_tile_delta
bindings:  src0(0x5018) -> x_rs ,  src1(0x5038) -> y_rs ,  dst(0x4020) -> z-rs
```

add_10x10 (3 tiles): src0 base 704 (=x_rs f13), src1 1088 (=y_rs f13), dst 0 (=z-rs f13);
per-tile deltas {tile0:0, tile1:0, tile2:192}. add_128x128 (4 add tiles): same bindings, deltas
{0,0,16384,0} into the 32768-byte z-rs.

**The baked regcmd address fields are IGNORED.** Proof: a hand-edited model where block1 had
baked dst=64000 was patched to base+0, and block2 with baked dst=0 was patched to base+192 — i.e.
the runtime used the original template's relocation deltas (0,0,192), not our edits. So the
binding + per-tile deltas live in a relocation structure the runtime builds from the model, NOT
in the regcmd words we can rewrite.

**The per-tile delta is a SURFACE offset (multi-surface tiling).** Decoding the add-tile dst
deltas against the NC1HWC2 surface size (z-rs byte size / C1):

```
add_10x10  : z-rs=384   surf=192   tile deltas {0,192}    = {0,1} surfaces
add_20x20  : z-rs=960   surf=320   tile deltas {0,640}    = {0,2} surfaces
add_128x128: z-rs=32768 surf=2048  tile deltas {0,16384}  = {0,8} surfaces
```

So the toolkit splits the Add across the `C1 = ceil(R/8)` channel surfaces (the "8/8/8
multi-surface split" = `RKNNMultiSurfacePass`), each tile covering a contiguous run of surfaces;
the per-tile delta is just `first_surface_index * surface_bytes`. The inter-block gaps are
constant PC-control words (not relocation), so the delta is COMPUTED by the runtime from the
tiling/partition, not stored as a literal we can patch.

**Why our toolkit-free generator caps at 8176:** the 1D-contiguous plan uses `[1,1,1,N,8]` =
**C1=1, a single surface**, so there is nothing to surface-tile — one ~8176-wide tile is the max.
To scale we must adopt the 2D NC1HWC2 layout (`C1=ceil(R/8)` real surfaces) AND reproduce the
multi-surface split: per-tile relocation deltas = surface offsets, the transpose/reshape/store
convert tiles that pack contiguous input into the surfaces, and the expanded task array. That is
`RKNNMultiSurfacePass` + memory planner + scheduler — the compiler's tiler. Mechanism now fully
characterised end-to-end; building it is a separate port. Tools: `capture_cmdqueue.py MODEL
--tiles` and `ioctl_sniff.so` + `CMDBUF_DIR=...` (dump runtime-patched regcmd .bin).

### TOOLKIT-FREE MULTI-SURFACE TILING (IMPLEMENTED, 8x the ceiling)
We do NOT need to port the relocation table after all. The runtime computes the per-surface
deltas ITSELF from the tensor's C1 dimension. So the recipe is simply: lay the work into
`C1 = ceil(N/8176)` channel surfaces by setting the working tensors' native shape to
`[1, C1, 1, W, 8]` (W = ceil(N/C1)), size = C1*W*16, and set each regcmd tile to width W. The
runtime then auto-tiles one DPU pass per surface and patches `tensor_base + surface_index*surf_bytes`
on its own. C1=1 is the original single-tile case.

Verified on the NPU (rknn_verify, 0 mismatches), all TOOLKIT-FREE:
```
N=100 (C1=1) .. 8176 (C1=1) .. 10000 (C1=2) .. 20000 (C1=3) .. 30000 (C1=4) ..
50000 (C1=7) .. 65408 (C1=8, W=8176)   ALL PASS
```
**UPDATE 3 - multi-tile splitting extends toolkit-free range to ~50M (HARD CAP at 6 tiles).**
Analysis of the vendor compiler's output for 4096x4096 shows it uses multiple add tiles,
each with CHANNEL <= 8191 (13-bit), and byte-offset base addresses that the runtime adds
to the tensor base (e.g., tile 1 at offset 0, tile 2 at offset 256*W*16 = 0x01000000).
Implemented in `rknn_add_gen.py`: when C1 > 1024, the 6 template blocks become up to 6
tiles, each covering ceil(C1/6) surfaces with its own CHANNEL and base address offset.
Ceiling: 6 tiles x 1024 surfaces x 8176 width = **N <= 50,231,944 (~50M)**.
Verified: 4096x4096 (N=16.8M, 3 tiles) and 6144x8176 (N=50.2M, 6 tiles) PASS with 0
mismatches, toolkit-free. For N > 50M, use `--allow-toolkit` to fall back to the vendor
compiler.

**UPDATE 4 - body extension beyond 6 blocks is blocked by runtime validation.**
Extensive testing confirmed that the runtime validates the body content against FlatBuffer-
encoded expectations. Any change to the body structure beyond the 6 template regcmd blocks
is rejected with "Invalid RKNN format" at load time. Approaches that ALL failed:

  - **Dynamic extension**: appending block+gap sequences past the 6th block -> rejected
  - **Rebuild from scratch**: constructing the entire body (FlatBuffer + regcmd + gaps +
    terminator) from scratch, both with the original FlatBuffer and with patched fields ->
    rejected
  - **FlatBuffer field patching**: modifying individual subgraph/node fields to reflect the
    new body structure -> rejected (fields are interdependent)

Root cause (fully diagnosed): the FlatBuffer embeds a **task descriptor table** with exactly
6 entries (one per regcmd block). This table is at body bytes 7160-7399 (40 bytes per entry),
followed by the regcmd_size (3840) at body[7420]. The body layout is:
  - FlatBuffer: 0-7423 (7424 bytes, includes task descriptor table)
  - Regcmd blocks: 7424-11263 (3840 bytes = 6 × 80 words × 8)
  - Task data: 11264-11519 (256 bytes)
  - Total: 11520 bytes
Each task descriptor entry contains: base_addr, block_words(69), channel_mask(0x1FFFF), stride
constants. The runtime computes expected body size from this table and rejects any body that
doesn't match. Adding more blocks requires extending the task descriptor table WITHIN the
FlatBuffer, which needs the full FlatBuffer schema (the table is a serialized vector of
FlatBuffer tables with vtable pointers). Patching sg.f7, sg.f10, sg.f4, or regcmd_size alone
is insufficient — the runtime validates the FlatBuffer structural integrity holistically.

Additionally tested: using the vendor 8192x8192 model (18 regcmd spans) as a template. Patching
add block register values works (model loads and structure is accepted), but the vendor model's
FlatBuffer has a complex multi-core operation graph (transpose/reshape/add/store) that cannot
be simplified to add-only without breaking the operation scheduler. The body extension path was
removed from `rknn_add_gen.py`; the 6-block cap is the definitive toolkit-free limit.

**UPDATE 2 - the 15-bit CHANNEL claim was wrong; it is 13-bit (mask 0x1FFF).**
The earlier "15-bit" ceiling was based on reverse-engineering guesswork, but the actual
register definitions in rkt_registers.h show all channel-carrying fields (DPU_DATA_CUBE_CHANNEL,
WDMA_SIZE_0, RDMA_DATA_CUBE_CHANNEL) use a 13-bit mask (0x1FFF, max 8191) with bits [15:13]
reserved. Empirically confirmed: 4096x4096 (C1=2053, ch=16423 > 8191) produces 96.5% mismatches
(the hardware truncates ch=16423=0x4027 to 13 bits -> 39, processing only ~5 of 2053 surfaces).
2048x2048 (C1=514, ch=4111 <= 8191) passes. Corrected ceiling: C1*8-1 <= 8191 => C1 <= 1024
per tile.

**UPDATE - the 8-surface cap was an artifact of a hardcoded CHANNEL value, now removed.**
The old code set the add tile DATA_CUBE_CHANNEL=0x00070007 (8 channels), which let the runtime
auto-tile only 8 surfaces. Setting CHANNEL = C1*8-1 (the FULL channel count) makes ONE add tile
span all C1 surfaces. `rknn_add_gen.py` does this automatically (`ch=C1*8-1` in make_tiles;
C1=1 -> ch=7 = the original single-surface value, so it unifies).
(The convert-tile reproduction in convert_tile.py remains valid and would be needed only to
exceed 50M via a vendor-built FlatBuffer template with more than 6 blocks.)

Old text below kept for context:
New ceiling: N <= 65408 (the runtime tiles at most 8 channel surfaces per submit; C1=16 fails
mid-buffer at the 8-surface boundary, not a buffer-size limit). `rknn_add_gen.py` does this
automatically (`C1=surface_split(N)`), one interface, instant, no compiler.

For N > 65408 (e.g. 256x256=65536 needs 9 surfaces), `--allow-toolkit` falls back to the on-device
build.

**Multi-core does NOT lift the cap (tested).** Built C1=12/16/24 models and ran them with
`rknn_set_core_mask(RKNN_NPU_CORE_0_1_2)` (`rknn_verify_mc`): identical failure to single core
(first 8 surfaces correct, the rest wrong; e.g. C1=12 N=98112 -> 32193 mismatches on both 1- and
3-core). So core_mask alone doesn't make the runtime tile >8 surfaces — that needs a model built
for multi-core, which our template isn't.

**Dense channel packing also doesn't trivially work.** Declaring `[1,1,1,ceil(N/8),8]` so the 8 C2
lanes carry data fails (rknn_inputs_set doesn't produce the exact NC1HWC2 channel interleave the
DPU reads). Our working scheme keeps 1 logical element per width slot and lets the C1 surface
dimension do the multiplying.

Going fully toolkit-free beyond 65408 therefore needs a bigger EMBEDDED TASK ARRAY (the toolkit
fits 65536 in one submit via dense 4-tile x 4-surface packing) — a larger effort; the on-device
fallback covers it today.

### How the vendor scales the command queue (captured)
The number of `DRM_IOCTL_RKNPU_SUBMIT` calls is ALWAYS 1, for every size 100 .. ~1,000,000. The
vendor does NOT split into multiple submits; it grows the `rknpu_task[]` array inside the single
submit. Each submit is a 4-stage pipeline (transpose=load+NC1HWC2 convert, reshape, matmul=Add
tiles, store=output convert), and more tasks of each stage are added as N grows:

```
              submits  transpose reshape add store  total
1D  8192         1        21       21     6    3      51
1D 24576         1        24       24    12    3      63
1D 65536         1        57       57    27    3     144
2D 256x256       1         3        3     4    2      12   (2D packs surfaces densely)
chain 1024x1024  1         -        -     -    -    3093   (~1M elems, 1024 chained adds)
```

The kernel walks first_task..last_task and runs them sequentially as ONE job; there is no
per-chunk re-submission from userspace. 2D [R,C] is far more compact than 1D for the same N
(12 vs 144 tasks at 65536) because it maps straight into C1=ceil(R/8) channel surfaces. So the
toolkit-free scaling lever is the embedded TASK ARRAY size (one submit), not the submit count —
which is exactly why reusing the 10x10 3-task template caps us at 65408.

### The convert-tile spec (decoded) and why one add tile = 8 surfaces
The pipeline tiles other than the add use DPU_DATA_FORMAT=0x24000001 (reformat/transpose mode;
the add uses 0x48000002). Decoded from add_128x128:
- transpose (op2): src=x (contiguous input buffer), dst=x_rs (feature). Reformats input->NC1HWC2.
  HEIGHT=R-1, CHANNEL=R-1, NOTCH=(R-1,R-1), WDMA_SIZE_0=0x080f007f packs (C2=8,C1-1,W-1),
  DST_SURF_STRIDE=16, SURFACE_ADD=128.
- reshape (op3): same, for y -> y_rs.
- store  (op5): src=z-rs, dst=z (contiguous output). NC1HWC2 -> contiguous.
add_256x256 (R=256 -> C1=32 surfaces) uses 4 ADD tiles => **each add tile covers 8 surfaces**.
That is our 8-surface cap exactly: one EW-Add tile spans 8 channel surfaces, and the vendor scales
by emitting MORE add tiles (4 tiles -> 32 surfaces -> 65536) plus the matching transpose/reshape/
store convert tiles, all in one submit.

So a fully toolkit-free large-N emitter = grow the task array: K add tiles (K*8 surfaces) + their
convert tiles + the expanded `task` descriptor. The convert geometry (WDMA_SIZE/NOTCH packings
above) must be computed from the dims. This is the remaining port; the field encodings are now
captured (use `ioctl_sniff.so` + decode), but it is a multi-day build, not a patch.

### MILESTONE: input-convert (transpose/reshape) tile reproduced (convert_tile.py)
First convert tile of the pipeline is now reproduced FROM DIMS and validated. The input-convert
tile (contiguous x[R,C] -> x_rs NC1HWC2 [1,ceil(R/8),1,C,8]) is one DPU reformat tile
(DATA_FORMAT=0x24000001, DATA_CUBE_WIDTH=0). Derived register formula (see convert_tile.py):
```
0x400c FEATURE_MODE = 0x1e5 | ((ceil(R/8)*ceil(C/8) >> 4) << 16)
0x4034 HEIGHT = R-1   0x4038 NOTCH = 0x403c CHANNEL = ((C-1)<<16)|(C-1)
0x4058 WDMA0  = (8<<24)|((ceil(C/8)-1)<<16)|(C-1)     0x405c WDMA1 = (ceil(R/8)-1)<<16
0x5010 RDMA_H = R-1   0x5014 RDMA_CH = C-1   0x5048 = (C-8)<<16
0x504c RDMA_SURF_NOTCH = -(R-1)*C*2 (two's complement)   0x4024=16  0x40c0=128  consts
```
Validated BYTE-EXACT vs the toolkit across 64x128/128x128/256x128/128x64/128x256 and the held-out
96x96, and VERIFIED ON THE NPU: zeroing then rebuilding both input-convert tiles of add_96x96 from
this formula and running gives 0 mismatches. (Transpose and reshape are the same tile, differing
only in the runtime-patched src/dst; the store tile is a separate reformat with WIDTH!=0.)
### MILESTONE 2: store tile reproduced -> WHOLE convert pipeline done
The output convert (store: z_rs NC1HWC2 -> z[R,C] contiguous) is the other reformat tile
(DATA_FORMAT=0x24000001, but DATA_CUBE_WIDTH != 0). Formula (convert_tile.py store_regs):
```
0x4024 DST_SURF_STRIDE = C*2     0x4030 WIDTH = ceil(C/8)-1   0x4034 HEIGHT = 7 (=C2-1)
0x403c CHANNEL = ((R-1)<<16)|(R-1)   0x4058 WDMA0 = (8<<24)|((ceil(R/8)-1)<<16)|(R-1)
0x405c WDMA1 = ((ceil(C/32)-1)<<16)|3   0x40c0 SURFACE_ADD = C*16   0x5014 RDMA_CH = R-1
0x400c FEATURE_MODE = same as input-convert
```
Validated BYTE-EXACT for C<=128 (64x128/128x128/256x128/128x64/96x96). For C>128 the store
splits into multiple width-tiles (WIDTH caps at 15) -- the one remaining sub-case. NPU-VERIFIED:
rebuilding ALL convert tiles of add_96x96 (2 input-convert + 1 store) from formula gives 0
mismatches. => the entire data-reformatting pipeline (contiguous<->NC1HWC2) is now reproducible
from dims.

Remaining for the full toolkit-free large-N port: emit K add tiles (K*8 surfaces) wired to the
convert tiles + grow the embedded `task` array + memory plan. The compute tiles (convert +
single add) are all reproduced; what's left is assembling a K-surface model.

### How the official library creates LARGE models (gdb + memory monitoring)
Confirmed with gdb (break ioctl, filter DRM_IOCTL_RKNPU_SUBMIT=0xc0686441) + ioctl_sniff
(mem_create/mem_map monitor). The library scales purely by GROWING THE EMBEDDED TASK ARRAY in
ONE submit -- no multi-submit, no memory-sync streaming:

  DMA objects per inference (mem_create):
    handle 1 (0x403): the .rknn model (regcmd blocks + task tensor).   16KB -> 314KB -> 2.4MB
    handle 2 (0x40b): the TASK array DMA buffer = page_round(n_tasks*40). 4KB(3) -> 8KB(144 tasks)
    handle 3 (0x403): feature/working buffer, scales with N.            4KB -> 557KB -> 4.4MB
    handles 4..6:     input/output buffers.
  gdb-confirmed task_number: 51 (fp16_add_8192), 144 (fp16_add_65536), all in ONE submit.
  Scaling: N=8192 -> 51 tasks / 21 blocks; 65536 -> 144 tasks / 48 blocks. Each rknpu_task =
  {op_idx, reg_amount, regcmd_addr}; reg_amount VARIES (69 full tile, 9/22/25 partial/edge);
  tasks reuse blocks (up to 3x). Pipeline = transpose+reshape (convert) -> add tiles -> store.

=> To beat our 6-tile cap, emit the SAME: many (varying-size) regcmd blocks + a large "task"
command-tensor of descriptors. Memory mechanism is plain (one job, one big feature buffer synced
whole); the only remaining decode is the task-tensor byte format. Memory sync is NOT the lever.

---

## 6b. radare2 / objdump confirmation (librknnc.so, 28 MB)

Static analysis of the compiler `…/rknn/api/lib/linux-aarch64/librknnc.so` corroborates
the empirically-derived model:

* **All 34 pass names present** as strings, incl. `RKNNSubGraphMemoryPlanPass`,
  `RKNNMultiSurfacePass`, `RKNNModelRegCmdbuildPass`, `RKNNModelExportPass`, and
  **`RKNNFlatcModelBuildPass`** (the flatc/FlatBuffer serialiser).
* **FlatBuffer schema types** (from the symbol table, demangled):
  `rknn::fbs::TensorLayout`, `rknn::fbs::TensorMemory`, `rknn::fbs::TensorType`.
  These map onto the tensor-table fields we reversed: TensorLayout = the native/logical
  shape vectors (f3/f4), TensorMemory = byte size + offset (f12/f13), TensorType = dtype.
* **The memory plan is a `std::unordered_map<std::string, rknn::fbs::TensorMemory>`**
  (tensor name -> {size, offset}) plus `vector<pair<int, vector<TensorLayout>>>` — i.e.
  exactly the name-keyed size/offset table that `rknn_add_gen.py` writes.
* **Layout/field tokens**: `NC1HWC2`, `c1c0`, `feat_stride`, `real_model_offset`,
  `align`, and conversion kernels `NC1HWC2_2_NCHW` / `NC1HWC2_2_NW1HCW2` (these are the
  `x_rs`/`z-rs` reshape nodes).
* **Plan-size logging string** names the regions of the body:
  `total plan size … regcmd buffer size … task buffer size … weight size … internal size … misc size`.

Not extracted: the exact byte-level allocator arithmetic (W-pad-to-4 etc.) — the 28 MB
binary is stripped of local symbols and resists cheap xref analysis (adrp+add data refs
weren't resolved by `aav`/`aar` within budget). Those constants were instead recovered
empirically from the 10x10/16x16/20x20/1024 models (section 3).

---

## 6c. Why the body still needs a TEMPLATE: the runtime's `Verify ModelBuffer` (librknnrt.so)

**Current design is HYBRID (not fully toolkit-free for the container).** `build_body()` clones
a raw reference body (`_body_add{n}_10x10.body`, ex-`_embedded_bodies.py`, now de-base64'd) and
PATCHES it: `_plan_memory` rewrites tensor size/offset/shape (f12/f13/f3/f4), `_patch_tiles`
rewrites the regcmd command blocks. So **the NPU commands + memory plan are ours; the FlatBuffer
metadata wrapper is still copied from a toolkit-built reference.**

Proven the *commands* are fully synthesisable (splice test, NPU-verified 0 mismatches):
`body = template_FB_skeleton + rc_template_gen.build_template(n) + _taskdesc(n)`, then
`_plan_memory`/`_patch_tiles` → loads and computes correctly. Only the FB skeleton is template-bound.

**Pure FlatBuffers rebuild status (`_build_body_scratch_flatbuffers`)** — reverse-engineered from
`librknnrt.so`:
* Load path does `parseRKNN` → "Verify ModelBuffer". Located at vaddr **0x2ad490** (loads the
  err string "Verify ModelBuffer failed!" @0x60b478 / xref @0x2ad570). Pre-checks: `size>11`,
  `buffer[4:8]=="RKNN"`, `0 < root_off <= size-1`, then calls the **flatc verifier @0x2d06b0**.
* It is **structural FlatBuffer verification, NOT a checksum** — proven because the patched
  reference bodies (we rewrite shapes + regcmd) still load. Verifier helpers identified:
  `0x2c8c58`=VerifyTableStart, `0x2a94e8`=VerifyField(scalar), `0x2c9958`=VerifyOffset,
  `0x2c9850`=VerifyString, `0x2c97d0`=VerifyVector.
* The 2-input pure path now loads and computes correctly **for all N** (tested 100, 10000,
  1048576) without copying the raw FlatBuffer skeleton. The key decoded quirks were:
  - subgraph f7 middle field is a vector of 8-byte `(offset, id)` pairs, not a flat
    `vector<uint32>`;
  - root f20/f21 are verifier-visible relative targets into the command/task regions, while
    later runtime code still depends on the raw value matching `HEADER_SIZE + region_start`;
  - the generated FlatBuffer needs the same root padding canonicalization before regcmd
    patching, otherwise final command words are patched at the wrong alignment.
* **Critical alignment constraint:** `_RC_TEMPLATES` contains DPU register command blocks at
  half-word (4-byte) offsets within the blob — they are NOT uint64-aligned when the blob
  starts at byte 0. The blocks are designed so that when the blob is placed at the correct
  body offset, the block starts land on uint64 word boundaries. Specifically, the template
  body for n_inputs=2 has `fb_end=6812` (≡4 mod 8), placing the first DPU block at body
  byte 7424 (≡0 mod 8, word 928). If the FlatBuffer prefix length has the wrong modulo-8
  residue, `_regcmd_spans` (which scans the entire body as uint64 words) finds zero spans
  and the regcmd is never patched, causing NPU "failed to submit" at runtime. The fix:
  pad the FlatBuffer prefix so that `fb_len % 8 == template_fb_end % 8`, derived
  dynamically from the reference template. Without this fix, whether a given N works or
  not depends on whether the variable-length FB content (JSON attrs strings containing N)
  happens to produce an fb_len with the correct alignment by chance.
* Pure FlatBuffers generation now supports **2 through 7 inputs** (all NPU-verified).
  For n_inputs=5,6,7 the regcmd template is extracted from the reference bodies
  (`_rc_add{5,6,7}.bin`) since `rc_template_gen.py` only covers 2-4. The remaining
  per-input-count specs are now **generated algorithmically** (no hardcoded tables):
  - `_SG_F7_SPECS` → `_generate_sg_f7_specs(n)`: IO offset/id pairs computed from
    base+step formula (input bases 55/61/141+(k-2)*80, output base 5+(n-2)*80,
    inter bases 5+(k-1)*80 and 135+(k-1)*80, all with step=80*n_adds).
  - `_EXSEC_F13` → `_generate_exsec_f13(n)`: inputs 0,1 get `320*n`; inputs ≥2
    cycle `[384, 768, 1152, None]` with period 4; output-rs = `384 if n%4==2 else None`.
    (This is a hardware bank pattern — 3 active secondary data banks + 1 idle slot.)
  - `_ROOT_F19` → inline formula `(192+(n-2)*64, 1472+(n-2)*320)`. The old
    hardcoded values for n=5,6,7 were wrong (huge garbage numbers); the runtime
    does not use field 19, so both correct and wrong values pass NPU verification.
  - The guard check now only requires `_RC_TEMPLATES` (regcmd binary blob).
* Two tables remain **hardcoded per input-count** (builder-calibrated, not derived from
  the template body):
  - `_ROOT_CMD_TARGET_DELTAS`: per-n deltas `(d20, d21)` that specify where in the
    RC/taskdesc sections the root command targets point. The template body has a
    universal constant `(64, 216)` for all n, but our FlatBuffer builder produces a
    different layout (different fb_len), requiring different absolute targets. The
    per-n values are calibrated empirically for our builder output. Attempting to use
    the template values causes "Invalid RKNN format" rejection by the runtime.
  - Alignment (`_root_command_offsets`): `fb_end % 8` from the template body
    determines the padding needed so DPU regcmd blocks land on uint64 word boundaries.
    The formula `even→0, odd→4` matches the template for all n=2..7, but the builder
    output changes when other tables change, making the formula fragile. The template-
    based lookup remains the safe approach.
* **n>7 support**: the ONLY remaining blocker is `_RC_TEMPLATES[n]` — the binary DPU
  register command stream. All other tables generate algorithmically for arbitrary n.
  To add n=8: (1) generate a reference model with `onnx` + `rknn-toolkit2` (via
  `rknn_add_gen.py --inputs 8`), (2) extract the regcmd blob to `_rc_add8.bin`,
  (3) extract the fb_end alignment and calibrate deltas from the reference body,
  (4) update `_ROOT_CMD_TARGET_DELTAS` and `_get_template_body`. No other tables
  need updating.
* `build_body()` now uses the pure FlatBuffers path (`_build_body_scratch_flatbuffers`)
  as its default — no template FB skeleton is copied at all. The old template-patching
  path is preserved as `build_body_scratch()` for reference. Verified: 2-7 inputs ×
  (10x10, 100x100, 1024x1024) = 18 tests, all PASS with 0 mismatches.

`build_body_scratch` now takes the pragmatic spec-conformant path: keep the decoded
toolkit FlatBuffer skeleton, regenerate the command region from `rc_template_gen.py`
and `_taskdesc()`, then patch shapes, memory, and regcmds. For 2-input Add this is
byte-identical to the live template-patching path and passes NPU verification.

## 6d. Multi-Op Support: Add / Sub / Mul / Div Element-Wise Operations

The generator now supports all 4 arithmetic element-wise ops (Add, Sub, Mul, Div) in
any combination — single-op or multi-op chained. CLI: `--op {Add,Sub,Mul,Div}` for
uniform ops or `--ops Mul,Add,Sub,Div` for per-operation assignment.

### Four op-specific DPU registers

Decoded from vendor reference models (Add/Sub/Mul/Div 10x10). Only 4 registers differ
per op type; all other 67+ DPU/RDMA registers in the EW block are identical:

| Register | Field | Add | Sub | Mul | Div |
|----------|-------|-----|-----|-----|-----|
| `0x4070` (DPU_EW_CFG) | op config | `0x108202c0` | `0x108402c0` | `0x108003c4` | `0x108303c0` |
| `0x403c` (DATA_CUBE_CHANNEL) | channel geometry | `(ch<<16)\|ch` | `((W-1)<<16)\|ch` | `((W-1)<<16)\|ch` | `((W-1)<<16)\|ch` |
| `0x4084` (DPU_OUT_RES) | output resolution | `0x00010001` | `0x00010001` | `0x00010001` | `0x00000001` |
| `0x5044` (RDMA_BN_MUL) | batch norm multiplier | `0x00017849` | `0x00017849` | `0x00017849` | `0x00017841` |

Key differences:
- **EW_CFG** (0x4070): each op has a unique bit pattern (CVT_BYPASS, TRUNC_NEG, etc.).
  These are **lookup table values** — NOT computed by bit manipulation (earlier attempts
  to flip bit 2 were wrong; Sub/Mul/Div have different bit patterns).
- **CHANNEL** (0x403c): Add uses `(ch<<16)|ch` (both fields = channel count - 1); all
  others use `((W-1)<<16)|ch` (upper = width-1). Note: our flat layout uses W = total
  elements (vendor uses a tiled W×C layout); both work on the NPU.
- **Div uniquely** changes OUT_RES from `0x00010001` to `0x00000001` and BN_MUL from
  `0x00017849` to `0x00017841`.

### Per-group CANON blocks in `rc_template_gen.py`

The template generator now accepts a per-group op list:

```python
build_template(n_inputs, ops=[EW_OP_MUL, EW_OP_ADD, EW_OP_SUB])
```

Each CANON block is parameterized with the correct EW_CFG, OUT_RES, and BN_MUL from
lookup dicts (`_EW_CFG`, `_DPU_OUT_RES`, `_RDMA_BN_MUL`). The template emits blocks
in **tile-major interleaved order**: `[tile0_op0, tile0_op1, ..., tile0_opK, tile1_op0, ...]`.

### Three bugs found and fixed (multi-op 4+ inputs)

NPU verification passed for 2-input and 3-input models but **failed** for 4+ inputs
(max_err ~1600, completely wrong arithmetic). Root causes:

**Bug 1 — Strided span grouping (CRITICAL).** The RC template emits blocks in
tile-major order, but `_patch_regcmd` / `_patch_ew_ops` / `make_tiles` grouped them
consecutively (`spans[g*k:(g+1)*k]`). This applied each op's config to a MIX of all
ops' blocks — 67% of tiles got the wrong EW_CFG. Fix: strided selection
`spans[g::n_adds]`. Affected:
- `rknn_flatbuf.py:_patch_regcmd` (line 869)
- `rknn_add_gen.py:_patch_ew_ops` (line 207)
- `rknn_add_gen.py:make_tiles` (line 309)

**Bug 2 — Missing f13 memory offset for d_rs (n_inputs=4).** A broken special case
`not (n_inputs == 4 and nm == "d")` at `rknn_flatbuf.py:616` skipped the f13 memory
offset field for the 4th input tensor. The runtime then read d_rs from offset 0 (aliased
with the output buffer) instead of its correct slot. Fix: always set `has_f13=True`.

**Bug 3 — Undersized task descriptor (n_inputs >= 4).** `_taskdesc()` hardcoded
`rec_in * 3` for all `n_inputs >= 3`, producing only 3 input reshape records. For 4+
inputs the hardware's task descriptor table was incomplete. Fix: `rec_in * n_inputs`.

### NPU verification results

All 16 tests pass (via `librknnrt.so` direct C API on the rk3588 NPU):

| Test | Shape | Inputs | Ops | max_err | Result |
|------|-------|--------|-----|---------|--------|
| Add | 10x10 | 2 | Add | 0.001356 | PASS |
| Sub | 10x10 | 2 | Sub | 0.001978 | PASS |
| Mul | 10x10 | 2 | Mul | 0.002302 | PASS |
| Div | 10x10 | 2 | Div | 0.017689 | PASS |
| Mul+Sub | 10x10 | 3 | Mul,Sub | 0.001951 | PASS |
| Div+Add+Mul | 10x10 | 4 | Div,Add,Mul | 0.064896 | PASS |
| All4 | 10x10 | 5 | Mul,Add,Sub,Div | 0.050186 | PASS |
| Mul | 32x32 | 2 | Mul | 0.006386 | PASS |
| Mul+Sub | 32x32 | 3 | Mul,Sub | 0.004517 | PASS |
| Div+Add+Mul | 32x32 | 4 | Div,Add,Mul | 0.110687 | PASS |
| Mul | 64x64 | 2 | Mul | 0.004256 | PASS |
| Mul+Sub | 64x64 | 3 | Mul,Sub | 0.004521 | PASS |
| Div+Add+Mul | 64x64 | 4 | Div,Add,Mul | 1.156982 | PASS |
| Mul | 128x128 | 2 | Mul | 0.005001 | PASS |
| Mul+Sub | 128x128 | 3 | Mul,Sub | 0.007370 | PASS |
| Div+Add+Mul | 128x128 | 4 | Div,Add,Mul | 6.939453 | PASS |

The max_err increase with size for Div-containing chains is expected fp16 precision
accumulation (Div amplifies small differences). All within `atol=0.1, rtol=0.02`.

### Register comparison: generated vs vendor (10x10)

Three critical op-type registers match **byte-exact** across all 4 ops:

| Op | Register | Generated | Vendor | Match |
|----|----------|-----------|--------|-------|
| Add | EW_CFG | `0x108202c0` | `0x108202c0` | exact |
| Sub | EW_CFG | `0x108402c0` | `0x108402c0` | exact |
| Mul | EW_CFG | `0x108003c4` | `0x108003c4` | exact |
| Div | EW_CFG | `0x108303c0` | `0x108303c0` | exact |
| Div | OUT_RES | `0x00000001` | `0x00000001` | exact |
| Div | BN_MUL | `0x00017841` | `0x00017841` | exact |

Geometry registers (WIDTH, CHANNEL, strides) differ because our model uses a flat
W=total_elements layout while the vendor uses a tiled W=rows layout. Both work on the
NPU — verified for all 4 ops.

---

## 7. Toolbox

| file | role |
|------|------|
| `rknn_add_gen.py` | toolkit-free generator (params -> raw-body template patches + trailer -> rknn), `--verify` |
| `rknn_flatbuf.py` | body builder: `build_body` = pure FlatBuffers (2-7 inputs, no template FB skeleton); algorithmic generators for `_SG_F7_SPECS`, `_EXSEC_F13`, `_ROOT_F19`; only `_RC_TEMPLATES`, alignment, and deltas are per-n hardcoded; `build_body_scratch` = spec-conformant skeleton + regenerated commands; old template-patching path preserved for reference |
| `_body_add{2..7}_10x10.body` | raw reference bodies (ex-base64 `_embedded_bodies.py`), 2..7-input |
| `_rc_add{5,6,7}.bin` | extracted regcmd templates for 5-7 input chained Add |
| `decode_embedded_bodies.py` | decode a `.body` into FB-prefix / regcmd-blocks / taskdesc-tail |
| `gen_ref_addn.py N` | mint a reference body for N inputs via on-device toolkit (names match `_io`) |
| `rc_template_gen.py` | READABLE regcmd template generator (replaces gzip `_RC_TEMPLATES`) |
| `decode_rc_templates.py` | decode/annotate/round-trip the regcmd templates (§4b) |
| `verify_rc_gen.py` | proves `build_template` == original after patching, all shapes |
| `oracle_required.py` / `scratch_oracle.py` / `fb_verify.py` / `type_oracle.py` | verifier-as-oracle tooling for the from-scratch FB RE (§6c) |
| `flatbuffer_to_rknn.py` | split/assemble container |
| `extract_rknn_build_queue.py` | FlatBuffer parser (FB class) |
| `decode_cmdbuf.py` | classify/decode regcmd blocks |
| `rknn_run_generic` / `.cpp`, `rknn_verify*` / `.cpp` | vendor-API NPU correctness harnesses (`rknn_verify_n` = N-input) |
| `/data/rk3588/rknn-header/rkt_registers.h` | rk3588 NPU register definitions |
