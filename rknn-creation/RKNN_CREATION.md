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

`rknn_flatbuf.py` used to carry the regcmd region as gzip+base64 constants
(`_RC_TEMPLATES`). These are **completely regular** and have been decoded into a
readable generator (`rc_template_gen.py`); `rknn_flatbuf.py` now does
`_RC_TEMPLATES = rc_template_gen.all_templates(max_n=16)`.

Each template (n_inputs = 2..64) is:

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
* Pure FlatBuffers generation now supports **2 through 64 inputs** (NPU-verified
  end-to-end for every n=2..64 at 10x10, plus spot checks at 64x64). `MAX_INPUTS=64`
  in both `rc_template_gen.py` and `rknn_flatbuf.py`; the CLI `--inputs` choice list
  tracks it automatically. The regcmd templates for all supported input counts are
  generated by `rc_template_gen.py`; every `_RC_TEMPLATES[n]` is algorithmic, not
  extracted from a toolkit reference blob. The remaining per-input-count specs are
  also **generated algorithmically** (no hardcoded tables):
  - `_SG_F7_SPECS` → `_generate_sg_f7_specs(n)`: IO offset/id pairs computed from
    base+step formula (input bases 55/61/141+(k-2)*80, output base 5+(n-2)*80,
    inter bases 5+(k-1)*80 and 135+(k-1)*80, all with step=80*n_adds).
  - `_EXSEC_F13` → `_generate_exsec_f13(n)`: inputs 0,1 get `320*n`; inputs ≥2
    cycle `[384, 768, 1152, None]` with period 4; output-rs = `384 if n%4==2 else None`.
    (This is a hardware bank pattern — 3 active secondary data banks + 1 idle slot.)
  - `_ROOT_F19` → inline formula `(192+(n-2)*64, 1472+(n-2)*320)`. The old
    hardcoded values for n=5,6,7 were wrong (huge garbage numbers); the runtime
    does not use field 19, so both correct and wrong values pass NPU verification.
  - The guard check now only requires `_RC_TEMPLATES` (generated regcmd blob).
* The old builder-calibrated root-delta tables and template-body alignment lookups
  have been replaced by direct computation from the generated RC/task sections plus
  parity-based alignment. Old `_rc_add{5,6,7}.bin` and `_body_add*_10x10.body`
  artifacts may exist from the reverse-engineering sprint, but they are not used by
  the production `build_body()` path.
* **n=10..16 support — the `anomaly` formula fix.** Extending pure generation past
  n=9 initially failed at load with "Verify ModelBuffer failed! / Invalid RKNN format"
  (in `init_runtime`, AFTER `load_rknn` succeeds). The cause was NOT the FlatBuffer:
  the **template-body path** (real toolkit FB skeleton + our generated RC) *also*
  failed for n=10, while the **unmodified** template body passed — isolating the bug to
  the regcmd PREFIX. Byte-diffing our `build_template(n)` against the n=4,5,10..16
  reference bodies showed the only meaningful difference was the `chain1`/`canon_base`
  chain addresses, all off by a multiple of 64. The PREFIX address step shrinks by 64
  for **every 4 inputs**, not capped at 64:
  ```python
  anomaly = 64 if n >= 6 else 0     # OLD: correct only for n=6..9
  anomaly = 64 * ((n - 2) // 4)     # NEW: 0 (n<6), 64 (n6..9), 128 (n10..13), 192 (n14..16)
  ```
  This is unchanged for n≤9 (`(n-2)//4 == 1` there), so the working range is preserved.
  After the one-line fix, the **pure** FlatBuffers path passes for the full range with
  no template files. (The complex 2000+-word PREFIX seen in a *freshly* toolkit-built
  n=10 model was a red herring — the simple ~350-word PREFIX structure is equally valid;
  matching the chain addresses is what the runtime verifies.)
* **n=17..64 support — two more limits removed.** Pushing past 16 (the `anomaly` term,
  RC PREFIX, memory plan, and SG/exsec/root specs are all continuous in n) surfaced two
  unrelated ceilings:
  - **26-letter alphabet in `_io()`.** Inputs/output were single chars `a..z`, so n=26
    overran `L[n]` (`IndexError`). `_io()` now draws from `_io_names()`: `a..z` first
    (keeps byte-exact parity for n≤25), then spreadsheet-style `aa, ab, ...`. None
    collide with the `t<k>-rs` intermediates (those always start with `t`+digit).
  - **`_taskdesc` bad-length pattern.** The cosmetic, never-patched task descriptor is
    `leading_zeros = n//3+1` zero words + 3 reshape records + a bracket zero. The
    runtime's ModelBuffer verifier rejects taskdescs of exactly **288 + 64·k bytes**
    (`leading_zeros` = 11, 19, 27, …); every neighbouring length loads. This hit
    n∈{30,31,32} (lead 11), {54,55,56} (lead 19), etc. Fix: nudge `leading_zeros += 1`
    whenever `leading_zeros ≥ 11 and (leading_zeros-11) % 8 == 0`. n≤25 is unaffected
    (lead < 11 there).
  With both fixes, n=2..64 build, load, and compute correctly on the NPU
  (`mismatches=0`). The cap is now an arbitrary `MAX_INPUTS=64`, not a known structural
  wall — deeper chains likely work but were not swept. Note deep **Mul/Div** chains can
  overflow fp16 (e.g. 40-deep Mul → `inf`); that is a numeric property of the op chain,
  not a model-validity issue (the model still loads and runs).
* `build_body()` now uses the pure FlatBuffers path (`_build_body_scratch_flatbuffers`)
  as its default — no template FB skeleton is copied at all. The old template-splice
  path is preserved as `build_body_scratch()` for reference. Verified: **n=2..64**
  load+init on the NPU and produce correct inference (`max_err≈0, 0 mismatches`) for
  Add (and Sub/Mul/Div where the chain doesn't overflow fp16), larger tensors
  (64x64, 32x32), and multi-op chains whose internal n maps to ≥10.

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

**Bug 3 — Task descriptor size is verifier-sensitive.** `_taskdesc()` is not a simple
`rec_in * n_inputs` formula: NPU load sweeps show `n_inputs=4..7` are accepted only with
3 input reshape records, while `n_inputs=8` accepts the full 8-record descriptor. The
current code encodes that runtime-verified rule.

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

## 6e. CPU-fallback ops: chained `And` (reshape-only NPU + CPU And nodes)

ONNX `And` exports with `Target=CPU`; the runtime executes it on a CPU kernel
(`CPU_OP_ENUMS["And"]=85`, node field `f7`). The NPU side runs only reshape/copy
blocks; there are **no NPU compute blocks**. An n-input chain
`out = (((a0 AND a1) AND a2) ... AND a_{n-1})` (shape `[1,4]`, bool/int8) has:

- **3n+1 nodes**: n `InputOperator`, 2 `Reshape` (first two inputs), then n-2
  interleaved `And`+`Reshape` pairs, 1 final `And`, 1 output `Reshape`, 1
  `OutputOperator`. Node ordering verified against toolkit references n=2..5.
- **5n+5 tensors**, `sg.f10 = [0,0,0,n+1,2n+2]`, n+1 reshape/copy command blocks.
- **f7 routing** maps every block's reshape *result* to word 5 and its *source*
  to word 55 of that 80-word block: input i -> `(i*80+55)`, rs i -> `(i*80+5)`;
  output -> `(n*80+5)`, out-rs -> `(n*80+55)`.

### Modular op dispatch (NOT a special-case path)

Op type is the single source of truth in `rc_template_gen.py`:
`CPU_OP_ENUMS`/`is_cpu_op`/`cpu_op_id` (CPU) vs `_OP_NAMES`/`ew_op_id` (NPU).
Both the FlatBuffer node builder and the regcmd generator dispatch on it, so any
op flows through the same modular path — no And/Add special method:

- **FlatBuffer node** (`rknn_flatbuf._op_node`): NPU ops emit DPU geometry +
  field `f3`; CPU ops emit field `f7 = cpu_op_id` + an empty attrs table `f8`,
  with geometry zeroed. `_add_node` is a thin alias.
- **regcmd** (`rc_template_gen.build_template`): an all-NPU chain emits compute
  `_canon` blocks; an all-CPU chain dispatches to `build_cpu_template`. Mixed
  NPU/CPU chains raise a clear `NotImplementedError` (FB nodes already handle
  mixing; the mixed RC schedule is future work).

### CPU regcmd is generated from scratch (byte-exact, no .rknn dependency)

`build_cpu_template(n)` reproduces the chained-op RC **byte-for-byte** for
n=2..64 (`verify_cpu_rc()` proves it when the references are present) as a
uint32 stream:

```
CPU descriptor prefix (_cpu_prefix_u32: compact lead schedule + closed-form records)
(n+1) x [ reshape_copy_canon[138 u32] + PC(chain) + PC14 + GAP71 ]   # last: +GAP69
CPU trailing (_cpu_trailing)
```

Key decodings:
- The reshape/copy **canon is the same 69-register DPU+RDMA block** as the NPU
  `_canon`, configured as a copy (DPU `0x4070=0x383`, RDMA `0x5044=0x7801`); for
  `[1,4]` bool it is fully shape-invariant (`reshape_copy_canon_words`).
- The PC chain-address + PC14 + GAP framing is **identical** to the NPU path.
- Working on a **uint32** basis handles even-n half-word alignment naturally
  (even-n streams carry one extra trailing 32-bit word).
- The descriptor prefix is generated structurally: `_cpu_fixed_header()` builds
  the constant header (length-prefixed `a_rs_i1` name + offset tables), and the
  address/copy-descriptor/DMA records are closed-form in n.
- The **alignment lead** is pure padding that re-aligns the reshape/copy canon to
  a 64-byte boundary. `_cpu_lead_u32(n, rc_word_off)` computes its length (and the
  descending offset suffix it carries) purely from the alignment math, where
  `rc_word_off` is the RC section's u32 file offset (equivalently FB-body length
  // 4). There is **no per-n lead/residue table** — `build_cpu_template(n,
  rc_word_off=...)` derives the lead from the actual layout offset, and the
  trailing descriptor absorbs the resulting length change so the total RC size
  matches the toolkit schedule.
- The CPU trailing descriptor is a prefix of the repeated `[1,1,1,4]` record
  stream, cut to the toolkit's total RC length schedule.

The **task descriptor** is likewise generated: `_taskdesc_cpu(n)` emits the same
8-word reshape-descriptor family as `_taskdesc` but with the `[1,4]` bool dims
(output record dims `[1,4]`, input records `[1,1,1,4]`). For n>=3 it is a
cyclic window whose offset is derived from the CPU regcmd alignment-lead length;
byte-exact vs the references n=2..64.

The end-to-end on-device models (n=2..64, `verify_and` PASS) are
built by `_build_cpu_body` in `rknn_flatbuf.py`. The **entire model — FlatBuffer
body, RC, and taskdesc — is fully generated from scratch** (no toolkit, no
fixed-template reference body). The RC and taskdesc are generated by
`rc_template_gen.build_cpu_template` + `rknn_flatbuf._taskdesc_cpu`; the
FlatBuffer body is emitted by the Python `flatbuffers.Builder` via
`_build_cpu_body` (dispatched from `_build_body_scratch_flatbuffers` when all ops
are CPU ops). NPU-verified for every n=2..64 (0 mismatches).

Three bugs were found and fixed during regression testing:

1. **ops string vs list** (`_build_body_scratch_flatbuffers`): the dispatcher
   passed the raw `ops` parameter (often a string like `'And'`) to
   `_build_cpu_body` instead of the normalized `op_names` list. String iteration
   caused `ops[k]` to return single characters (`'A'`, `'n'`, `'d'`) instead of
   full op names, corrupting node names and RC geometry. Fix: pass `op_names`.
2. **`verify_and` hardcoded 2 inputs**: the verifier only set 2 of `n` inputs;
   for n≥3 the unset inputs defaulted to zero, making AND always return 0. Fix:
   dynamically set all `io.n_input` inputs (extras initialized to all-1s).
3. **Non-ASCII tensor names for n≥32** (`_cpu_io`): used `chr(97+i)` which
   produces non-ASCII characters (`chr(128)` = `\x80`) for n≥32, crashing the
   runtime's JSON parser with "json value must be object". Fix: use `_io_names`
   which produces safe a..z, aa..ab.. style names.

**f20/f21 are absolute FILE offsets**, not FlatBuffer-relative targets. Patch
with `value = HEADER_SIZE + body_offset_of_section` (production
`_patch_root_f20_f21`).

### RESOLVED: from-scratch FlatBuffer body (CPU-only path)

The Python `flatbuffers.Builder` FlatBuffer body is **accepted by the runtime**
for all n=2..64. The original "Verify ModelBuffer failed!" claim was a red
herring — the body loads fine. The real blocker was an **RC alignment-lead
parity mismatch**: the Python builder produces a FlatBuffer body ~60 bytes larger
than the toolkit's (different vtable layout / string packing), which changes
`rc_word_off % 16` (the RC section's u32 file offset modulo 16). The CPU RC
alignment lead (`_cpu_lead_u32`) has its length determined by this residue; the
different residue flipped the lead length parity (e.g. 14→15 words), which in
turn flipped the prefix parity, making `total_u32 - canon_total - prefix_words`
odd. Since the CPU RC trailing is generated as u64→u32 pairs (always even), it
cannot absorb a 1-word difference, producing an RC section 1 word too long and
causing "failed to submit!" at `rknn_run`.

**Fix** (`rknn_flatbuf._build_cpu_body`): after building the FlatBuffer body,
check whether `total_u32 - canon_total - prefix_words` is odd; if so, pad the
FB body by 4 bytes (1 u32 word). This increments `rc_word_off` by 1, which
flips the alignment lead parity (adding 4 bytes always changes the lead by an
odd amount: ±1 or ±15 depending on the current pad value), restoring the even
trailing and making the RC byte-identical to what the toolkit emits.

## 6f. Mixed CPU(And) + NPU(Add) in one model ("AND and ADD at the same time")

A single `.rknn` can hold both a CPU-fallback op (And) and an NPU element-wise op
(Add). The toolkit accepts every shape we tried — chained (And→Cast→Add,
Add→Greater→And) and **parallel** (independent `out1 = a AND b` and
`out2 = x + y`). The parallel form is the clean canonical case and is NPU-VERIFIED
(`verify_parallel.cpp`, 0 mismatches): inputs a,b (bool/INT8) and x,y (fp16),
outputs out1 (bool) and out2 (fp16).

### What the vendor produces (decoded, the command barely differs)

The op-graph is just the **union of the per-branch node chains**, and the
register-command (RC) block region is the **concatenation** of every branch's
blocks — all the CPU copy/reshape blocks first, then all the NPU compute blocks:

```
And(2-in)+Add(2-in)  : 3 copy + 6  compute = 9   blocks  sg.f10=[0,0,0,4,9]
And(3-in)+Add(2-in)  : 4 copy + 6  compute = 10  blocks
And(2-in)+Add(3-in)  : 3 copy + 12 compute = 15  blocks  sg.f10=[0,0,0,5,12]
```

* **copy block** = the exact `_canon(EW_OP_COPY)` block (DPU_MODE=0,
  EW_CFG=0x383, RDMA_MODE=1) the chained-And path already emits — one per CPU
  reshape (n_in inputs + 1 output = n_in+1 blocks per And branch).
* **compute block** = the exact per-tile Add block (`DPU_MODE=0x48000002`,
  EW_CFG=0x108202c0) that `build_template` emits — 6 tiles × n_adds, with the
  per-tile-last block a 69-block and the rest 71-blocks.
* In a mixed stream every copy block is a 71-block (its PC preamble chains to the
  next); only the final block of the whole stream is a 69-block.
* `sg.f10 = [0,0,0, n_cpu_blocks + n_npu_ops, total_blocks]`. The taskdesc is the
  same 8-word reshape-record family (`_taskdesc`/`_taskdesc_cpu`), unchanged.

So the per-op DPU/RDMA register config is **identical** to the standalone And and
Add models — only the schedule (block ordering + descriptors) is combined. The
decoded block schedule is reproduced byte-exact by
`rc_template_gen.mixed_parallel_block_schedule(cpu_branches, npu_branches)`
(matches all references for And2+Add2, And3+Add2, And2+Add3).

### What still needs the compiler: the combined RC PREFIX

The mixed RC **prefix** (the descriptor + DMA schedule before the blocks) is
runtime-validated — probed on-device against the vendor parallel body:

```
zero copy-descriptor chain addresses -> rknn_init fails (-6, load rejected)
zero the descending size table       -> SEGFAULT in the runtime
zero DMA base fields                 -> loads, but output all-zero (bases are real)
zero DMA 'count' fields              -> STILL PASSES (count is cosmetic)
```

The prefix is regular (it generalises the CPU `_cpu_prefix`: a size table with
one entry per block, then `n_blocks+2` copy descriptors with chain addresses, then
3×n_blocks DMA records), but its chain addresses + size table must be byte-exact.
Reproducing it from scratch is the same `RKNNSubGraphMemoryPlanPass` +
relocation-table port that gates large-N Add (section 6) — not yet done.

### Current generator: hybrid + from-scratch (works today, NPU-verified)

`rknn_mixed_gen.py` builds the parallel And+Add model via two paths:

* **Hybrid** (default): reuses a toolkit-built reference FlatBuffer body + RC for
  the mixed graph (`_ref_parallel_and_add.rknn`) and generates the container
  header, JSON trailer, and memory plan from scratch. NPU-verified
  (`verify_parallel`, 0 mismatches), toolkit-free at generation time.
* **From-scratch** (`--scratch`): generates the **entire FlatBuffer body from
  scratch** via `_build_mixed_and_add_body()` — all 27 tensors, 14 nodes,
  subgraph, and root table emitted by the Python `flatbuffers.Builder`. The
  reference RC prefix + taskdesc are spliced in (the FB-body-to-RC boundary is
  patched via `_patch_root_command_offsets`). NPU-verified (`verify_parallel`,
  0 mismatches).

The from-scratch FB body (11296 bytes) uses this tensor/node layout:

* **27 tensors**: [0]=empty, [1-6]=rsi1 (CPU f0=7,f2=5), [7-8]=CPU ext inputs
  (f0=3), [9-10]=NPU ext inputs (f0=10), [11-16]=CPU workspace/exsec/rs (f0=3),
  [17]=CPU output (f0=3,f2=2), [18-23]=NPU exsec/rs (f0=10), [24]=NPU output
  (f0=10,f2=2), [25]=regcmd (f18=7), [26]=task (f18=8).
* **14 nodes**: [0-3]=inputs, [4-5]=CPU reshape (f3=2), [6]=And (f7=85),
  [7]=CPU reshape, [8]=CPU output, [9-10]=NPU reshape (f3=None), [11]=Add
  (f3=2, custom f10/f12 geometry), [12]=NPU reshape, [13]=NPU output.
* **Subgraph f7**: 6 CPU single-pair entries + 3 NPU 6-pair entries (shifted
  by 240=3×80 from standalone n=2 base). `_MIXED_NPU_IO_OFF=[0,0,32,0,32,48]`,
  `_MIXED_CPU_SHIFT=240`.
* **Root**: f19=(384,384), sg.f10=[0,0,0,4,9], sg.f12=[(out1,3),(out2,4)].
* `_op_node` and `_cpu_ext_output` are parameterized with `npu_f10`/`npu_f12`
  and `f13` kwargs for the mixed model's non-standard geometry (backward-
  compatible: existing callers are unaffected).

The remaining from-scratch work is the combined RC prefix (the mixed-graph
descriptor + DMA schedule is a superset of both the CPU `_cpu_prefix` and the
NPU `_build_prefix`; its chain addresses + size table must be byte-exact).

---

## 6g. Multi-dtype support on the NPU element-wise path (`build_body(dtype=...)`)

The pure-FlatBuffers element-wise generator now emits multiple tensor data types
in addition to fp16, fully from `rknn_flatbuf.py` + `rc_template_gen.py` (no
toolkit). `build_body(N, n_inputs, ops=..., dtype=...)` and `rknn_add_gen.py
--dtype {float16,float32,int8,bool}` build models the **vendor `librknnrt.so`
loads, runs, and reports with the right `rknn_tensor_type`** (`query_attr`).

### Per-dtype status (NPU-verified against librknnrt.so, Add mismatches=0)

| dtype | body f0 | I/O bytes | NPU status |
|-------|---------|-----------|------------|
| `float16` | 10 | 2 | native DPU fp16 — **WORKS** |
| `float32` | 10 | 4 | runtime converts fp32↔fp16 — **WORKS** (fp16 precision) |
| `int8` | 3 | 1 | DPU int8 element-wise — **WORKS** (exact integer) |
| `bool` | 3 | 1 | int8 enum; `Add≈OR`, `Mul≠AND` — **WORKS** (int8 arithmetic) |
| `uint8` | — | 1 | input normalizes but **output uint8 dtype unsupported** by EW path |
| `int16`/`int32`/`int64` | 4/6/8 | 2/4/8 | recognized input, but native pack is **C2=4** not 8 → the fp16 RC template mis-computes; needs a per-width RC template + memory plan |
| `uint16`/`uint32`/`int4`/`bfloat16` | — | — | **not recognized** by this librknnrt's dtype string parser (report `UNKNOW(12)`) |

The 4 working dtypes are in `DTYPES`; the rest are listed in
`_DTYPE_KNOWN_UNSUPPORTED` so `build_body` raises a specific
`NotImplementedError` (and the CLI `--dtype` choice list shows only the working
set). Key decoded facts (below) were established by sweeping `f0` enum and
dtype-string candidates against the real runtime, not guessed.

The `int32` reference (`rknn-reg/int/int32add.rknn`) confirms the C2=4 native
pack: its working tensors are `[1,1,1,W,4]` (4 fp16-byte lanes), versus fp16's
`[1,1,1,W,8]`. Supporting int16/32/64 on the NPU therefore requires emitting a
width-specific RC block (different `DATA_CUBE_CHANNEL`/stride) and memory plan —
the same compiler-tiler territory as large-N Add — and is left as future work.

### Where the dtype lives (decoded)

* **Reported dtype = root fields `f12`/`f13`** (the `dtype_in`/`dtype_out` JSON
  strings). Proven by sweep: patching only the per-tensor `f0` enum, or only the
  root `f3` attrs JSON, leaves the runtime reporting FP16; only changing
  `f12`/`f13` makes `rknn_query` return the new type.
* **Per-tensor `f0` = internal FlatBuffer `TensorType` enum** (decoded via
  `extract_rknn_build_queue.tensor_info`): fp16=`10`, int8/bool=`3`, int32=`6`.
  This is *not* the public `rknn_tensor_type` (where `BOOL=9`); the body uses its
  own enum. `DTYPES` maps name -> `{fb_type, elem_bytes, str}` and
  `_patch_tensor_dtype` rewrites every data tensor's `f0` after the FB is built
  (a no-op for fp16/fp32, which share the fp16 internal class).

### Matching f0 to the dtype is mandatory (decoded from the runtime errors)

The body `f0` enum AND the reported dtype string must agree. If they disagree the
runtime tries to convert on input and fails: e.g. an int8-declared input over an
fp16-class body raises
`Normalize does not support for this data type. src type(INT8), dst type fbs::TensorType_FLOAT16`.
The internal enum names come straight from the library
(`strings librknnrt.so | grep TensorType_`): FLOAT, FLOAT16, INT8, UINT8, INT16,
INT64, BOOL, BFLOAT16, TF32. `float32` keeps the fp16 internal class (f0=10)
because the runtime losslessly normalises fp32↔fp16 at the I/O boundary.

### int8 / bool integer semantics (NPU-verified truth tables)

For 1-byte dtypes the DPU computes **int8-style integer** element-wise (output is
1 byte/elem; `want_float` converts the raw byte, e.g. Add(1,1)=2.0):

```
op    (0,0) (0,1) (1,0) (1,1)   matches
Add     0     1     1     2     integer a+b   (= OR when clamped nonzero)
Sub     0     0     1     0     integer a-b   (saturates at 0)
Div     1     0     0     1     integer a/b   (1/1=1, x/0 -> 1, 0/x -> 0)
Mul     0     0     0     0     BROKEN (1*1 should be 1)  -- Mul cfg is fp16-only
```

So **Add/Sub/Div are valid 8-bit integer element-wise ops** (`Add ≈ logical OR`
for bool), but **Mul does not yield logical AND** — its `EW_CFG=0x108003c4`
expects fp16 operand encoding, so integer-byte multiply mis-computes. For genuine
logical ops on bool data, use the CPU-fallback `And` chain (§6e), which the
runtime reports as `BOOL` and executes correctly on a CPU kernel.

### Status

* `build_body` → `_build_root` thread `dtype`; `_resolve_dtype` validates it,
  `_root_attrs` emits consistent attrs/quant dtype, `_patch_tensor_dtype` sets the
  body `f0` (no-op for fp16/fp32). fp16 remains the default and byte-unchanged
  (regression: 10x10 / 20x20 / 32x32 / 1x2048 Add still 0 mismatches).
* Verified end-to-end on the NPU: **float16, float32, int8, bool** (n=2 and n=3
  chains, 1x4 / 10x10). Unsupported dtypes raise a specific `NotImplementedError`.
* Tools added: `query_attr.cpp` (dump runtime-reported tensor dtype/size),
  `verify_bool.cpp` (bool truth tables), `verify_int.cpp` (integer Add check).

---

## 7. Toolbox

| file | role |
|------|------|
| `rknn_add_gen.py` | toolkit-free CLI generator (`build_body` + trailer -> `.rknn`), `--verify` uses `rknn_verify_n`, `--dtype {float16,float32,int8,bool}` (§6g) |
| `rknn_flatbuf.py` | body builder: `build_body(...,dtype=)` = pure FlatBuffers (2-64 inputs; fp16/fp32/int8/bool, no template FB skeleton); `DTYPES` + `_DTYPE_KNOWN_UNSUPPORTED` + `_resolve_dtype`/`_patch_tensor_dtype`; algorithmic generators for tensor metadata, RC templates, alignment, and task descriptors; `build_body_scratch` = legacy template-splice reference path |
| `query_attr.cpp` | dump the vendor runtime's reported per-tensor dtype/size/fmt (§6g) |
| `verify_bool.cpp` | feed/read 1-byte bool buffers; classify NPU bool op semantics (§6g) |
| `verify_int.cpp` | integer-dtype element-wise Add correctness harness (§6g) |
| `_body_add*_10x10.body` | legacy raw reference bodies from reverse-engineering; not used by production `build_body()` |
| `_rc_add*.bin` | legacy extracted regcmd blobs from reverse-engineering; not used by production `build_body()` |
| `decode_embedded_bodies.py` | decode a `.body` into FB-prefix / regcmd-blocks / taskdesc-tail |
| `gen_ref_addn.py N` | mint a reference body for N inputs via on-device toolkit (names match `_io`) |
| `rc_template_gen.py` | READABLE regcmd template generator (replaces gzip `_RC_TEMPLATES`) |
| `decode_rc_templates.py` | decode/annotate/round-trip the regcmd templates (§4b) |
| `verify_rc_gen.py` | proves `build_template` == original after patching, all shapes |
| `oracle_required.py` / `scratch_oracle.py` / `fb_verify.py` / `type_oracle.py` | verifier-as-oracle tooling for the from-scratch FB RE (§6c) |
| `flatbuffer_to_rknn.py` | split/assemble container |
| `extract_rknn_build_queue.py` | FlatBuffer parser (FB class) |
| `decode_cmdbuf.py` | classify/decode regcmd blocks |
| `rknn_run_generic` / `.cpp`, `rknn_verify*` / `.cpp` | vendor-API NPU correctness harnesses (`rknn_verify_n` = N-input, op-aware) |
| `rknn_flatbuf._build_cpu_body(n, ops=)` | pure-FlatBuffers CPU-fallback chained-`And` builder (§6e); **fully from scratch** — no toolkit, no fixed template; n=2..64 PASS on-device |
| `rknn_flatbuf._build_mixed_and_add_body()` | pure-FlatBuffers mixed CPU(And)+NPU(Add) builder (§6f); 27 tensors + 14 nodes from scratch |
| `rknn_flatbuf.build_mixed_and_add()` | splice from-scratch FB body with reference RC+taskdesc (§6f) |
| `build_and_chain_ref_n.py N` | mint a chained-`And` reference model via on-device toolkit (n inputs) |
| `verify_and` / `.cpp` | n-input chained-`And` correctness harness (sets all n inputs, AND-of-all check); n=2..64 PASS |
| `test_and_chain_n.cpp` | n-input chained-`And` correctness harness (AND-of-all-inputs check) |
| `rknn_mixed_gen.py` | mixed CPU(And)+NPU(Add) parallel generator (§6f); hybrid ref-body + from-scratch container/trailer; `--scratch` for fully from-scratch FB body; PASS on-device |
| `_ref_parallel_and_add.rknn` | toolkit-built reference for the parallel And+Add graph (hybrid FB body source for `rknn_mixed_gen.py`) |
| `rc_template_gen.mixed_parallel_block_schedule()` | decoded mixed RC block schedule (copy+compute), byte-exact vs vendor (§6f) |
| `/data/rk3588/rknn-header/rkt_registers.h` | rk3588 NPU register definitions |
