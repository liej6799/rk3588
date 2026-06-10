# librknnrt.so Runtime Internals — Reverse Engineering Spec

Reverse-engineered from `librknnrt.so` v2.3.2 (2025-04-09) on RK3588 aarch64.
Methods: static disassembly (radare2/objdump), dynamic tracing (gdb + LD_PRELOAD shim).
Verified against live NPU inference of Add, Mul, And (CPU), and hybrid (CPU+NPU) models.

---

## 1. Library API Surface (exported symbols)

```
rknn_init           @ 0x0c5160   # load model, alloc buffers, build task array
rknn_query          @ 0x0cf458   # query IO attrs, perf info
rknn_inputs_set     @ 0x0d6400   # reshape inputs contiguous->NC1HWC2, sync TO_DEV
rknn_run            @ 0x0cc040   # patch regcmd, submit tasks, execute CPU ops
rknn_outputs_get    @ 0x0d8e10   # sync FROM_DEV, convert NC1HWC2->contiguous
rknn_outputs_release@ 0x0bea48
rknn_destroy        @ 0x0c1db0
rknn_set_core_mask  @ 0x0bf3f0
rknn_create_mem     @ 0x0c0888
rknn_destroy_mem    @ 0x0bef88
rknn_mem_sync       @ 0x0c0898
rknn_set_io_mem     @ 0x0cb1f8
```

---

## 2. DRM IOCTL Interface (rknpu driver v0.9.x)

The runtime uses the **DRM char-dev** path (`/dev/dri/card1`). All ioctls use type `'d'` (0x64)
with `DRM_COMMAND_BASE` (0x40):

| Ioctl | Number | Struct size | Purpose |
|-------|--------|-------------|---------|
| `ACTION` | `0xc0086440` | 8 | HW version, reset, power on/off |
| `SUBMIT` | `0xc0686441` | 104 | Submit NPU work |
| `MEM_CREATE` | `0xc0306442` | 48 | Allocate DMA buffer (GEM object) |
| `MEM_MAP` | `0xc0106443` | 16 | Get mmap fake offset |
| `MEM_DESTROY` | `0xc0106444` | 16 | Free DMA buffer |
| `MEM_SYNC` | `0xc0206445` | 32 | Cache sync (TO_DEV=1, FROM_DEV=2, BIDIR=3) |

Additionally used: `DRM_IOCTL_VERSION (0xc0406400)`, `GEM_FLINK (0xc008640a)`,
`PRIME_HANDLE_TO_FD (0xc00c642d)`.

### 2a. Submit Wrapper (dual interface)

The submit wrapper at `0x2e9948` checks a flag byte at `ctx[+8]`:
- **flag != 0**: `ioctl(fd, 0xc0686441, submit)` — DRM path (standard)
- **flag == 0**: `ioctl(fd, 0xc0687201, submit)` — char-dev path (`'r'` magic = `/dev/rknpu`)

The DRM path is the default on mainline kernels.

---

## 3. Buffer Plan (rknn_init)

`rknn_init` allocates **6 DMA buffers** (via MEM_CREATE → GEM_FLINK → PRIME_HANDLE_TO_FD
→ MEM_MAP → mmap64) for a 2-input EW model:

| Handle | Purpose | Size (add_10x10) | Contents |
|--------|---------|-------------------|----------|
| 1 | **Model buffer** | 12861 (= .rknn file) | FlatBuffer body + regcmd blocks + task descriptors |
| 2 | **Task array** | 240 (6 × 40B records) | `rknpu_task[]` for DRM_IOCTL_RKNPU_SUBMIT |
| 3 | **Feature/working buffer** | 1472 | NC1HWC2 working tensors (`x_rs`, `y_rs`, `z-rs`) |
| 4 | **Input 0** | 320 (100 × 2B fp16) | External tensor `x` |
| 5 | **Input 1** | 320 | External tensor `y` |
| 6 | **Output** | 256 | External tensor `z` |

Handles 4-6 scale with N (element count); handle 3 scales with the NC1HWC2 memory plan;
handle 1 is the full `.rknn` file. Buffer sizes are page-rounded to 4096 by the kernel.

For CPU-op models (And), the layout is similar but handle 1 is the task array (task buffer
is allocated first) and handle counts may differ.

---

## 4. rknpu_task[] Construction

The runtime expands the FlatBuffer's embedded "task" tensor into an array of `rknpu_task`
records (40 bytes each). The `op_idx` field is the **FlatBuffer node index**, NOT an opcode.

### EW Add model (add_10x10): 6 records, 3 active (task_number=3)

| task | op_idx | node | enable_mask | int_mask | regcfg_amount | regcmd_addr |
|------|--------|------|-------------|----------|---------------|-------------|
| 0 | 4 | Add:add1 | 0x18 | 0x300 | 69 | model_dma + 7488 + 0×640 |
| 1 | 4 | Add:add1 | 0x18 | 0x300 | 69 | model_dma + 7488 + 1×640 |
| 2 | 4 | Add:add1 | 0x18 | 0x300 | 69 | model_dma + 7488 + 2×640 |
| 3-5 | 4 | (ping/pong) | 0x18 | 0x300 | 69 | model_dma + 7488 + 3-5×640 |

Constants across all EW ops (Add/Sub/Mul/Div):
- `flags = 0`
- `enable_mask = 0x18` (bits 3+4 = DPU + CORE)
- `int_mask = 0x300` (bits 8+9)
- `int_clear = 0x1ffff` (= RKNPU_INT_CLEAR)
- `regcfg_amount = 69` (69 register words per block)
- `regcfg_offset = 0`

The op type (Add vs Mul vs Sub vs Div) is encoded in the **regcmd register values**
(DPU_EW_CFG at reg 0x4070), NOT in the task metadata.

`regcmd_addr` = DMA address of the model buffer + body regcmd region offset + block_index × 640
(640 = 80 words × 8 bytes per slot).

### CPU-op model (and_chain2): 9 records, 6 populated

| task | op_idx | node | meaning |
|------|--------|------|---------|
| 0 | 2 | Reshape:a_rs | NPU copy/reshape input a |
| 1 | 3 | Reshape:b_rs | NPU copy/reshape input b |
| 2 | 5 | Reshape:out-rs | NPU copy/reshape output |
| 3-8 | 2,3,5 | (ping/pong repeats) | |

**The And node (op_idx=4) is NOT in the task array.** CPU ops are executed by the
runtime's CPU kernel between NPU submits.

---

## 5. rknpu_submit Structure

```c
struct rknpu_submit {       // 104 bytes
    uint32_t flags;         // 0x5 = PC | PINGPONG (standard for EW/CPU-op models)
    uint32_t timeout;       // 6000 ms (default), or max(30 * task_number, ctx[+2428])
    uint32_t task_start;    // first task index to run
    uint32_t task_number;   // number of tasks to submit
    uint32_t task_counter;  // 0
    int32_t  priority;      // 0
    uint64_t task_obj_addr; // kernel obj_addr of task DMA buffer
    uint64_t regcfg_obj_addr; // 0 (unused in v2.3.2)
    uint64_t task_base_addr;  // 0 (unused)
    uint64_t user_data;       // 0
    uint32_t core_mask;     // 0 = auto (single core)
    int32_t  fence_fd;      // -1
    struct { uint32_t task_start, task_number; } subcore_task[5];
};
```

### Task chunking (0x2f1c08)

The runtime reads a task cap from `ctx[+584]`: `max_tasks_per_submit = (1 << ctx[+584])`.
If `task_number > max_tasks_per_submit`, it loops, submitting chunks of
`min(max_tasks, remaining)` with incrementing `task_start`.

### Submit call chain

```
rknn_run (0x0cc040)
  → internal dispatcher (0x0d79f0 / 0x0d7e04)
    → task_chunker (0x2f1c08)    builds flags from ctx[+12..16], computes chunks
      → submit_job (0x3075f0)    assembles rknpu_submit on stack, sets timeout/core_mask
        → submit_wrapper (0x2e9948)  calls ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit)
```

---

## 6. CPU/NPU Operation Split

**The split is determined at model load time** from the FlatBuffer node graph. Each node
has a target field (`f3` for NPU nodes with DPU geometry, `f7` for CPU ops with a CPU kernel ID).

### NPU-only model (all EW ops): 1 submit per rknn_run

```
inputs_set:  CPU reshape (contiguous → NC1HWC2) → sync TO_DEV
rknn_run:    SUBMIT(task_start=0, task_number=3)
outputs_get: sync FROM_DEV → CPU convert (NC1HWC2 → contiguous)
```

### CPU-only model (And): 2 submits per rknn_run

```
rknn_init:   SUBMIT(task_start=0, task_number=6)        ← reshape all inputs (init-time!)
rknn_run:    [nothing visible before first submit]
             ← CPU And kernel executes between submits
             SUBMIT(task_start=2, task_number=3)         ← reshape output
outputs_get: sync FROM_DEV → read output
```

### Hybrid model (Add + And, 3 inputs): 2 submits per rknn_run

```
rknn_run:    SUBMIT(task_start=0, task_number=3)        ← NPU Add + input reshapes
             ← CPU And kernel executes
             SUBMIT(task_start=1, task_number=3)        ← output reshape
outputs_get: sync FROM_DEV → convert
```

**Key insight**: CPU ops create **submit boundaries**. The runtime splits the graph at
CPU op nodes: NPU tasks before the CPU op run in one submit, CPU executes, then NPU tasks
after the CPU op run in a second submit. The `task_start` field advances to skip already-
executed tasks.

---

## 7. inputs_set: Contiguous → NC1HWC2 Reshape (CPU-side)

`rknn_inputs_set` performs the data format conversion **entirely on the CPU**. The sequence
(traced via MEM_SYNC for add_10x10):

1. Read external input `x` from input buffer (handle 4, 320 B)
2. Read model metadata region (handle 1, off=11520, 32 B) — task descriptor/attr data
3. Read `x_exSecondary` status byte (handle 3, off=640)
4. **Write NC1HWC2 tensor `x_rs`** to working buffer (handle 3, off=704, 384 B)
5. Repeat for input `y` → `y_rs` (handle 3, off=1088, 384 B)
6. Sync regcmd region TO_DEV (handle 1, off=7488, 3840 B)

The NC1HWC2 layout for 100 fp16 elements (10×10):
- Native shape: `[1, 2, 1, 10, 8]` = `[N, C1, H, W, C2]`
- `C2 = 8` (8 fp16 lanes = 16 bytes, the DPU atom)
- `C1 = ceil(R/8) = 2` (2 channel surfaces for 10 rows)
- Working buffer size = `C1 × pad4(C) × 16 = 2 × 12 × 16 = 384` bytes

**No NPU submit during inputs_set.** The reshape is purely CPU memcpy with lane packing.

---

## 8. outputs_get: NC1HWC2 → Contiguous Conversion (CPU-side)

After SUBMIT returns, `outputs_get` converts the NPU result back:

1. Sync `z-rs` FROM_DEV (handle 3, off=0, 384 B) — the NC1HWC2 output
2. Read model metadata (handle 1, off=11392, 16 B)
3. Read `z-rs_exSecondary` status byte (handle 3, off=384)
4. **Write contiguous output `z`** to output buffer (handle 6, 256 B)
5. Sync output buffer FROM_DEV for user read

The conversion is CPU-side NC1HWC2→contiguous, inverse of inputs_set.

---

## 9. DMA Address Patching

The runtime patches **only 3 DMA base addresses per regcmd block** at submit time:

| Register | Field | Tensor |
|----------|-------|--------|
| `0x5018` | `DPU_RDMA_RDMA_SRC_BASE` | `x_rs` (input 0) |
| `0x5038` | `DPU_RDMA_RDMA_EW_BASE` | `y_rs` (input 1) |
| `0x4020` | `DPU_DST_BASE_ADDR` | `z-rs` (output) |

Patched address = `feature_buffer_dma + tensor.f13_offset`.

The geometry registers (width, height, channel, stride, notch) are **baked in the file**
and never modified at runtime. The runtime reads the tensor table's `f13` (byte offset) and
`f12` (byte size) to compute the DMA addresses. Per-tile deltas for multi-surface models
are computed automatically from the tensor's `C1` dimension.

---

## 10. Feature Buffer Layout (Working Memory)

The feature buffer (handle 3) is a flat allocation containing all working tensors at their
planned offsets (from the FlatBuffer tensor table `f13`):

```
add_10x10 feature buffer (1472 bytes):
  off=0     z-rs       384 B   NC1HWC2 [1,2,1,10,8]
  off=384   z-rs_exSec   1 B   status byte
  off=448   z (output)  256 B   contiguous [10,10]
  off=640   x_exSec      1 B   status byte (shared with y_exSec)
  off=704   x_rs       384 B   NC1HWC2 [1,2,1,10,8]
  off=1088  y_rs       384 B   NC1HWC2 [1,2,1,10,8]
```

Some offsets are aliased (e.g., `x_exSecondary` and `y_exSecondary` share offset 640).
This is the compiler's memory plan output (`RKNNSubGraphMemoryPlanPass`).

---

## 11. Regcmd Block Layout

The regcmd region starts at `body_offset + FlatBuffer_size` within the model buffer.
For add_10x10: body starts at file offset 0x40, FlatBuffer is 7424 bytes, so regcmd
starts at `0x40 + 7424 = 7488` bytes from buffer start.

Each block slot = **80 words × 8 bytes = 640 bytes**:
- 52 DPU registers (`0x4004..0x412c`)
- 17 DPU_RDMA registers (`0x500c..0x506c`)
- = 69 active register words (= `regcfg_amount`)
- + 11 padding/gap words

6 slots = 3840 bytes total regcmd region.

---

## 12. Static Call Chain (librknnrt.so addresses)

```
rknn_init       @ 0x0c5160
  parseRKNN     → FlatBuffer verification @ 0x2ad490 (flatc verifier @ 0x2d06b0)
  → mem_create × 6 → mmap64 → mem_sync

rknn_inputs_set @ 0x0d6400
  → CPU reshape (contiguous → NC1HWC2)
  → mem_sync TO_DEV (input bufs + working buf + regcmd)

rknn_run        @ 0x0cc040
  → run dispatcher @ 0x0d79f0
    → task_chunker @ 0x2f1c08
      → submit_job @ 0x3075f0
        → submit_wrapper @ 0x2e9948
          → ioctl(fd, 0xc0686441, &rknpu_submit)
  [if hybrid: CPU kernel → second submit]

rknn_outputs_get @ 0x0d8e10
  → mem_sync FROM_DEV
  → CPU convert (NC1HWC2 → contiguous)

rknn_destroy    @ 0x0c1db0
  → mem_destroy × 6
```

---

## 13. What a From-Scratch Runtime Must Do

To replace `librknnrt.so`, implement:

1. **Model loader**: parse `.rknn` container (64B header + FlatBuffer body + JSON trailer).
   Extract the tensor table (shapes, sizes, offsets), node graph, regcmd blob, and task
   descriptor from the FlatBuffer.

2. **Buffer allocator**: MEM_CREATE 6 buffers (model, task, feature, inputs, output),
   MEM_MAP + mmap64 each, copy the .rknn into the model buffer.

3. **Task builder**: expand the embedded task descriptor into `rknpu_task[]` records.
   Set `op_idx` from node index, `enable_mask=0x18`, `int_mask=0x300`,
   `int_clear=0x1ffff`, `regcfg_amount=69`, `regcmd_addr = model_dma + regcmd_offset +
   block_idx × 640`.

4. **Input reshape**: CPU-side contiguous→NC1HWC2 conversion. Pack flat fp16 input into
   `[1, C1, 1, W, 8]` layout, write to feature buffer at tensor offset.

5. **DMA patching**: write `feature_dma + tensor.offset` into regcmd base-address fields
   (0x4020, 0x5018, 0x5038) for each regcmd block.

6. **Submit**: fill `rknpu_submit` struct, call `DRM_IOCTL_RKNPU_SUBMIT`.

7. **CPU op execution** (hybrid models): between NPU submits, run CPU kernels
   (And/Or/Xor → bitwise, Cast → type conversion) on the feature buffer data.

8. **Output convert**: CPU-side NC1HWC2→contiguous, read from feature buffer, write to
   output buffer.

### Already implemented (reusable components):

- `.rknn` generation: `rknn-creation/rknn_flatbuf.py`, `rc_template_gen.py`
- `.rknn` decode: `rknn-decode/helpers/rknn_decode.py`
- Direct ioctl submit: `rknn-ops/rknnops.h`, `/data/rkt/examples/simple_add.py`
- Register dictionary: `rknn-header/rkt_registers.h`
- NC1HWC2 layout math: `rknn-creation/RKNN_CREATION.md` §3

### Not yet implemented:

- Arbitrary `.rknn` loader → ioctl submit pipeline
- CPU op kernel library (And/Or/Xor/Cast/int32/fp32)
- NC1HWC2 reshape for arbitrary shapes
- Multi-submit graph executor for hybrid models

---

## 14. Tooling Artifacts

- `/tmp/opencode/npu_trace.so` — LD_PRELOAD shim that decodes all RKNPU ioctls and dumps
  `rknpu_task[]` contents. Usage: `LD_PRELOAD=npu_trace.so ./rknn_verify_n model.rknn`
- `/tmp/opencode/librknnrt.dis` — full objdump disassembly of librknnrt.so (1.4M lines)
