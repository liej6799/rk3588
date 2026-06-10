#!/usr/bin/env python3
"""
rknn_runtime.py — From-scratch replacement for librknnrt.so.

Loads a .rknn file, parses the FlatBuffer graph, allocates DMA buffers,
performs NC1HWC2 reshape, patches DMA addresses, submits NPU tasks via DRM
ioctl, and executes CPU-op nodes in Python (fixing the broken int32 path).

Replicates the librknnrt.so flow:
  init → inputs_set → run → outputs_get

Usage:
    from rknn_runtime import RKNNRuntime
    rt = RKNNRuntime("model.rknn")
    rt.inputs_set([a_array, b_array])
    rt.run()
    result = rt.outputs_get()
    rt.destroy()
"""
import ctypes, mmap, os, struct, sys
from fcntl import ioctl
import numpy as np

# ── DRM IOCTL ────────────────────────────────────────────────────────────────
RKNPU_MEM_KERNEL_MAPPING = 8
RKNPU_MEM_NON_CACHEABLE = 0
RKNPU_ACT_RESET = 1
RKNPU_JOB_PC = 1
RKNPU_JOB_PINGPONG = 4

class rknpu_mem_create(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("size", ctypes.c_uint64), ("obj_addr", ctypes.c_uint64),
                ("dma_addr", ctypes.c_uint64), ("sram_size", ctypes.c_uint64)]

class rknpu_mem_map(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
                ("offset", ctypes.c_uint64)]

class rknpu_mem_sync(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("offset", ctypes.c_uint64), ("size", ctypes.c_uint64)]

class rknpu_subcore_task(ctypes.Structure):
    _fields_ = [("task_start", ctypes.c_uint32), ("task_number", ctypes.c_uint32)]

class rknpu_submit(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("timeout", ctypes.c_uint32),
                ("task_start", ctypes.c_uint32), ("task_number", ctypes.c_uint32),
                ("task_counter", ctypes.c_uint32), ("priority", ctypes.c_int32),
                ("task_obj_addr", ctypes.c_uint64),
                ("iommu_domain_id", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
                ("task_base_addr", ctypes.c_uint64), ("hw_elapse_time", ctypes.c_int64),
                ("core_mask", ctypes.c_uint32), ("fence_fd", ctypes.c_int32),
                ("subcore_task", rknpu_subcore_task * 5)]

class rknpu_task(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("op_idx", ctypes.c_uint32),
                ("enable_mask", ctypes.c_uint32), ("int_mask", ctypes.c_uint32),
                ("int_clear", ctypes.c_uint32), ("int_status", ctypes.c_uint32),
                ("regcfg_amount", ctypes.c_uint32), ("regcfg_offset", ctypes.c_uint32),
                ("regcmd_addr", ctypes.c_uint64)]

class rknpu_action(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("value", ctypes.c_uint32)]

def _IOWR(ty, nr, sz): return (3 << 30) | (ord(ty) << 8) | nr | (sz << 16)
IOCTL_MEM_CREATE = _IOWR('d', 0x42, ctypes.sizeof(rknpu_mem_create))
IOCTL_MEM_MAP    = _IOWR('d', 0x43, ctypes.sizeof(rknpu_mem_map))
IOCTL_SUBMIT     = _IOWR('d', 0x41, ctypes.sizeof(rknpu_submit))
IOCTL_MEM_SYNC   = _IOWR('d', 0x44, ctypes.sizeof(rknpu_mem_sync))
IOCTL_ACTION     = _IOWR('d', 0x40, ctypes.sizeof(rknpu_action))

# ── FlatBuffer Reader (minimal) ──────────────────────────────────────────────
class FB:
    def __init__(self, data):
        self.b = data
    def u16(self, o): return struct.unpack_from("<H", self.b, o)[0]
    def u32(self, o): return struct.unpack_from("<I", self.b, o)[0]
    def i32(self, o): return struct.unpack_from("<i", self.b, o)[0]
    def root(self): return self.u32(0)

    def _vt(self, pos):
        return pos - self.i32(pos)

    def field_abs(self, pos, field):
        vt = self._vt(pos)
        entry = vt + 4 + field * 2
        if entry + 2 > vt + self.u16(vt):
            return None
        off = self.u16(entry)
        return pos + off if off else None

    def scalar_u32(self, pos, field, default=None):
        a = self.field_abs(pos, field)
        return self.u32(a) if a is not None else default

    def string(self, pos, field):
        a = self.field_abs(pos, field)
        if a is None: return None
        t = a + self.u32(a)
        n = self.u32(t)
        return self.b[t + 4:t + 4 + n].decode("ascii", "replace")

    def vec_u32(self, pos, field):
        a = self.field_abs(pos, field)
        if a is None: return None
        v = a + self.u32(a)
        n = self.u32(v)
        return [self.u32(v + 4 + i * 4) for i in range(n)]

    def vec_tables(self, pos, field):
        a = self.field_abs(pos, field)
        if a is None: return []
        v = a + self.u32(a)
        n = self.u32(v)
        return [v + 4 + i * 4 + self.u32(v + 4 + i * 4) for i in range(n)]

# ── Tensor / Node / Block types ─────────────────────────────────────────────
class Tensor:
    __slots__ = ("idx", "name", "native", "logical", "size", "offset", "n_elems", "_dma_addr", "_buf")
    def __init__(self, idx, name, native, logical, size, offset):
        self.idx = idx; self.name = name
        self.native = native or []; self.logical = logical or []
        self.size = size; self.offset = offset
        self.n_elems = 1
        for d in (logical or native or [1]):
            self.n_elems *= d
        self._dma_addr = None
        self._buf = None
    def __repr__(self):
        return f"Tensor({self.idx}, {self.name}, native={self.native}, logical={self.logical}, {self.size}@{self.offset})"

class Node:
    __slots__ = ("idx", "op", "name", "inputs", "outputs", "target", "cpu_kernel")
    def __init__(self, idx, op, name):
        self.idx = idx; self.op = op; self.name = name
        self.inputs = []; self.outputs = []
        self.target = None; self.cpu_kernel = None
    def __repr__(self):
        return f"Node({self.idx}, {self.op}, {self.name}, target={self.target})"

class CommandBlock:
    __slots__ = ("word_offset", "n_words", "words", "kind")
    def __init__(self, word_offset, n_words, words, kind):
        self.word_offset = word_offset; self.n_words = n_words
        self.words = words; self.kind = kind

# ── RKNN Container Parser ───────────────────────────────────────────────────
HEADER_SIZE = 0x40

def parse_rknn(data):
    data = bytes(data)
    if data[:4] != b"RKNN":
        raise ValueError("not an RKNN file")
    version = struct.unpack_from("<Q", data, 0x08)[0]
    body_size = struct.unpack_from("<Q", data, 0x10)[0]
    body = data[HEADER_SIZE:HEADER_SIZE + body_size]

    fb = FB(body)
    root = fb.root()
    subgraphs = fb.vec_tables(root, 2)
    if not subgraphs:
        raise ValueError("no subgraphs")
    sg = subgraphs[0]

    tensors = []
    for i, p in enumerate(fb.vec_tables(sg, 0)):
        tensors.append(Tensor(
            idx=i,
            name=fb.string(p, 5),
            native=fb.vec_u32(p, 3),
            logical=fb.vec_u32(p, 4),
            size=fb.scalar_u32(p, 12),
            offset=fb.scalar_u32(p, 13),
        ))

    nodes = []
    for i, p in enumerate(fb.vec_tables(sg, 1)):
        n = Node(i, fb.string(p, 1) or "", fb.string(p, 2) or "")
        n.target = fb.scalar_u32(p, 6, default=None)
        n.cpu_kernel = fb.scalar_u32(p, 7, default=None)
        n.inputs = fb.vec_u32(p, 4) or []
        n.outputs = fb.vec_u32(p, 5) or []
        nodes.append(n)

    n_words = len(body) // 8
    all_words = list(struct.unpack_from(f"<{n_words}Q", body, 0))

    NPU_TARGETS = {0x0101, 0x0201, 0x0801, 0x1001, 0x2001, 0x4001, 0x8001}
    blocks = []
    i = 0
    while i < n_words:
        if ((all_words[i] >> 48) & 0xFFFF) in NPU_TARGETS:
            j = i
            while j < n_words and ((all_words[j] >> 48) & 0xFFFF) in NPU_TARGETS:
                j += 1
            if j - i >= 20:
                kind = _classify_block(all_words[i:j])
                blocks.append(CommandBlock(i, j - i, all_words[i:j], kind))
            i = j
        else:
            i += 1

    return {"version": version, "body": body, "body_size": body_size,
            "tensors": tensors, "nodes": nodes, "blocks": blocks}

def _classify_block(words):
    reg_map = {}
    for w in words:
        target = (w >> 48) & 0xFFFF
        reg = w & 0xFFFF
        val = (w >> 16) & 0xFFFFFFFF
        reg_map[(target, reg)] = val
    fmt = reg_map.get((0x1001, 0x4010), 0)
    ew = reg_map.get((0x1001, 0x4070), 0)
    is_binary = (ew & 0x40) and not (ew & 0x01)
    if fmt == 0x48000002 and is_binary:
        return "EW_BINARY"
    return "COPY"

# ── NPU Device ───────────────────────────────────────────────────────────────
class NPUDevice:
    def __init__(self, dev="/dev/dri/card1"):
        self.fd = os.open(dev, os.O_RDWR)

    def mem_alloc(self, size, flags=0):
        mc = rknpu_mem_create(flags=flags, size=size)
        ctypes.memset(ctypes.addressof(mc), 0, ctypes.sizeof(mc))
        mc.flags = flags; mc.size = size
        ioctl(self.fd, IOCTL_MEM_CREATE, mc)
        mm = rknpu_mem_map(handle=mc.handle)
        ctypes.memset(ctypes.addressof(mm), 0, ctypes.sizeof(mm))
        mm.handle = mc.handle
        ioctl(self.fd, IOCTL_MEM_MAP, mm)
        buf = mmap.mmap(self.fd, mc.size, mmap.MAP_SHARED,
                        mmap.PROT_READ | mmap.PROT_WRITE, offset=mm.offset)
        return buf, mc

    def mem_sync(self, handle, size, offset=0, flags=1):
        s = rknpu_mem_sync(handle=handle, flags=flags, offset=offset, size=size)
        ioctl(self.fd, IOCTL_MEM_SYNC, s)

    def reset(self):
        ioctl(self.fd, IOCTL_ACTION, rknpu_action(flags=RKNPU_ACT_RESET, value=0))

    def submit(self, task_obj_addr, task_start, task_number):
        self.reset()
        s = rknpu_submit(
            flags=RKNPU_JOB_PC | RKNPU_JOB_PINGPONG, timeout=6000,
            task_start=task_start, task_number=task_number,
            task_counter=0, priority=0, task_obj_addr=task_obj_addr,
            core_mask=0, fence_fd=-1)
        s.subcore_task[0] = rknpu_subcore_task(task_start=0, task_number=1)
        s.subcore_task[1] = rknpu_subcore_task(task_start=0, task_number=1)
        s.subcore_task[2] = rknpu_subcore_task(task_start=0, task_number=1)
        return ioctl(self.fd, IOCTL_SUBMIT, s)

    def close(self):
        os.close(self.fd)

# ── NC1HWC2 Layout Conversion ───────────────────────────────────────────────
def _elem_size_from_native(native):
    """Determine bytes per element from NC1HWC2 shape [N, C1, H, W, C2]."""
    if len(native) == 5:
        return 16 // native[4]  # atom is 16 bytes, C2 elements per atom
    return 2

def _contiguous_to_nc1hwc2(src, native_shape, stride_atoms=None):
    """Pack flat data into NC1HWC2 layout with optional stride padding.
    native_shape = [N, C1, H, W, C2]. stride_atoms = atoms per C1 row (default W)."""
    if not native_shape or len(native_shape) < 5:
        n = len(src)
        out = np.zeros(n * max(src.dtype.itemsize, 2), dtype=np.uint8)
        out[:n * src.dtype.itemsize] = src.view(np.uint8)
        return out

    N, C1, H, W, C2 = native_shape
    row_atoms = stride_atoms if stride_atoms else W
    total_atoms = N * C1 * H * row_atoms
    elem_bytes = src.dtype.itemsize
    out = np.zeros(total_atoms * C2 * elem_bytes, dtype=np.uint8).view(src.dtype)

    flat = src.ravel()
    idx = 0
    for n in range(N):
        for c1 in range(C1):
            for h in range(H):
                for w in range(W):
                    atom = ((n * C1 + c1) * H + h) * row_atoms + w
                    take = min(C2, len(flat) - idx)
                    if take <= 0:
                        break
                    out[atom * C2:atom * C2 + take] = flat[idx:idx + take]
                    idx += take

    return out.view(np.uint8)

def _nc1hwc2_to_contiguous(packed, native_shape, dtype, stride_atoms=None):
    """Unpack NC1HWC2 layout with stride back to flat contiguous data."""
    if not native_shape or len(native_shape) < 5:
        n = len(packed) // np.dtype(dtype).itemsize
        return packed[:n * np.dtype(dtype).itemsize].view(dtype).copy()

    N, C1, H, W, C2 = native_shape
    row_atoms = stride_atoms if stride_atoms else W
    src = packed.view(dtype) if packed.dtype != dtype else packed.copy()

    result = []
    for n in range(N):
        for c1 in range(C1):
            for h in range(H):
                for w in range(W):
                    atom = ((n * C1 + c1) * H + h) * row_atoms + w
                    for c2 in range(C2):
                        si = atom * C2 + c2
                        if si < len(src):
                            result.append(src[si])

    return np.array(result, dtype=dtype)

# ── CPU Op Kernels ───────────────────────────────────────────────────────────
CPU_KERNELS = {
    "Add": lambda inputs: inputs[0] + inputs[1],
    "Sub": lambda inputs: inputs[0] - inputs[1],
    "Mul": lambda inputs: inputs[0] * inputs[1],
    "Div": lambda inputs: inputs[0] / inputs[1],
    "And": lambda inputs: np.bitwise_and(inputs[0], inputs[1]),
    "Or":  lambda inputs: np.bitwise_or(inputs[0], inputs[1]),
    "Xor": lambda inputs: np.bitwise_xor(inputs[0], inputs[1]),
    "Not": lambda inputs: np.bitwise_not(inputs[0]),
    "Neg": lambda inputs: -inputs[0],
}

# ── DMA Address Patching ────────────────────────────────────────────────────
REG_DST_BASE  = 0x4020  # DPU_DST_BASE_ADDR
REG_RDMA_SRC  = 0x5018  # DPU_RDMA_RDMA_SRC_BASE_ADDR
REG_RDMA_EW   = 0x5038  # DPU_RDMA_RDMA_EW_BASE_ADDR
DPU_TARGET    = 0x1001
RDMA_TARGET   = 0x2001

def _patch_dma_addr(words, reg, target, addr):
    for i, w in enumerate(words):
        t = (w >> 48) & 0xFFFF
        r = w & 0xFFFF
        if t == target and r == reg:
            val = (addr & 0xFFFFFFFF)
            words[i] = (target << 48) | (val << 16) | reg
            return True
    return False

# ── RKNN Runtime ─────────────────────────────────────────────────────────────
class RKNNRuntime:
    def __init__(self, rknn_path):
        with open(rknn_path, "rb") as f:
            self.raw = f.read()
        self.model = parse_rknn(self.raw)
        self.dev = NPUDevice()
        self._buffers = {}
        self._feature_buf = None
        self._feature_mc = None
        self._input_bufs = []
        self._output_bufs = []
        self._task_buf = None
        self._task_mc = None
        self._model_buf = None
        self._model_mc = None
        self._graph = None
        self._npu_submit_order = []
        self._cpu_ops = []
        self._input_tensors = []
        self._output_tensors = []
        self._init()

    def _init(self):
        m = self.model
        tensors = m["tensors"]
        nodes = m["nodes"]
        blocks = m["blocks"]

        # Classify tensors
        for t in tensors:
            if t.name and t.name.startswith("InputOperator") or (t.logical and len(t.logical) >= 2 and t.name and t.name in [n.name.split(":")[-1] for n in nodes if n.op == "InputOperator"]):
                pass

        # Find input/output tensors by name
        input_node_names = [n.name for n in nodes if n.op == "InputOperator"]
        output_node_names = [n.name for n in nodes if n.op == "OutputOperator"]

        # Input tensors: named "x", "y" etc or linked from InputOperator nodes
        # Input tensors: external inputs named x/y or a/b or with InputOperator
        self._input_tensors = [t for t in tensors
            if t.name and t.name.lower() in ("x", "y", "a", "b", "input0", "input1")]
        self._output_tensors = [t for t in tensors
            if t.name and t.name.lower() in ("z", "c", "output0", "output")]

        # If no named matches, use position: first external tensors with logical shape
        if not self._input_tensors:
            self._input_tensors = [t for t in tensors if t.logical and len(t.logical) >= 2 and t.offset is None and "rs" not in (t.name or "")][:2]
        if not self._output_tensors:
            self._output_tensors = [t for t in tensors if t.logical and len(t.logical) >= 2 and t.offset is None and "rs" not in (t.name or "")][-1:]

        # Classify: NPU-op vs CPU-op
        cpu_ops = []
        for n in nodes:
            if n.op in ("InputOperator", "OutputOperator", "Reshape"):
                continue
            cpu_ops.append(n)

        # Decide execution path:
        # - If blocks are EW_BINARY with fp16/int8 precision → NPU can execute them
        # - If blocks are COPY or EW with int32/fp32 precision → CPU fallback
        has_ew_binary = any(blk.kind == "EW_BINARY" for blk in blocks)
        # Check precision from first block's DATA_FORMAT register
        npu_compatible = False
        proc_prec = None
        if has_ew_binary and blocks:
            fmt_val = None
            for w in blocks[0].words:
                target = (w >> 48) & 0xFFFF
                reg = w & 0xFFFF
                val = (w >> 16) & 0xFFFFFFFF
                if target == DPU_TARGET and reg == 0x4010:  # DATA_FORMAT
                    fmt_val = val
                    break
            if fmt_val is not None:
                proc_prec = fmt_val & 0xF
                npu_compatible = proc_prec in (2, 0)  # fp16=2, int8=0

        self._cpu_ops = cpu_ops

        all_copy = all(blk.kind == "COPY" for blk in blocks) if blocks else False
        has_cpu_ops = any(n.op in CPU_KERNELS for n in nodes)
        if all_copy and has_cpu_ops:
            self._graph = True
            self._all_cpu = False
            self._init_graph()
            return

        # Element-wise ops (Add/Sub/Mul/Div) whose baked blocks are fp16/int8 run
        # on the NPU directly from the .rknn body — the register commands are
        # extracted from the file (no synthesis), inputs are NC1HWC2-reshaped, the
        # 3 DMA bases are patched, and the body's task blocks are submitted. This
        # holds for ANY size baked into the file: a larger .rknn simply carries
        # more blocks. fp16 EW therefore NEVER falls back to CPU.
        all_cpu = not npu_compatible
        self._all_cpu = all_cpu

        # For NPU path: determine how many tasks to submit
        # Vendor runtime submits len(blocks)//2 tasks (ping half only)
        if not all_cpu and len(blocks) >= 2:
            self._npu_task_count = len(blocks) // 2
        else:
            self._npu_task_count = 0

        # Allocate DMA buffers
        # 1. Model buffer (holds .rknn data including regcmd blocks)
        self._model_buf, self._model_mc = self.dev.mem_alloc(
            max(len(self.raw), 4096), RKNPU_MEM_KERNEL_MAPPING)

        # Copy model data into model buffer
        mv = memoryview(self._model_buf)
        mv[:len(self.raw)] = self.raw

        # 2. Feature/working buffer
        feature_tensors = [t for t in tensors if t.offset is not None and t.size and "rs" in (t.name or "")]
        if not feature_tensors:
            feature_tensors = [t for t in tensors if t.offset is not None and t.size]
        rs_tensors_nc1hwc2 = {t.name: t for t in tensors
                              if t.name and "rs" in t.name and t.native and len(t.native) == 5
                              and t.name not in (n.name for n in feature_tensors if n.name)}
        for name, t in rs_tensors_nc1hwc2.items():
            if t.offset is None:
                max_off = max((ft.offset + ft.size for ft in feature_tensors), default=0)
                aligned = (max_off + 63) & ~63
                t.offset = aligned
                feature_tensors.append(t)

        # Also assign offsets for input/output tensors that have offset=None
        for t in list(self._input_tensors) + list(self._output_tensors):
            if t.offset is None and t.size:
                max_off = max((ft.offset + ft.size for ft in feature_tensors), default=0)
                aligned = (max_off + 63) & ~63
                t.offset = aligned
                feature_tensors.append(t)

        feature_size = max((t.offset + t.size for t in feature_tensors), default=4096)
        feature_size = max(feature_size, 4096)

        # For NPU path: allocate separate DMA buffers per RS tensor (like vendor)
        self._rs_bufs = {}
        if not all_cpu:
            for t in feature_tensors:
                if t.native and len(t.native) == 5:
                    sz = max(t.size, 4096)
                    buf, mc = self.dev.mem_alloc(sz, RKNPU_MEM_KERNEL_MAPPING)
                    t.offset = 0
                    t._dma_addr = mc.dma_addr
                    t._buf = buf
                    self._rs_bufs[t.name] = (buf, mc, t)
            # Also allocate for any NC1HWC2 tensor that was added
            self._feature_buf, self._feature_mc = self.dev.mem_alloc(4096, RKNPU_MEM_KERNEL_MAPPING)
        else:
            self._feature_buf, self._feature_mc = self.dev.mem_alloc(feature_size, RKNPU_MEM_KERNEL_MAPPING)
            for t in feature_tensors:
                t._dma_addr = self._feature_mc.dma_addr

        # 3. Input buffers (one per input tensor)
        for t in self._input_tensors:
            sz = max(t.size or (np.prod(t.logical) * 4), 4096)
            buf, mc = self.dev.mem_alloc(sz)
            self._input_bufs.append((buf, mc, t))

        # 4. Output buffer
        for t in self._output_tensors:
            sz = max(t.size or (np.prod(t.logical) * 4), 4096)
            buf, mc = self.dev.mem_alloc(sz)
            self._output_bufs.append((buf, mc, t))

        # 5. Task array buffer
        n_tasks = len(blocks)
        task_size = max(n_tasks * 40, 4096)
        self._task_buf, self._task_mc = self.dev.mem_alloc(task_size, RKNPU_MEM_KERNEL_MAPPING)

        # Build task array from blocks
        tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(self._task_buf)),
                            ctypes.POINTER(rknpu_task))
        body = m["body"]
        regcmd_offset = blocks[0].word_offset * 8 if blocks else 0

        for i, blk in enumerate(blocks):
            tasks[i].flags = 0
            tasks[i].op_idx = 4
            tasks[i].enable_mask = 0x18
            tasks[i].int_mask = 0x300
            tasks[i].int_clear = 0x1ffff
            tasks[i].int_status = 0
            tasks[i].regcfg_amount = blk.n_words
            tasks[i].regcfg_offset = 0
            tasks[i].regcmd_addr = self._model_mc.dma_addr + HEADER_SIZE + blk.word_offset * 8

        self._tasks = tasks
        self._n_tasks = n_tasks
        self._blocks = blocks
        self._feature_tensors = feature_tensors

        # Extract NPU stride from first block (for stride-aware NC1HWC2 packing)
        self._npu_stride = None
        if blocks:
            for w in blocks[0].words:
                target = (w >> 48) & 0xFFFF
                reg = w & 0xFFFF
                val = (w >> 16) & 0xFFFFFFFF
                if target == DPU_TARGET and reg == 0x4024:
                    self._npu_stride = val
                    break

        # Fix z-rs/c-rs offset if None (vendor runtime places output RS at offset 0)
        z_rs = self._get_feature_tensor("z-rs") or self._get_feature_tensor("c-rs")
        if z_rs and z_rs.offset is None:
            z_rs.offset = 0
            print(f"  Fixed {z_rs.name} offset to 0")

        print(f"RKNNRuntime init: {len(nodes)} nodes, {len(blocks)} blocks, "
              f"{len(cpu_ops)} CPU ops, all_cpu={all_cpu}")
        print(f"  inputs: {[t.name for t in self._input_tensors]}")
        print(f"  outputs: {[t.name for t in self._output_tensors]}")
        print(f"  feature_buf: {feature_size} bytes, model_buf: {len(self.raw)} bytes")

    def _get_feature_tensor(self, name):
        alt_map = {"x_rs": "a_rs", "a_rs": "x_rs",
                   "y_rs": "b_rs", "b_rs": "y_rs",
                   "z-rs": "c-rs", "c-rs": "z-rs"}
        candidates = [name, name.lower(), name.upper()]
        # Also add alternative names
        for n in list(candidates):
            if n in alt_map:
                candidates.extend([alt_map[n], alt_map[n].upper(), alt_map[n].lower()])
        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        candidates = unique

        # Exact match first (case-insensitive)
        for c in candidates:
            for t in self._feature_tensors:
                if t.name and t.name.lower() == c.lower():
                    return t
        # Then NC1HWC2 prefix match
        for c in candidates:
            nc1hwc2 = [t for t in self._feature_tensors
                       if t.name and t.name.lower().startswith(c.lower()) and t.native and len(t.native) == 5]
            if nc1hwc2:
                return nc1hwc2[0]
        return None

    def _detect_model_dtype(self):
        for n in self.model["nodes"]:
            if n.op == "InputOperator":
                for ti in n.outputs:
                    t = self.model["tensors"][ti]
                    if t.native and len(t.native) == 5:
                        c2 = t.native[4]
                        if c2 == 8: return np.float16
                        elif c2 == 4: return np.int32
                        elif c2 == 2: return np.float16
                        elif c2 == 1: return np.int8
        return np.int32

    def _init_graph(self):
        m = self.model
        tensors = m["tensors"]
        nodes = m["nodes"]
        blocks = m["blocks"]

        self._input_tensors = []
        self._output_tensors = []
        for n in nodes:
            if n.op == "InputOperator":
                for ti in n.outputs:
                    self._input_tensors.append(tensors[ti])
            elif n.op == "OutputOperator":
                for ti in n.inputs:
                    self._output_tensors.append(tensors[ti])

        self._graph_dtype = self._detect_model_dtype()
        elem_size = self._graph_dtype().itemsize

        data_tensor_indices = set()
        for n in nodes:
            if n.op == "InputOperator":
                data_tensor_indices.update(n.outputs)
            elif n.op == "OutputOperator":
                data_tensor_indices.update(n.inputs)
            else:
                data_tensor_indices.update(n.outputs)

        self._tensor_bufs = {}
        for ti in sorted(data_tensor_indices):
            t = tensors[ti]
            sz = max(t.size or 0,
                     int(np.prod(t.logical)) * elem_size if t.logical else 0,
                     4096)
            buf, mc = self.dev.mem_alloc(sz, RKNPU_MEM_KERNEL_MAPPING)
            t._dma_addr = mc.dma_addr
            t._buf = buf
            self._tensor_bufs[ti] = (buf, mc)

        self._model_buf, self._model_mc = self.dev.mem_alloc(
            max(len(self.raw), 4096), RKNPU_MEM_KERNEL_MAPPING)
        mv = memoryview(self._model_buf)
        mv[:len(self.raw)] = self.raw

        n_tasks = len(blocks)
        task_size = max(n_tasks * 40, 4096)
        self._task_buf, self._task_mc = self.dev.mem_alloc(task_size, RKNPU_MEM_KERNEL_MAPPING)

        tasks = ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(self._task_buf)),
                            ctypes.POINTER(rknpu_task))
        for i, blk in enumerate(blocks):
            tasks[i].flags = 0
            tasks[i].op_idx = 4
            tasks[i].enable_mask = 0x18
            tasks[i].int_mask = 0x300
            tasks[i].int_clear = 0x1ffff
            tasks[i].int_status = 0
            tasks[i].regcfg_amount = blk.n_words
            tasks[i].regcfg_offset = 0
            tasks[i].regcmd_addr = self._model_mc.dma_addr + HEADER_SIZE + blk.word_offset * 8

        self._tasks = tasks
        self._n_tasks = n_tasks
        self._blocks = blocks

        block_to_node = {}
        bi = 0
        for n in nodes:
            if n.op in ("InputOperator", "OutputOperator"):
                continue
            if n.op in CPU_KERNELS:
                continue
            if bi < len(blocks):
                block_to_node[bi] = n
                bi += 1

        hdr_qwords = HEADER_SIZE // 8
        model_view = (ctypes.c_uint64 * (len(self.raw) // 8)).from_buffer(self._model_buf)

        for bi, blk in enumerate(blocks):
            node = block_to_node.get(bi)
            if node is None:
                continue
            data_inputs = [ti for ti in node.inputs if ti in data_tensor_indices]
            data_outputs = [ti for ti in node.outputs if ti in data_tensor_indices]
            if not data_inputs or not data_outputs:
                continue
            src_ti = data_inputs[0]
            dst_ti = data_outputs[0]
            src_addr = tensors[src_ti]._dma_addr
            dst_addr = tensors[dst_ti]._dma_addr
            for wi in range(blk.n_words):
                abs_idx = hdr_qwords + blk.word_offset + wi
                w = model_view[abs_idx]
                target = (w >> 48) & 0xFFFF
                reg = w & 0xFFFF
                old_val = (w >> 16) & 0xFFFFFFFF
                if target == RDMA_TARGET and reg == REG_RDMA_SRC:
                    model_view[abs_idx] = (target << 48) | (((src_addr + old_val) & 0xFFFFFFFF) << 16) | reg
                elif target == DPU_TARGET and reg == REG_DST_BASE:
                    model_view[abs_idx] = (target << 48) | (((dst_addr + old_val) & 0xFFFFFFFF) << 16) | reg

        self.dev.mem_sync(self._model_mc.handle, self._model_mc.size)

        schedule = []
        task_start = 0
        pending_tasks = 0
        for n in nodes:
            if n.op in ("InputOperator", "OutputOperator"):
                continue
            if n.op in CPU_KERNELS:
                if pending_tasks > 0:
                    schedule.append(("npu", task_start, pending_tasks))
                    task_start += pending_tasks
                    pending_tasks = 0
                schedule.append(("cpu", n))
            else:
                pending_tasks += 1
        if pending_tasks > 0:
            schedule.append(("npu", task_start, pending_tasks))

        self._graph_schedule = schedule
        print(f"Graph executor: {len(blocks)} COPY blocks, "
              f"{len([s for s in schedule if s[0]=='cpu'])} CPU ops, dtype={self._graph_dtype}")
        print(f"  inputs:  {[t.name for t in self._input_tensors]}")
        print(f"  outputs: {[t.name for t in self._output_tensors]}")
        print(f"  data_tensors: {sorted(data_tensor_indices)}")
        print(f"  schedule: {[(s[0], s[1].op if s[0]=='cpu' else f'tasks {s[1]}-{s[1]+s[2]-1}') for s in schedule]}")

    def _inputs_set_graph(self, input_arrays):
        for i, arr in enumerate(input_arrays):
            arr = np.asarray(arr)
            t = self._input_tensors[i]
            buf = t._buf
            if t.native and len(t.native) >= 5:
                packed = _contiguous_to_nc1hwc2(arr, t.native)
            else:
                packed = arr.view(np.uint8)
            n_bytes = packed.nbytes if isinstance(packed, np.ndarray) else len(packed)
            ct = (ctypes.c_uint8 * n_bytes).from_buffer(buf)
            ct[:n_bytes] = packed.tobytes() if isinstance(packed, np.ndarray) else bytes(packed)
            mc = self._tensor_bufs[t.idx][1]
            self.dev.mem_sync(mc.handle, mc.size)
        self._graph_user_dtype = input_arrays[0].dtype if input_arrays else self._graph_dtype

    def _run_graph(self):
        tensors = self.model["tensors"]
        for item in self._graph_schedule:
            if item[0] == "npu":
                _, start, count = item
                for ti in range(start, start + count):
                    blk = self._blocks[ti]
                    self._tasks[0].regcmd_addr = (self._model_mc.dma_addr
                                                   + HEADER_SIZE + blk.word_offset * 8)
                    self._tasks[0].regcfg_amount = blk.n_words
                    self.dev.submit(self._task_mc.obj_addr, 0, 1)
            elif item[0] == "cpu":
                _, node = item
                self._execute_graph_cpu_op(node, tensors)

    def _execute_graph_cpu_op(self, node, tensors):
        kernel = CPU_KERNELS.get(node.op)
        if kernel is None:
            return
        dtype = self._graph_user_dtype if hasattr(self, '_graph_user_dtype') else self._graph_dtype
        dt = np.dtype(dtype)
        elem_size = dt.itemsize

        data_inputs = [ti for ti in node.inputs if ti in self._tensor_bufs]
        data_outputs = [ti for ti in node.outputs if ti in self._tensor_bufs]

        input_data = []
        for ti in data_inputs:
            t = tensors[ti]
            buf = t._buf
            n_bytes = t.size or (int(np.prod(t.logical)) * elem_size if t.logical else 4096)
            n_elems = n_bytes // elem_size
            raw = (ctypes.c_uint8 * n_bytes).from_buffer(buf)
            input_data.append(np.frombuffer(raw, dtype=dt, count=n_elems).copy())

        result = kernel(input_data)

        for oi, ti in enumerate(data_outputs):
            t = tensors[ti]
            buf = t._buf
            out_data = result if len(data_outputs) == 1 else result[oi]
            n_bytes = len(out_data) * elem_size
            ct = (ctypes.c_uint8 * n_bytes).from_buffer(buf)
            ct[:n_bytes] = out_data.view(np.uint8).tobytes()
            mc = self._tensor_bufs[ti][1]
            self.dev.mem_sync(mc.handle, mc.size)

    def _outputs_get_graph(self):
        results = []
        dtype = self._graph_user_dtype if hasattr(self, '_graph_user_dtype') else self._graph_dtype
        dt = np.dtype(dtype)
        elem_size = dt.itemsize
        for t in self._output_tensors:
            buf = t._buf
            if t.native and len(t.native) >= 5:
                sz = t.size or 4096
                raw = (ctypes.c_uint8 * sz).from_buffer(buf)
                packed = np.frombuffer(raw, dtype=np.uint8, count=sz).copy()
                data = _nc1hwc2_to_contiguous(packed, t.native, dt)
            else:
                n_elems = int(np.prod(t.logical)) if t.logical else 1
                n_bytes = n_elems * elem_size
                raw = (ctypes.c_uint8 * n_bytes).from_buffer(buf)
                data = np.frombuffer(raw, dtype=dt, count=n_elems).copy()
            results.append(data.reshape(t.logical) if t.logical else data)
        return results

    def inputs_set(self, input_arrays):
        """Write input data into the model (contiguous → NC1HWC2 + DMA patching)."""
        if self._graph:
            return self._inputs_set_graph(input_arrays)
        # Write raw input data to input buffers
        for i, arr in enumerate(input_arrays):
            if i >= len(self._input_bufs):
                break
            buf, mc, tensor = self._input_bufs[i]
            arr = np.asarray(arr)
            n_bytes = arr.nbytes
            ct = (ctypes.c_uint8 * n_bytes).from_buffer(buf)
            ct[:n_bytes] = arr.view(np.uint8).tobytes()

        # Reshape inputs into NC1HWC2 and write to feature buffer
        # For each input tensor, find its NC1HWC2 version (x_rs, y_rs)
        for i, arr in enumerate(input_arrays):
            arr = np.asarray(arr)
            name = self._input_tensors[i].name if i < len(self._input_tensors) else f"input{i}"
            rs_names = [f"{name}_rs", f"{name}-rs"]
            alt_map = {"x": "a", "a": "x", "y": "b", "b": "y"}
            if name.lower() in alt_map:
                rs_names.append(f"{alt_map[name.lower()]}_rs")
                rs_names.append(f"{alt_map[name.lower()]}-rs")
            rs_tensor = None
            for rn in rs_names:
                rs_tensor = self._get_feature_tensor(rn)
                if rs_tensor:
                    break

            if rs_tensor is None or rs_tensor.native is None:
                # No RS tensor: write raw input data directly to feature buffer
                inp_t = self._input_tensors[i]
                if inp_t.offset is not None:
                    raw = arr.view(np.uint8).tobytes()
                    n_bytes = min(len(raw), inp_t.size or len(raw))
                    ct = (ctypes.c_uint8 * n_bytes).from_buffer(self._feature_buf, inp_t.offset)
                    ct[:n_bytes] = raw[:n_bytes]
                continue

            packed = _contiguous_to_nc1hwc2(arr, rs_tensor.native, self._stride_for(rs_tensor))
            buf = getattr(rs_tensor, '_buf', None) or self._feature_buf
            off = rs_tensor.offset
            n_bytes = min(len(packed), rs_tensor.size)
            ct = (ctypes.c_uint8 * n_bytes).from_buffer(buf, off)
            ct[:n_bytes] = packed[:n_bytes].tobytes() if isinstance(packed, np.ndarray) else bytes(packed[:n_bytes])

        # Patch DMA addresses in regcmd blocks
        # Patch DST_BASE_ADDR (0x4020) → z-rs offset in feature buffer
        # Patch RDMA_SRC (0x5018) → x_rs offset in feature buffer
        # Patch RDMA_EW (0x5038) → y_rs offset in feature buffer
        x_rs = self._get_feature_tensor("x_rs")
        y_rs = self._get_feature_tensor("y_rs")
        z_rs = self._get_feature_tensor("z-rs")

        x_dma = getattr(x_rs, '_dma_addr', None) or (self._feature_mc.dma_addr + x_rs.offset if x_rs else None)
        y_dma = getattr(y_rs, '_dma_addr', None) or (self._feature_mc.dma_addr + y_rs.offset if y_rs else None)
        z_dma = getattr(z_rs, '_dma_addr', None) or (self._feature_mc.dma_addr + z_rs.offset if z_rs else None)

        # Patch all blocks in the model buffer (modifying the mmap'd copy)
        # Block word offsets are relative to the body, model buffer has 64B header
        hdr_qwords = HEADER_SIZE // 8
        model_view = (ctypes.c_uint64 * (len(self.raw) // 8)).from_buffer(self._model_buf)
        for blk in self._blocks:
            for wi in range(blk.n_words):
                abs_idx = hdr_qwords + blk.word_offset + wi
                w = model_view[abs_idx]
                target = (w >> 48) & 0xFFFF
                reg = w & 0xFFFF
                old_val = (w >> 16) & 0xFFFFFFFF

                if target == RDMA_TARGET and reg == REG_RDMA_SRC and x_dma is not None:
                    model_view[abs_idx] = (RDMA_TARGET << 48) | (((x_dma + old_val) & 0xFFFFFFFF) << 16) | reg

                elif target == RDMA_TARGET and reg == REG_RDMA_EW and y_dma is not None:
                    model_view[abs_idx] = (RDMA_TARGET << 48) | (((y_dma + old_val) & 0xFFFFFFFF) << 16) | reg

                elif target == DPU_TARGET and reg == REG_DST_BASE and z_dma is not None:
                    model_view[abs_idx] = (DPU_TARGET << 48) | (((z_dma + old_val) & 0xFFFFFFFF) << 16) | reg

        for name, (buf, mc, t) in self._rs_bufs.items():
            self.dev.mem_sync(mc.handle, mc.size)
        self.dev.mem_sync(self._model_mc.handle, self._model_mc.size)

    def run(self):
        """Execute the model: NPU submit or CPU fallback."""
        if self._graph:
            return self._run_graph()
        if self._all_cpu:
            for cpu_node in self._cpu_ops:
                self._execute_cpu_op(cpu_node)
        else:
            # NPU path: submit EW_BINARY blocks
            n_submit = self._npu_task_count or self._n_tasks
            self.dev.submit(self._task_mc.obj_addr, 0, n_submit)

    def _execute_cpu_op(self, node):
        op = node.op
        if op not in CPU_KERNELS:
            print(f"  CPU op '{op}' not implemented, skipping")
            return

        x_rs = self._get_feature_tensor("x_rs")
        y_rs = self._get_feature_tensor("y_rs")
        z_rs = self._get_feature_tensor("z-rs")

        if x_rs is not None and z_rs is not None:
            self._execute_cpu_op_rs(op, x_rs, y_rs, z_rs)
        else:
            self._execute_cpu_op_flat(op)

    def _execute_cpu_op_flat(self, op):
        x_t = self._input_tensors[0]
        y_t = self._input_tensors[1] if len(self._input_tensors) > 1 else None
        z_t = self._output_tensors[0]

        n_elems = np.prod(x_t.logical) if x_t.logical else x_t.n_elems
        dtype = self._detect_dtype(x_t, n_elems)

        x_data = self._read_flat(x_t, dtype, n_elems)

        if y_t:
            y_data = self._read_flat(y_t, dtype, n_elems)
            result = CPU_KERNELS[op]([x_data, y_data])
        else:
            result = CPU_KERNELS[op]([x_data])

        self._write_flat(z_t, result, dtype)

    def _is_nc1hwc2(self, tensor):
        if not tensor.native or len(tensor.native) != 5:
            return False
        N, C1, H, W, C2 = tensor.native
        nc1hwc2_total = N * C1 * H * W * C2
        logical_total = int(np.prod(tensor.logical)) if tensor.logical else 0
        return nc1hwc2_total > logical_total and C2 in (1, 2, 4, 8)

    def _detect_dtype(self, tensor, n_elems):
        if self._is_nc1hwc2(tensor):
            C2 = tensor.native[4]
            if C2 == 8: return np.float16
            elif C2 == 4: return np.int32
            elif C2 == 1: return np.int8
        elem_bytes = (tensor.size or 0) // max(n_elems, 1)
        if elem_bytes == 2: return np.float16
        elif elem_bytes == 1: return np.int8
        return np.int32

    def _read_flat(self, tensor, dtype, n_elems):
        buf = getattr(tensor, '_buf', None) or self._feature_buf
        off = tensor.offset or 0
        sz = tensor.size or (n_elems * dtype().itemsize)
        raw = (ctypes.c_uint8 * sz).from_buffer(buf, off)
        return np.frombuffer(raw, dtype=dtype, count=n_elems).copy()

    def _write_flat(self, tensor, data, dtype):
        buf = getattr(tensor, '_buf', None) or self._feature_buf
        off = tensor.offset or 0
        n_bytes = min(len(data) * dtype().itemsize, tensor.size or len(data) * dtype().itemsize)
        ct = (ctypes.c_uint8 * n_bytes).from_buffer(buf, off)
        ct[:n_bytes] = data[:n_bytes // dtype().itemsize].view(np.uint8).tobytes()

    def _execute_cpu_op_rs(self, op, x_rs, y_rs, z_rs):
        c2 = x_rs.native[4] if len(x_rs.native) >= 5 else 4
        if c2 == 8:
            dtype = np.float16
        elif c2 == 4:
            dtype = np.int32
        elif c2 == 2:
            dtype = np.float16
        elif c2 == 1:
            dtype = np.int8
        else:
            dtype = np.int32

        x_data = self._read_nc1hwc2(x_rs, dtype)

        if y_rs is not None:
            y_data = self._read_nc1hwc2(y_rs, dtype)
            result = CPU_KERNELS[op]([x_data, y_data])
        else:
            result = CPU_KERNELS[op]([x_data])

        self._write_nc1hwc2(z_rs, result)

    def _stride_for(self, tensor):
        if self._npu_stride and tensor.native and len(tensor.native) == 5:
            W = tensor.native[3]
            stride_atoms = self._npu_stride // 16
            if stride_atoms >= W:
                return stride_atoms
        return None

    def _read_nc1hwc2(self, tensor, dtype):
        buf = getattr(tensor, '_buf', None) or self._feature_buf
        off = tensor.offset
        sz = tensor.size
        raw = (ctypes.c_uint8 * sz).from_buffer(buf, off)
        packed = np.frombuffer(raw, dtype=np.uint8, count=sz).copy()
        return _nc1hwc2_to_contiguous(packed, tensor.native, dtype, self._stride_for(tensor))

    def _write_nc1hwc2(self, tensor, data):
        buf = getattr(tensor, '_buf', None) or self._feature_buf
        packed = _contiguous_to_nc1hwc2(data, tensor.native, self._stride_for(tensor))
        off = tensor.offset
        n = min(len(packed), tensor.size)
        ct = (ctypes.c_uint8 * n).from_buffer(buf, off)
        ct[:n] = packed[:n].tobytes() if isinstance(packed, np.ndarray) else bytes(packed[:n])

    def _get_output_rs_offset(self):
        """Determine the z-rs/c-rs offset in the feature buffer.
        The vendor runtime places the output NC1HWC2 tensor at offset 0.
        The FlatBuffer may have it as None or we compute from the memory plan."""
        # Check all known output RS tensor name patterns
        for prefix in ("z-rs", "c-rs", "out-rs", "output_rs"):
            for t in self._feature_tensors:
                if t.name and t.name.startswith(prefix) and t.offset is not None:
                    return t.offset, t.size, t.native
        # The vendor always puts output RS at offset 0 in the feature buffer
        # Find the tensor by size: it's the first NC1HWC2 tensor with the output size
        z_rs = self._get_feature_tensor("z-rs") or self._get_feature_tensor("c-rs")
        if z_rs:
            # Find from the output tensor's size
            for t in self._output_tensors:
                expected_size = z_rs.size
                # Offset 0 is the default for the output RS tensor
            return 0, z_rs.size, z_rs.native
        return None, None, None

    def outputs_get(self):
        if self._graph:
            return self._outputs_get_graph()
        results = []
        for i, (buf, mc, tensor) in enumerate(self._output_bufs):
            z_rs = self._get_feature_tensor("z-rs") or self._get_feature_tensor("c-rs")
            if z_rs and z_rs.native and len(z_rs.native) >= 5:
                c2 = z_rs.native[4]
                if c2 == 8:
                    dtype = np.float16
                elif c2 == 4:
                    dtype = np.int32
                elif c2 == 2:
                    dtype = np.float16
                elif c2 == 1:
                    dtype = np.int8
                else:
                    dtype = np.int32
                data = self._read_nc1hwc2(z_rs, dtype)
                n_out = np.prod(tensor.logical) if tensor.logical else len(data)
                results.append(data[:n_out])
            else:
                out_t = self._output_tensors[0] if self._output_tensors else tensor
                n_out = np.prod(out_t.logical) if out_t.logical else (out_t.size // 4)
                inp_t = self._input_tensors[0] if self._input_tensors else out_t
                n_inp = np.prod(inp_t.logical) if inp_t.logical else n_out
                dtype = self._detect_dtype(inp_t, n_inp)
                data = self._read_flat(out_t, dtype, n_out)
                results.append(data)
        return results

    def destroy(self):
        self.dev.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.destroy()


# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rknn_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/int32_add.rknn"

    print(f"=== Testing {rknn_path} ===\n")
    with RKNNRuntime(rknn_path) as rt:
        # The input/output shapes (and hence the size) come from the .rknn itself.
        # fp16 element-wise models execute on the NPU using the body's baked
        # register blocks; int32/fp32/CPU-op models use the CPU executor.
        # Detect dtype and shape from tensors
        x_rs = rt._get_feature_tensor("x_rs")
        if not x_rs:
            x_rs = rt._get_feature_tensor("a_rs")
        n_elems = rt._input_tensors[0].n_elems if rt._input_tensors else 10
        if x_rs and len(x_rs.native) >= 5:
            c2 = x_rs.native[4]
            if c2 == 8:
                dtype = np.float16
            elif c2 == 4:
                dtype = np.int32
            else:
                dtype = np.int32
        else:
            dtype = rt._detect_dtype(rt._input_tensors[0], n_elems)

        a = np.arange(n_elems, dtype=dtype).reshape(rt._input_tensors[0].logical) * 3
        b = np.arange(n_elems, dtype=dtype).reshape(rt._input_tensors[0].logical) * 7
        expected = a + b
        print(f"Input a: {a}")
        print(f"Input b: {b}")
        print(f"Expected: {expected}")

        rt.inputs_set([a, b])
        rt.run()
        results = rt.outputs_get()

        got = results[0].ravel()
        n = min(len(got), len(expected.ravel()))
        print(f"Got:      {got[:n]}")
        if dtype in (np.float16, np.float32):
            match = np.allclose(got[:n], expected.ravel()[:n], atol=0.5)
        else:
            match = np.array_equal(got[:n], expected.ravel()[:n])
        print(f"MATCH: {match}")
        if not match:
            for i in range(n):
                g = got[i]; e = expected.ravel()[i]
                if dtype in (np.float16, np.float32):
                    if abs(float(g) - float(e)) > 0.5:
                        print(f"  [{i}] got={g} expected={e}")
                else:
                    if g != e:
                        print(f"  [{i}] got={g} expected={e}")
