"""Helper: disassemble an RKNN model into its FlatBuffer graph + NPU command queue.

An .rknn (RKNPU2, toolkit 2.3.2 / rk3588) container is three concatenated parts:

    [0x00] 64-byte header   "RKNN" magic, u64 version @0x08, u64 bodySize @0x10
    [0x40] FlatBuffer body  (bodySize bytes) -- the compiled model graph
    [end]  u64 jsonLen + JSON trailer        -- graph "connection"/io metadata

The body is a FlatBuffer (root -> subgraph[0] -> {tensors, nodes}). Two of the
tensors, "regcmd" and "task", carry the compiled NPU command stream: regcmd is a
flat array of 64-bit register-command words grouped into per-tile blocks. This module
parses the container + FlatBuffer (no external deps) and decodes the regcmd words into
a readable command queue, labelling registers from refs/rkt_registers.h.

    from helpers.rknn_decode import decode_rknn, print_rknn_disasm
    m = decode_rknn("model.rknn")     # or bytes
    print_rknn_disasm(m)

This is offline/static: it never calls the runtime or submits anything to the NPU.
"""
import json
import os
import re
import struct

HEADER_SIZE = 0x40
MAGIC = b"RKNN"

# regcmd word layout (see RKNN_CREATION notes): target<<48 | value<<16 | reg
TARGET_MAP = {
  0x0101: "PC", 0x0201: "CNA", 0x0801: "CORE", 0x1001: "DPU",
  0x2001: "DPU_RDMA", 0x4001: "PPU", 0x8001: "PPU_RDMA",
  0x10001: "DDMA", 0x20001: "SDMA", 0x40001: "GLOBAL",
}
_NPU_TARGETS = set(TARGET_MAP)

# registers that identify / parameterize a DPU tile (see decode notes)
_DPU_MODE  = ("DPU", 0x4010)       # DATA_FORMAT: 0x48000002 = ALU op, 0x24000001 = reformat/transpose
_DPU_WIDTH = ("DPU", 0x4030)       # element count - 1
_DPU_OUT   = ("DPU", 0x4020)       # output DMA base (runtime-patched)
_DPU_EWCFG = ("DPU", 0x4070)       # EW_CFG: nonzero => elementwise (binary) datapath active
_RDMA_SRC  = ("DPU_RDMA", 0x5018)  # operand 0 DMA base (runtime-patched)
_RDMA_EW   = ("DPU_RDMA", 0x5034)  # ERDMA_CFG: bits[31:30] data-mode set => 2nd operand (live/patched form)
_RDMA_SRC2 = ("DPU_RDMA", 0x5038)  # operand 1 DMA base (runtime-patched)
_ALU_FORMAT, _REFORMAT = 0x48000002, 0x24000001
_MODE_NAMES = {_ALU_FORMAT: "alu", _REFORMAT: "reformat"}


def _load_reg_names():
  """Map reg offset (low 16 bits of a regcmd word) -> name, from refs/rkt_registers.h."""
  path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "refs", "rkt_registers.h")
  names = {}
  pat = re.compile(r"^#define\s+REG_([A-Z0-9_]+)\s+0x0000([0-9a-fA-F]{4})\b")
  with open(path) as f:
    for line in f:
      m = pat.match(line)
      if m:
        names[int(m.group(2), 16)] = m.group(1)
  return names

REG_NAMES = _load_reg_names()


# --------------------------------------------------------------------------- #
# container
# --------------------------------------------------------------------------- #
def split_container(data: bytes):
  """Split a .rknn into (version, body, trailer). `data` is the raw container bytes."""
  if data[:4] != MAGIC:
    raise ValueError("not an RKNN container (bad magic)")
  version = struct.unpack_from("<Q", data, 0x08)[0]
  body_size = struct.unpack_from("<Q", data, 0x10)[0]
  body = data[HEADER_SIZE:HEADER_SIZE + body_size]
  trailer = data[HEADER_SIZE + body_size:]
  return version, body, trailer


def parse_trailer(trailer: bytes):
  """Parse the trailing `u64 jsonLen + JSON` config; returns the decoded object or None."""
  if len(trailer) < 8:
    return None
  n = struct.unpack_from("<Q", trailer, 0)[0]
  try:
    return json.loads(trailer[8:8 + n].decode("utf-8", "replace"))
  except (ValueError, UnicodeDecodeError):
    return None


# --------------------------------------------------------------------------- #
# minimal FlatBuffer reader (just what the rknn body needs)
# --------------------------------------------------------------------------- #
class _FB:
  def __init__(self, body): self.b = body
  def u16(self, o): return struct.unpack_from("<H", self.b, o)[0]
  def u32(self, o): return struct.unpack_from("<I", self.b, o)[0]
  def i32(self, o): return struct.unpack_from("<i", self.b, o)[0]

  def _field_off(self, pos, field):
    vt = pos - self.i32(pos)
    vt_size = self.u16(vt)
    entry = vt + 4 + field * 2
    return 0 if entry + 2 > vt + vt_size else self.u16(entry)

  def field_abs(self, pos, field):
    off = self._field_off(pos, field)
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

  def _vec_target(self, pos, field):
    a = self.field_abs(pos, field)
    return a + self.u32(a) if a is not None else None

  def vec_u32(self, pos, field):
    v = self._vec_target(pos, field)
    if v is None: return None
    n = self.u32(v)
    return [self.u32(v + 4 + i * 4) for i in range(n)]

  def vec_tables(self, pos, field):
    v = self._vec_target(pos, field)
    if v is None: return []
    n = self.u32(v)
    return [v + 4 + i * 4 + self.u32(v + 4 + i * 4) for i in range(n)]

  def root(self): return self.u32(0)


def _tensor(fb, pos, idx):
  return {
    "index": idx,
    "name": fb.string(pos, 5),
    "native": fb.vec_u32(pos, 3),       # NC1HWC2 hardware shape
    "logical": fb.vec_u32(pos, 4),      # logical shape
    "size": fb.scalar_u32(pos, 12),     # padded buffer byte size
    "offset": fb.scalar_u32(pos, 13),   # offset in the feature DMA buffer
  }


# --------------------------------------------------------------------------- #
# regcmd decode
# --------------------------------------------------------------------------- #
def decode_reg(word: int):
  """Decode one 64-bit regcmd word into (target_name, reg_off, reg_name, value)."""
  target = (word >> 48) & 0xFFFF
  value = (word >> 16) & 0xFFFFFFFF
  reg = word & 0xFFFF
  tname = TARGET_MAP.get(target, f"UNK(0x{target:x})")
  return tname, reg, REG_NAMES.get(reg, f"REG_0x{reg:04x}"), value


def _regcmd_spans(body: bytes, min_words=20):
  """Find runs of >=min_words consecutive NPU-target words: (word_index, n_words)."""
  n = len(body) // 8
  words = struct.unpack_from(f"<{n}Q", body, 0)
  spans, i = [], 0
  while i < n:
    if ((words[i] >> 48) & 0xFFFF) in _NPU_TARGETS:
      j = i
      while j < n and ((words[j] >> 48) & 0xFFFF) in _NPU_TARGETS:
        j += 1
      if j - i >= min_words:
        spans.append((i, j - i))
      i = j
    else:
      i += 1
  return words, spans


def _classify(regs):
  """Functional label for a DPU command block from its register signature."""
  m = {(t, r): v for t, r, _, v in regs}
  mode = m.get(_DPU_MODE)
  if mode is None:
    targets = {}
    for t, _, _, _ in regs:
      targets[t] = targets.get(t, 0) + 1
    return "OTHER", " ".join(f"{t}:{c}" for t, c in sorted(targets.items()))
  width = m.get(_DPU_WIDTH, 0) + 1
  # binary (2-operand) elementwise: live-patched form sets ERDMA data-mode bit30; the
  # stored template instead shows ALU data-format + a configured EW_CFG datapath.
  binary = ((m.get(_RDMA_EW, 0) & 0xC0000000) != 0
            or (mode == _ALU_FORMAT and m.get(_DPU_EWCFG, 0) != 0))
  mode_name = _MODE_NAMES.get(mode, f"mode=0x{mode:x}")
  kind = ("EW_BINARY" if binary else "COPY") + f"/{mode_name}"
  detail = f"w={width} out=0x{m.get(_DPU_OUT, 0):08x} in0=0x{m.get(_RDMA_SRC, 0):08x}"
  if binary:
    detail += f" in1=0x{m.get(_RDMA_SRC2, 0):08x}"
  return kind, detail


def build_command_queue(body: bytes):
  """Decode the regcmd region of a body into a list of command blocks (tiles).

  Each block is {index, word_offset, n_words, kind, detail, regs:[(target,off,name,value)]}.
  """
  words, spans = _regcmd_spans(body)
  queue = []
  for bi, (start, count) in enumerate(spans):
    regs = [decode_reg(words[start + k]) for k in range(count)]
    kind, detail = _classify(regs)
    queue.append({
      "index": bi,
      "word_offset": start,
      "byte_offset": start * 8,
      "n_words": count,
      "kind": kind,
      "detail": detail,
      "regs": regs,
    })
  return queue


# --------------------------------------------------------------------------- #
# top-level
# --------------------------------------------------------------------------- #
def decode_rknn(model):
  """Disassemble an .rknn (bytes or path) into its graph + NPU command queue.

  Returns a dict: version, body_size, trailer, tensors, nodes, command_queue.
  """
  data = model if isinstance(model, (bytes, bytearray)) else open(model, "rb").read()
  version, body, trailer = split_container(bytes(data))

  fb = _FB(body)
  root = fb.root()
  subgraphs = fb.vec_tables(root, 2)
  if not subgraphs:
    raise ValueError("no subgraphs in RKNN FlatBuffer body")
  sg = subgraphs[0]
  tensors = [_tensor(fb, p, i) for i, p in enumerate(fb.vec_tables(sg, 0))]
  nodes = [{"index": i, "op": fb.string(p, 1), "name": fb.string(p, 2)}
           for i, p in enumerate(fb.vec_tables(sg, 1))]

  return {
    "version": version,
    "body_size": len(body),
    "target_platform": fb.string(root, 8),
    "framework": fb.string(root, 9),
    "toolkit": fb.string(root, 7),
    "trailer": parse_trailer(trailer),
    "tensors": tensors,
    "nodes": nodes,
    "command_queue": build_command_queue(body),
  }


def print_rknn_disasm(decoded, regs: bool = False):
  """Pretty-print a decoded model: graph nodes, command tensors, and the command queue.

  Pass regs=True to also dump every decoded register of every command block.
  """
  d = decoded
  print(f"=== RKNN model (version {d['version']}, body {d['body_size']} bytes) ===")
  print(f"target={d['target_platform']} framework={d['framework']} toolkit={d['toolkit']}")

  print(f"\nnodes ({len(d['nodes'])}):")
  for n in d["nodes"]:
    print(f"  {n['index']:2d}: {str(n['op']):16s} {n['name']}")

  named = [t for t in d["tensors"] if t["name"]]
  print(f"\ntensors ({len(named)} named): name  native -> logical  size@offset")
  for t in named:
    off = "?" if t["offset"] is None else t["offset"]
    print(f"  [{t['index']:3d}] {str(t['name']):20s} {t['native']} -> {t['logical']}  "
          f"{t['size']}@{off}")

  q = d["command_queue"]
  print(f"\ncommand queue ({len(q)} blocks):")
  for blk in q:
    print(f"  block[{blk['index']:2d}] @word {blk['word_offset']:5d} "
          f"({blk['n_words']:3d} words)  {blk['kind']:18s} {blk['detail']}")
    if regs:
      for t, off, name, val in blk["regs"]:
        print(f"        {t:9s} 0x{off:04x} {name:32s} = 0x{val:08x} ({val})")
