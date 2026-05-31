#!/usr/bin/env python3
"""Inspect the build-time queue embedded in an RKNN model.

This is intentionally offline: it parses the exported .rknn FlatBuffer payload
and does not call init_runtime/inference or submit anything to the kernel.
"""
import argparse
import json
import struct
from pathlib import Path


ROOT_OFFSET = 0x40


class FB:
  def __init__(self, data):
    self.data = data

  def u16(self, off):
    return struct.unpack_from("<H", self.data, off)[0]

  def u32(self, off):
    return struct.unpack_from("<I", self.data, off)[0]

  def i32(self, off):
    return struct.unpack_from("<i", self.data, off)[0]

  def table_info(self, pos):
    vtable = pos - self.i32(pos)
    vtable_size = self.u16(vtable)
    object_size = self.u16(vtable + 2)
    return vtable, vtable_size, object_size

  def field_offset(self, pos, field):
    vtable, vtable_size, _ = self.table_info(pos)
    entry = vtable + 4 + field * 2
    if entry + 2 > vtable + vtable_size:
      return 0
    return self.u16(entry)

  def field_abs(self, pos, field):
    rel = self.field_offset(pos, field)
    return pos + rel if rel else None

  def scalar_u32(self, pos, field, default=0):
    abspos = self.field_abs(pos, field)
    return self.u32(abspos) if abspos is not None else default

  def scalar_u8(self, pos, field, default=0):
    abspos = self.field_abs(pos, field)
    return self.data[abspos] if abspos is not None else default

  def target(self, abspos):
    return abspos + self.u32(abspos)

  def string_at(self, off):
    n = self.u32(off)
    raw = self.data[off + 4:off + 4 + n]
    return raw.decode("utf-8", "replace")

  def string(self, pos, field):
    abspos = self.field_abs(pos, field)
    if abspos is None:
      return None
    return self.string_at(self.target(abspos))

  def table_dump(self, pos, max_fields=20):
    vtable, vtable_size, object_size = self.table_info(pos)
    fields = []
    for field in range(max_fields):
      rel = self.field_offset(pos, field)
      if rel:
        abspos = pos + rel
        raw = self.data[abspos:abspos + 8]
        fields.append({
          "field": field,
          "rel": rel,
          "abs": abspos,
          "u8": raw[0],
          "u32": self.u32(abspos) if abspos + 4 <= len(self.data) else None,
          "bytes": raw.hex(),
        })
    return {
      "table_offset": pos,
      "vtable_offset": vtable,
      "vtable_size": vtable_size,
      "object_size": object_size,
      "object_bytes": self.data[pos:pos + object_size].hex(),
      "fields": fields,
    }

  def vector_target(self, pos, field):
    abspos = self.field_abs(pos, field)
    return self.target(abspos) if abspos is not None else None

  def vector_table_positions(self, pos, field):
    vec = self.vector_target(pos, field)
    if vec is None:
      return []
    n = self.u32(vec)
    out = []
    for i in range(n):
      entry = vec + 4 + i * 4
      out.append(entry + self.u32(entry))
    return out

  def vector_u32(self, off):
    n = self.u32(off)
    return [self.u32(off + 4 + i * 4) for i in range(n)]

  def string_vector(self, pos, field):
    vec = self.vector_target(pos, field)
    if vec is None:
      return []
    n = self.u32(vec)
    out = []
    for i in range(n):
      entry = vec + 4 + i * 4
      out.append(self.string_at(entry + self.u32(entry)))
    return out


def looks_like_rknn(data):
  return data.startswith(b"RKNN") and data[ROOT_OFFSET:ROOT_OFFSET + 4] == b"p\x00\x00\x00"


def root_pos(fb):
  return ROOT_OFFSET + fb.u32(ROOT_OFFSET)


def node_info(fb, pos, idx):
  info = {
    "index": idx,
    "table_offset": pos,
    "op": fb.string(pos, 1),
    "name": fb.string(pos, 2),
  }
  info["table"] = fb.table_dump(pos, max_fields=12)
  return info


def tensor_info(fb, pos, idx):
  info = {
    "index": idx,
    "table_offset": pos,
    "name": fb.string(pos, 5),
    "raw_type": fb.scalar_u8(pos, 0),
    "raw_target": fb.scalar_u8(pos, 1),
    "raw_dtype": fb.scalar_u8(pos, 2),
  }
  info["table"] = fb.table_dump(pos, max_fields=20)
  return info


def extract(path):
  data = Path(path).read_bytes()
  fb = FB(data)
  if not looks_like_rknn(data):
    raise SystemExit(f"{path}: not the expected RKNN container header")

  root = root_pos(fb)
  subgraphs = fb.vector_table_positions(root, 2)
  if not subgraphs:
    raise SystemExit("No subgraphs found in RKNN FlatBuffer payload")
  subgraph = subgraphs[0]

  tensors = [tensor_info(fb, p, i) for i, p in enumerate(fb.vector_table_positions(subgraph, 0))]
  nodes = [node_info(fb, p, i) for i, p in enumerate(fb.vector_table_positions(subgraph, 1))]
  command_tensors = [t for t in tensors if t["name"] in ("task", "regcmd")]

  return {
    "file": str(path),
    "size": len(data),
    "root_offset": root,
    "root": fb.table_dump(root, max_fields=16),
    "target": fb.string(root, 1),
    "target_platform": fb.string(root, 8),
    "framework": fb.string(root, 9),
    "toolkit": fb.string(root, 7),
    "subgraph_count": len(subgraphs),
    "subgraphs": [fb.table_dump(s, max_fields=16) for s in subgraphs],
    "nodes": nodes,
    "command_tensors": command_tensors,
    "npu_nodes": [n for n in nodes if n["op"] not in ("InputOperator", "OutputOperator", "Reshape")],
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("rknn", nargs="?", default="/data/test/fp16_add_10x10.rknn")
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args()

  info = extract(args.rknn)
  if args.json:
    print(json.dumps(info, indent=2))
    return

  print(f"file: {info['file']} ({info['size']} bytes)")
  print(f"target: {info['target']} / {info['target_platform']}")
  print(f"framework: {info['framework']}")
  print(f"toolkit: {info['toolkit']}")
  print()
  print("build-time execution queue:")
  for n in info["nodes"]:
    marker = "NPU" if n in info["npu_nodes"] else "CPU/meta"
    print(f"  {n['index']:2d}: {marker:8s} {n['op'] or '?':16s} {n['name'] or '?'}")
  print()
  print("command payload tensors:")
  for t in info["command_tensors"]:
    print(
      f"  tensor[{t['index']:2d}] {t['name']:6s} "
      f"raw_type={t['raw_type']} raw_target={t['raw_target']} raw_dtype={t['raw_dtype']}"
    )
  print()
  print("npu command candidates:")
  for n in info["npu_nodes"]:
    print(f"  node[{n['index']}] {n['op']} {n['name']}")


if __name__ == "__main__":
  main()
