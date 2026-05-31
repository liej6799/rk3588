#!/usr/bin/env python3
"""Convert between a raw FlatBuffer body and a runnable .rknn container.

An .rknn (RKNPU2, toolkit 2.3.2) is three concatenated sections:

    [0x00] 64-byte header
           0x00  b"RKNN" magic (+4 zero pad)
           0x08  u64 version  (= 6)
           0x10  u64 bodySize (= length of the FlatBuffer body)
           0x18..0x40  zero
    [0x40] FlatBuffer body            (bodySize bytes) -- the model graph
    [end]  u64 jsonLen + JSON config  (graph "connection"/io metadata)

The FlatBuffer body alone is NOT a valid .rknn: it needs the header (whose
bodySize the native parseRKNN validates) and the trailing JSON config. This
tool rebuilds the header from scratch, so the output is independent of any
existing .rknn except for the two payload sections.

Subcommands
    extract  RKNN  --body B.fb --trailer T.bin   split a .rknn into its body + trailer
    build    --body B.fb --trailer T.bin  -o OUT  wrap body + trailer into a .rknn
"""
import argparse
import struct
import sys
from pathlib import Path

HEADER_SIZE = 0x40
MAGIC = b"RKNN"
DEFAULT_VERSION = 6


def read_header(data):
  if data[:4] != MAGIC:
    raise ValueError("not an RKNN container (bad magic)")
  version = struct.unpack_from("<Q", data, 0x08)[0]
  body_size = struct.unpack_from("<Q", data, 0x10)[0]
  return version, body_size


def build_header(version, body_size):
  h = bytearray(HEADER_SIZE)
  h[0:4] = MAGIC
  struct.pack_into("<Q", h, 0x08, version)
  struct.pack_into("<Q", h, 0x10, body_size)
  return bytes(h)


def cmd_extract(args):
  data = Path(args.rknn).read_bytes()
  version, body_size = read_header(data)
  body = data[HEADER_SIZE:HEADER_SIZE + body_size]
  trailer = data[HEADER_SIZE + body_size:]
  Path(args.body).write_bytes(body)
  Path(args.trailer).write_bytes(trailer)
  print(f"extracted from {args.rknn} (version {version}):")
  print(f"  body    -> {args.body}    ({len(body)} bytes)")
  print(f"  trailer -> {args.trailer} ({len(trailer)} bytes)")


def cmd_build(args):
  body = Path(args.body).read_bytes()
  trailer = Path(args.trailer).read_bytes()
  header = build_header(args.version, len(body))
  out = header + body + trailer
  Path(args.output).write_bytes(out)
  # sanity: re-read our own header
  v, bs = read_header(out)
  assert bs == len(body) and v == args.version
  print(f"built {args.output}: {len(out)} bytes "
        f"(header {HEADER_SIZE} + body {len(body)} + trailer {len(trailer)}), "
        f"version {v}, bodySize {bs}")


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)

  pe = sub.add_parser("extract", help="split a .rknn into FlatBuffer body + trailer")
  pe.add_argument("rknn")
  pe.add_argument("--body", required=True)
  pe.add_argument("--trailer", required=True)
  pe.set_defaults(func=cmd_extract)

  pb = sub.add_parser("build", help="wrap a FlatBuffer body + trailer into a .rknn")
  pb.add_argument("--body", required=True)
  pb.add_argument("--trailer", required=True)
  pb.add_argument("-o", "--output", required=True)
  pb.add_argument("--version", type=int, default=DEFAULT_VERSION)
  pb.set_defaults(func=cmd_build)

  args = ap.parse_args()
  args.func(args)
  return 0


if __name__ == "__main__":
  sys.exit(main())
