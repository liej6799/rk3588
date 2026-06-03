"""Helper: recreate a .rknn container from a decoded representation.

The inverse of `rknn_decode`. It demonstrates that the parts the decoder understands
are losslessly reconstructable:

  * the 64-byte container header is rebuilt **from scratch** (magic + version + bodySize),
  * the embedded NPU register-command stream is re-emitted from the decoded command
    queue (each 64-bit word repacked as target<<48 | value<<16 | reg).

`recreate_rknn(model)` decodes a model and rebuilds the bytes; for a library-generated
model the result is **byte-identical** to the original (see tests/test_recreate.py).

Scope / honesty: the FlatBuffer body's metadata (vtables, tensor table, JSON attrs) and
the trailing task tensor are carried through verbatim — those are the compiler-owned
parts. Synthesising them without the toolkit is a separate, much larger effort (the
memory plan + flatc serialisation); a fully toolkit-free body can only be made
*functionally* (NPU-) equivalent, not byte-identical, since it embeds the compiler's
version string and its own memory-plan offsets.
"""
import struct

from .rknn_decode import HEADER_SIZE, MAGIC, TARGET_MAP, decode_rknn, split_container

_CODE_BY_NAME = {name: code for code, name in TARGET_MAP.items()}


def encode_reg(target_name: str, reg_off: int, value: int) -> int:
  """Inverse of decode_reg: pack (target, reg, value) into a 64-bit regcmd word."""
  code = _CODE_BY_NAME[target_name]
  return (code << 48) | ((value & 0xFFFFFFFF) << 16) | (reg_off & 0xFFFF)


def encode_container(version: int, body: bytes, trailer: bytes) -> bytes:
  """Rebuild a .rknn container from its parts. The 64-byte header is built from scratch."""
  h = bytearray(HEADER_SIZE)
  h[0:4] = MAGIC
  struct.pack_into("<Q", h, 0x08, version)
  struct.pack_into("<Q", h, 0x10, len(body))
  return bytes(h) + body + trailer


def reemit_command_stream(body: bytes, command_queue) -> bytes:
  """Rewrite the regcmd region of `body` from a decoded command queue.

  Each block's decoded registers are repacked into 64-bit words and written back at the
  block's byte offset. Words between blocks (PC control / gaps) are left as in `body`.
  Returns the new body. With an unmodified queue this reproduces `body` exactly; edit a
  block's `regs` first to author a modified command stream.
  """
  out = bytearray(body)
  for blk in command_queue:
    words = [encode_reg(t, off, val) for t, off, _name, val in blk["regs"]]
    struct.pack_into(f"<{len(words)}Q", out, blk["byte_offset"], *words)
  return bytes(out)


def recreate_rknn(model) -> bytes:
  """Decode a .rknn (bytes or path) and rebuild it from the decoded representation.

  Header is rebuilt from scratch; the command stream is re-emitted from the decoded
  command queue; FlatBuffer metadata + trailer are carried through. For a
  library-generated model the result is byte-identical to the input.
  """
  data = model if isinstance(model, (bytes, bytearray)) else open(model, "rb").read()
  data = bytes(data)
  version, body, trailer = split_container(data)
  decoded = decode_rknn(data)
  body = reemit_command_stream(body, decoded["command_queue"])
  return encode_container(version, body, trailer)
