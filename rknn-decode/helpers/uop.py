"""Minimal, self-contained UOp library for the rknn-decode project.

This is a trimmed-down, dependency-free reimplementation of the pieces of
tinygrad's `tinygrad/uop/ops.py` + `tinygrad/dtype.py` that we actually need to
represent and manipulate linearized UOps. It intentionally does NOT import
tinygrad so this project can stand on its own.

Provided: Ops, dtypes/DType/PtrDType, AxisType, KernelInfo, and a small UOp.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto

# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------
class Ops(Enum):
  # buffers / values
  PARAM = auto(); CONST = auto(); DEFINE_VAR = auto()
  # control / indexing
  RANGE = auto(); END = auto(); SINK = auto(); INDEX = auto(); GEP = auto(); CAST = auto(); STACK = auto()
  # memory
  LOAD = auto(); STORE = auto()
  # alu
  ADD = auto(); SUB = auto(); MUL = auto(); SHL = auto(); MAX = auto(); CMPLT = auto()

class AxisType(Enum):
  GLOBAL = auto(); LOCAL = auto(); LOOP = auto(); REDUCE = auto(); UPCAST = auto(); UNROLL = auto()
  def __repr__(self) -> str: return f"AxisType.{self.name}"

# ---------------------------------------------------------------------------
# dtypes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DType:
  name: str
  itemsize: int
  count: int = 1                      # vector width (1 == scalar)
  @property
  def base(self) -> "DType": return DType(self.name, self.itemsize // self.count) if self.count > 1 else self
  @property
  def size(self) -> int: return self.count
  def vec(self, n: int) -> "DType": return DType(self.name, self.itemsize * n, n)
  def ptr(self, size: int) -> "PtrDType": return PtrDType(self, size)
  def __repr__(self) -> str: return f"dtypes.{self.name}" + (f".vec({self.count})" if self.count > 1 else "")

@dataclass(frozen=True)
class PtrDType:
  pointee: DType
  size: int
  @property
  def base(self) -> DType: return self.pointee.base
  def __repr__(self) -> str: return f"{self.pointee!r}.ptr({self.size})"

class dtypes:
  void  = DType("void", 0)
  bool  = DType("bool", 1)
  int   = DType("int", 4)
  half  = DType("half", 2)
  float = DType("float", 4)

# ---------------------------------------------------------------------------
# kernel metadata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KernelInfo:
  name: str = ""
  axis_types: tuple = ()
  dont_use_locals: bool = False
  applied_opts: tuple = ()
  opts_to_apply: object = None
  estimates: object = None
  beam: int = 0

# ---------------------------------------------------------------------------
# UOp
# ---------------------------------------------------------------------------
class UOp:
  __slots__ = ("op", "dtype", "src", "arg")
  def __init__(self, op: Ops, dtype: "DType|PtrDType" = dtypes.void, src: tuple = (), arg=None):
    self.op, self.dtype, self.src, self.arg = op, dtype, tuple(src), arg

  # --- builders (mirror the tinygrad helpers we use) ---
  @staticmethod
  def const(dtype, b) -> "UOp": return UOp(Ops.CONST, dtype, (), b)
  @staticmethod
  def sink(*srcs: "UOp", arg=None) -> "UOp": return UOp(Ops.SINK, dtypes.void, tuple(s for s in srcs if s is not None), arg)
  @staticmethod
  def range(end: int, axis_id: int, axis_type: AxisType) -> "UOp":
    return UOp(Ops.RANGE, dtypes.int, (UOp.const(dtypes.int, end),), (axis_id, axis_type))
  def index(self, idx: "UOp") -> "UOp": return UOp(Ops.INDEX, self.dtype, (self, idx))
  def load(self) -> "UOp": return UOp(Ops.LOAD, self.dtype.base if isinstance(self.dtype, PtrDType) else self.dtype, (self,))
  def store(self, val: "UOp") -> "UOp": return UOp(Ops.STORE, dtypes.void, (self, val))
  def end(self, *rngs: "UOp") -> "UOp": return UOp(Ops.END, dtypes.void, (self, *rngs))

  # --- graph utilities ---
  def toposort(self) -> list["UOp"]:
    seen: set[int] = set(); out: list[UOp] = []
    def visit(u: UOp):
      if id(u) in seen: return
      seen.add(id(u))
      for s in u.src: visit(s)
      out.append(u)
    visit(self)
    return out

  def _rebuild(self, src: tuple) -> "UOp":
    # reuse self when sources are unchanged, so untouched subgraphs (e.g. PARAMs) stay shared
    if len(src) == len(self.src) and all(a is b for a, b in zip(src, self.src)): return self
    return UOp(self.op, self.dtype, src, self.arg)

  def substitute(self, dvars: dict["UOp", "UOp"]) -> "UOp":
    memo: dict[int, UOp] = {}
    def go(u: UOp) -> UOp:
      for k, v in dvars.items():
        if k is u: return v
      if id(u) in memo: return memo[id(u)]
      memo[id(u)] = new = u._rebuild(tuple(go(s) for s in u.src))
      return new
    return go(self)

  def simplify(self) -> "UOp":
    # bottom-up constant folding of integer ALU ops (enough to collapse unrolled index math)
    memo: dict[int, UOp] = {}
    def go(u: UOp) -> UOp:
      if id(u) in memo: return memo[id(u)]
      u2 = u._rebuild(tuple(go(s) for s in u.src))
      if u2.op in _ALU_FOLD and all(s.op is Ops.CONST for s in u2.src):
        u2 = UOp.const(u2.dtype, _ALU_FOLD[u2.op](*[s.arg for s in u2.src]))
      memo[id(u)] = u2
      return u2
    return go(self)

  def __repr__(self) -> str: return f"UOp({self.op.name}, {self.dtype!r}, arg={self.arg!r})"

# integer ALU folding table used by UOp.simplify()
_ALU_FOLD = {
  Ops.ADD: lambda a, b: a + b,
  Ops.SUB: lambda a, b: a - b,
  Ops.MUL: lambda a, b: a * b,
  Ops.SHL: lambda a, b: a << b,
  Ops.MAX: max,
}
