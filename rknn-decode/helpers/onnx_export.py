"""Helper: build an ONNX graph directly from a fully-unrolled UOp list.

The unrolled graph is scalar/elementwise: each output element is a small expression
over gathered input elements. We translate it literally:

    LOAD(INDEX(param, off))  ->  Gather(input, [off])      # one [1] element
    ADD/MUL/SUB/MAX(...)     ->  Add/Mul/Sub/Max node
    CONST                    ->  initializer
    the STOREs               ->  Concat of the per-element results, in output order

onnx is imported lazily so the rest of the project stays dependency-free.
"""
from .uop import Ops

_ONNX_ELEM = {"half": 10, "float": 1}            # TensorProto.FLOAT16 / FLOAT
_ALU_ONNX = {Ops.ADD: "Add", Ops.MUL: "Mul", Ops.SUB: "Sub", Ops.MAX: "Max"}

# Handles both the scalar full-unroll (LOAD(INDEX)->Gather, op->node, STOREs->Concat) and
# the vectorized UPCAST form (CAST vec-ptr, vec LOAD, GEP, STACK): a vec(n) LOAD becomes a
# Gather of n contiguous indices, GEP a Gather of one lane, STACK a Concat.

def uops_to_onnx(uops: list, name: str = "kernel"):
  import numpy as np
  from onnx import helper, numpy_helper, TensorProto

  assert not any(u.op is Ops.RANGE for u in uops), "uops_to_onnx needs a fully-unrolled (range-free) graph"
  params = {u.arg: u for u in uops if u.op is Ops.PARAM}
  out_param = params[0]
  size = out_param.dtype.size
  elem = out_param.dtype.base.name
  onnx_dt = _ONNX_ELEM[elem]
  np_dt = {"half": np.float16, "float": np.float32}[elem]

  inputs = [helper.make_tensor_value_info(f"input{a}", onnx_dt, [size]) for a in sorted(params) if a != 0]
  output = helper.make_tensor_value_info("output", onnx_dt, [size])

  nodes, inits, memo, ctr = [], [], {}, [0]
  def newname(p): ctr[0] += 1; return f"{p}{ctr[0]}"

  def _index_of(u):
    """Unwrap an (optionally CAST) INDEX(param, const) -> (param, offset)."""
    if u.op is Ops.CAST: u = u.src[0]
    assert u.op is Ops.INDEX and u.src[0].op is Ops.PARAM and u.src[1].op is Ops.CONST, \
      "expected (CAST of) INDEX(param, const)"
    return u.src[0], u.src[1].arg

  def emit(u) -> str:
    if id(u) in memo: return memo[id(u)]
    if u.op is Ops.CAST:                                    # vec-ptr cast: a no-op for values
      out = emit(u.src[0])
    elif u.op is Ops.LOAD:                                  # scalar or vec(n): Gather n contiguous
      param, off = _index_of(u.src[0])
      n = u.dtype.count
      iname = newname("idx"); inits.append(helper.make_tensor(iname, TensorProto.INT64, [n], list(range(off, off + n))))
      out = newname("ld"); nodes.append(helper.make_node("Gather", [f"input{param.arg}", iname], [out], axis=0))
    elif u.op is Ops.GEP:                                   # pick one lane out of a vector
      iname = newname("idx"); inits.append(helper.make_tensor(iname, TensorProto.INT64, [1], [u.arg[0]]))
      out = newname("gep"); nodes.append(helper.make_node("Gather", [emit(u.src[0]), iname], [out], axis=0))
    elif u.op is Ops.STACK:                                 # pack lanes back into a vector
      out = newname("stk"); nodes.append(helper.make_node("Concat", [emit(s) for s in u.src], [out], axis=0))
    elif u.op is Ops.MULACC:                               # FMA a*b+c -> Mul then Add (no ONNX FMA)
      x, y, z = (emit(s) for s in u.src)
      t = newname("mul"); nodes.append(helper.make_node("Mul", [x, y], [t]))
      out = newname("acc"); nodes.append(helper.make_node("Add", [t, z], [out]))
    elif u.op in _ALU_ONNX:
      out = newname("alu"); nodes.append(helper.make_node(_ALU_ONNX[u.op], [emit(s) for s in u.src], [out]))
    elif u.op is Ops.CONST:
      out = newname("c"); inits.append(numpy_helper.from_array(np.array([u.arg], dtype=np_dt), out))
    else:
      raise NotImplementedError(f"uops_to_onnx: unsupported op {u.op}")
    memo[id(u)] = out
    return out

  # each STORE places its value (1 or n elements) at a constant output offset; concat in order.
  # a vec store covers `count` contiguous offsets, so the offsets must tile 0..size-1.
  stores = [st for st in uops if st.op is Ops.STORE]
  placed = sorted((_index_of(st.src[0])[1], emit(st.src[1]), st.src[1].dtype.count) for st in stores)
  pos = 0
  for off, _name, count in placed:
    assert off == pos, f"store offsets must tile contiguously from 0 (gap at {pos})"
    pos += count
  assert pos == size, f"stores cover {pos} elements, expected {size}"
  nodes.append(helper.make_node("Concat", [name for _off, name, _n in placed], ["output"], axis=0))

  graph = helper.make_graph(nodes, name, inputs, [output], initializer=inits)
  return helper.make_model(graph, producer_name="rknn-decode", opset_imports=[helper.make_opsetid("", 13)])

def print_onnx_graph(model) -> None:
  """Print an ONNX ModelProto as a readable node listing (op_type, inputs -> outputs).

  Initializer (constant) inputs are marked with a trailing '*'.
  """
  g = model.graph
  inits = {i.name for i in g.initializer}
  print(f"=== ONNX graph '{g.name}' ({len(g.node)} nodes) ===")
  print(f"inputs : {[i.name for i in g.input]}")
  print(f"outputs: {[o.name for o in g.output]}")
  for i, n in enumerate(g.node):
    ins = [f"{s}*" if s in inits else s for s in n.input]      # '*' = initializer/constant
    print(f"{i:3d}: {n.op_type:8s} {ins} -> {list(n.output)}")
