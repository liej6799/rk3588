// Verify integer-tensor element-wise Add on the vendor NPU runtime.
// Feeds small integers in the model's reported dtype and checks integer a+b.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "rknn_api.h"

static void ck(int r, const char *w) {
  if (r < 0) { std::fprintf(stderr, "%s failed: %d\n", w, r); std::exit(1); }
}

static int elem_bytes(rknn_tensor_type t) {
  switch (t) {
    case RKNN_TENSOR_INT8: case RKNN_TENSOR_UINT8: case RKNN_TENSOR_BOOL: return 1;
    case RKNN_TENSOR_INT16: case RKNN_TENSOR_UINT16: case RKNN_TENSOR_FLOAT16: return 2;
    case RKNN_TENSOR_INT32: case RKNN_TENSOR_UINT32: case RKNN_TENSOR_FLOAT32: return 4;
    case RKNN_TENSOR_INT64: return 8;
    default: return 0;
  }
}

static void put(void *buf, int i, rknn_tensor_type t, long v) {
  switch (t) {
    case RKNN_TENSOR_INT8: case RKNN_TENSOR_UINT8: case RKNN_TENSOR_BOOL:
      ((uint8_t *)buf)[i] = (uint8_t)v; break;
    case RKNN_TENSOR_INT16: case RKNN_TENSOR_UINT16:
      ((uint16_t *)buf)[i] = (uint16_t)v; break;
    case RKNN_TENSOR_INT32: case RKNN_TENSOR_UINT32:
      ((uint32_t *)buf)[i] = (uint32_t)v; break;
    case RKNN_TENSOR_INT64:
      ((int64_t *)buf)[i] = (int64_t)v; break;
    default: break;
  }
}

static long get(void *buf, int i, rknn_tensor_type t) {
  switch (t) {
    case RKNN_TENSOR_INT8: return ((int8_t *)buf)[i];
    case RKNN_TENSOR_UINT8: case RKNN_TENSOR_BOOL: return ((uint8_t *)buf)[i];
    case RKNN_TENSOR_INT16: return ((int16_t *)buf)[i];
    case RKNN_TENSOR_UINT16: return ((uint16_t *)buf)[i];
    case RKNN_TENSOR_INT32: return ((int32_t *)buf)[i];
    case RKNN_TENSOR_UINT32: return ((uint32_t *)buf)[i];
    case RKNN_TENSOR_INT64: return (long)((int64_t *)buf)[i];
    default: return 0;
  }
}

int main(int argc, char **argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s MODEL.rknn\n", argv[0]); return 2; }
  rknn_context ctx = 0;
  ck(rknn_init(&ctx, (void *)argv[1], 0, 0, nullptr), "rknn_init");
  rknn_input_output_num io = {};
  ck(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "IN_OUT_NUM");
  rknn_tensor_attr a0 = {}; a0.index = 0;
  ck(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &a0, sizeof(a0)), "INPUT_ATTR");
  rknn_tensor_attr o0 = {}; o0.index = 0;
  ck(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &o0, sizeof(o0)), "OUTPUT_ATTR");
  uint32_t n = a0.n_elems;
  int eb = elem_bytes(a0.type);
  std::printf("type=%s n=%u in_size=%u out_size=%u\n",
              get_type_string(a0.type), n, a0.size, o0.size);
  if (!eb) { std::printf("unsupported type\n"); return 2; }

  std::vector<std::vector<uint8_t>> in(io.n_input, std::vector<uint8_t>(n * eb, 0));
  for (uint32_t i = 0; i < n; i++) {
    put(in[0].data(), i, a0.type, (i % 5));
    for (uint32_t j = 1; j < io.n_input; j++) put(in[j].data(), i, a0.type, (i % 3) + 1);
  }
  std::vector<rknn_input> inputs(io.n_input);
  std::memset(inputs.data(), 0, inputs.size() * sizeof(rknn_input));
  for (uint32_t i = 0; i < io.n_input; i++) {
    inputs[i].index = i; inputs[i].type = a0.type; inputs[i].fmt = a0.fmt;
    inputs[i].buf = in[i].data(); inputs[i].size = n * eb;
  }
  ck(rknn_inputs_set(ctx, io.n_input, inputs.data()), "inputs_set");
  ck(rknn_run(ctx, nullptr), "run");

  rknn_output out = {}; out.index = 0; out.want_float = 0;
  ck(rknn_outputs_get(ctx, 1, &out, nullptr), "outputs_get");

  int bad = 0; long fg = 0, fw = 0; uint32_t fi = 0;
  for (uint32_t i = 0; i < n; i++) {
    long want = get(in[0].data(), i, a0.type);
    for (uint32_t j = 1; j < io.n_input; j++) want += get(in[j].data(), i, a0.type);
    long g = get(out.buf, i, o0.type);
    if (g != want) { if (!bad) { fi = i; fg = g; fw = want; } bad++; }
  }
  std::printf("Add(int) mismatches=%d %s", bad, bad ? "FAIL" : "PASS");
  if (bad) std::printf("  first @%u got %ld want %ld", fi, fg, fw);
  std::printf("\n");
  rknn_outputs_release(ctx, 1, &out);
  rknn_destroy(ctx);
  return bad ? 1 : 0;
}
