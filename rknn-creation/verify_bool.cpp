// Verify a boolean-tensor element-wise model on the vendor NPU runtime.
// Feeds 1-byte bool inputs, reads 1-byte bool output, and reports what the
// hardware actually computed for each element-wise op semantics it might match.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "rknn_api.h"

static void check(int r, const char *w) {
  if (r < 0) { std::fprintf(stderr, "%s failed: %d\n", w, r); std::exit(1); }
}

int main(int argc, char **argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s MODEL.rknn\n", argv[0]); return 2; }
  rknn_context ctx = 0;
  check(rknn_init(&ctx, (void *)argv[1], 0, 0, nullptr), "rknn_init");

  rknn_input_output_num io = {};
  check(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "IN_OUT_NUM");

  rknn_tensor_attr a0 = {}; a0.index = 0;
  check(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &a0, sizeof(a0)), "INPUT_ATTR");
  uint32_t n = a0.n_elems;
  std::printf("inputs=%u outputs=%u n_elems=%u in_type=%s\n",
              io.n_input, io.n_output, n, get_type_string(a0.type));

  // Build deterministic boolean inputs covering all 4 combinations of (a,b).
  std::vector<std::vector<uint8_t>> in(io.n_input, std::vector<uint8_t>(n));
  for (uint32_t i = 0; i < n; i++) {
    in[0][i] = (i & 1) ? 1 : 0;
    if (io.n_input > 1) in[1][i] = (i & 2) ? 1 : 0;
    for (uint32_t j = 2; j < io.n_input; j++) in[j][i] = (i & (1 << j)) ? 1 : 0;
  }

  std::vector<rknn_input> inputs(io.n_input);
  std::memset(inputs.data(), 0, inputs.size() * sizeof(rknn_input));
  for (uint32_t i = 0; i < io.n_input; i++) {
    inputs[i].index = i;
    inputs[i].type = RKNN_TENSOR_BOOL;
    inputs[i].fmt = a0.fmt;
    inputs[i].buf = in[i].data();
    inputs[i].size = n;  // 1 byte per bool elem
  }
  check(rknn_inputs_set(ctx, io.n_input, inputs.data()), "rknn_inputs_set");
  check(rknn_run(ctx, nullptr), "rknn_run");

  rknn_output out = {};
  out.index = 0;
  out.want_float = 0;            // raw bool bytes
  check(rknn_outputs_get(ctx, 1, &out, nullptr), "rknn_outputs_get");
  uint8_t *got = static_cast<uint8_t *>(out.buf);

  // Compare against candidate boolean semantics.
  int mismatch_and = 0, mismatch_or = 0, mismatch_xor = 0, mismatch_add = 0;
  for (uint32_t i = 0; i < n; i++) {
    int a = in[0][i] ? 1 : 0;
    int b = (io.n_input > 1) ? (in[1][i] ? 1 : 0) : 0;
    int g = got[i] ? 1 : 0;
    if (g != (a & b)) mismatch_and++;
    if (g != (a | b)) mismatch_or++;
    if (g != (a ^ b)) mismatch_xor++;
    if (g != ((a + b) ? 1 : 0)) mismatch_add++;
  }
  std::printf("first 8 out bytes:");
  for (uint32_t i = 0; i < n && i < 8; i++) std::printf(" %d", got[i]);
  std::printf("\n");
  std::printf("mismatches vs AND=%d OR=%d XOR=%d ADD(nonzero)=%d\n",
              mismatch_and, mismatch_or, mismatch_xor, mismatch_add);

  rknn_outputs_release(ctx, 1, &out);
  rknn_destroy(ctx);
  return 0;
}
