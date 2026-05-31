// Run any 2-input/1-output fp16 elementwise-add .rknn through the VENDOR
// runtime C API (rknn_init/rknn_inputs_set/rknn_run/rknn_outputs_get).
// Sizes come from the queried tensor attrs, so it works for 10x10, 1024, etc.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "rknn_api.h"

static void check(int ret, const char *what) {
  if (ret < 0) { std::fprintf(stderr, "%s failed: %d\n", what, ret); std::exit(1); }
}

static uint16_t f2h(float f) {
  uint32_t x; std::memcpy(&x, &f, sizeof(x));
  uint32_t sign = (x >> 16) & 0x8000;
  int exp = int((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = x & 0x7fffff;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7c00;
  return uint16_t(sign | (uint32_t(exp) << 10) | (mant >> 13));
}

static void dump_attr(const char *label, const rknn_tensor_attr &a) {
  std::printf("%s[%u] name=%s dims=[", label, a.index, a.name);
  for (uint32_t i = 0; i < a.n_dims; i++) std::printf("%s%d", i ? "," : "", a.dims[i]);
  std::printf("] elems=%d size=%d fmt=%s type=%s\n", a.n_elems, a.size,
              get_format_string(a.fmt), get_type_string(a.type));
}

int main(int argc, char **argv) {
  const char *model = argc > 1 ? argv[1] : "/data/test/fp16_add_10x10.rknn";
  std::printf("== vendor rknn_run: %s ==\n", model);

  rknn_context ctx = 0;
  check(rknn_init(&ctx, (void *)model, 0, 0, nullptr), "rknn_init");
  check(rknn_set_core_mask(ctx, RKNN_NPU_CORE_0), "rknn_set_core_mask");

  rknn_input_output_num io = {};
  check(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "QUERY_IN_OUT_NUM");
  std::printf("inputs=%u outputs=%u\n", io.n_input, io.n_output);

  std::vector<rknn_tensor_attr> in(io.n_input), out(io.n_output);
  for (uint32_t i = 0; i < io.n_input; i++) {
    in[i].index = i;
    check(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in[i], sizeof(rknn_tensor_attr)), "INPUT_ATTR");
    dump_attr("input", in[i]);
  }
  for (uint32_t i = 0; i < io.n_output; i++) {
    out[i].index = i;
    check(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out[i], sizeof(rknn_tensor_attr)), "OUTPUT_ATTR");
    dump_attr("output", out[i]);
  }

  uint32_t n = in[0].n_elems;
  std::vector<uint16_t> ah(n), bh(n);
  for (uint32_t i = 0; i < n; i++) { ah[i] = f2h(float(i)); bh[i] = f2h(10.0f); }

  std::vector<rknn_input> inputs(io.n_input);
  std::memset(inputs.data(), 0, inputs.size() * sizeof(rknn_input));
  for (uint32_t i = 0; i < io.n_input; i++) {
    inputs[i].index = i;
    inputs[i].type = RKNN_TENSOR_FLOAT16;
    inputs[i].fmt = in[i].fmt;
    inputs[i].pass_through = 0;
    inputs[i].buf = (i == 0) ? (void *)ah.data() : (void *)bh.data();
    inputs[i].size = n * sizeof(uint16_t);
  }
  check(rknn_inputs_set(ctx, io.n_input, inputs.data()), "rknn_inputs_set");

  check(rknn_run(ctx, nullptr), "rknn_run");

  rknn_output o = {};
  o.index = 0; o.want_float = 1; o.is_prealloc = 0;
  check(rknn_outputs_get(ctx, 1, &o, nullptr), "rknn_outputs_get");

  float *r = static_cast<float *>(o.buf);
  bool ok = true;
  for (uint32_t i = 0; i < n; i++)
    if (std::fabs(r[i] - (float(i) + 10.0f)) > 1e-2f) { ok = false; break; }

  std::printf("correct %s\nfirst", ok ? "true" : "false");
  for (uint32_t i = 0; i < std::min<uint32_t>(8, n); i++) std::printf(" %.0f", r[i]);
  std::printf("\nlast");
  for (uint32_t i = (n > 8 ? n - 8 : 0); i < n; i++) std::printf(" %.0f", r[i]);
  std::printf("\n");

  rknn_outputs_release(ctx, 1, &o);
  rknn_destroy(ctx);
  return ok ? 0 : 1;
}
