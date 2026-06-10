#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "rknn_api.h"

static void ck(int r, const char *w) {
  if (r < 0) { fprintf(stderr, "%s failed: %d\n", w, r); exit(1); }
}

static void f16_to_bytes(uint16_t h, uint8_t *out) {
  out[0] = h & 0xFF;
  out[1] = (h >> 8) & 0xFF;
}

static uint16_t float_to_f16(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = (x >> 13) & 0x3FF;
  if (exp <= 0) { mant |= 0x400; int shift = 1 - exp; mant >>= shift; exp = 0; }
  if (exp > 31) exp = 31;
  return sign | ((exp & 0x1F) << 10) | mant;
}

static float f16_to_float(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0) { exp = 1; }
  exp = exp - 15 + 127;
  uint32_t f = sign | (exp << 23) | (mant << 13);
  float r;
  memcpy(&r, &f, 4);
  return r;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s MODEL.rknn\n", argv[0]); return 2; }

  rknn_context ctx = 0;
  ck(rknn_init(&ctx, (void *)argv[1], 0, 0, nullptr), "rknn_init");

  rknn_input_output_num io = {};
  ck(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "IN_OUT_NUM");

  rknn_tensor_attr a0 = {}; a0.index = 0;
  ck(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &a0, sizeof(a0)), "INPUT_ATTR");
  rknn_tensor_attr o0 = {}; o0.index = 0;
  ck(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &o0, sizeof(o0)), "OUTPUT_ATTR");

  uint32_t n = a0.n_elems;
  int is_fp16 = (a0.type == RKNN_TENSOR_FLOAT16);
  int eb = is_fp16 ? 2 : 4;

  printf("model=%s type=%s n=%u in_size=%u out_size=%u n_inputs=%u\n",
         argv[1], get_type_string(a0.type), n, a0.size, o0.size, io.n_input);

  std::vector<uint8_t> in0(n * eb, 0), in1(n * eb, 0);
  for (uint32_t i = 0; i < n; i++) {
    if (is_fp16) {
      float va = (float)(i % 100) * 3.0f;
      float vb = (float)(i % 100) * 7.0f;
      uint16_t ha = float_to_f16(va);
      uint16_t hb = float_to_f16(vb);
      f16_to_bytes(ha, &in0[i * 2]);
      f16_to_bytes(hb, &in1[i * 2]);
    } else {
      int32_t va = (i % 100) * 3;
      int32_t vb = (i % 100) * 7;
      memcpy(&in0[i * 4], &va, 4);
      memcpy(&in1[i * 4], &vb, 4);
    }
  }

  rknn_input inputs[2] = {};
  inputs[0].index = 0;
  inputs[0].type = a0.type;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].buf = in0.data();
  inputs[0].size = n * eb;
  inputs[1].index = 1;
  inputs[1].type = a0.type;
  inputs[1].fmt = RKNN_TENSOR_NHWC;
  inputs[1].buf = in1.data();
  inputs[1].size = n * eb;

  ck(rknn_inputs_set(ctx, io.n_input, inputs), "inputs_set");
  ck(rknn_run(ctx, nullptr), "run");

  rknn_output out = {}; out.index = 0; out.want_float = 0;
  ck(rknn_outputs_get(ctx, 1, &out, nullptr), "outputs_get");

  int bad = 0;
  for (uint32_t i = 0; i < n; i++) {
    float got, want;
    if (is_fp16) {
      uint16_t hg;
      memcpy(&hg, (uint8_t *)out.buf + i * 2, 2);
      got = f16_to_float(hg);
      uint16_t ha, hb;
      memcpy(&ha, &in0[i * 2], 2);
      memcpy(&hb, &in1[i * 2], 2);
      want = f16_to_float(ha) + f16_to_float(hb);
    } else {
      int32_t g;
      memcpy(&g, (uint8_t *)out.buf + i * 4, 4);
      got = (float)g;
      int32_t va, vb;
      memcpy(&va, &in0[i * 4], 4);
      memcpy(&vb, &in1[i * 4], 4);
      want = (float)(va + vb);
    }
    float diff = got - want;
    if (diff < 0) diff = -diff;
    if (diff > 0.5f) {
      if (bad < 5) printf("  [%u] got=%.1f want=%.1f\n", i, got, want);
      bad++;
    }
  }

  printf("%s mismatches=%d %s\n", argv[1], bad, bad ? "FAIL" : "PASS");
  rknn_outputs_release(ctx, 1, &out);
  rknn_destroy(ctx);
  return bad ? 1 : 0;
}
