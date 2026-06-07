// verify_hybrid.cpp - Verify a parallel CPU logical + NPU arithmetic RKNN model.
//
// Usage: verify_hybrid MODEL.rknn CPU_OP NPU_OP
//   CPU_OP: And | Or | Xor
//   NPU_OP: Add | Sub | Div
//
// Inputs are fixed to the mixed model convention:
//   a,b: bool/int8[4]   x,y: fp16[4]
// Outputs are fixed to out1 (bool) then out2 (float).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "rknn_api.h"

static uint16_t f2h(float f) {
  uint32_t x; std::memcpy(&x, &f, 4);
  uint32_t s = (x >> 16) & 0x8000;
  int e = int((x >> 23) & 0xff) - 127 + 15;
  uint32_t m = x & 0x7fffff;
  if (e <= 0) return s;
  if (e >= 31) return s | 0x7c00;
  return s | (uint32_t(e) << 10) | (m >> 13);
}

static int cpu_expected(const char* op, int a, int b) {
  if (!std::strcmp(op, "And")) return a & b;
  if (!std::strcmp(op, "Or")) return a | b;
  if (!std::strcmp(op, "Xor")) return a ^ b;
  std::fprintf(stderr, "unknown CPU_OP: %s\n", op);
  std::exit(2);
}

static float npu_expected(const char* op, float x, float y) {
  if (!std::strcmp(op, "Add")) return x + y;
  if (!std::strcmp(op, "Sub")) return x - y;
  if (!std::strcmp(op, "Div")) return x / y;
  std::fprintf(stderr, "unknown NPU_OP: %s\n", op);
  std::exit(2);
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s MODEL.rknn CPU_OP NPU_OP\n", argv[0]);
    return 2;
  }
  const char* cpu_op = argv[2];
  const char* npu_op = argv[3];

  FILE* fp = std::fopen(argv[1], "rb");
  if (!fp) { std::perror(argv[1]); return 2; }
  std::fseek(fp, 0, SEEK_END); long sz = std::ftell(fp); std::fseek(fp, 0, SEEK_SET);
  void* model = std::malloc(sz); std::fread(model, 1, sz, fp); std::fclose(fp);

  rknn_context ctx = 0;
  int ret = rknn_init(&ctx, model, sz, 0, NULL);
  std::printf("rknn_init: %d\n", ret);
  if (ret < 0) { std::free(model); return 1; }

  rknn_input_output_num io; std::memset(&io, 0, sizeof(io));
  rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
  std::printf("inputs=%u outputs=%u\n", io.n_input, io.n_output);

  int8_t a[4] = {1, 0, 1, 1};
  int8_t b[4] = {1, 1, 0, 1};
  float xf[4] = {1.5f, 2.0f, 0.25f, 3.0f};
  float yf[4] = {0.5f, 1.0f, 4.0f, 2.0f};
  uint16_t xh[4], yh[4];
  for (int i = 0; i < 4; i++) { xh[i] = f2h(xf[i]); yh[i] = f2h(yf[i]); }

  rknn_input ins[4]; std::memset(ins, 0, sizeof(ins));
  ins[0].index = 0; ins[0].type = RKNN_TENSOR_INT8; ins[0].size = 4;
  ins[0].fmt = RKNN_TENSOR_NCHW; ins[0].buf = a; ins[0].pass_through = 1;
  ins[1].index = 1; ins[1].type = RKNN_TENSOR_INT8; ins[1].size = 4;
  ins[1].fmt = RKNN_TENSOR_NCHW; ins[1].buf = b; ins[1].pass_through = 1;
  ins[2].index = 2; ins[2].type = RKNN_TENSOR_FLOAT16; ins[2].size = 8;
  ins[2].fmt = RKNN_TENSOR_NCHW; ins[2].buf = xh;
  ins[3].index = 3; ins[3].type = RKNN_TENSOR_FLOAT16; ins[3].size = 8;
  ins[3].fmt = RKNN_TENSOR_NCHW; ins[3].buf = yh;
  ret = rknn_inputs_set(ctx, 4, ins);
  std::printf("inputs_set: %d\n", ret);
  if (ret < 0) return 1;
  ret = rknn_run(ctx, NULL);
  std::printf("run: %d\n", ret);
  if (ret < 0) return 1;

  rknn_output outs[2]; std::memset(outs, 0, sizeof(outs));
  outs[0].index = 0; outs[0].want_float = 0;
  outs[1].index = 1; outs[1].want_float = 1;
  ret = rknn_outputs_get(ctx, 2, outs, NULL);
  std::printf("outputs_get: %d\n", ret);
  if (ret < 0) return 1;

  int bad = 0;
  int8_t* o1 = static_cast<int8_t*>(outs[0].buf);
  float* o2 = static_cast<float*>(outs[1].buf);
  std::printf("%s out:", cpu_op);
  for (int i = 0; i < 4; i++) std::printf(" %d", o1[i] ? 1 : 0);
  std::printf("  (want");
  for (int i = 0; i < 4; i++) std::printf(" %d", cpu_expected(cpu_op, a[i] != 0, b[i] != 0));
  std::printf(")\n");
  std::printf("%s out:", npu_op);
  for (int i = 0; i < 4; i++) std::printf(" %.3f", o2[i]);
  std::printf("  (want");
  for (int i = 0; i < 4; i++) std::printf(" %.3f", npu_expected(npu_op, xf[i], yf[i]));
  std::printf(")\n");

  for (int i = 0; i < 4; i++) {
    if ((o1[i] != 0) != (cpu_expected(cpu_op, a[i] != 0, b[i] != 0) != 0)) bad++;
    float want = npu_expected(npu_op, xf[i], yf[i]);
    if (o2[i] != o2[i] || std::fabs(o2[i] - want) > 0.12f) bad++;
  }
  std::printf("%s mismatches=%d\n", bad ? "FAIL" : "PASS", bad);
  rknn_outputs_release(ctx, 2, outs);
  rknn_destroy(ctx);
  std::free(model);
  return bad ? 1 : 0;
}
