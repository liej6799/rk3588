// Generic verifier for fp16 element-wise Add/Sub/Mul/Div RKNN chains.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "rknn_api.h"

static void check(int ret, const char *what) {
  if (ret < 0) {
    std::fprintf(stderr, "%s failed: %d\n", what, ret);
    std::exit(1);
  }
}

static uint16_t f2h(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  uint32_t sign = (x >> 16) & 0x8000;
  int exp = int((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = x & 0x7fffff;
  if (exp <= 0) return uint16_t(sign);
  if (exp >= 31) return uint16_t(sign | 0x7c00);
  return uint16_t(sign | (uint32_t(exp) << 10) | (mant >> 13));
}

static float h2f(uint16_t h) {
  uint32_t sign = (uint32_t(h) & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x03ff;
  uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      exp = 1;
      while ((mant & 0x0400) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x03ff;
      out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    out = sign | 0x7f800000 | (mant << 13);
  } else {
    out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

static std::vector<std::string> split_ops(const std::string &s) {
  std::vector<std::string> ops;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) ops.push_back(item);
  }
  return ops;
}

static float seed_value(uint32_t elem, uint32_t input_idx) {
  float base = 0.5f + 0.125f * float(input_idx);
  float wave = float((elem * 3 + input_idx * 7) % 23) * 0.03125f;
  return base + wave;
}

static float apply_op(float a, float b, const std::string &op) {
  if (op == "Add") return a + b;
  if (op == "Sub") return a - b;
  if (op == "Mul") return a * b;
  if (op == "Div") return a / b;
  std::fprintf(stderr, "unknown op: %s\n", op.c_str());
  std::exit(2);
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "usage: %s MODEL.rknn [Add,Sub,...]\n", argv[0]);
    return 2;
  }
  const char *model = argv[1];

  rknn_context ctx = 0;
  check(rknn_init(&ctx, (void *)model, 0, 0, nullptr), "rknn_init");

  rknn_input_output_num io = {};
  check(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "QUERY_IN_OUT_NUM");
  if (io.n_output != 1 || io.n_input < 2) {
    std::fprintf(stderr, "unsupported model IO: inputs=%u outputs=%u\n", io.n_input, io.n_output);
    return 2;
  }

  std::vector<std::string> ops;
  if (argc == 3) {
    ops = split_ops(argv[2]);
  } else {
    ops.assign(io.n_input - 1, "Add");
  }
  if (ops.size() < io.n_input - 1) {
    std::fprintf(stderr, "expected at least %u ops for %u inputs, got %zu\n",
                 io.n_input - 1, io.n_input, ops.size());
    return 2;
  }

  std::vector<rknn_tensor_attr> attrs(io.n_input);
  for (uint32_t i = 0; i < io.n_input; i++) {
    attrs[i] = {};
    attrs[i].index = i;
    check(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attrs[i], sizeof(rknn_tensor_attr)),
          "QUERY_INPUT_ATTR");
  }
  uint32_t n = attrs[0].n_elems;
  for (uint32_t i = 1; i < io.n_input; i++) {
    if (attrs[i].n_elems != n) {
      std::fprintf(stderr, "input element count mismatch: input0=%u input%u=%u\n",
                   n, i, attrs[i].n_elems);
      return 2;
    }
  }

  std::vector<std::vector<uint16_t>> input_half(io.n_input, std::vector<uint16_t>(n));
  std::vector<std::vector<float>> input_float(io.n_input, std::vector<float>(n));
  for (uint32_t j = 0; j < io.n_input; j++) {
    for (uint32_t i = 0; i < n; i++) {
      input_half[j][i] = f2h(seed_value(i, j));
      input_float[j][i] = h2f(input_half[j][i]);
    }
  }

  std::vector<rknn_input> inputs(io.n_input);
  std::memset(inputs.data(), 0, inputs.size() * sizeof(rknn_input));
  for (uint32_t i = 0; i < io.n_input; i++) {
    inputs[i].index = i;
    inputs[i].type = RKNN_TENSOR_FLOAT16;
    inputs[i].fmt = attrs[i].fmt;
    inputs[i].buf = input_half[i].data();
    inputs[i].size = n * sizeof(uint16_t);
  }
  check(rknn_inputs_set(ctx, io.n_input, inputs.data()), "rknn_inputs_set");
  check(rknn_run(ctx, nullptr), "rknn_run");

  rknn_output out = {};
  out.index = 0;
  out.want_float = 1;
  check(rknn_outputs_get(ctx, 1, &out, nullptr), "rknn_outputs_get");

  float *got = static_cast<float *>(out.buf);
  int bad = 0;
  uint32_t first_bad = 0;
  float first_got = 0.0f, first_want = 0.0f, max_err = 0.0f;
  const float atol = 0.1f;
  const float rtol = 0.02f;
  for (uint32_t i = 0; i < n; i++) {
    float want = input_float[0][i];
    for (size_t j = 0; j < ops.size(); j++) {
      uint32_t operand = 1 + j % (io.n_input - 1);
      want = apply_op(want, input_float[operand][i], ops[j]);
    }
    float err = std::fabs(got[i] - want);
    if (err > max_err) max_err = err;
    if (err > atol + rtol * std::fabs(want)) {
      if (bad == 0) {
        first_bad = i;
        first_got = got[i];
        first_want = want;
      }
      bad++;
    }
  }

  std::printf("N=%u inputs=%u ops=%s max_err=%.6f mismatches=%d %s\n",
              n, io.n_input, argc == 3 ? argv[2] : "Add", max_err, bad,
              bad ? "FAIL" : "PASS");
  if (bad) {
    std::printf("  first bad @%u: got %.6f want %.6f\n", first_bad, first_got, first_want);
  }

  rknn_outputs_release(ctx, 1, &out);
  rknn_destroy(ctx);
  return bad ? 1 : 0;
}
