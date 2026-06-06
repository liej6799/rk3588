// Query and print input/output tensor attrs (dtype etc.) for any .rknn.
#include <cstdio>
#include <cstdlib>
#include "rknn_api.h"

static void check(int r, const char *w) {
  if (r < 0) { std::fprintf(stderr, "%s failed: %d\n", w, r); std::exit(1); }
}

int main(int argc, char **argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: %s MODEL.rknn\n", argv[0]); return 2; }
  rknn_context ctx = 0;
  check(rknn_init(&ctx, (void *)argv[1], 0, 0, nullptr), "rknn_init");
  rknn_input_output_num io = {};
  check(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "IN_OUT_NUM");
  std::printf("inputs=%u outputs=%u\n", io.n_input, io.n_output);
  for (uint32_t i = 0; i < io.n_input; i++) {
    rknn_tensor_attr a = {}; a.index = i;
    check(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &a, sizeof(a)), "INPUT_ATTR");
    std::printf("  in[%u] name=%s type=%s(%d) n_elems=%u size=%u fmt=%d\n",
                i, a.name, get_type_string(a.type), a.type, a.n_elems, a.size, a.fmt);
  }
  for (uint32_t i = 0; i < io.n_output; i++) {
    rknn_tensor_attr a = {}; a.index = i;
    check(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a)), "OUTPUT_ATTR");
    std::printf("  out[%u] name=%s type=%s(%d) n_elems=%u size=%u fmt=%d\n",
                i, a.name, get_type_string(a.type), a.type, a.n_elems, a.size, a.fmt);
  }
  rknn_destroy(ctx);
  return 0;
}
