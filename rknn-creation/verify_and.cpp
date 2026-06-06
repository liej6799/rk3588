// verify_and.cpp - Verify And-only RKNN model on NPU
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "rknn_api.h"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.rknn>\n", argv[0]); return 2; }
    FILE* fp = fopen(argv[1], "rb");
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    void* model = malloc(sz); fread(model, 1, sz, fp); fclose(fp);

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model, sz, 0, NULL);
    printf("rknn_init: %d\n", ret);
    if (ret < 0) { free(model); return 1; }

    rknn_input_output_num io; memset(&io, 0, sizeof(io));
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    printf("inputs=%u outputs=%u\n", io.n_input, io.n_output);

    for (uint32_t i = 0; i < io.n_input; i++) {
        rknn_tensor_attr a; memset(&a, 0, sizeof(a)); a.index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &a, sizeof(a));
        printf("  in[%u] name=%s n_elems=%u type=%d fmt=%d\n", i, a.name, a.n_elems, a.type, a.fmt);
    }
    for (uint32_t i = 0; i < io.n_output; i++) {
        rknn_tensor_attr a; memset(&a, 0, sizeof(a)); a.index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a));
        printf("  out[%u] name=%s n_elems=%u type=%d fmt=%d\n", i, a.name, a.n_elems, a.type, a.fmt);
    }

    // a=[1,0,1,0], b=[1,1,0,0] -> AND = [1,0,0,0]
    int8_t a[4] = {1, 0, 1, 0};
    int8_t b[4] = {1, 1, 0, 0};

    rknn_input ins[2]; memset(ins, 0, sizeof(ins));
    ins[0].index = 0; ins[0].type = RKNN_TENSOR_INT8; ins[0].size = 4;
    ins[0].fmt = RKNN_TENSOR_NCHW; ins[0].buf = a; ins[0].pass_through = 1;
    ins[1].index = 1; ins[1].type = RKNN_TENSOR_INT8; ins[1].size = 4;
    ins[1].fmt = RKNN_TENSOR_NCHW; ins[1].buf = b; ins[1].pass_through = 1;
    ret = rknn_inputs_set(ctx, 2, ins);
    printf("inputs_set: %d\n", ret);

    ret = rknn_run(ctx, NULL);
    printf("run: %d\n", ret);

    rknn_output out; memset(&out, 0, sizeof(out));
    out.want_float = 0;
    ret = rknn_outputs_get(ctx, 1, &out, NULL);
    printf("outputs_get: %d size=%d\n", ret, out.size);

    int8_t* od = (int8_t*)out.buf;
    printf("AND output: %d %d %d %d\n", od[0], od[1], od[2], od[3]);

    int8_t expected[4] = {1, 0, 0, 0};
    int bad = 0;
    for (int i = 0; i < 4; i++) {
        if (od[i] != expected[i]) {
            printf("  MISMATCH [%d]: got %d, expected %d\n", i, od[i], expected[i]);
            bad++;
        }
    }
    printf("%s\n", bad ? "FAIL" : "PASS");

    rknn_outputs_release(ctx, 1, &out);
    rknn_destroy(ctx);
    free(model);
    return bad;
}
