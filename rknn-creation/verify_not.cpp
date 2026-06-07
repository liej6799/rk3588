// verify_not.cpp - Verify a unary Not (NOT gate) RKNN model on the NPU.
// Feeds a = [1,0,1,0] and checks out == ~a == [0,1,0,1].
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

    int8_t a[4] = {1, 0, 1, 0};
    rknn_input ins[1]; memset(ins, 0, sizeof(ins));
    ins[0].index = 0; ins[0].type = RKNN_TENSOR_INT8; ins[0].size = 4;
    ins[0].fmt = RKNN_TENSOR_NCHW; ins[0].buf = a; ins[0].pass_through = 1;
    ret = rknn_inputs_set(ctx, io.n_input, ins);
    printf("inputs_set: %d\n", ret);

    ret = rknn_run(ctx, NULL);
    printf("run: %d\n", ret);

    rknn_output out; memset(&out, 0, sizeof(out));
    out.want_float = 0;
    ret = rknn_outputs_get(ctx, 1, &out, NULL);
    printf("outputs_get: %d size=%d\n", ret, out.size);

    int8_t* od = (int8_t*)out.buf;
    printf("NOT output: %d %d %d %d\n", od[0], od[1], od[2], od[3]);

    int8_t expected[4] = {0, 1, 0, 1};   // ~a (bool)
    int bad = 0;
    for (int i = 0; i < 4; i++) {
        if ((od[i] ? 1 : 0) != expected[i]) {
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
