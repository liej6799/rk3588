// verify_xor.cpp - Verify an n-input fused XOR (parity) RKNN model on the NPU.
// out[i] = a0[i] ^ a1[i] ^ ... ^ a{n-1}[i].  Uses 4 elements with per-input bit
// patterns so every input differs:  in[j][i] = (i >> j) & 1  (for j < n, i < 4..).
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
    rknn_tensor_attr a0; memset(&a0, 0, sizeof(a0)); a0.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &a0, sizeof(a0));
    uint32_t N = a0.n_elems;
    printf("inputs=%u outputs=%u n_elems=%u\n", io.n_input, io.n_output, N);

    int8_t in[64][64];
    memset(in, 0, sizeof(in));
    for (uint32_t j = 0; j < io.n_input && j < 64; j++)
        for (uint32_t i = 0; i < N && i < 64; i++)
            in[j][i] = (int8_t)((i >> j) & 1);

    rknn_input ins[64]; memset(ins, 0, sizeof(ins));
    for (uint32_t j = 0; j < io.n_input && j < 64; j++) {
        ins[j].index = j; ins[j].type = RKNN_TENSOR_INT8; ins[j].size = N;
        ins[j].fmt = RKNN_TENSOR_NCHW; ins[j].buf = in[j]; ins[j].pass_through = 1;
    }
    ret = rknn_inputs_set(ctx, io.n_input, ins);
    printf("inputs_set: %d\n", ret);
    ret = rknn_run(ctx, NULL);
    printf("run: %d\n", ret);

    rknn_output out; memset(&out, 0, sizeof(out));
    out.want_float = 0;
    ret = rknn_outputs_get(ctx, 1, &out, NULL);
    printf("outputs_get: %d size=%d\n", ret, out.size);
    int8_t* od = (int8_t*)out.buf;

    printf("XOR output:");
    for (uint32_t i = 0; i < N && i < 8; i++) printf(" %d", od[i]);
    printf("\n");

    int bad = 0;
    for (uint32_t i = 0; i < N; i++) {
        int parity = 0;
        for (uint32_t j = 0; j < io.n_input; j++) parity ^= (in[j][i] & 1);
        if ((od[i] ? 1 : 0) != parity) {
            printf("  MISMATCH [%u]: got %d, expected %d\n", i, od[i], parity);
            bad++;
        }
    }
    printf("%s\n", bad ? "FAIL" : "PASS");

    rknn_outputs_release(ctx, 1, &out);
    rknn_destroy(ctx);
    free(model);
    return bad;
}
