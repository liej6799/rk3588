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

    // a=[1,0,1,0], b=[1,1,0,0], rest=[1,1,1,1] -> AND = [1,0,0,0]
    int8_t in_data[64][4];
    memset(in_data, 1, sizeof(in_data));
    in_data[0][0]=1; in_data[0][1]=0; in_data[0][2]=1; in_data[0][3]=0;
    in_data[1][0]=1; in_data[1][1]=1; in_data[1][2]=0; in_data[1][3]=0;

    rknn_input ins[64]; memset(ins, 0, sizeof(ins));
    for (uint32_t i = 0; i < io.n_input && i < 64; i++) {
        ins[i].index = i; ins[i].type = RKNN_TENSOR_INT8; ins[i].size = 4;
        ins[i].fmt = RKNN_TENSOR_NCHW; ins[i].buf = in_data[i]; ins[i].pass_through = 1;
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
