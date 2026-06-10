#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "rknn_api.h"

int main() {
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, (void*)"_int32_add.rknn", 0, 0, nullptr);
    printf("rknn_init: %d\n", ret);
    if (ret) return 1;

    rknn_input_output_num io = {};
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    printf("inputs=%u outputs=%u\n", io.n_input, io.n_output);

    rknn_tensor_attr ia[2] = {}, oa[1] = {};
    ia[0].index = 0; ia[1].index = 1;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &ia[0], sizeof(rknn_tensor_attr));
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &ia[1], sizeof(rknn_tensor_attr));
    oa[0].index = 0;
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &oa[0], sizeof(rknn_tensor_attr));
    printf("in0: name=%s type=%s elems=%d dims=[%d,%d,%d,%d]\n", ia[0].name,
           get_type_string(ia[0].type), ia[0].n_elems, ia[0].dims[0], ia[0].dims[1], ia[0].dims[2], ia[0].dims[3]);
    printf("in1: name=%s type=%s elems=%d dims=[%d,%d,%d,%d]\n", ia[1].name,
           get_type_string(ia[1].type), ia[1].n_elems, ia[1].dims[0], ia[1].dims[1], ia[1].dims[2], ia[1].dims[3]);
    printf("out: name=%s type=%s elems=%d dims=[%d,%d,%d,%d]\n", oa[0].name,
           get_type_string(oa[0].type), oa[0].n_elems, oa[0].dims[0], oa[0].dims[1], oa[0].dims[2], oa[0].dims[3]);

    int32_t a[4] = {1, 100, 1000, 32000};
    int32_t b[4] = {1, 200, 2000, 1};
    printf("\ninput a: [%d, %d, %d, %d]\n", a[0], a[1], a[2], a[3]);
    printf("input b: [%d, %d, %d, %d]\n", b[0], b[1], b[2], b[3]);

    rknn_input inputs[2] = {};
    inputs[0].index = 0; inputs[0].type = RKNN_TENSOR_INT32; inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = a; inputs[0].size = sizeof(a);
    inputs[1].index = 1; inputs[1].type = RKNN_TENSOR_INT32; inputs[1].fmt = RKNN_TENSOR_NHWC;
    inputs[1].buf = b; inputs[1].size = sizeof(b);
    rknn_inputs_set(ctx, 2, inputs);

    ret = rknn_run(ctx, NULL);
    printf("rknn_run: %d\n", ret);

    int32_t out[4] = {};
    rknn_output outputs[1] = {};
    outputs[0].index = 0; outputs[0].want_float = 0; outputs[0].is_prealloc = 1;
    outputs[0].buf = out; outputs[0].size = sizeof(out);
    rknn_outputs_get(ctx, 1, outputs, NULL);

    printf("output:  [%d, %d, %d, %d]\n", out[0], out[1], out[2], out[3]);
    printf("expect:  [2, 300, 3000, 32001]\n");
    int pass = (out[0]==2 && out[1]==300 && out[2]==3000 && out[3]==32001);
    printf("%s\n", pass ? "PASS" : "FAIL");

    rknn_outputs_release(ctx, 1, outputs);
    rknn_destroy(ctx);
    return pass ? 0 : 1;
}
