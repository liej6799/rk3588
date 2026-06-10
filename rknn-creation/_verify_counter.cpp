#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "rknn_api.h"

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "_counter_increment.rknn";

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, (void*)model_path, 0, 0, nullptr);
    printf("rknn_init: %d\n", ret);
    if (ret != RKNN_SUCC) return 1;

    rknn_input_output_num io_num = {};
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("inputs: %d  outputs: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[1] = {}, output_attrs[1] = {};
    input_attrs[0].index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[0], sizeof(rknn_tensor_attr));
    output_attrs[0].index = 0;
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[0], sizeof(rknn_tensor_attr));
    printf("input:  name=%s type=%d n_elems=%d dims=[%d,%d,%d,%d]\n",
           input_attrs[0].name, input_attrs[0].type,
           input_attrs[0].n_elems, input_attrs[0].dims[0], input_attrs[0].dims[1],
           input_attrs[0].dims[2], input_attrs[0].dims[3]);
    printf("output: name=%s type=%d n_elems=%d dims=[%d,%d,%d,%d]\n",
           output_attrs[0].name, output_attrs[0].type,
           output_attrs[0].n_elems, output_attrs[0].dims[0], output_attrs[0].dims[1],
           output_attrs[0].dims[2], output_attrs[0].dims[3]);

    // Test 1: lo=0xFFFFFFFE, hi=0 -> expected: lo=0, hi=1 (overflow)
    uint32_t input_data[2] = {0xFFFFFFFE, 0};
    printf("\n--- Test 1: overflow case ---\n");
    printf("input:  [0x%08X, 0x%08X]  (uint32: [%u, %u])\n", input_data[0], input_data[1], input_data[0], input_data[1]);

    rknn_input inputs[1] = {};
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_INT32;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = input_data;
    inputs[0].size = sizeof(input_data);
    rknn_inputs_set(ctx, 1, inputs);

    ret = rknn_run(ctx, NULL);
    printf("rknn_run: %d\n", ret);

    int32_t output_data[2] = {};
    rknn_output outputs[1] = {};
    outputs[0].index = 0;
    outputs[0].want_float = 0;
    outputs[0].is_prealloc = 1;
    outputs[0].buf = output_data;
    outputs[0].size = sizeof(output_data);
    rknn_outputs_get(ctx, 1, outputs, NULL);

    printf("output: [0x%08X, 0x%08X]  (uint32: [%u, %u])\n",
           (uint32_t)output_data[0], (uint32_t)output_data[1],
           (uint32_t)output_data[0], (uint32_t)output_data[1]);

    uint32_t exp_lo = 0, exp_hi = 1;
    int pass = ((uint32_t)output_data[0] == exp_lo && (uint32_t)output_data[1] == exp_hi);
    printf("expected: [0x%08X, 0x%08X]\n", exp_lo, exp_hi);
    printf("%s\n", pass ? "PASS" : "FAIL");

    // Test 2: lo=10, hi=0 -> expected: lo=12, hi=0 (no overflow)
    printf("\n--- Test 2: no overflow case ---\n");
    input_data[0] = 10; input_data[1] = 0;
    printf("input:  [%u, %u]\n", input_data[0], input_data[1]);
    rknn_inputs_set(ctx, 1, inputs);
    rknn_run(ctx, NULL);
    rknn_outputs_get(ctx, 1, outputs, NULL);
    printf("output: [%u, %u]\n", (uint32_t)output_data[0], (uint32_t)output_data[1]);
    int pass2 = ((uint32_t)output_data[0] == 12 && (uint32_t)output_data[1] == 0);
    printf("expected: [12, 0]\n");
    printf("%s\n", pass2 ? "PASS" : "FAIL");

    rknn_outputs_release(ctx, 1, outputs);
    rknn_destroy(ctx);
    return (pass && pass2) ? 0 : 1;
}
