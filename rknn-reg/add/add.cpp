
#include <set>
#include <string>
#include <vector>
#include <sys/time.h>
#include <cmath>
#include <cstring>
#include <typeinfo>
#include <sys/types.h>
#include <sys/stat.h>

#include "rknn_api.h"


static unsigned char *load_npy(rknn_tensor_attr *input_attr, int *input_type, int *input_size,
    int *type_bytes)
{


float* test = (float*)malloc(100 * sizeof(float));

for (size_t i = 0; i < 100; ++i) {
test[i] = 10.0f;
}

// Dynamically allocate destination buffer
float* data = (float*)malloc(100 * sizeof(float));

// Copy the data
//memcpy(data, test, 1048576 * sizeof(float)); // Does not hit SegFault
memcpy(data, test, 100 * sizeof(float)); // Hit SegFault
//memcpy(data, pr, 16777216);
// memcpy(testData, reinterpret_cast<unsigned char*>(ptr), 16777216);

// *input_size = npy_data.num_bytes();

// for (int i = 0; i < 30; i++) 
// {
//   printf("DATA: %f ", float_data[i]);
// }

// // *input_size = 240;

// std::cout << "Type of a: " << typeid(float_data).name() << std::endl;

// std::cout << "Type of b: " << typeid(ptr).name() << std::endl;

// std::cout << "size of a: " << sizeof(float_data) << std::endl;

// std::cout << "size of b: " << sizeof(ptr) << std::endl;




// float* Pf = test;  // Pf now points to the first element of A30_f

unsigned char* byte_data = (unsigned char*)data;
printf("DATA: %f", data[0]);

return byte_data;
//return reinterpret_cast<unsigned char*>(Pf);
}


static inline int64_t getCurrentTimeUs()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

void dump_tensor_attr(rknn_tensor_attr* attr)
{
  std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
  for (int i = 1; i < attr->n_dims; ++i) {
    shape_str += ", " + std::to_string(attr->dims[i]);
  }

  printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
         "type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
         attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static void* load_file(const char* file_path, size_t* file_size)
{
  FILE* fp = fopen(file_path, "rb");
  if (fp == NULL) {
    printf("failed to open file: %s\n", file_path);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  size_t size = (size_t)ftell(fp);
  fseek(fp, 0, SEEK_SET);

  void* file_data = malloc(size);
  if (file_data == NULL) {
    fclose(fp);
    printf("failed allocate file size: %zu\n", size);
    return NULL;
  }

  if (fread(file_data, 1, size, fp) != size) {
    fclose(fp);
    free(file_data);
    printf("failed to read file data!\n");
    return NULL;
  }

  fclose(fp);

  *file_size = size;

  return file_data;
}


int main() {
    printf("Hello, World!\n");

    // Load model file
    const char* model_path = "../add.rknn";
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        printf("Failed to open model file: %s\n", model_path);
        return -1;
    }
    struct stat st;
    stat(model_path, &st);
    size_t model_size = st.st_size;
    void* model_data = malloc(model_size);
    if (!model_data) {
        printf("Failed to allocate memory for model\n");
        fclose(fp);
        return -1;
    }
    fread(model_data, 1, model_size, fp);
    fclose(fp);

    // RKNN init
    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);
    if (ret < 0) {
      printf("rknn_init fail! ret=%d\n", ret);
      return -1;
    }

    printf("rknn_init success!\n");

  // Get sdk and driver version
  rknn_sdk_version sdk_ver;
  ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  printf("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);

  rknn_mem_size mem_size;
  ret = rknn_query(ctx, RKNN_QUERY_MEM_SIZE, &mem_size, sizeof(mem_size));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  printf("total weight size: %u, total internal size: %u\n", mem_size.total_weight_size, mem_size.total_internal_size);
  printf("total dma used size: %zu\n", (size_t)mem_size.total_dma_allocated_size);


  // Get Model Input Output Info
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  printf("input tensors:\n");
  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
  for (uint32_t i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0) {
      printf("rknn_query error! ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&input_attrs[i]);
  }

  printf("output tensors:\n");
  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
  for (uint32_t i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&output_attrs[i]);
  }


  unsigned char* input_data[io_num.n_input];
  int            input_type[io_num.n_input];  
  int            input_layout[io_num.n_input];
  int            input_size[io_num.n_input];
  int            type_bytes[io_num.n_input];
  for (int i = 0; i < io_num.n_input; i++) {
    input_data[i]   = NULL;
    input_type[i]   = RKNN_TENSOR_FLOAT32;
    input_layout[i] = RKNN_TENSOR_UNDEFINED;
    input_size[i]   = input_attrs[i].n_elems * sizeof(float);
    type_bytes[i] = 4;
  }

 
for (int i = 0; i < io_num.n_input; i++) {
    // Load npy file 
    input_data[i] = load_npy(&input_attrs[i], &input_type[i], 
                            &input_size[i], &type_bytes[i]);

    if (!input_data[i]) {
    return -1;
    }
}


  rknn_input inputs[io_num.n_input];
  memset(inputs, 0, io_num.n_input * sizeof(rknn_input));
  for (int i = 0; i < io_num.n_input; i++) {
    inputs[i].index        = i;
    inputs[i].pass_through = 0;
    inputs[i].type         = (rknn_tensor_type)input_type[i];
    inputs[i].fmt          = RKNN_TENSOR_UNDEFINED;
    inputs[i].buf          = input_data[i];
    inputs[i].size         = input_size[i];
  }
  

  // Set input
  ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
  if (ret < 0) {
    printf("rknn_input_set fail! ret=%d\n", ret);
    return -1;
  }

  rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
  // Run
  printf("Begin perf ...\n");
  double total_time = 0;
  for (int i = 0; i < 1; ++i) {
    int64_t start_us  = getCurrentTimeUs();
    ret               = rknn_run(ctx, NULL);
    int64_t elapse_us = getCurrentTimeUs() - start_us; if (ret < 0) {
      printf("rknn run error %d\n", ret);
      return -1;
    }
    total_time += elapse_us / 1000.f;
    printf("%4d: Elapse Time = %.2fms, FPS = %.2f\n", i, elapse_us / 1000.f, 1000.f * 1000.f / elapse_us);
  }
  printf("Avg elapse Time = %.3fms\n", total_time / 1);
  printf("Avg FPS = %.3f\n", 1 * 1000.f / total_time);

  // Get output
  rknn_output outputs[io_num.n_output];
  memset(outputs, 0, io_num.n_output * sizeof(rknn_output));
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    outputs[i].want_float  = 1;
    outputs[i].index       = i;
    outputs[i].is_prealloc = 0;
  }

  ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
  if (ret < 0) {
    printf("rknn_outputs_get fail! ret=%d\n", ret);
    return ret;
  }
  
  const auto out_elems = output_attrs[0].n_elems; 

  for (size_t idx=0; idx<out_elems;idx++) {
    const auto buf_data = (float *)outputs[0].buf;
    printf("%f ",buf_data[idx]);

  }
    return 0;
}