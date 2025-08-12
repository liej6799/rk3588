/*
 * Copyright (C) 2024  Jasbir Matharu, <jasjnuk@gmail.com>
 *
 * This file is part of rk3588-npu.
 *
 * rk3588-npu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rk3588-npu is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with rk3588-npu.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef RKNNOPS_H
#define RKNNOPS_H

#include <stdint.h>
#include <stddef.h>
#include <typeinfo>

#include <libdrm/drm.h>
#include "rknpu-ioctl.h"
#include "rknn_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * RK3588 NPU Operations Library
 * ============================================================================
 * This library provides high-level operations for the RockChip RK3588 NPU
 * including memory management, data operations, and NPU task execution.
 */

/* ============================================================================
 * Basic Data Types and Operations
 * ============================================================================ */

 
 struct MemHandles {
    void* input;
    void* weights;
    void* output;
    uint64_t input_dma, input_obj;
    uint64_t weights_dma, weights_obj;
    uint64_t output_dma, output_obj;
    uint64_t tasks_obj;
};

int get_type_size(rknn_tensor_type type){
    switch (type){
        case RKNN_TENSOR_INT8:
            return sizeof(int8_t);
        case RKNN_TENSOR_UINT8:
            return sizeof(uint8_t);
        case RKNN_TENSOR_INT16:
            return sizeof(int16_t);
        case RKNN_TENSOR_UINT16:
            return sizeof(uint16_t);
        case RKNN_TENSOR_INT32:
            return sizeof(int32_t);
        case RKNN_TENSOR_UINT32:
            return sizeof(uint32_t);
        case RKNN_TENSOR_INT64:
            return sizeof(int64_t);
        case RKNN_TENSOR_FLOAT16:
            return sizeof(__fp16);
        case RKNN_TENSOR_FLOAT32:
            return sizeof(float);
        default:
            printf("    get_type_size error: not support dtype %d\n", type);
            return 0;
    }
}


 void* mem_allocate(int fd, size_t size, uint64_t *dma_addr, uint64_t *obj, uint32_t flags) {

    int ret;
    struct rknpu_mem_create mem_create = {
      .flags = flags | RKNPU_MEM_NON_CACHEABLE,
      .size = size,
    };
  
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create);
    if(ret < 0)  {
      printf("RKNPU_MEM_CREATE failed %d\n",ret);
      return NULL;
    }
  
    struct rknpu_mem_map mem_map = { .handle = mem_create.handle, .offset=0 };
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mem_map);
    if(ret < 0) {
      printf("RKNPU_MEM_MAP failed %d\n",ret);
      return NULL;
    }	
    void *map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mem_map.offset);
  
    *dma_addr = mem_create.dma_addr;
    *obj = mem_create.obj_addr;
    return map;
  }
  
  void mem_destroy(int fd, uint32_t handle, uint64_t obj_addr) {
  
    int ret;
    struct rknpu_mem_destroy destroy = {
      .handle = handle ,
      .obj_addr = obj_addr
    };
  
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &destroy);
    if (ret <0) {
      printf("RKNPU_MEM_DESTROY failed %d\n",ret);
    }
  }

 
 uint64_t npu_regs[] = {
    0x10010000000e4004, // 0
    0x20010000000e5004, // 1
    0x1001000001e5400c, // 2
    0x1001480000024010, // 3
    0x1001000000004014, // 4
    0x1001000000004020, // 5
    0x1001000000c04024, // 6
    0x1001000000094030, // 7
    0x1001000000004034, // 8
    0x1001000000004038, // 9
    0x100100070007403c, // 10
    0x1001000000534040, // 11
    0x1001000000004044, // 12
    0x1001000000004048, // 13
    0x100100000000404c, // 14
    0x1001000000024050, // 15
    0x1001000000004054, // 16
    0x1001000000074058, // 17
    0x100100000009405c, // 18
    0x1001000000534060, // 19
    0x1001000000004064, // 20
    0x1001000000004068, // 21
    0x100100000000406c, // 22
    0x1001108202c04070, // 23 // 0x1001108003c44070 0x1001108202c04070 0x1001108402c04070
    0x1001000000004074, // 24
    0x1001000000014078, // 25
    0x100100000000407c, // 26
    0x1001000000004080, // 27
    0x1001000100014084, // 28
    0x1001000000004088, // 29
    0x1001000000004090, // 30
    0x1001000000004094, // 31
    0x1001000000004098, // 32
    0x100100000000409c, // 33
    0x10010000000040a0, // 34
    0x10010000000040a4, // 35
    0x10010000000040a8, // 36
    0x10010000000040ac, // 37
    0x1001000000c040c0, // 38
    0x10010000000040c4, // 39
    0x1001000000004100, // 40
    0x1001000000004104, // 41
    0x1001000000004108, // 42
    0x100100000000410c, // 43
    0x1001000000004110, // 44
    0x1001000000004114, // 45
    0x1001000000004118, // 46
    0x100100000000411c, // 47
    0x1001000000004120, // 48
    0x1001000000004124, // 49
    0x1001000000004128, // 50
    0x100100000000412c, // 51
    0x200100000009500c, // 52
    0x2001000000005010, // 53
    0x2001000000075014, // 54
    0x2001000000005018, // 55
    0x200100000000501c, // 56
    0x2001000000005020, // 57
    0x2001000000005028, // 58
    0x200100000000502c, // 59
    0x2001400000085034, // 60
    0x2001000000005038, // 61
    0x2001000000c05040, // 62
    0x2001000178495044, // 63
    0x2001000000005048, // 64
    0x200100000020504c, // 65
    0x2001000000005064, // 66
    0x2001010101015068, // 67
    0x200100000020506c, // 68
    0x0000000000000000, // 69
    0x0101000000000014, // 70
    0x0041000000000000, // 71
    0x0081000000180008, // 72
    };

int getDeviceFd()
{
    int fd = open("/dev/dri/card1", O_RDWR);
    if(fd<0) {
      printf("Failed to open /dev/dri/card1 %d\n",errno);
      exit(1);
    }
    return fd;  
}

MemHandles createRegCmd(int fd, int type_size)
{
    uint64_t tasks_dma, tasks_obj;
    struct rknpu_task *tasks = static_cast<struct rknpu_task*>(mem_allocate(fd, 1024, &tasks_dma, &tasks_obj, RKNPU_MEM_KERNEL_MAPPING));
  
    uint64_t regcmd_dma, regcmd_obj;
    uint64_t *regcmd = static_cast<uint64_t*>(mem_allocate(fd, 1024, &regcmd_dma, &regcmd_obj, 0));
  
    uint64_t input_dma, input_obj;
    void *input = mem_allocate(fd, type_size, &input_dma, &input_obj, 0);  

    uint64_t weights_dma, weights_obj;
    void *weights = mem_allocate(fd, type_size, &weights_dma, &weights_obj, 0);

    uint64_t output_dma, output_obj;
    void *output = mem_allocate(fd, type_size, &output_dma, &output_obj, 0);

    // To return input, weights, and output, you can use output parameters or a struct.
    // Example: define a struct to hold them and return it.
    // Set input, weights and output physical memory locations. Note limited to 
    // a 32 bit address size (4GB)
    npu_regs[55] = npu_regs[55] | ((input_dma & 0xFFFFFFFF) <<16);
    npu_regs[61] = npu_regs[61] | ((weights_dma & 0xFFFFFFFF)  <<16);
    npu_regs[5] = npu_regs[5] | ((output_dma & 0xFFFFFFFF) <<16);

        
    memcpy(regcmd,npu_regs,sizeof(npu_regs));

    tasks[0].flags  = 0;
    tasks[0].op_idx = 4;
    tasks[0].enable_mask = 0x18;
    tasks[0].int_mask = 0x300; // wait for DPU to finish
    tasks[0].int_clear = 0x1ffff;
    tasks[0].int_status = 0;
    tasks[0].regcfg_amount = sizeof(npu_regs)/sizeof(uint64_t); //nInstrs - 1;
    tasks[0].regcfg_offset = 0;
    tasks[0].regcmd_addr = regcmd_dma;
    
    MemHandles handles;
    handles.input = input;
    handles.weights = weights;
    handles.output = output;
    handles.input_dma = input_dma;
    handles.input_obj = input_obj;
    handles.weights_dma = weights_dma;
    handles.weights_obj = weights_obj;
    handles.output_dma = output_dma;
    handles.output_obj = output_obj;
    handles.tasks_obj = tasks_obj;
    return handles;
}

int submitTask(int fd, uint64_t tasks_obj)
{
  struct rknpu_submit submit = {
    .flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,
    .timeout = 6000,
    .task_start = 0,
    .task_number = 1,
    .task_counter = 0,
    .priority = 0,
    .task_obj_addr = tasks_obj,
    .regcfg_obj_addr = 0,
    .task_base_addr = 0,
    .user_data = 0,
    .core_mask = 1,
    .fence_fd = -1,
    .subcore_task = { // Only use core 1, nothing for core 2/3
     {
       .task_start = 0,
       .task_number = 1,
     }, { 1, 0}, {2, 0}
   },
  
  };

   return ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &submit);
}

/**
 * @brief Float16 addition operation
 * @param a First float16 operand
 * @param b Second float16 operand
 * @return Sum of a and b in float16 format
 */
__fp16* float16_add_op(__fp16* a, __fp16* b)
{
    int fd = getDeviceFd();
    rknn_tensor_type dtype = RKNN_TENSOR_FLOAT16;

    MemHandles handles = createRegCmd(fd, get_type_size(dtype));
    __fp16 *weights_fp16 = static_cast<__fp16*>(handles.weights);
    __fp16 *feature_data_fp16 = static_cast<__fp16*>(handles.input);
    __fp16 *output_data = static_cast<__fp16*>(handles.output);
    
    memcpy(weights_fp16, a, get_type_size(dtype));
    memcpy(feature_data_fp16, b, get_type_size(dtype));

    int ret = submitTask(fd, handles.tasks_obj);
    if(ret < 0) {
        printf("RKNPU_SUBMIT failed %d\n",ret);
        return NULL;
    }
    return output_data;
}



#ifdef __cplusplus
}
#endif

#endif /* RKNNOPS_H */