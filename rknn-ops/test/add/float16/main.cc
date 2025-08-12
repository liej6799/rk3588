/*
 * Copyright (C) 2024  Jasbir Matharu, <jasjnuk@gmail.com>
 *
 * This file is part of rk3588-npu.
 *
 * rk3588-npu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * rk3588-npu is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with rk3588-npu.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
 

 

#include "rknnops.h"
 
  // Test currently runs against kernel 5.10 haven't tested 6.1 kernel.
 

 
  // Hand crafted register definitions for a simple fp 16 convolution which
  // can be done with single NPU task because the input cube and weights are
  // small. Feature data is 4x1x40 and weights 1x1x40x16, output is 4x1x16.
  // Note: numerous registers require changes if the input cube or weight
  // dimensions are altered.

  // Updated npu_regs[] to match the values from the latest npu_regs_map2 dump
 

int main(int argc, char **argv) {

    __fp16* a = (__fp16*)malloc(10 * sizeof(__fp16));
    __fp16* b = (__fp16*)malloc(10 * sizeof(__fp16));
    for (size_t i = 0; i < 10; i++) {
        a[i] = 1.1f;
        b[i] = 2.2f;
    }
    __fp16* result = float16_add_op(a, b);
    for (size_t i = 0; i < 10; i++) {
        printf("%f ", result[i]);
    }

 

  return 0;
}