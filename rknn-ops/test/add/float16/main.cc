
#include <stdio.h>
#include "rknnops.h"
 

int main(int argc, char **argv) {

    __fp16* a = (__fp16*)malloc(10 * sizeof(__fp16));
    __fp16* b = (__fp16*)malloc(10 * sizeof(__fp16));
    for (size_t i = 0; i < 10; i++) {
        a[i] = 1187.1123f;
        b[i] = 3256.3123f;
    }
    __fp16* result = float16_add_op(a, b);
    for (size_t i = 0; i < 10; i++) {
        printf("%f ", result[i]);
    }
  return 0;
}