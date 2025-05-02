#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>

extern "C" {

int (*my_ioctl)(int filedes, unsigned long request, void *argp) = NULL;
#undef ioctl
int ioctl(int filedes, unsigned long request, void *argp) {
  if (my_ioctl == NULL) my_ioctl = reinterpret_cast<decltype(my_ioctl)>(dlsym(RTLD_NEXT, "ioctl"));

  uint8_t type = (request >> 8) & 0xFF;
  uint8_t nr = (request >> 0) & 0xFF;
  uint16_t size = (request >> 16) & 0xFFF;

  // run first
  int ret = my_ioctl(filedes, request, argp);

  printf("%3d 0x%3x ", filedes, size);
  printf("%3d ", ret);

  return ret;
}

}