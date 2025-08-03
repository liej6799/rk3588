#include <iostream>

int main() {
    struct rknpu_mem_create mem_create = {
    .flags = flags | RKNPU_MEM_NON_CACHEABLE,
    .size = size,
    };

    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create);
    if(ret < 0)  {
    printf("RKNPU_MEM_CREATE failed %d\n",ret);
    return NULL;
    }
 
      
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
