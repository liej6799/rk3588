import os
print("from import 1")

import ctypes, ctypes.util
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.extra.run import get_struct, format_struct
@ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long)
def _mmap(addr, length, prot, flags, fd, offset):
  mmap_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long)
  orig_mmap = mmap_type(ctypes.addressof(orig_mmap_mv))
  ret = orig_mmap(addr, length, prot, flags, fd, offset)
  # ll = os.readlink(f"/proc/self/fd/{fd}") if fd >= 0 else ""
  print(f"mmap {addr=}, {length=}, {prot=}, {flags=}, {fd=}, {offset=} {ret=}")
  return ret

#install_hook(libc.ioctl, ioctl)
#orig_mmap_mv = install_hook(libc.mmap, _mmap)
print("import done 1")
import mmap

alloc_sizes = {}
mmaped = {}
memory_mappings = {}  # Global dict to track

def handle_ioctl(fd, request, argp, ret):
    fn = os.readlink(f"/proc/self/fd/{fd}")
    idir, size, itype, nr = (request>>30), (request>>16)&0x3FFF, (request>>8)&0xFF, request&0xFF
    
    if fn == "/dev/dri/card1":
        if nr == 64: # 0x40 + 0x00
            st = get_struct(argp, rk.struct_rknpu_action)
            print(ret, "struct_rknpu_action", format_struct(st))   

        elif nr == 65: # 0x40 + 0x01
            st = get_struct(argp, rk.struct_rknpu_submit)
  
            print(ret, "struct_rknpu_submit", format_struct(st))     
            # Print sub_core_task (struct_rknpu_subcore_task_Array_5)
            subcore_task = st.subcore_task
            print("subcore_task:")
            for i, sct in enumerate(subcore_task):
                print(f"  [{i}] task_start: {sct.task_start}, task_number: {sct.task_number}")
# Get first 1024 bytes of the mmap
            # Convert mmaped data to struct_rknpu_task
   
          

        elif nr == 66: # 0x40 + 0x02 
            st = get_struct(argp, rk.struct_rknpu_mem_create)
            print(ret, "struct_rknpu_mem_create", format_struct(st))
            alloc_sizes[st.handle] = st.size

        elif nr == 67: # 0x40 + 0x03
            st = get_struct(argp, rk.struct_rknpu_mem_map)
            print(ret, "struct_rknpu_mem_map", format_struct(st))        
  

        elif nr == 68: # 0x40 + 0x04     
            st = get_struct(argp, rk.struct_rknpu_mem_destroy)
            print(ret, "struct_rknpu_mem_destroy", format_struct(st))
 
  # Map 1024 bytes at offset 0x100001000 and print as hex, 32 bytes per line, 4 bytes per group
            # try:
            #     mm = mmap.mmap(fd, 1024, access=mmap.ACCESS_WRITE, offset=0x100000000)
            #     first_1024 = mm[:1024]
            #     print("First 1024 bytes from map:")
            #     for i in range(1024):
            #         print(f"{first_1024[i]:02x}", end="")
            #         if (i+1) % 32 == 0:
            #             print()
            #         elif (i+1) % 4 == 0:
            #             print(" ", end="")
            #     print()
            #     mm.close()

            #     mm = mmap.mmap(fd, 1024, access=mmap.ACCESS_WRITE, offset=0x100001000)
            #     first_1024 = mm[:1024]
            #     print("First 1024 bytes from map:")
            #     for i in range(1024):
            #         print(f"{first_1024[i]:02x}", end="")
            #         if (i+1) % 32 == 0:
            #             print()
            #         elif (i+1) % 4 == 0:
            #             print(" ", end="")
            #     print()
            #     mm.close()

            #     mm = mmap.mmap(fd, 1024, access=mmap.ACCESS_WRITE, offset=0x100003000)
            #     first_1024 = mm[:1024]
            #     print("First 1024 bytes from map:")
            #     for i in range(1024):
            #         print(f"{first_1024[i]:02x}", end="")
            #         if (i+1) % 32 == 0:
            #             print()
            #         elif (i+1) % 4 == 0:
            #             print(" ", end="")
            #     print()
            #     mm.close()
            # except Exception as e:
            #     print(f"Failed to mmap and print 1024 bytes: {e}")

        elif nr == 69: # 0x40 + 0x05    
            st = get_struct(argp, rk.struct_rknpu_mem_sync)
            print(ret, "struct_rknpu_mem_sync", format_struct(st))




        elif nr == 0: # 0x00
            st = get_struct(argp, rk.struct_drm_version)
            print(ret, "struct_drm_version", format_struct(st))    
        elif nr == 1: # 0x01
            st = get_struct(argp, rk.struct_drm_unique)
            print(ret, "struct_drm_unique", format_struct(st))

        elif nr == 10: # 0x0a
            st = get_struct(argp, rk.struct_drm_gem_flink)
            print(ret, "struct_drm_gem_flink", format_struct(st))

        elif nr == 45: # 0x2d
            st = get_struct(argp, rk.struct_drm_prime_handle)
            print(ret, "struct_drm_prime_handle", format_struct(st))        
        else:
            print("ioctl", f"{idir=} {size=} {itype=} {nr=} {fd=} {ret=} {argp=}", fn)
    else:
        print("ioctl", f"{idir=} {size=} {itype=} {nr=} {fd=} {ret=} {argp=}", fn) 
