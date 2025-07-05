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

def handle_ioctl(fd, request, argp, ret):
  fn = os.readlink(f"/proc/self/fd/{fd}")
  idir, size, itype, nr = (request>>30), (request>>16)&0x3FFF, (request>>8)&0xFF, request&0xFF
  
  if fn == "/dev/dri/card1":
    
    if nr == 64: # 0x40 + 0x00
        st = get_struct(argp, rk.struct_rknpu_action)
        print(ret, "struct_rknpu_action", format_struct(st))   

    elif nr == 65: # 0x40 + 0x01
        st = get_struct(argp, rk.struct_rknpu_submit)
        data = format_struct(st)
        print(ret, "struct_rknpu_action", format_struct(st))  
        addr_str = data[6].split(":")[1].strip()
        if addr_str.startswith("0x") or addr_str.startswith("0X"):
            addr_int = int(addr_str, 16)
        else:
            addr_int = int(addr_str)
        data_ptr = ctypes.cast(addr_str, ctypes.POINTER(rk.struct_rknpu_task))
        data = data_ptr.contents
        print('FLAGS', data.flags)
        # print('INT_MASK', data.int_mask)
        # print('OP_IDX', data.op_idx)
        # print('ENABLE_MASK', data.enable_mask)
        # print('INT_CLEAR', data.int_clear)
        # print('INT_STATUS', data.int_status)
        # print('REGCFG_AMOUNT', data.regcfg_amount)
        # print('REGCFG_OFFSET', data.regcfg_offset)
        # print('REGCFG_ADDR', data.regcmd_addr)


# FLAGS 2991891568
# INT_MASK 70
# OP_IDX 65535
# ENABLE_MASK 70
# INT_CLEAR 70
# INT_STATUS 70

        

    elif nr == 66: # 0x40 + 0x02 
        st = get_struct(argp, rk.struct_rknpu_mem_create)
        print(ret, "struct_rknpu_mem_create", format_struct(st))

    elif nr == 67: # 0x40 + 0x03
        st = get_struct(argp, rk.struct_rknpu_mem_map)
        print(ret, "struct_rknpu_mem_map", format_struct(st))        
      
    elif nr == 68: # 0x40 + 0x04     
        st = get_struct(argp, rk.struct_rknpu_mem_destroy)
        print(ret, "struct_rknpu_mem_destroy", format_struct(st))

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