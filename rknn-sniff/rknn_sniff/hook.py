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
        data = format_struct(st)
        print(ret, "struct_rknpu_submit", format_struct(st))  
        
        # Handle the mmap data - check if it's a dict (new format) or just offset (old format)
        if st.task_start in mmaped:
            mmap_data = mmaped[st.task_start]
            if isinstance(mmap_data, dict):
                print(f"Task {st.task_start} mmap: offset=0x{mmap_data['offset']:x}, size={mmap_data['size']}")

                print('mmap_obj', mmap_data['mmap_obj'])
                
                # Debug: Let's see what's actually in the mmap
                mmap_obj = mmap_data['mmap_obj']
                mmap_obj.seek(0)
                raw_data = mmap_obj.read(64)  # Read first 64 bytes
                print(f"Raw mmap data (first 64 bytes): {raw_data.hex()}")
                
                # Try to create the struct from buffer
                try:
                    my_struct = rk.struct_rknpu_task.from_buffer(mmap_obj)
                    print('my_struct.int_clear:', my_struct.int_clear)
                    print('my_struct.flags:', my_struct.flags)
                    print('my_struct.op_idx:', my_struct.op_idx)
                    print('my_struct.enable_mask:', my_struct.enable_mask)
                    print('my_struct.int_mask:', my_struct.int_mask)
                    print('my_struct.int_status:', my_struct.int_status)
                    print('my_struct.regcfg_amount:', my_struct.regcfg_amount)
                    print('my_struct.regcfg_offset:', my_struct.regcfg_offset)
                    print('my_struct.regcmd_addr:', hex(my_struct.regcmd_addr))
                except Exception as e:
                    print(f"Failed to create struct from buffer: {e}")
                
                # Reset position for next read
                mmap_obj.seek(0)
                




                
            else:
                print(f"Task {st.task_start} mmap offset: 0x{mmap_data:x}")
        else:
            print(f"No mmap data found for task {st.task_start}")



# FLAGS 2991891568
# INT_MASK 70
# OP_IDX 65535
# ENABLE_MASK 70
# INT_CLEAR 70
# INT_STATUS 70

        

    elif nr == 66: # 0x40 + 0x02 
        st = get_struct(argp, rk.struct_rknpu_mem_create)
        print(ret, "struct_rknpu_mem_create", format_struct(st))
        alloc_sizes[st.handle] = st.size


    elif nr == 67: # 0x40 + 0x03
        st = get_struct(argp, rk.struct_rknpu_mem_map)
        print(ret, "struct_rknpu_mem_map", format_struct(st))   
        mmaped[st.handle] = st.offset
        
        # Create Python mmap object for this handle
        try:
            # We need the file descriptor (fd) and the offset from the struct
            # The fd is the same one used in the ioctl call
            size = alloc_sizes.get(st.handle, 1024)  # Default to 1024 if not found
            
                        # Create mmap object using Python's mmap module
            # This is equivalent to: void *regmap = mmap(NULL, 1024, PROT_READ | PROT_WRITE, MAP_SHARED, fd, regcmd_offset);
            mmap_obj = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE, offset=st.offset)
            
            # Get the pointer address

            print(f"Created Python mmap for handle {st.handle}:")
            print(f"  Size: {size}, Offset: 0x{st.offset:x}")
            print(f"  Mmap object: {mmap_obj}")
            
            # Store the mmap object for later use
            mmaped[st.handle] = {
                'offset': st.offset,
                'size': size,
                'mmap_obj': mmap_obj,

            }
            
            # If this looks like a regcmd mapping (1024 bytes), try to parse it
            if size == 1024:
                print(f"Detected potential regcmd mapping for handle {st.handle}")
                # Check if this is actually regcmd data or task data
                mmap_obj.seek(0)
                first_bytes = mmap_obj.read(16)
                print(f"First 16 bytes: {first_bytes.hex()}")
                
                # Try to interpret as different structures
                try:
                    # Try as struct_rknpu_task
                    mmap_obj.seek(0)
                    task_struct = rk.struct_rknpu_task.from_buffer(mmap_obj)
                    print(f"Interpreted as task struct - int_clear: {task_struct.int_clear}")
                except:
                    print("Not a valid task struct")
                
                mmap_obj.seek(0)  # Reset position
                
        except Exception as e:
            print(f"Failed to create mmap for handle {st.handle}: {e}")
            mmaped[st.handle] = st.offset


      
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