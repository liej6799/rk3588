0 struct_rknpu_mem_create ['handle:0x1', 'flags:0x0', 'size:0x1000', 'obj_addr:0xFFFF0001012A8000', 'dma_addr:0xFFFFF000', 'sram_size:0x0']
0 struct_rknpu_mem_map ['handle:0x1', 'reserved:0x0', 'offset:0x100000000']
0 struct_rknpu_mem_create ['handle:0x2', 'flags:0x8', 'size:0x1000', 'obj_addr:0xFFFF0001012A9400', 'dma_addr:0xFFFFE000', 'sram_size:0x0']
0 struct_rknpu_mem_map ['handle:0x2', 'reserved:0x0', 'offset:0x100001000']
0 struct_rknpu_mem_create ['handle:0x3', 'flags:0x0', 'size:0x1000', 'obj_addr:0xFFFF0001012AA000', 'dma_addr:0xFFFFD000', 'sram_size:0x0']
0 struct_rknpu_mem_map ['handle:0x3', 'reserved:0x0', 'offset:0x100002000']
0 struct_rknpu_mem_create ['handle:0x4', 'flags:0x0', 'size:0x1000', 'obj_addr:0xFFFF0001012A8800', 'dma_addr:0xFFFFC000', 'sram_size:0x0']
0 struct_rknpu_mem_map ['handle:0x4', 'reserved:0x0', 'offset:0x100003000']
0 struct_rknpu_mem_create ['handle:0x5', 'flags:0x0', 'size:0x1000', 'obj_addr:0xFFFF0001012AC400', 'dma_addr:0xFFFFB000', 'sram_size:0x0']
0 struct_rknpu_mem_map ['handle:0x5', 'reserved:0x0', 'offset:0x100004000']


## add.cpp
# mmap location 0x100000000
# Tasks Obj

# mmap Location 0x100001000
# 0x10010000000e4004
# NPU REG
# mmap location 0x100002000 ??

# mmap location 0x100003000 -> input 1.0
# npu_regs_map2[0]: 0x3c003c003c003c00
# npu_regs_map2[1]: 0x3c003c003c003c00
# npu_regs_map2[2]: 0x000000003c003c00

# mmap location 0x100004000 -> input 2.0
# npu_regs_map2[0]: 0x4000400040004000
# npu_regs_map2[1]: 0x4000400040004000
# npu_regs_map2[2]: 0x0000000040004000

# mmap locatoin 0x100005000 = result
# if add 0x100002000 is empty
# if mul 0x100002000 has value


1070 -> CNA Feature Address
5038 -> MMU Feature Address