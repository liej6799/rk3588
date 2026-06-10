This is an attempt reverse engineering librknnrt.so

1. As we have completed recreate rknn file from scratch (can refer to /data/rk3588/rknn-creation) Able to test few scenario such as 
- EW Ops (Element Wise Operations including ADD/SUB/MUL/DIV)
- CPU Ops (AND, OR, XOR)
- Hybrid Ops (Combination on EW and CPU Ops)

2. We are able to integrate into tinygrad's UOP operations (can refer to /data/rk3588/rknn-decode) Able to perform unroll, break down complex command into simpler one that can be represented under ONNX Graph and run on the NPU with the RKNN file created from scratch from the earlier reverse engineering attempt.

3. There are some limitation as currently calling the vendor runtime library which is still closed (librknnrt.so)

4. You can use these tools to help check what does this library do (the function) and how it manages to load rknn file and seperate out the CPU and NPU Ops, reshape the tensor call NPU Submit IOCTL Operation. 
Tools:
- radare2
- gdb
- ioctl

5. Inside the library, there are some function such as eval performance to show the whole graph for your reference after load the rknn file.

6. Our goal should be a function that can replicate the said library. And are able to integrate with tinygrad uops, handle hybrid ops, reshape and resize if necessary, handle int32, float32 structure (this will be handled under CPUOps due to NPU limitation).  

7. To understand the rknpu raw command (rc_template) can refer to the /data/rkt folder for the documentation.



