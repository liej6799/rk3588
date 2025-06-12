 LD_LIBRARY_PATH="/usr/lib/python3.10" LD_PRELOAD="/root/tinygrad/extra/rockchip/preload_python.so" ./install/rknn_api_demo_Linux/rknn_create_mem_demo model/RK3588/mobilenet_v1.rknn model/dog_224x224.jpg 1
./install/rknn_api_demo_Linux/rknn_create_mem_demo model/RK3588/mobilenet_v1.rknn model/dog_224x224.jpg 1

./install/rknn_api_test_custom_cpu_op_Linux/rknn_api_test_custom_cpu_op model/RK3588/dual_residual_custom.rknn model/dog_224x224.jpg 1
