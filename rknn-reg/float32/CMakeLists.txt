cmake_minimum_required(VERSION 3.6)

project(float32_addition_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Linker flags to allow undefined shared libs
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../rknn-header/)
set(RKNN_RT_LIB ${RKNN_API_PATH}/librknnrt.so)
include_directories(${RKNN_API_PATH})

# Add libdrm include path
include_directories(/usr/include/libdrm)
include_directories(/usr/include)

set(CMAKE_INSTALL_RPATH "lib")

# Build 'float32add' demo
add_executable(float32add float32add.cpp)
target_link_libraries(float32add ${RKNN_RT_LIB})

# Install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/float32_demo_${CMAKE_SYSTEM_NAME})
install(TARGETS float32add DESTINATION ./)

# Install libraries
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)