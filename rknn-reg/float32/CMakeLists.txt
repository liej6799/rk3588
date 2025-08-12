cmake_minimum_required(VERSION 3.6)

project(float32_addition_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Linker flags to allow undefined shared libs
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# rknn api - handle both standalone and root build
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # Building standalone
    set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../rknn-header/)
else()
    # Building from root
    set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../rknn-header/)
endif()
set(RKNN_RT_LIB ${RKNN_API_PATH}/librknnrt.so)
include_directories(${RKNN_API_PATH})

# Add libdrm include path
include_directories(/usr/include/libdrm)
include_directories(/usr/include)

set(CMAKE_INSTALL_RPATH "lib")

# Build 'float32add' demo
add_executable(float32add float32add.cpp)
target_link_libraries(float32add ${RKNN_RT_LIB})

# Build 'float32addreg' demo
add_executable(float32addreg float32.cpp)
target_link_libraries(float32addreg ${RKNN_RT_LIB} drm)

# Install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/float32_demo_${CMAKE_SYSTEM_NAME})
install(TARGETS float32add float32addreg DESTINATION ./)

# Install libraries
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)