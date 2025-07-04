cmake_minimum_required(VERSION 3.6)

project(rknn_api_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Linker flags to allow undefined shared libs
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../rknn-header/)
set(RKNN_RT_LIB ${RKNN_API_PATH}/librknnrt.so)
include_directories(${RKNN_API_PATH})

# stb
include_directories(${CMAKE_SOURCE_DIR}/include/)

set(CMAKE_INSTALL_RPATH "lib")

# Build 'add' demo
add_executable(add add.cpp)
target_link_libraries(add ${RKNN_RT_LIB})

# Install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_api_demo_${CMAKE_SYSTEM_NAME})
install(TARGETS add DESTINATION ./)

# Install model and image files
install(DIRECTORY model/${TARGET_SOC} DESTINATION ./model)
file(GLOB IMAGE_FILES "model/*.jpg")
install(FILES ${IMAGE_FILES} DESTINATION ./model/)

# Install libraries
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)