cmake_minimum_required(VERSION 3.6)

project(rknn_ops)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build rknnops program
add_executable(rknnops rknnops.c)

# Install target
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
install(TARGETS rknnops DESTINATION bin)