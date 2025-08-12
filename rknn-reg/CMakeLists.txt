cmake_minimum_required(VERSION 3.6)

project(rknn_regression_tests)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Global settings
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# Add subdirectories for each demo
add_subdirectory(float32)
add_subdirectory(add)
add_subdirectory(int)
add_subdirectory(andor)
add_subdirectory(matmul)

# Optional: Create a custom target to build all demos
add_custom_target(all_demos
    DEPENDS float32add float32addreg
    COMMENT "Building all rknn regression test demos"
)