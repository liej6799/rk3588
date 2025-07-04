cmake_minimum_required(VERSION 3.6)

project(rknn_api_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../rknn-header/)
set(RKNN_RT_LIB ${RKNN_API_PATH}/librknnrt.so)
include_directories(${RKNN_API_PATH})

# stb
include_directories(${CMAKE_SOURCE_DIR}/include/)


set(CMAKE_INSTALL_RPATH "lib")

# rknn_create_mem_demo
add_executable(rknn_create_mem_demo
  src/rknn_create_mem_demo.cpp
)

target_link_libraries(rknn_create_mem_demo
  ${RKNN_RT_LIB}
)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_api_demo_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_create_mem_demo DESTINATION ./)


# rknn_api_test_custom_cpu_op
add_executable(rknn_api_test_custom_cpu_op
  src/rknn_api_test_custom_cpu_op.cpp
  ${CMAKE_SOURCE_DIR}/include/cnpy/cnpy.cpp
)


target_link_libraries(rknn_api_test_custom_cpu_op
  ${RKNN_RT_LIB}
)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_api_test_custom_cpu_op_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn_api_test_custom_cpu_op DESTINATION ./)


# At present, mmz　demo is only available under Android, but not for Linux temporarily,
# mainly because libmpimmz.so has no Linux implementation now. The API of the NPU itself supports Linux.
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  # rknn_with_mmz_demo
  add_executable(rknn_with_mmz_demo
    src/rknn_with_mmz_demo.cpp
  )

  target_link_libraries(rknn_with_mmz_demo
    ${RKNN_RT_LIB}
    ${MPI_MMZ_LIB}
  )

  # install target and libraries
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_api_demo_${CMAKE_SYSTEM_NAME})
  install(TARGETS rknn_with_mmz_demo DESTINATION ./)


  # rknn_set_internal_mem_from_fd_demo
  add_executable(rknn_set_internal_mem_from_fd_demo
    src/rknn_set_internal_mem_from_fd_demo.cpp
    ${CMAKE_SOURCE_DIR}/include/cnpy/cnpy.cpp
  )

  target_link_libraries(rknn_set_internal_mem_from_fd_demo
    ${RKNN_RT_LIB}
    ${MPI_MMZ_LIB}
  )

  # install target and libraries
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_api_demo_${CMAKE_SYSTEM_NAME})
  install(TARGETS rknn_set_internal_mem_from_fd_demo DESTINATION ./)
endif()

install(DIRECTORY model/${TARGET_SOC} DESTINATION ./model)
file(GLOB IMAGE_FILES "model/*.jpg")
install(FILES ${IMAGE_FILES} DESTINATION ./model/)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)