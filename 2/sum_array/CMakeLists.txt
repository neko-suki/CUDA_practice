cmake_minimum_required(VERSION 3.1)
project(sum_array)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDA REQUIRED)
message("--CUDA--")
message("Version: " ${CUDA_VERSION})
message("Library: " ${CUDA_CUDA_LIBRARY})
message("Runtime: " ${CUDA_CUDAART_LIBRARY})


if (CMAKE_COMPILER_IS_GNUCXX)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Werror -Wfatal-errors")
endif()

# To Be fixed
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=-O3, -Wall, -Wextra, -Werror, -Wfatal-errors")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -arch=compute_61")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")

file(GLOB SOURCES "*.cpp" "*.hpp" "*.cu")
cuda_add_executable(sum_array ${SOURCES})

