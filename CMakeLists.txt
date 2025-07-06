cmake_minimum_required(VERSION 3.10)
project(block_sparse LANGUAGES CXX ASM)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/matmul.cpp
    src/tensor.cpp
    src/sparse_micro_kernel.s
    src/bindings.cpp
)

# Create the Python module
pybind11_add_module(block_sparse ${SOURCES})

# Link libraries (if needed, e.g., for OpenMP)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(block_sparse PRIVATE OpenMP::OpenMP_CXX)
endif()
set(SOURCES2
    src/tensor.cpp
    src/matmul.cpp
    src/sparse_micro_kernel.s
)
add_library(block_sparse_test STATIC ${SOURCES2})
# 
# Set output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
