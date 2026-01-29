cmake_minimum_required(VERSION 3.18)
project(GPUImageFilters LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures (adjust based on your GPU)
# Common options:
#   - 60: Pascal (GTX 10xx, Tesla P100)
#   - 70: Volta (Tesla V100)
#   - 75: Turing (RTX 20xx, GTX 16xx)
#   - 80: Ampere (RTX 30xx, A100)
#   - 86: Ampere (RTX 30xx mobile)
#   - 89: Ada Lovelace (RTX 40xx)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create CUDA library for image filters
add_library(gpu_image_filters STATIC
    src/image_filters.cu
)

# Set CUDA-specific properties
set_target_properties(gpu_image_filters PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(gpu_image_filters PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gpu_image_filters PUBLIC
    CUDA::cudart
)

# Enable verbose output for debugging
set(CMAKE_VERBOSE_MAKEFILE ON)

