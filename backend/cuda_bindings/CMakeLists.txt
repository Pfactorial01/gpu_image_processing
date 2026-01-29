cmake_minimum_required(VERSION 3.18)
project(GPUFiltersPython LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via find_package, trying pip install location")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(pybind11_DIR)
        message(STATUS "Found pybind11 at: ${pybind11_DIR}")
        set(pybind11_DIR ${pybind11_DIR} CACHE PATH "Path to pybind11 CMake files")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
    endif()
endif()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include the main library
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../cuda_lib/include)

# Create Python module
pybind11_add_module(gpu_filters bindings.cpp)

# Link against our CUDA library
target_link_libraries(gpu_filters PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../build/cuda_lib/libgpu_image_filters.a
    CUDA::cudart
    CUDA::cudart_static
)

# Enable CUDA device linking
set_target_properties(gpu_filters PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Set output directory
set_target_properties(gpu_filters PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)

message(STATUS "Python module will be built as: gpu_filters")
message(STATUS "Install with: pip install -e .")

