cmake_minimum_required(VERSION 3.18)
project(GPUImageProcessing LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Add CUDA library subdirectory
add_subdirectory(cuda_lib)

# Build synthetic test executable
add_executable(test_gaussian_blur
    tests/test_gaussian_blur.cu
)

target_link_libraries(test_gaussian_blur PRIVATE
    gpu_image_filters
)

target_include_directories(test_gaussian_blur PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_lib/include
)

# Build real image test executable
add_executable(test_real_image
    tests/test_real_image.cu
)

target_link_libraries(test_real_image PRIVATE
    gpu_image_filters
)

target_include_directories(test_real_image PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_lib/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)

# Build comparison test (Level 1 vs Level 2)
add_executable(test_comparison
    tests/test_comparison.cu
)

target_link_libraries(test_comparison PRIVATE
    gpu_image_filters
)

target_include_directories(test_comparison PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_lib/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)

# Build box blur test
add_executable(test_box_blur
    tests/test_box_blur.cu
)

target_link_libraries(test_box_blur PRIVATE
    gpu_image_filters
)

target_include_directories(test_box_blur PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_lib/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
)

