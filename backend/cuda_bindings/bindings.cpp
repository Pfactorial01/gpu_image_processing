#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include "image_filters.h"

namespace py = pybind11;

/**
 * Python wrapper for Gaussian Blur
 * Accepts NumPy array, processes on GPU, returns NumPy array + metrics
 */
py::dict gaussian_blur_py(
    py::array_t<unsigned char> input_array,
    float sigma,
    int radius,
    int level
) {
    // Get array info
    py::buffer_info buf = input_array.request();
    
    if (buf.ndim != 3) {
        throw std::runtime_error("Input must be 3D array (height, width, channels)");
    }
    
    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = buf.shape[2];
    
    if (channels != 1 && channels != 3 && channels != 4) {
        throw std::runtime_error("Channels must be 1, 3, or 4");
    }
    
    size_t imageSize = height * width * channels;
    unsigned char* h_input = static_cast<unsigned char*>(buf.ptr);
    
    // Allocate GPU memory
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    
    // Copy input to GPU
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);
    
    // Determine optimization level
    OptimizationLevel opt_level;
    switch(level) {
        case 1: opt_level = NAIVE; break;
        case 2: opt_level = TEXTURE_MEMORY; break;
        default:
            cudaFree(d_input);
            cudaFree(d_output);
            throw std::runtime_error("Level must be 1 (naive) or 2 (texture_memory) for Gaussian blur");
    }
    
    // Run Gaussian blur
    PerformanceMetrics metrics;
    cudaError_t err = gaussianBlur(
        d_input, d_output,
        width, height, channels,
        sigma, radius,
        opt_level,
        &metrics
    );
    
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
    
    // Allocate output array
    py::array_t<unsigned char> output_array({height, width, channels});
    py::buffer_info out_buf = output_array.request();
    unsigned char* h_output = static_cast<unsigned char*>(out_buf.ptr);
    
    // Copy result from GPU
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    
    // Return result + metrics as dict
    py::dict result;
    result["image"] = output_array;
    result["time_ms"] = metrics.time_ms;
    result["bandwidth_gbps"] = metrics.bandwidth_gbps;
    result["fps"] = metrics.fps;
    
    return result;
}

/**
 * Python wrapper for Box Blur
 */
py::dict box_blur_py(
    py::array_t<unsigned char> input_array,
    int radius,
    int level
) {
    py::buffer_info buf = input_array.request();
    
    if (buf.ndim != 3) {
        throw std::runtime_error("Input must be 3D array (height, width, channels)");
    }
    
    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = buf.shape[2];
    
    if (channels != 1 && channels != 3 && channels != 4) {
        throw std::runtime_error("Channels must be 1, 3, or 4");
    }
    
    size_t imageSize = height * width * channels;
    unsigned char* h_input = static_cast<unsigned char*>(buf.ptr);
    
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);
    
    OptimizationLevel opt_level;
    switch(level) {
        case 1: opt_level = NAIVE; break;
        case 2: opt_level = SHARED_MEMORY; break;
        default:
            cudaFree(d_input);
            cudaFree(d_output);
            throw std::runtime_error("Level must be 1 (naive) or 2 (shared_memory)");
    }
    
    PerformanceMetrics metrics;
    cudaError_t err = boxBlur(
        d_input, d_output,
        width, height, channels,
        radius,
        opt_level,
        &metrics
    );
    
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
    
    py::array_t<unsigned char> output_array({height, width, channels});
    py::buffer_info out_buf = output_array.request();
    unsigned char* h_output = static_cast<unsigned char*>(out_buf.ptr);
    
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    py::dict result;
    result["image"] = output_array;
    result["time_ms"] = metrics.time_ms;
    result["bandwidth_gbps"] = metrics.bandwidth_gbps;
    result["fps"] = metrics.fps;
    
    return result;
}

/**
 * Python wrapper for Sobel Edge Detection
 */
py::dict sobel_edge_detection_py(
    py::array_t<unsigned char> input_array,
    int level
) {
    py::buffer_info buf = input_array.request();
    
    if (buf.ndim != 3) {
        throw std::runtime_error("Input must be 3D array (height, width, channels)");
    }
    
    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = buf.shape[2];
    
    if (channels != 1 && channels != 3 && channels != 4) {
        throw std::runtime_error("Channels must be 1, 3, or 4");
    }
    
    size_t imageSize = height * width * channels;
    unsigned char* h_input = static_cast<unsigned char*>(buf.ptr);
    
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);
    
    OptimizationLevel opt_level;
    switch(level) {
        case 1: opt_level = NAIVE; break;
        case 2: opt_level = SHARED_MEMORY; break;
        default:
            cudaFree(d_input);
            cudaFree(d_output);
            throw std::runtime_error("Level must be 1 (naive) or 2 (shared_memory) for Sobel edge detection");
    }
    
    PerformanceMetrics metrics;
    cudaError_t err = sobelEdgeDetection(
        d_input, d_output,
        width, height, channels,
        opt_level,
        &metrics
    );
    
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
    
    py::array_t<unsigned char> output_array({height, width, channels});
    py::buffer_info out_buf = output_array.request();
    unsigned char* h_output = static_cast<unsigned char*>(out_buf.ptr);
    
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    py::dict result;
    result["image"] = output_array;
    result["time_ms"] = metrics.time_ms;
    result["bandwidth_gbps"] = metrics.bandwidth_gbps;
    result["fps"] = metrics.fps;
    
    return result;
}

// Python module definition
PYBIND11_MODULE(gpu_filters, m) {
    m.doc() = "GPU Image Processing Library - Python Bindings";
    
    m.def("gaussian_blur", &gaussian_blur_py,
          py::arg("image"),
          py::arg("sigma") = 2.0,
          py::arg("radius") = 3,
          py::arg("level") = 1,
          "Apply Gaussian blur to image using GPU\n\n"
          "Parameters:\n"
          "  image: NumPy array (height, width, channels)\n"
          "  sigma: Blur strength (default: 2.0)\n"
          "  radius: Kernel radius (default: 3)\n"
          "  level: Optimization level 1=naive, 2=texture_memory (default: 1)\n\n"
          "Returns:\n"
          "  dict with 'image' (blurred image) and metrics (time_ms, bandwidth_gbps, fps)");
    
    m.def("box_blur", &box_blur_py,
          py::arg("image"),
          py::arg("radius") = 3,
          py::arg("level") = 1,
          "Apply Box blur to image using GPU\n\n"
          "Parameters:\n"
          "  image: NumPy array (height, width, channels)\n"
          "  radius: Kernel radius (default: 3)\n"
          "  level: Optimization level 1=naive, 2=shared_memory (default: 1)\n\n"
          "Returns:\n"
          "  dict with 'image' (blurred image) and metrics (time_ms, bandwidth_gbps, fps)");
    
    m.def("sobel_edge_detection", &sobel_edge_detection_py,
          py::arg("image"),
          py::arg("level") = 1,
          "Apply Sobel edge detection to image using GPU\n\n"
          "Parameters:\n"
          "  image: NumPy array (height, width, channels)\n"
          "  level: Optimization level 1=naive, 2=shared_memory (default: 1)\n\n"
          "Returns:\n"
          "  dict with 'image' (edge map) and metrics (time_ms, bandwidth_gbps, fps)");
    
    // Expose optimization levels as constants
    m.attr("NAIVE") = py::int_(1);
    m.attr("SHARED_MEMORY") = py::int_(2);
    m.attr("TEXTURE_MEMORY") = py::int_(3);
}

