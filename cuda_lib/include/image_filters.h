#ifndef IMAGE_FILTERS_H
#define IMAGE_FILTERS_H

#include <cuda_runtime.h>

/**
 * GPU Image Processing Library
 * 
 * A high-performance CUDA-based image processing library
 * with multiple optimization levels for various filters.
 * 
 * Current filters: Gaussian Blur, Box Blur
 * Future: Sobel, Canny, Bilateral, Median
 */

// Struct to hold performance metrics
struct PerformanceMetrics {
    float time_ms;              // Execution time in milliseconds
    float bandwidth_gbps;       // Memory bandwidth in GB/s
    float fps;                  // Frames per second
};

// Optimization levels
enum OptimizationLevel {
    NAIVE = 1,           // Level 1: Global memory only
    SHARED_MEMORY = 2,   // Level 2: Shared memory optimization
    TEXTURE_MEMORY = 3,  // Level 3: Texture memory
    ADVANCED = 4         // Level 4: All optimizations
};

/**
 * Apply Gaussian blur to an image (grayscale or color)
 * 
 * @param d_input       Device pointer to input image
 * @param d_output      Device pointer to output image
 * @param width         Image width in pixels
 * @param height        Image height in pixels
 * @param channels      Number of channels (1=grayscale, 3=RGB, 4=RGBA)
 * @param sigma         Gaussian sigma (controls blur strength)
 * @param kernelRadius  Kernel radius (kernel size = 2*radius + 1)
 * @param level         Optimization level to use
 * @param metrics       Output performance metrics
 * @return cudaError_t  CUDA error code
 */
cudaError_t gaussianBlur(
    unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    float sigma,
    int kernelRadius,
    OptimizationLevel level,
    PerformanceMetrics* metrics
);

/**
 * Apply Box blur to an image (grayscale or color)
 * Box blur is a simple average of all pixels within the kernel radius
 * 
 * @param d_input       Device pointer to input image
 * @param d_output      Device pointer to output image
 * @param width         Image width in pixels
 * @param height        Image height in pixels
 * @param channels      Number of channels (1=grayscale, 3=RGB, 4=RGBA)
 * @param kernelRadius  Kernel radius (kernel size = 2*radius + 1)
 * @param level         Optimization level to use
 * @param metrics       Output performance metrics
 * @return cudaError_t  CUDA error code
 */
cudaError_t boxBlur(
    unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    int kernelRadius,
    OptimizationLevel level,
    PerformanceMetrics* metrics
);

/**
 * Apply Sobel edge detection to an image (grayscale or color)
 * 
 * Sobel edge detection computes image gradients using two 3x3 kernels:
 * - Gx (horizontal gradient): detects vertical edges
 * - Gy (vertical gradient): detects horizontal edges
 * 
 * The output is the gradient magnitude: sqrt(Gx^2 + Gy^2)
 * Higher values indicate stronger edges.
 * 
 * For color images, the image is first converted to grayscale using
 * the standard weights: 0.299*R + 0.587*G + 0.114*B
 * 
 * @param d_input       Device pointer to input image
 * @param d_output      Device pointer to output image (edge map)
 * @param width         Image width in pixels
 * @param height        Image height in pixels
 * @param channels      Number of channels (1=grayscale, 3=RGB, 4=RGBA)
 * @param level         Optimization level to use
 * @param metrics       Output performance metrics
 * @return cudaError_t  CUDA error code
 */
cudaError_t sobelEdgeDetection(
    unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    OptimizationLevel level,
    PerformanceMetrics* metrics
);

#endif // IMAGE_FILTERS_H

