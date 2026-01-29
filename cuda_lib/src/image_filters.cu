#include "image_filters.h"
#include <cuda_runtime.h>
#include <texture_types.h>
#include <cmath>
#include <cstdio>
#include <cstring>

// ============================================================================
// Constant Memory for Gaussian Kernel
// ============================================================================

// Constant memory for Gaussian kernel (max 64 elements = radius up to 31)
__constant__ float c_gaussianKernel[64];
__constant__ int c_kernelRadius;
__constant__ int c_kernelSize;

// ============================================================================
// Helper function: Generate 1D Gaussian kernel
// ============================================================================

/**
 * Compute 1D Gaussian kernel weights
 * Formula: G(x) = (1/sqrt(2*pi*sigma^2)) * e^(-x^2 / (2*sigma^2))
 */
void generateGaussianKernel(float* kernel, int radius, float sigma) {
    float sum = 0.0f;
    
    // Calculate raw Gaussian values
    for (int i = -radius; i <= radius; i++) {
        float x = static_cast<float>(i);
        float value = expf(-(x * x) / (2.0f * sigma * sigma));
        kernel[radius + i] = value;
        sum += value;
    }
    
    // Normalize so weights sum to 1.0
    for (int i = 0; i < 2 * radius + 1; i++) {
        kernel[i] /= sum;
    }
    
    // Debug: Print kernel weights
    printf("Gaussian kernel (sigma=%.1f, radius=%d): [", sigma, radius);
    for (int i = 0; i < 2 * radius + 1; i++) {
        printf("%.3f", kernel[i]);
        if (i < 2 * radius) printf(", ");
    }
    printf("]\n");
}

// ============================================================================
// LEVEL 1: NAIVE KERNELS (Global Memory Only)
// ============================================================================

/**
 * Naive horizontal blur kernel (supports multi-channel images)
 * Each thread:
 *   1. Calculates its pixel position (x, y)
 *   2. Reads neighbors from global memory for each channel
 *   3. Applies Gaussian weights
 *   4. Writes result for all channels
 * 
 * Memory layout: interleaved (e.g., RGB: RGBRGBRGB...)
 */
__global__ void gaussianBlurHorizontalNaive(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels,
    const float* kernel,
    int radius
) {
    // Calculate pixel position this thread handles
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check
    if (x >= width || y >= height) return;
    
    // Process each channel independently
    for (int c = 0; c < channels; c++) {
        // Accumulate weighted sum for this channel
        float sum = 0.0f;
        
        // Convolve with horizontal kernel
        for (int i = -radius; i <= radius; i++) {
            int neighborX = x + i;
            
            // Clamp to image boundaries
            if (neighborX < 0) neighborX = 0;
            if (neighborX >= width) neighborX = width - 1;
            
            // Read pixel from global memory (SLOW - this is why it's naive)
            // Index: (y * width + neighborX) * channels + c
            unsigned char pixelValue = input[(y * width + neighborX) * channels + c];
            
            // Multiply by kernel weight and accumulate
            sum += pixelValue * kernel[radius + i];
        }
        
        // Write result for this channel
        output[(y * width + x) * channels + c] = (unsigned char)(sum + 0.5f);
    }
}

/**
 * Naive vertical blur kernel (supports multi-channel images)
 * Nearly identical to horizontal, but moves vertically instead
 */
__global__ void gaussianBlurVerticalNaive(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels,
    const float* kernel,
    int radius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Process each channel independently
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        // Convolve with vertical kernel
        for (int i = -radius; i <= radius; i++) {
            int neighborY = y + i;
            
            // Clamp to boundaries
            if (neighborY < 0) neighborY = 0;
            if (neighborY >= height) neighborY = height - 1;
            
            // Read from global memory
            unsigned char pixelValue = input[(neighborY * width + x) * channels + c];
            
            sum += pixelValue * kernel[radius + i];
        }
        
        output[(y * width + x) * channels + c] = (unsigned char)(sum + 0.5f);
    }
}

// ============================================================================
// LEVEL 2: TEXTURE MEMORY + CONSTANT MEMORY + VECTORIZED ACCESS
// ============================================================================

// Texture object for input image (will be created at runtime)
// Using texture objects (modern CUDA API) instead of texture references

/**
 * Level 2 horizontal blur kernel with:
 * - Texture memory for input (better caching, automatic boundary handling) - for single channel
 * - Constant memory for kernel (cached, broadcast efficiently)
 * - Vectorized access for multi-channel images (optimized global memory)
 */
__global__ void gaussianBlurHorizontalLevel2(
    const unsigned char* input,  // For multi-channel vectorized access
    unsigned char* output,
    int width,
    int height,
    int channels
) {
    // Calculate pixel position this thread handles
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check
    if (x >= width || y >= height) return;
    
    // Get kernel radius and size from constant memory
    int radius = c_kernelRadius;
    
    // Vectorized processing for 4-channel images (RGBA)
    if (channels == 4) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        // Convolve with horizontal kernel using vectorized global memory access
        for (int i = -radius; i <= radius; i++) {
            int neighborX = x + i;
            
            // Clamp to image boundaries
            neighborX = max(0, min(width - 1, neighborX));
            
            // Vectorized load: read 4 channels at once
            uchar4 pixel = *reinterpret_cast<const uchar4*>(
                &input[(y * width + neighborX) * channels]
            );
            
            float weight = c_gaussianKernel[radius + i];
            
            sum.x += pixel.x * weight;
            sum.y += pixel.y * weight;
            sum.z += pixel.z * weight;
            sum.w += pixel.w * weight;
        }
        
        // Write result using vectorized store
        uchar4 result = make_uchar4(
            (unsigned char)(sum.x + 0.5f),
            (unsigned char)(sum.y + 0.5f),
            (unsigned char)(sum.z + 0.5f),
            (unsigned char)(sum.w + 0.5f)
        );
        
        *reinterpret_cast<uchar4*>(&output[(y * width + x) * channels]) = result;
    }
    // Optimized processing for 3-channel images (RGB)
    // Note: Can't use uchar4 for RGB due to alignment (3 bytes != 4-byte aligned)
    // Still benefits from constant memory for kernel weights
    else if (channels == 3) {
        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        
        for (int i = -radius; i <= radius; i++) {
            int neighborX = x + i;
            neighborX = max(0, min(width - 1, neighborX));
            
            // Read RGB channels individually (still coalesced, benefits from constant memory)
            int pixelIdx = (y * width + neighborX) * channels;
            unsigned char r = input[pixelIdx + 0];
            unsigned char g = input[pixelIdx + 1];
            unsigned char b = input[pixelIdx + 2];
            
            float weight = c_gaussianKernel[radius + i];
            
            sum.x += r * weight;
            sum.y += g * weight;
            sum.z += b * weight;
        }
        
        // Write result (3 channels)
        int outIdx = (y * width + x) * channels;
        output[outIdx + 0] = (unsigned char)(sum.x + 0.5f);
        output[outIdx + 1] = (unsigned char)(sum.y + 0.5f);
        output[outIdx + 2] = (unsigned char)(sum.z + 0.5f);
    }
    // Scalar processing for grayscale (1 channel) - use optimized global memory
    else {
        float sum = 0.0f;
        
        for (int i = -radius; i <= radius; i++) {
            int neighborX = x + i;
            neighborX = max(0, min(width - 1, neighborX));
            
            // Optimized global memory access (still benefits from constant memory for kernel)
            unsigned char pixel = input[(y * width + neighborX) * channels];
            sum += pixel * c_gaussianKernel[radius + i];
        }
        
        output[(y * width + x) * channels] = (unsigned char)(sum + 0.5f);
    }
}

/**
 * Level 2 vertical blur kernel with same optimizations
 */
__global__ void gaussianBlurVerticalLevel2(
    const unsigned char* input,  // For multi-channel vectorized access
    unsigned char* output,
    int width,
    int height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int radius = c_kernelRadius;
    
    // Vectorized processing for 4-channel images (RGBA)
    if (channels == 4) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        for (int i = -radius; i <= radius; i++) {
            int neighborY = y + i;
            neighborY = max(0, min(height - 1, neighborY));
            
            // Vectorized load: read 4 channels at once
            uchar4 pixel = *reinterpret_cast<const uchar4*>(
                &input[(neighborY * width + x) * channels]
            );
            
            float weight = c_gaussianKernel[radius + i];
            
            sum.x += pixel.x * weight;
            sum.y += pixel.y * weight;
            sum.z += pixel.z * weight;
            sum.w += pixel.w * weight;
        }
        
        uchar4 result = make_uchar4(
            (unsigned char)(sum.x + 0.5f),
            (unsigned char)(sum.y + 0.5f),
            (unsigned char)(sum.z + 0.5f),
            (unsigned char)(sum.w + 0.5f)
        );
        
        *reinterpret_cast<uchar4*>(&output[(y * width + x) * channels]) = result;
    }
    // Optimized processing for 3-channel images (RGB)
    // Note: Can't use uchar4 for RGB due to alignment (3 bytes != 4-byte aligned)
    // Still benefits from constant memory for kernel weights
    else if (channels == 3) {
        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        
        for (int i = -radius; i <= radius; i++) {
            int neighborY = y + i;
            neighborY = max(0, min(height - 1, neighborY));
            
            // Read RGB channels individually (still coalesced, benefits from constant memory)
            int pixelIdx = (neighborY * width + x) * channels;
            unsigned char r = input[pixelIdx + 0];
            unsigned char g = input[pixelIdx + 1];
            unsigned char b = input[pixelIdx + 2];
            
            float weight = c_gaussianKernel[radius + i];
            
            sum.x += r * weight;
            sum.y += g * weight;
            sum.z += b * weight;
        }
        
        // Write result (3 channels)
        int outIdx = (y * width + x) * channels;
        output[outIdx + 0] = (unsigned char)(sum.x + 0.5f);
        output[outIdx + 1] = (unsigned char)(sum.y + 0.5f);
        output[outIdx + 2] = (unsigned char)(sum.z + 0.5f);
    }
    // Scalar processing for grayscale (1 channel) - use optimized global memory
    else {
        float sum = 0.0f;
        
        for (int i = -radius; i <= radius; i++) {
            int neighborY = y + i;
            neighborY = max(0, min(height - 1, neighborY));
            
            // Optimized global memory access (still benefits from constant memory for kernel)
            unsigned char pixel = input[(neighborY * width + x) * channels];
            sum += pixel * c_gaussianKernel[radius + i];
        }
        
        output[(y * width + x) * channels] = (unsigned char)(sum + 0.5f);
    }
}

// ============================================================================
// LEVEL 2: SHARED MEMORY KERNELS (for reference, not used for Gaussian)
// ============================================================================

// ============================================================================
// BOX BLUR KERNELS
// ============================================================================

/**
 * Naive horizontal box blur kernel
 * Box blur = simple average (no weights needed!)
 * Simpler and faster than Gaussian blur
 */
__global__ void boxBlurHorizontalNaive(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels,
    int radius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int kernelSize = 2 * radius + 1;
    float invKernelSize = 1.0f / kernelSize;
    
    // Process each channel
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        // Simple sum (no weights!)
        for (int i = -radius; i <= radius; i++) {
            int neighborX = x + i;
            
            // Clamp to boundaries
            if (neighborX < 0) neighborX = 0;
            if (neighborX >= width) neighborX = width - 1;
            
            sum += input[(y * width + neighborX) * channels + c];
        }
        
        // Simple average: divide by kernel size
        output[(y * width + x) * channels + c] = (unsigned char)(sum * invKernelSize + 0.5f);
    }
}

/**
 * Naive vertical box blur kernel
 */
__global__ void boxBlurVerticalNaive(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels,
    int radius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int kernelSize = 2 * radius + 1;
    float invKernelSize = 1.0f / kernelSize;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int i = -radius; i <= radius; i++) {
            int neighborY = y + i;
            
            if (neighborY < 0) neighborY = 0;
            if (neighborY >= height) neighborY = height - 1;
            
            sum += input[(neighborY * width + x) * channels + c];
        }
        
        output[(y * width + x) * channels + c] = (unsigned char)(sum * invKernelSize + 0.5f);
    }
}

// ============================================================================
// LEVEL 2: SHARED MEMORY BOX BLUR KERNELS
// ============================================================================

/**
 * Shared memory horizontal box blur kernel
 * 
 * Optimizations:
 * - Tile-based processing with shared memory
 * - Cooperative loading with halo regions (left/right)
 * - Reduced global memory access
 * 
 * Shared memory layout: (BLOCK_SIZE + 2*radius) × channels bytes per row
 * This includes a radius-pixel halo on left and right sides.
 */
__global__ void boxBlurHorizontalShared(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int channels,
    int radius
) {
    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global pixel coordinates
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    // Shared memory width (tile + halo on each side)
    int s_width = blockDim.x + 2 * radius;
    
    // Allocate shared memory (stores all channels interleaved)
    extern __shared__ unsigned char s_tile[];
    
    // Helper macro for shared memory indexing
    // Layout: row-major with interleaved channels
    // s_tile[row * s_width * channels + col * channels + ch]
    #define S_IDX(row, col, ch) ((row) * s_width * channels + (col) * channels + (ch))
    
    // ========================================================================
    // Cooperative tile loading with halo regions
    // ========================================================================
    
    // Each thread loads its main pixel (with channel data)
    int s_col = tx + radius;  // Position in shared memory (offset by halo)
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            s_tile[S_IDX(ty, s_col, c)] = input[(y * width + x) * channels + c];
        }
    }
    
    // Load left halo (threads with tx < radius load left halo pixels)
    if (tx < radius) {
        int haloX = blockIdx.x * blockDim.x - radius + tx;  // Global X for halo
        haloX = max(0, haloX);  // Clamp to left boundary
        
        if (y < height) {
            for (int c = 0; c < channels; c++) {
                s_tile[S_IDX(ty, tx, c)] = input[(y * width + haloX) * channels + c];
            }
        }
    }
    
    // Load right halo (threads with tx >= blockDim.x - radius load right halo pixels)
    if (tx >= blockDim.x - radius) {
        int haloOffset = tx - (blockDim.x - radius);  // 0, 1, ..., radius-1
        int haloX = blockIdx.x * blockDim.x + blockDim.x + haloOffset;  // Global X for halo
        haloX = min(width - 1, haloX);  // Clamp to right boundary
        
        int s_halo_col = blockDim.x + radius + haloOffset;  // Position in shared memory
        
        if (y < height && x < width) {
            for (int c = 0; c < channels; c++) {
                s_tile[S_IDX(ty, s_halo_col, c)] = input[(y * width + haloX) * channels + c];
            }
        }
    }
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    // ========================================================================
    // Compute box blur average using shared memory
    // ========================================================================
    
    // Boundary check
    if (x >= width || y >= height) return;
    
    int kernelSize = 2 * radius + 1;
    float invKernelSize = 1.0f / kernelSize;
    
    // Process each channel
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        // Sum pixels from shared memory (much faster than global memory)
        for (int i = 0; i < kernelSize; i++) {
            // s_col - radius + i gives us the range [s_col - radius, s_col + radius]
            // But we need to handle boundary clamping
            int col_idx = s_col - radius + i;
            
            // Handle boundary clamping for shared memory indices
            // This is needed when the global position would be out of bounds
            int globalNeighborX = x - radius + i;
            if (globalNeighborX < 0) {
                col_idx = 0;  // Clamp to first element in shared memory row
            } else if (globalNeighborX >= width) {
                col_idx = s_col + (width - 1 - x);  // Clamp to last valid element
            }
            
            sum += s_tile[S_IDX(ty, col_idx, c)];
        }
        
        // Write result to global memory
        output[(y * width + x) * channels + c] = (unsigned char)(sum * invKernelSize + 0.5f);
    }
    
    #undef S_IDX
}

/**
 * Shared memory vertical box blur kernel
 * 
 * Optimizations:
 * - Tile-based processing with shared memory
 * - Cooperative loading with halo regions (top/bottom)
 * - Reduced global memory access
 * 
 * Shared memory layout: (BLOCK_SIZE + 2*radius) × channels bytes per column
 * This includes a radius-pixel halo on top and bottom sides.
 */
__global__ void boxBlurVerticalShared(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int channels,
    int radius
) {
    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global pixel coordinates
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    // Shared memory height (tile + halo on each side)
    int s_height = blockDim.y + 2 * radius;
    
    // Allocate shared memory (stores all channels interleaved)
    extern __shared__ unsigned char s_tile[];
    
    // Helper macro for shared memory indexing
    // Layout: column-major style for vertical processing
    // s_tile[col * s_height * channels + row * channels + ch]
    #define S_IDX(row, col, ch) ((col) * s_height * channels + (row) * channels + (ch))
    
    // ========================================================================
    // Cooperative tile loading with halo regions
    // ========================================================================
    
    // Each thread loads its main pixel (with channel data)
    int s_row = ty + radius;  // Position in shared memory (offset by halo)
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            s_tile[S_IDX(s_row, tx, c)] = input[(y * width + x) * channels + c];
        }
    }
    
    // Load top halo (threads with ty < radius load top halo pixels)
    if (ty < radius) {
        int haloY = blockIdx.y * blockDim.y - radius + ty;  // Global Y for halo
        haloY = max(0, haloY);  // Clamp to top boundary
        
        if (x < width) {
            for (int c = 0; c < channels; c++) {
                s_tile[S_IDX(ty, tx, c)] = input[(haloY * width + x) * channels + c];
            }
        }
    }
    
    // Load bottom halo (threads with ty >= blockDim.y - radius load bottom halo pixels)
    if (ty >= blockDim.y - radius) {
        int haloOffset = ty - (blockDim.y - radius);  // 0, 1, ..., radius-1
        int haloY = blockIdx.y * blockDim.y + blockDim.y + haloOffset;  // Global Y for halo
        haloY = min(height - 1, haloY);  // Clamp to bottom boundary
        
        int s_halo_row = blockDim.y + radius + haloOffset;  // Position in shared memory
        
        if (x < width && y < height) {
            for (int c = 0; c < channels; c++) {
                s_tile[S_IDX(s_halo_row, tx, c)] = input[(haloY * width + x) * channels + c];
            }
        }
    }
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    // ========================================================================
    // Compute box blur average using shared memory
    // ========================================================================
    
    // Boundary check
    if (x >= width || y >= height) return;
    
    int kernelSize = 2 * radius + 1;
    float invKernelSize = 1.0f / kernelSize;
    
    // Process each channel
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        // Sum pixels from shared memory (much faster than global memory)
        for (int i = 0; i < kernelSize; i++) {
            // s_row - radius + i gives us the range [s_row - radius, s_row + radius]
            int row_idx = s_row - radius + i;
            
            // Handle boundary clamping for shared memory indices
            int globalNeighborY = y - radius + i;
            if (globalNeighborY < 0) {
                row_idx = 0;  // Clamp to first element in shared memory column
            } else if (globalNeighborY >= height) {
                row_idx = s_row + (height - 1 - y);  // Clamp to last valid element
            }
            
            sum += s_tile[S_IDX(row_idx, tx, c)];
        }
        
        // Write result to global memory
        output[(y * width + x) * channels + c] = (unsigned char)(sum * invKernelSize + 0.5f);
    }
    
    #undef S_IDX
}

// ============================================================================
// Main Gaussian Blur Function
// ============================================================================

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
) {
    cudaError_t err = cudaSuccess;
    
    // Validate optimization level
    if (level != NAIVE && level != TEXTURE_MEMORY) {
        fprintf(stderr, "Error: Only NAIVE and TEXTURE_MEMORY (Level 2) levels are currently implemented for Gaussian blur\n");
        return cudaErrorNotSupported;
    }
    
    // ========================================================================
    // Step 1: Generate Gaussian kernel on host
    // ========================================================================
    
    int kernelSize = 2 * kernelRadius + 1;
    float* h_kernel = new float[kernelSize];
    generateGaussianKernel(h_kernel, kernelRadius, sigma);
    
    // ========================================================================
    // Step 2: Copy kernel to device (constant memory for Level 2, global for Level 1)
    // ========================================================================
    
    float* d_kernel = nullptr;
    
    if (level == NAIVE) {
        // Level 1: Use global memory for kernel
        err = cudaMalloc(&d_kernel, kernelSize * sizeof(float));
        if (err != cudaSuccess) {
            delete[] h_kernel;
            return err;
        }
        
        err = cudaMemcpy(d_kernel, h_kernel, kernelSize * sizeof(float), 
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_kernel);
            delete[] h_kernel;
            return err;
        }
    } else if (level == TEXTURE_MEMORY) {
        // Level 2: Use constant memory for kernel
        if (kernelSize > 64) {
            fprintf(stderr, "Error: Kernel size %d exceeds constant memory limit (64)\n", kernelSize);
            delete[] h_kernel;
            return cudaErrorInvalidValue;
        }
        
        err = cudaMemcpyToSymbol(c_gaussianKernel, h_kernel, kernelSize * sizeof(float));
        if (err != cudaSuccess) {
            delete[] h_kernel;
            return err;
        }
        
        // Also copy radius and size to constant memory
        err = cudaMemcpyToSymbol(c_kernelRadius, &kernelRadius, sizeof(int));
        if (err != cudaSuccess) {
            delete[] h_kernel;
            return err;
        }
        
        err = cudaMemcpyToSymbol(c_kernelSize, &kernelSize, sizeof(int));
        if (err != cudaSuccess) {
            delete[] h_kernel;
            return err;
        }
    }
    
    // ========================================================================
    // Step 3: Allocate temporary buffer for intermediate result
    // ========================================================================
    
    unsigned char* d_temp;
    size_t tempSize = width * height * channels * sizeof(unsigned char);
    err = cudaMalloc(&d_temp, tempSize);
    if (err != cudaSuccess) {
        if (d_kernel) cudaFree(d_kernel);
        delete[] h_kernel;
        return err;
    }
    
    // ========================================================================
    // Step 4: Configure thread blocks
    // ========================================================================
    
    // Use 16x16 thread blocks (256 threads per block)
    dim3 blockSize(16, 16);
    
    // Calculate grid size to cover entire image
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    printf("Launching kernels: Grid(%d, %d), Block(%d, %d)\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    
    // ========================================================================
    // Step 5: Level 2 optimizations are:
    // - Constant memory for kernel (already done above)
    // - Vectorized access for multi-channel (in kernel code)
    // Note: Texture memory setup removed for simplicity - constant memory
    // and vectorized access provide the main performance benefits
    // ========================================================================
    
    // ========================================================================
    // Step 6: Create CUDA events for timing
    // ========================================================================
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========================================================================
    // Step 7: Execute blur passes
    // ========================================================================
    
    cudaEventRecord(start);
    
    if (level == NAIVE) {
        // ====================================================================
        // LEVEL 1: Naive implementation (global memory only)
        // ====================================================================
        
        gaussianBlurHorizontalNaive<<<gridSize, blockSize>>>(
            d_input,
            d_temp,
            width,
            height,
            channels,
            d_kernel,
            kernelRadius
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Horizontal kernel launch failed: %s\n", 
                    cudaGetErrorString(err));
            cudaFree(d_temp);
            if (d_kernel) cudaFree(d_kernel);
            delete[] h_kernel;
            return err;
        }
        
        gaussianBlurVerticalNaive<<<gridSize, blockSize>>>(
            d_temp,
            d_output,
            width,
            height,
            channels,
            d_kernel,
            kernelRadius
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Vertical kernel launch failed: %s\n", 
                    cudaGetErrorString(err));
            cudaFree(d_temp);
            if (d_kernel) cudaFree(d_kernel);
            delete[] h_kernel;
            return err;
        }
        
    } else if (level == TEXTURE_MEMORY) {
        // ====================================================================
        // LEVEL 2: Constant memory + vectorized access
        // ====================================================================
        
        // First pass: horizontal blur (input -> temp)
        gaussianBlurHorizontalLevel2<<<gridSize, blockSize>>>(
            d_input,
            d_temp,
            width,
            height,
            channels
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Horizontal Level 2 kernel launch failed: %s\n", 
                    cudaGetErrorString(err));
            cudaFree(d_temp);
            delete[] h_kernel;
            return err;
        }
        
        // Second pass: vertical blur (temp -> output)
        gaussianBlurVerticalLevel2<<<gridSize, blockSize>>>(
            d_temp,
            d_output,
            width,
            height,
            channels
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Vertical Level 2 kernel launch failed: %s\n", 
                    cudaGetErrorString(err));
            cudaFree(d_temp);
            delete[] h_kernel;
            return err;
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // ========================================================================
    // Step 8: Calculate performance metrics
    // ========================================================================
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate memory bandwidth
    // We read the entire image twice and write twice (2 passes)
    size_t bytesTransferred = width * height * channels * sizeof(unsigned char) * 4;
    float bandwidth_gbps = (bytesTransferred / (milliseconds / 1000.0f)) / (1024.0f * 1024.0f * 1024.0f);
    
    // Calculate FPS
    float fps = 1000.0f / milliseconds;
    
    if (metrics) {
        metrics->time_ms = milliseconds;
        metrics->bandwidth_gbps = bandwidth_gbps;
        metrics->fps = fps;
    }
    
    const char* levelName = (level == NAIVE) ? "Level 1 (Naive)" : 
                            (level == TEXTURE_MEMORY) ? "Level 2 (Texture + Constant + Vectorized)" : "Unknown";
    
    printf("%s Performance:\n", levelName);
    printf("  Time: %.3f ms\n", milliseconds);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth_gbps);
    printf("  FPS: %.2f\n", fps);
    
    // ========================================================================
    // Step 9: Cleanup
    // ========================================================================
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Cleanup (no texture to unbind - using constant memory and vectorized access)
    
    cudaFree(d_temp);
    if (d_kernel) cudaFree(d_kernel);
    delete[] h_kernel;
    
    return cudaSuccess;
}

// ============================================================================
// Main Box Blur Function
// ============================================================================

cudaError_t boxBlur(
    unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    int kernelRadius,
    OptimizationLevel level,
    PerformanceMetrics* metrics
) {
    cudaError_t err = cudaSuccess;
    
    // Validate optimization level
    if (level != NAIVE && level != SHARED_MEMORY) {
        fprintf(stderr, "Error: Only NAIVE (Level 1) and SHARED_MEMORY (Level 2) are implemented for box blur\n");
        return cudaErrorNotSupported;
    }
    
    printf("Box Blur Parameters:\n");
    printf("  Kernel size: %dx%d (radius=%d)\n", 2*kernelRadius+1, 2*kernelRadius+1, kernelRadius);
    printf("  Note: Box blur uses equal weights (simple average)\n");
    printf("  Optimization level: %s\n", level == NAIVE ? "Naive (global memory)" : "Shared Memory");
    
    // ========================================================================
    // Allocate temporary buffer
    // ========================================================================
    
    unsigned char* d_temp;
    size_t tempSize = width * height * channels * sizeof(unsigned char);
    err = cudaMalloc(&d_temp, tempSize);
    if (err != cudaSuccess) {
        return err;
    }
    
    // ========================================================================
    // Create CUDA events for timing
    // ========================================================================
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========================================================================
    // Configure thread blocks
    // ========================================================================
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    printf("Launching kernels: Grid(%d, %d), Block(%d, %d)\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    
    // ========================================================================
    // Execute blur passes
    // ========================================================================
    
    cudaEventRecord(start);
    
    if (level == NAIVE) {
        // ====================================================================
        // LEVEL 1: Naive implementation (global memory only)
        // ====================================================================
        
        boxBlurHorizontalNaive<<<gridSize, blockSize>>>(
            d_input, d_temp,
            width, height, channels, kernelRadius
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Box blur horizontal naive failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_temp);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return err;
        }
        
        boxBlurVerticalNaive<<<gridSize, blockSize>>>(
            d_temp, d_output,
            width, height, channels, kernelRadius
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Box blur vertical naive failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_temp);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return err;
        }
        
    } else if (level == SHARED_MEMORY) {
        // ====================================================================
        // LEVEL 2: Shared memory implementation
        // ====================================================================
        
        // Calculate shared memory sizes for each pass
        // Horizontal: (blockDim.x + 2*radius) * blockDim.y * channels
        // Vertical: blockDim.x * (blockDim.y + 2*radius) * channels
        size_t sharedMemHorizontal = (blockSize.x + 2 * kernelRadius) * blockSize.y * channels * sizeof(unsigned char);
        size_t sharedMemVertical = blockSize.x * (blockSize.y + 2 * kernelRadius) * channels * sizeof(unsigned char);
        
        printf("Shared memory usage: Horizontal=%zu bytes, Vertical=%zu bytes per block\n", 
               sharedMemHorizontal, sharedMemVertical);
        
        // First pass: horizontal blur (input -> temp)
        boxBlurHorizontalShared<<<gridSize, blockSize, sharedMemHorizontal>>>(
            d_input, d_temp,
            width, height, channels, kernelRadius
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Box blur horizontal shared failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_temp);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return err;
        }
        
        // Second pass: vertical blur (temp -> output)
        boxBlurVerticalShared<<<gridSize, blockSize, sharedMemVertical>>>(
            d_temp, d_output,
            width, height, channels, kernelRadius
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Box blur vertical shared failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_temp);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return err;
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // ========================================================================
    // Calculate performance metrics
    // ========================================================================
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    size_t bytesTransferred = width * height * channels * sizeof(unsigned char) * 4;
    float bandwidth_gbps = (bytesTransferred / (milliseconds / 1000.0f)) / (1024.0f * 1024.0f * 1024.0f);
    float fps = 1000.0f / milliseconds;
    
    if (metrics) {
        metrics->time_ms = milliseconds;
        metrics->bandwidth_gbps = bandwidth_gbps;
        metrics->fps = fps;
    }
    
    const char* levelName = (level == NAIVE) ? "Level 1 (Naive)" : "Level 2 (Shared Memory)";
    printf("Box Blur %s Performance:\n", levelName);
    printf("  Time: %.3f ms\n", milliseconds);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth_gbps);
    printf("  FPS: %.2f\n", fps);
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_temp);
    
    return cudaSuccess;
}

// ============================================================================
// SOBEL EDGE DETECTION KERNELS
// ============================================================================

/**
 * Naive Sobel edge detection kernel
 * 
 * Sobel edge detection computes image gradients using two 3x3 kernels:
 * 
 * Gx (horizontal gradient):
 *   [-1  0  1]
 *   [-2  0  2]
 *   [-1  0  1]
 * 
 * Gy (vertical gradient):
 *   [-1 -2 -1]
 *   [ 0  0  0]
 *   [ 1  2  1]
 * 
 * For each pixel:
 *   1. Convolve with Gx kernel to get horizontal gradient
 *   2. Convolve with Gy kernel to get vertical gradient
 *   3. Compute magnitude: sqrt(Gx^2 + Gy^2)
 *   4. Output magnitude as edge strength
 * 
 * For color images, we convert to grayscale first by computing
 * weighted average: 0.299*R + 0.587*G + 0.114*B
 * 
 * This is a naive implementation using global memory only.
 * Each thread handles one output pixel.
 */
__global__ void sobelEdgeDetectionNaive(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels
) {
    // Calculate pixel position this thread handles
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check (Sobel needs 3x3 neighborhood, so skip edge pixels)
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        // Set edge pixels to 0 (or could copy input)
        if (x < width && y < height) {
            if (channels == 1) {
                output[y * width + x] = 0;
            } else {
                for (int c = 0; c < channels; c++) {
                    output[(y * width + x) * channels + c] = 0;
                }
            }
        }
        return;
    }
    
    // Sobel kernels (3x3, hardcoded for efficiency)
    // Gx: horizontal gradient
    // Gy: vertical gradient
    // We'll compute these directly in the convolution loop
    
    if (channels == 1) {
        // Grayscale image - direct computation
        float gx = 0.0f;
        float gy = 0.0f;
        
        // Convolve with 3x3 Sobel kernels
        // Gx kernel:
        //   [-1  0  1]
        //   [-2  0  2]
        //   [-1  0  1]
        // Gy kernel:
        //   [-1 -2 -1]
        //   [ 0  0  0]
        //   [ 1  2  1]
        
        // Top row
        gx += -1.0f * input[((y - 1) * width + (x - 1))];
        gx +=  0.0f * input[((y - 1) * width + x)];
        gx +=  1.0f * input[((y - 1) * width + (x + 1))];
        
        gy += -1.0f * input[((y - 1) * width + (x - 1))];
        gy += -2.0f * input[((y - 1) * width + x)];
        gy += -1.0f * input[((y - 1) * width + (x + 1))];
        
        // Middle row
        gx += -2.0f * input[(y * width + (x - 1))];
        gx +=  0.0f * input[(y * width + x)];
        gx +=  2.0f * input[(y * width + (x + 1))];
        
        gy +=  0.0f * input[(y * width + (x - 1))];
        gy +=  0.0f * input[(y * width + x)];
        gy +=  0.0f * input[(y * width + (x + 1))];
        
        // Bottom row
        gx += -1.0f * input[((y + 1) * width + (x - 1))];
        gx +=  0.0f * input[((y + 1) * width + x)];
        gx +=  1.0f * input[((y + 1) * width + (x + 1))];
        
        gy +=  1.0f * input[((y + 1) * width + (x - 1))];
        gy +=  2.0f * input[((y + 1) * width + x)];
        gy +=  1.0f * input[((y + 1) * width + (x + 1))];
        
        // Compute magnitude: sqrt(Gx^2 + Gy^2)
        float magnitude = sqrtf(gx * gx + gy * gy);
        
        // Clamp to [0, 255] and convert to unsigned char
        // Note: Sobel output can exceed 255, so we need to normalize
        // For now, we'll clamp it (could also normalize by max value)
        magnitude = fminf(magnitude, 255.0f);
        output[y * width + x] = (unsigned char)(magnitude + 0.5f);
        
    } else {
        // Color image - convert to grayscale first, then apply Sobel
        // Standard grayscale conversion: 0.299*R + 0.587*G + 0.114*B
        
        float gx = 0.0f;
        float gy = 0.0f;
        
        // Convolve with Sobel kernels on grayscale values
        // Top row
        {
            int idx = ((y - 1) * width + (x - 1)) * channels;
            float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
            gx += -1.0f * gray;
            gy += -1.0f * gray;
        }
        {
            int idx = ((y - 1) * width + x) * channels;
            float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
            gx +=  0.0f * gray;
            gy += -2.0f * gray;
        }
        {
            int idx = ((y - 1) * width + (x + 1)) * channels;
            float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
            gx +=  1.0f * gray;
            gy += -1.0f * gray;
        }
        
        // Middle row
        {
            int idx = (y * width + (x - 1)) * channels;
            float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
            gx += -2.0f * gray;
            gy +=  0.0f * gray;
        }
        {
            int idx = (y * width + x) * channels;
            float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
            gx +=  0.0f * gray;
            gy +=  0.0f * gray;
        }
        {
            int idx = (y * width + (x + 1)) * channels;
            float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
            gx +=  2.0f * gray;
            gy +=  0.0f * gray;
        }
        
        // Bottom row
        {
            int idx = ((y + 1) * width + (x - 1)) * channels;
            float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
            gx += -1.0f * gray;
            gy +=  1.0f * gray;
        }
        {
            int idx = ((y + 1) * width + x) * channels;
            float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
            gx +=  0.0f * gray;
            gy +=  2.0f * gray;
        }
        {
            int idx = ((y + 1) * width + (x + 1)) * channels;
            float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
            gx +=  1.0f * gray;
            gy +=  1.0f * gray;
        }
        
        // Compute magnitude
        float magnitude = sqrtf(gx * gx + gy * gy);
        magnitude = fminf(magnitude, 255.0f);
        unsigned char edgeValue = (unsigned char)(magnitude + 0.5f);
        
        // For color output, we can either:
        // 1. Output grayscale edge map (same value in all channels)
        // 2. Output single channel
        // We'll output grayscale edge map (same value in all channels)
        for (int c = 0; c < channels; c++) {
            output[(y * width + x) * channels + c] = edgeValue;
        }
    }
}

/**
 * Level 2: Shared memory Sobel edge detection kernel
 * 
 * Optimizations:
 * - Tile-based processing with shared memory
 * - Pre-compute grayscale for color images during loading
 * - Coalesced memory access during tile loading
 * - Reduced register pressure
 * 
 * Shared memory layout: (BLOCK_SIZE + 2) × (BLOCK_SIZE + 2) pixels
 * This includes a 1-pixel halo on all sides for the 3×3 Sobel kernel.
 */
__global__ void sobelEdgeDetectionShared(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height,
    int channels
) {
    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Global pixel coordinates
    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;
    
    // Shared memory dimensions (tile + 1-pixel halo on each side)
    int s_width = blockDim.x + 2;
    int s_height = blockDim.y + 2;
    
    // Allocate shared memory (grayscale values only, regardless of input channels)
    extern __shared__ unsigned char s_tile[];
    
    // Helper macro for shared memory indexing
    #define S_IDX(row, col) ((row) * s_width + (col))
    
    // ========================================================================
    // Cooperative tile loading with grayscale pre-computation
    // ========================================================================
    
    if (channels == 1) {
        // Grayscale image: load directly
        
        // Load main tile pixel
        if (x < width && y < height) {
            s_tile[S_IDX(ty + 1, tx + 1)] = input[y * width + x];
        }
        
        // Load top halo (by top row threads)
        if (ty == 0) {
            int haloY = y - 1;
            if (haloY < 0) haloY = 0;
            if (x < width) {
                s_tile[S_IDX(0, tx + 1)] = input[haloY * width + x];
            }
        }
        
        // Load bottom halo (by bottom row threads)
        if (ty == blockDim.y - 1) {
            int haloY = y + 1;
            if (haloY >= height) haloY = height - 1;
            if (x < width && y < height) {
                s_tile[S_IDX(s_height - 1, tx + 1)] = input[haloY * width + x];
            }
        }
        
        // Load left halo (by left column threads)
        if (tx == 0) {
            int haloX = x - 1;
            if (haloX < 0) haloX = 0;
            if (y < height) {
                s_tile[S_IDX(ty + 1, 0)] = input[y * width + haloX];
            }
        }
        
        // Load right halo (by right column threads)
        if (tx == blockDim.x - 1) {
            int haloX = x + 1;
            if (haloX >= width) haloX = width - 1;
            if (x < width && y < height) {
                s_tile[S_IDX(ty + 1, s_width - 1)] = input[y * width + haloX];
            }
        }
        
        // Load corner halos (by corner threads)
        if (tx == 0 && ty == 0) {
            // Top-left corner
            int haloX = (x - 1 < 0) ? 0 : (x - 1);
            int haloY = (y - 1 < 0) ? 0 : (y - 1);
            s_tile[S_IDX(0, 0)] = input[haloY * width + haloX];
        }
        if (tx == blockDim.x - 1 && ty == 0) {
            // Top-right corner
            int haloX = (x + 1 >= width) ? (width - 1) : (x + 1);
            int haloY = (y - 1 < 0) ? 0 : (y - 1);
            if (x < width) {
                s_tile[S_IDX(0, s_width - 1)] = input[haloY * width + haloX];
            }
        }
        if (tx == 0 && ty == blockDim.y - 1) {
            // Bottom-left corner
            int haloX = (x - 1 < 0) ? 0 : (x - 1);
            int haloY = (y + 1 >= height) ? (height - 1) : (y + 1);
            if (y < height) {
                s_tile[S_IDX(s_height - 1, 0)] = input[haloY * width + haloX];
            }
        }
        if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
            // Bottom-right corner
            int haloX = (x + 1 >= width) ? (width - 1) : (x + 1);
            int haloY = (y + 1 >= height) ? (height - 1) : (y + 1);
            if (x < width && y < height) {
                s_tile[S_IDX(s_height - 1, s_width - 1)] = input[haloY * width + haloX];
            }
        }
        
    } else {
        // Color image: convert RGB→grayscale during loading
        // Standard weights: 0.299*R + 0.587*G + 0.114*B
        
        // Load main tile pixel
        if (x < width && y < height) {
            int idx = (y * width + x) * channels;
            float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
            s_tile[S_IDX(ty + 1, tx + 1)] = (unsigned char)(gray + 0.5f);
        }
        
        // Load top halo
        if (ty == 0) {
            int haloY = (y - 1 < 0) ? 0 : (y - 1);
            if (x < width) {
                int idx = (haloY * width + x) * channels;
                float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
                s_tile[S_IDX(0, tx + 1)] = (unsigned char)(gray + 0.5f);
            }
        }
        
        // Load bottom halo
        if (ty == blockDim.y - 1) {
            int haloY = (y + 1 >= height) ? (height - 1) : (y + 1);
            if (x < width && y < height) {
                int idx = (haloY * width + x) * channels;
                float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
                s_tile[S_IDX(s_height - 1, tx + 1)] = (unsigned char)(gray + 0.5f);
            }
        }
        
        // Load left halo
        if (tx == 0) {
            int haloX = (x - 1 < 0) ? 0 : (x - 1);
            if (y < height) {
                int idx = (y * width + haloX) * channels;
                float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
                s_tile[S_IDX(ty + 1, 0)] = (unsigned char)(gray + 0.5f);
            }
        }
        
        // Load right halo
        if (tx == blockDim.x - 1) {
            int haloX = (x + 1 >= width) ? (width - 1) : (x + 1);
            if (x < width && y < height) {
                int idx = (y * width + haloX) * channels;
                float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
                s_tile[S_IDX(ty + 1, s_width - 1)] = (unsigned char)(gray + 0.5f);
            }
        }
        
        // Load corner halos
        if (tx == 0 && ty == 0) {
            int haloX = (x - 1 < 0) ? 0 : (x - 1);
            int haloY = (y - 1 < 0) ? 0 : (y - 1);
            int idx = (haloY * width + haloX) * channels;
            float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
            s_tile[S_IDX(0, 0)] = (unsigned char)(gray + 0.5f);
        }
        if (tx == blockDim.x - 1 && ty == 0) {
            int haloX = (x + 1 >= width) ? (width - 1) : (x + 1);
            int haloY = (y - 1 < 0) ? 0 : (y - 1);
            if (x < width) {
                int idx = (haloY * width + haloX) * channels;
                float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
                s_tile[S_IDX(0, s_width - 1)] = (unsigned char)(gray + 0.5f);
            }
        }
        if (tx == 0 && ty == blockDim.y - 1) {
            int haloX = (x - 1 < 0) ? 0 : (x - 1);
            int haloY = (y + 1 >= height) ? (height - 1) : (y + 1);
            if (y < height) {
                int idx = (haloY * width + haloX) * channels;
                float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
                s_tile[S_IDX(s_height - 1, 0)] = (unsigned char)(gray + 0.5f);
            }
        }
        if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
            int haloX = (x + 1 >= width) ? (width - 1) : (x + 1);
            int haloY = (y + 1 >= height) ? (height - 1) : (y + 1);
            if (x < width && y < height) {
                int idx = (haloY * width + haloX) * channels;
                float gray = 0.299f * input[idx + 0] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2];
                s_tile[S_IDX(s_height - 1, s_width - 1)] = (unsigned char)(gray + 0.5f);
            }
        }
    }
    
    // Synchronize all threads before computation
    __syncthreads();
    
    // ========================================================================
    // Compute Sobel edge detection using shared memory
    // ========================================================================
    
    // Boundary check (skip edge pixels that don't have full 3×3 neighborhood)
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        if (x < width && y < height) {
            // Set edge pixels to 0
            if (channels == 1) {
                output[y * width + x] = 0;
            } else {
                for (int c = 0; c < channels; c++) {
                    output[(y * width + x) * channels + c] = 0;
                }
            }
        }
        return;
    }
    
    // Compute Sobel gradients using shared memory
    // Shared memory indices are offset by 1 (due to halo)
    int s_row = ty + 1;
    int s_col = tx + 1;
    
    // Gx kernel (horizontal gradient):
    //   [-1  0  1]
    //   [-2  0  2]
    //   [-1  0  1]
    float gx = 
        -1.0f * s_tile[S_IDX(s_row - 1, s_col - 1)] +
         0.0f * s_tile[S_IDX(s_row - 1, s_col)] +
         1.0f * s_tile[S_IDX(s_row - 1, s_col + 1)] +
        -2.0f * s_tile[S_IDX(s_row, s_col - 1)] +
         0.0f * s_tile[S_IDX(s_row, s_col)] +
         2.0f * s_tile[S_IDX(s_row, s_col + 1)] +
        -1.0f * s_tile[S_IDX(s_row + 1, s_col - 1)] +
         0.0f * s_tile[S_IDX(s_row + 1, s_col)] +
         1.0f * s_tile[S_IDX(s_row + 1, s_col + 1)];
    
    // Gy kernel (vertical gradient):
    //   [-1 -2 -1]
    //   [ 0  0  0]
    //   [ 1  2  1]
    float gy = 
        -1.0f * s_tile[S_IDX(s_row - 1, s_col - 1)] +
        -2.0f * s_tile[S_IDX(s_row - 1, s_col)] +
        -1.0f * s_tile[S_IDX(s_row - 1, s_col + 1)] +
         0.0f * s_tile[S_IDX(s_row, s_col - 1)] +
         0.0f * s_tile[S_IDX(s_row, s_col)] +
         0.0f * s_tile[S_IDX(s_row, s_col + 1)] +
         1.0f * s_tile[S_IDX(s_row + 1, s_col - 1)] +
         2.0f * s_tile[S_IDX(s_row + 1, s_col)] +
         1.0f * s_tile[S_IDX(s_row + 1, s_col + 1)];
    
    // Compute magnitude: sqrt(Gx^2 + Gy^2)
    float magnitude = sqrtf(gx * gx + gy * gy);
    magnitude = fminf(magnitude, 255.0f);
    unsigned char edgeValue = (unsigned char)(magnitude + 0.5f);
    
    // Write output
    if (channels == 1) {
        output[y * width + x] = edgeValue;
    } else {
        // For color output, write same edge value to all channels
        for (int c = 0; c < channels; c++) {
            output[(y * width + x) * channels + c] = edgeValue;
        }
    }
    
    #undef S_IDX
}

// ============================================================================
// Main Sobel Edge Detection Function
// ============================================================================

cudaError_t sobelEdgeDetection(
    unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int channels,
    OptimizationLevel level,
    PerformanceMetrics* metrics
) {
    cudaError_t err = cudaSuccess;
    
    // Validate optimization level
    if (level != NAIVE && level != SHARED_MEMORY) {
        fprintf(stderr, "Error: Only NAIVE and SHARED_MEMORY levels are currently implemented for Sobel edge detection\n");
        return cudaErrorNotSupported;
    }
    
    // Sobel uses a fixed 3x3 kernel, so no kernel generation needed
    
    // ========================================================================
    // Configure thread blocks
    // ========================================================================
    
    // Use 16x16 thread blocks (256 threads per block)
    dim3 blockSize(16, 16);
    
    // Calculate grid size to cover entire image
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
    
    printf("Launching Sobel kernel: Grid(%d, %d), Block(%d, %d)\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    
    // ========================================================================
    // Create CUDA events for timing
    // ========================================================================
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========================================================================
    // Execute Sobel edge detection
    // ========================================================================
    
    cudaEventRecord(start);
    
    if (level == NAIVE) {
        // LEVEL 1: Naive implementation (global memory only)
        sobelEdgeDetectionNaive<<<gridSize, blockSize>>>(
            d_input,
            d_output,
            width,
            height,
            channels
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Sobel kernel launch failed: %s\n", 
                    cudaGetErrorString(err));
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return err;
        }
        
    } else if (level == SHARED_MEMORY) {
        // LEVEL 2: Shared memory optimization
        // Shared memory size: (blockDim.x + 2) × (blockDim.y + 2) bytes
        // (tile + 1-pixel halo on all sides for 3×3 Sobel kernel)
        size_t sharedMemSize = (blockSize.x + 2) * (blockSize.y + 2) * sizeof(unsigned char);
        
        printf("Shared memory usage: %zu bytes per block\n", sharedMemSize);
        
        sobelEdgeDetectionShared<<<gridSize, blockSize, sharedMemSize>>>(
            d_input,
            d_output,
            width,
            height,
            channels
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Sobel shared memory kernel launch failed: %s\n", 
                    cudaGetErrorString(err));
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return err;
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // ========================================================================
    // Calculate performance metrics
    // ========================================================================
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate memory bandwidth
    // We read the entire image once (3x3 neighborhood per pixel) and write once
    // For Sobel, each pixel reads 9 neighbors, so approximate as 9x read + 1x write
    // But for simplicity, we'll use 2x (read + write) like other filters
    size_t bytesTransferred = width * height * channels * sizeof(unsigned char) * 2;
    float bandwidth_gbps = (bytesTransferred / (milliseconds / 1000.0f)) / (1024.0f * 1024.0f * 1024.0f);
    
    // Calculate FPS
    float fps = 1000.0f / milliseconds;
    
    if (metrics) {
        metrics->time_ms = milliseconds;
        metrics->bandwidth_gbps = bandwidth_gbps;
        metrics->fps = fps;
    }
    
    const char* levelName = (level == NAIVE) ? "Level 1 (Naive)" : 
                            (level == SHARED_MEMORY) ? "Level 2 (Shared Memory)" : "Unknown";
    
    printf("%s Performance:\n", levelName);
    printf("  Time: %.3f ms\n", milliseconds);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth_gbps);
    printf("  FPS: %.2f\n", fps);
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return cudaSuccess;
}

