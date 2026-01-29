#include "image_filters.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Create a simple test image with a white square on black background
 */
void createTestImage(unsigned char* image, int width, int height) {
    // Initialize to black
    memset(image, 0, width * height);
    
    // Draw white square in the center
    int squareSize = width / 4;
    int startX = width / 2 - squareSize / 2;
    int startY = height / 2 - squareSize / 2;
    
    for (int y = startY; y < startY + squareSize && y < height; y++) {
        for (int x = startX; x < startX + squareSize && x < width; x++) {
            image[y * width + x] = 255;
        }
    }
}

/**
 * Print a small portion of the image for debugging
 */
void printImageRegion(const unsigned char* image, int width, int height, 
                      int startX, int startY, int regionSize) {
    printf("\nImage region (%d,%d) to (%d,%d):\n", 
           startX, startY, startX + regionSize - 1, startY + regionSize - 1);
    
    for (int y = startY; y < startY + regionSize && y < height; y++) {
        for (int x = startX; x < startX + regionSize && x < width; x++) {
            printf("%3d ", image[y * width + x]);
        }
        printf("\n");
    }
}

/**
 * Save image as PGM (Portable GrayMap) format - simple ASCII format
 */
void savePGM(const char* filename, const unsigned char* image, int width, int height) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Could not open file %s for writing\n", filename);
        return;
    }
    
    // PGM header
    fprintf(f, "P2\n");
    fprintf(f, "%d %d\n", width, height);
    fprintf(f, "255\n");
    
    // Pixel data
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            fprintf(f, "%d ", image[y * width + x]);
            if ((x + 1) % 16 == 0) fprintf(f, "\n");
        }
        fprintf(f, "\n");
    }
    
    fclose(f);
    printf("Saved image to %s\n", filename);
}

int main(int argc, char** argv) {
    printf("=== Gaussian Blur Test (Level 1: Naive) ===\n\n");
    
    // ========================================================================
    // Test Parameters
    // ========================================================================
    
    const int width = 1920;      // Full HD width
    const int height = 1080;     // Full HD height
    const float sigma = 2.0f;    // Blur strength
    const int radius = 3;        // Kernel radius (7x7 kernel)
    
    printf("Test configuration:\n");
    printf("  Resolution: %dx%d\n", width, height);
    printf("  Sigma: %.1f\n", sigma);
    printf("  Kernel radius: %d (kernel size: %dx%d)\n", radius, 2*radius+1, 2*radius+1);
    printf("  Image size: %.2f MB\n\n", (width * height) / (1024.0f * 1024.0f));
    
    // ========================================================================
    // Query GPU properties
    // ========================================================================
    
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return EXIT_FAILURE;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("GPU Information:\n");
    printf("  Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Max Threads per Block: %d\n\n", prop.maxThreadsPerBlock);
    
    // ========================================================================
    // Allocate host memory
    // ========================================================================
    
    size_t imageSize = width * height * sizeof(unsigned char);
    unsigned char* h_input = (unsigned char*)malloc(imageSize);
    unsigned char* h_output = (unsigned char*)malloc(imageSize);
    
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    
    // ========================================================================
    // Create test image
    // ========================================================================
    
    printf("Creating test image...\n");
    createTestImage(h_input, width, height);
    
    // Print a small region of the input
    int centerX = width / 2 - 5;
    int centerY = height / 2 - 5;
    printf("\nInput image (center region):");
    printImageRegion(h_input, width, height, centerX, centerY, 10);
    
    // ========================================================================
    // Allocate device memory
    // ========================================================================
    
    printf("\nAllocating GPU memory...\n");
    unsigned char* d_input;
    unsigned char* d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    
    // ========================================================================
    // Copy input to device
    // ========================================================================
    
    printf("Copying input to GPU...\n");
    CUDA_CHECK(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));
    
    // ========================================================================
    // Run Gaussian blur
    // ========================================================================
    
    printf("\n=== Running Gaussian Blur (Level 1: Naive) ===\n");
    
    PerformanceMetrics metrics;
    cudaError_t err = gaussianBlur(
        d_input,
        d_output,
        width,
        height,
        1,           // channels (1 = grayscale)
        sigma,
        radius,
        NAIVE,
        &metrics
    );
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Gaussian blur failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        free(h_output);
        return EXIT_FAILURE;
    }
    
    // ========================================================================
    // Copy result back to host
    // ========================================================================
    
    printf("\nCopying result from GPU...\n");
    CUDA_CHECK(cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    // ========================================================================
    // Display results
    // ========================================================================
    
    printf("\nOutput image (center region):");
    printImageRegion(h_output, width, height, centerX, centerY, 10);
    
    printf("\n=== Final Results ===\n");
    printf("Execution time: %.3f ms\n", metrics.time_ms);
    printf("Memory bandwidth: %.2f GB/s\n", metrics.bandwidth_gbps);
    printf("Throughput: %.2f FPS\n", metrics.fps);
    printf("Latency per frame: %.3f ms\n", 1000.0f / metrics.fps);
    
    // Check if blur actually worked
    int centerPixelInput = h_input[height/2 * width + width/2];
    int centerPixelOutput = h_output[height/2 * width + width/2];
    printf("\nVerification:\n");
    printf("  Center pixel input: %d\n", centerPixelInput);
    printf("  Center pixel output: %d\n", centerPixelOutput);
    printf("  Result: %s\n", 
           (centerPixelInput != centerPixelOutput) ? "PASS (image changed)" : "FAIL (no change)");
    
    // ========================================================================
    // Save output images
    // ========================================================================
    
    printf("\nSaving images...\n");
    savePGM("test_input.pgm", h_input, width, height);
    savePGM("test_output_naive.pgm", h_output, width, height);
    
    printf("\nYou can view these images with:\n");
    printf("  - ImageMagick: display test_input.pgm\n");
    printf("  - GIMP: gimp test_input.pgm\n");
    printf("  - Convert to PNG: convert test_input.pgm test_input.png\n");
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    printf("\n=== Test Complete ===\n");
    return EXIT_SUCCESS;
}

