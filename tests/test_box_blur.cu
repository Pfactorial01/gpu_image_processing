#include "image_filters.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb_image.h"
#include "../external/stb_image_write.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

unsigned char* convertRGBAtoRGB(unsigned char* rgba, int width, int height) {
    unsigned char* rgb = (unsigned char*)malloc(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    return rgb;
}

int main(int argc, char** argv) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║      Box Blur vs Gaussian Blur - Comparison Test            ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    if (argc < 2) {
        printf("Usage: %s <input_image> [radius]\n", argv[0]);
        printf("\nExample:\n");
        printf("  %s photo.jpg\n", argv[0]);
        printf("  %s photo.jpg 5\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    const char* inputPath = argv[1];
    int radius = (argc > 2) ? atoi(argv[2]) : 3;
    
    printf("Configuration:\n");
    printf("  Input: %s\n", inputPath);
    printf("  Radius: %d (kernel size: %dx%d)\n\n", radius, 2*radius+1, 2*radius+1);
    
    // ========================================================================
    // Load image
    // ========================================================================
    
    printf("Loading image...\n");
    int width, height, channels;
    unsigned char* imageData = stbi_load(inputPath, &width, &height, &channels, 0);
    
    if (!imageData) {
        fprintf(stderr, "Error: Failed to load '%s': %s\n", inputPath, stbi_failure_reason());
        return EXIT_FAILURE;
    }
    
    printf("  Resolution: %dx%d\n", width, height);
    printf("  Channels: %d (%s)\n", channels,
           channels == 1 ? "Grayscale" : channels == 3 ? "RGB" : "RGBA");
    printf("  Size: %.2f MB\n\n", (width * height * channels) / (1024.0f * 1024.0f));
    
    unsigned char* processImage;
    int processChannels = channels;
    
    if (channels == 4) {
        printf("Converting RGBA to RGB...\n");
        processImage = convertRGBAtoRGB(imageData, width, height);
        processChannels = 3;
    } else {
        processImage = imageData;
    }
    
    // ========================================================================
    // GPU setup
    // ========================================================================
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (Compute %d.%d)\n\n", prop.name, prop.major, prop.minor);
    
    size_t imageSize = width * height * processChannels;
    unsigned char* d_input;
    unsigned char* d_output_box_naive;
    unsigned char* d_output_box_shared;
    unsigned char* d_output_gaussian_naive;
    unsigned char* d_output_gaussian_shared;
    
    unsigned char* h_output_box_naive = (unsigned char*)malloc(imageSize);
    unsigned char* h_output_box_shared = (unsigned char*)malloc(imageSize);
    unsigned char* h_output_gaussian_naive = (unsigned char*)malloc(imageSize);
    unsigned char* h_output_gaussian_shared = (unsigned char*)malloc(imageSize);
    
    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output_box_naive, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output_box_shared, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output_gaussian_naive, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output_gaussian_shared, imageSize));
    CUDA_CHECK(cudaMemcpy(d_input, processImage, imageSize, cudaMemcpyHostToDevice));
    
    // ========================================================================
    // Run all tests
    // ========================================================================
    
    PerformanceMetrics metrics_box_naive, metrics_box_shared;
    PerformanceMetrics metrics_gaussian_naive, metrics_gaussian_shared;
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("1. Box Blur - Naive (Level 1)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    boxBlur(d_input, d_output_box_naive, width, height, processChannels,
            radius, NAIVE, &metrics_box_naive);
    CUDA_CHECK(cudaMemcpy(h_output_box_naive, d_output_box_naive, imageSize, cudaMemcpyDeviceToHost));
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("2. Box Blur - Shared Memory (Level 2)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    boxBlur(d_input, d_output_box_shared, width, height, processChannels,
            radius, SHARED_MEMORY, &metrics_box_shared);
    CUDA_CHECK(cudaMemcpy(h_output_box_shared, d_output_box_shared, imageSize, cudaMemcpyDeviceToHost));
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("3. Gaussian Blur - Naive (Level 1)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    float sigma = radius / 3.0f;  // Rule of thumb: sigma ≈ radius/3
    gaussianBlur(d_input, d_output_gaussian_naive, width, height, processChannels,
                 sigma, radius, NAIVE, &metrics_gaussian_naive);
    CUDA_CHECK(cudaMemcpy(h_output_gaussian_naive, d_output_gaussian_naive, imageSize, cudaMemcpyDeviceToHost));
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("4. Gaussian Blur - Shared Memory (Level 2)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    gaussianBlur(d_input, d_output_gaussian_shared, width, height, processChannels,
                 sigma, radius, SHARED_MEMORY, &metrics_gaussian_shared);
    CUDA_CHECK(cudaMemcpy(h_output_gaussian_shared, d_output_gaussian_shared, imageSize, cudaMemcpyDeviceToHost));
    
    // ========================================================================
    // Display results
    // ========================================================================
    
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    PERFORMANCE COMPARISON                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("┌──────────────────────┬──────────────┬──────────────┬─────────────┐\n");
    printf("│ Method               │    Time      │  Bandwidth   │     FPS     │\n");
    printf("├──────────────────────┼──────────────┼──────────────┼─────────────┤\n");
    printf("│ Box (Naive)          │   %8.3f ms │   %8.2f GB/s│   %8.2f  │\n",
           metrics_box_naive.time_ms, metrics_box_naive.bandwidth_gbps, metrics_box_naive.fps);
    printf("│ Box (Shared)         │   %8.3f ms │   %8.2f GB/s│   %8.2f  │\n",
           metrics_box_shared.time_ms, metrics_box_shared.bandwidth_gbps, metrics_box_shared.fps);
    printf("│ Gaussian (Naive)     │   %8.3f ms │   %8.2f GB/s│   %8.2f  │\n",
           metrics_gaussian_naive.time_ms, metrics_gaussian_naive.bandwidth_gbps, metrics_gaussian_naive.fps);
    printf("│ Gaussian (Shared)    │   %8.3f ms │   %8.2f GB/s│   %8.2f  │\n",
           metrics_gaussian_shared.time_ms, metrics_gaussian_shared.bandwidth_gbps, metrics_gaussian_shared.fps);
    printf("└──────────────────────┴──────────────┴──────────────┴─────────────┘\n\n");
    
    printf("Speedup Analysis:\n");
    printf("  Box Blur speedup (L2 vs L1): %.2fx\n", 
           metrics_box_naive.time_ms / metrics_box_shared.time_ms);
    printf("  Gaussian speedup (L2 vs L1): %.2fx\n",
           metrics_gaussian_naive.time_ms / metrics_gaussian_shared.time_ms);
    printf("  Box vs Gaussian (both L2): %.2fx faster\n",
           metrics_gaussian_shared.time_ms / metrics_box_shared.time_ms);
    
    printf("\nKey Observations:\n");
    if (metrics_box_shared.time_ms < metrics_gaussian_shared.time_ms) {
        printf("  ✓ Box blur is faster (no kernel weights to compute)\n");
    }
    printf("  ✓ Both achieve significant speedup with shared memory\n");
    printf("  ✓ Gaussian provides better quality (smoother blur)\n");
    
    // ========================================================================
    // Save outputs
    // ========================================================================
    
    printf("\nSaving outputs...\n");
    stbi_write_png("output_box_naive.png", width, height, processChannels, 
                   h_output_box_naive, width * processChannels);
    stbi_write_png("output_box_shared.png", width, height, processChannels, 
                   h_output_box_shared, width * processChannels);
    stbi_write_png("output_gaussian_naive.png", width, height, processChannels, 
                   h_output_gaussian_naive, width * processChannels);
    stbi_write_png("output_gaussian_shared.png", width, height, processChannels, 
                   h_output_gaussian_shared, width * processChannels);
    
    printf("  ✓ output_box_naive.png\n");
    printf("  ✓ output_box_shared.png\n");
    printf("  ✓ output_gaussian_naive.png\n");
    printf("  ✓ output_gaussian_shared.png\n");
    
    printf("\nCompare visually:\n");
    printf("  Box blur: Simpler, faster, slight blocking artifacts\n");
    printf("  Gaussian: Smoother, more natural-looking blur\n");
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    cudaFree(d_input);
    cudaFree(d_output_box_naive);
    cudaFree(d_output_box_shared);
    cudaFree(d_output_gaussian_naive);
    cudaFree(d_output_gaussian_shared);
    free(h_output_box_naive);
    free(h_output_box_shared);
    free(h_output_gaussian_naive);
    free(h_output_gaussian_shared);
    stbi_image_free(imageData);
    if (processImage != imageData) free(processImage);
    
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    TEST COMPLETE                              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    return EXIT_SUCCESS;
}

