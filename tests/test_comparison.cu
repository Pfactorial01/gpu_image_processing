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

/**
 * Convert RGBA to RGB
 */
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
    printf("║  GPU Gaussian Blur - Level 1 vs Level 2 Comparison          ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    // ========================================================================
    // Parse arguments
    // ========================================================================
    
    if (argc < 2) {
        printf("Usage: %s <input_image> [sigma] [radius]\n", argv[0]);
        printf("\nExample:\n");
        printf("  %s photo.jpg\n", argv[0]);
        printf("  %s photo.jpg 3.0 5\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    const char* inputPath = argv[1];
    float sigma = (argc > 2) ? atof(argv[2]) : 2.0f;
    int radius = (argc > 3) ? atoi(argv[3]) : 3;
    
    printf("Configuration:\n");
    printf("  Input: %s\n", inputPath);
    printf("  Sigma: %.1f\n", sigma);
    printf("  Radius: %d\n\n", radius);
    
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
    
    // Handle RGBA
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
    // GPU info
    // ========================================================================
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Shared Memory per Block: %zu KB\n\n", prop.sharedMemPerBlock / 1024);
    
    // ========================================================================
    // Allocate GPU memory
    // ========================================================================
    
    size_t imageSize = width * height * processChannels;
    unsigned char* d_input;
    unsigned char* d_output_naive;
    unsigned char* d_output_shared;
    unsigned char* h_output_naive = (unsigned char*)malloc(imageSize);
    unsigned char* h_output_shared = (unsigned char*)malloc(imageSize);
    
    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output_naive, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output_shared, imageSize));
    CUDA_CHECK(cudaMemcpy(d_input, processImage, imageSize, cudaMemcpyHostToDevice));
    
    // ========================================================================
    // Run Level 1 (Naive)
    // ========================================================================
    
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Running LEVEL 1: Naive (Global Memory)\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    PerformanceMetrics metrics_naive;
    cudaError_t err = gaussianBlur(
        d_input, d_output_naive,
        width, height, processChannels,
        sigma, radius,
        NAIVE,
        &metrics_naive
    );
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Level 1 failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    
    CUDA_CHECK(cudaMemcpy(h_output_naive, d_output_naive, imageSize, cudaMemcpyDeviceToHost));
    
    // ========================================================================
    // Run Level 2 (Shared Memory)
    // ========================================================================
    
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Running LEVEL 2: Shared Memory\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    
    PerformanceMetrics metrics_shared;
    err = gaussianBlur(
        d_input, d_output_shared,
        width, height, processChannels,
        sigma, radius,
        SHARED_MEMORY,
        &metrics_shared
    );
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Level 2 failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    
    CUDA_CHECK(cudaMemcpy(h_output_shared, d_output_shared, imageSize, cudaMemcpyDeviceToHost));
    
    // ========================================================================
    // Compare Results
    // ========================================================================
    
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    PERFORMANCE COMPARISON                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("┌────────────────────┬──────────────┬──────────────┬─────────────┐\n");
    printf("│ Metric             │   Level 1    │   Level 2    │   Speedup   │\n");
    printf("├────────────────────┼──────────────┼──────────────┼─────────────┤\n");
    printf("│ Time (ms)          │   %8.3f   │   %8.3f   │    %.2fx    │\n",
           metrics_naive.time_ms, metrics_shared.time_ms,
           metrics_naive.time_ms / metrics_shared.time_ms);
    printf("│ Bandwidth (GB/s)   │   %8.2f   │   %8.2f   │    %.2fx    │\n",
           metrics_naive.bandwidth_gbps, metrics_shared.bandwidth_gbps,
           metrics_shared.bandwidth_gbps / metrics_naive.bandwidth_gbps);
    printf("│ FPS                │   %8.2f   │   %8.2f   │    %.2fx    │\n",
           metrics_naive.fps, metrics_shared.fps,
           metrics_shared.fps / metrics_naive.fps);
    printf("└────────────────────┴──────────────┴──────────────┴─────────────┘\n\n");
    
    float speedup = metrics_naive.time_ms / metrics_shared.time_ms;
    
    if (speedup >= 5.0f) {
        printf("🚀 EXCELLENT! Shared memory gives %.1fx speedup!\n\n", speedup);
    } else if (speedup >= 3.0f) {
        printf("✓ GOOD! Shared memory gives %.1fx speedup!\n\n", speedup);
    } else if (speedup >= 1.5f) {
        printf("✓ Shared memory gives %.1fx speedup\n\n", speedup);
    } else {
        printf("⚠ Speedup is only %.1fx (expected 5-10x)\n\n", speedup);
    }
    
    // ========================================================================
    // Verify correctness
    // ========================================================================
    
    printf("Verifying correctness...\n");
    
    int differences = 0;
    int maxDiff = 0;
    
    for (size_t i = 0; i < imageSize; i++) {
        int diff = abs((int)h_output_naive[i] - (int)h_output_shared[i]);
        if (diff > 0) differences++;
        if (diff > maxDiff) maxDiff = diff;
    }
    
    printf("  Pixels with differences: %d / %zu (%.2f%%)\n",
           differences, imageSize, 100.0f * differences / imageSize);
    printf("  Maximum difference: %d\n", maxDiff);
    
    if (maxDiff <= 1) {
        printf("  ✓ Results match (differences within rounding error)\n\n");
    } else {
        printf("  ⚠ WARNING: Results differ by more than rounding error!\n\n");
    }
    
    // ========================================================================
    // Save outputs
    // ========================================================================
    
    printf("Saving outputs...\n");
    stbi_write_png("output_naive.png", width, height, processChannels, h_output_naive, width * processChannels);
    stbi_write_png("output_shared.png", width, height, processChannels, h_output_shared, width * processChannels);
    printf("  Saved: output_naive.png\n");
    printf("  Saved: output_shared.png\n\n");
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    cudaFree(d_input);
    cudaFree(d_output_naive);
    cudaFree(d_output_shared);
    free(h_output_naive);
    free(h_output_shared);
    stbi_image_free(imageData);
    if (processImage != imageData) free(processImage);
    
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    COMPARISON COMPLETE                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    return EXIT_SUCCESS;
}

