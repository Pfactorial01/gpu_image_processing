#include "image_filters.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Define these before including stb_image
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb_image.h"
#include "../external/stb_image_write.h"

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
 * Convert RGBA to RGB (remove alpha channel)
 */
unsigned char* convertRGBAtoRGB(unsigned char* rgba, int width, int height) {
    unsigned char* rgb = (unsigned char*)malloc(width * height * 3);
    
    for (int i = 0; i < width * height; i++) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];  // R
        rgb[i * 3 + 1] = rgba[i * 4 + 1];  // G
        rgb[i * 3 + 2] = rgba[i * 4 + 2];  // B
        // Discard alpha channel
    }
    
    return rgb;
}

/**
 * Print usage information
 */
void printUsage(const char* progName) {
    printf("Usage: %s <input_image> [output_image] [sigma] [radius]\n", progName);
    printf("\nArguments:\n");
    printf("  input_image   : Path to input image (JPG, PNG, BMP, etc.)\n");
    printf("  output_image  : Path to output image (default: output.png)\n");
    printf("  sigma         : Blur strength (default: 2.0)\n");
    printf("  radius        : Kernel radius (default: 3)\n");
    printf("\nExamples:\n");
    printf("  %s photo.jpg\n", progName);
    printf("  %s photo.jpg blurred.png\n", progName);
    printf("  %s photo.jpg blurred.png 5.0 7\n", progName);
    printf("\nSupported formats:\n");
    printf("  Input:  JPG, PNG, BMP, GIF, PSD, TGA, HDR, PIC, PNM\n");
    printf("  Output: PNG, JPG, BMP, TGA\n");
}

int main(int argc, char** argv) {
    printf("=== Real Image Gaussian Blur Test ===\n\n");
    
    // ========================================================================
    // Parse command-line arguments
    // ========================================================================
    
    if (argc < 2) {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    
    const char* inputPath = argv[1];
    const char* outputPath = (argc > 2) ? argv[2] : "output.png";
    float sigma = (argc > 3) ? atof(argv[3]) : 2.0f;
    int radius = (argc > 4) ? atoi(argv[4]) : 3;
    
    // Validate parameters
    if (sigma <= 0.0f || sigma > 20.0f) {
        fprintf(stderr, "Error: sigma must be between 0 and 20\n");
        return EXIT_FAILURE;
    }
    
    if (radius < 1 || radius > 15) {
        fprintf(stderr, "Error: radius must be between 1 and 15\n");
        return EXIT_FAILURE;
    }
    
    printf("Configuration:\n");
    printf("  Input:  %s\n", inputPath);
    printf("  Output: %s\n", outputPath);
    printf("  Sigma:  %.1f\n", sigma);
    printf("  Radius: %d (kernel size: %dx%d)\n\n", radius, 2*radius+1, 2*radius+1);
    
    // ========================================================================
    // Load image
    // ========================================================================
    
    printf("Loading image...\n");
    
    int width, height, channels;
    unsigned char* imageData = stbi_load(inputPath, &width, &height, &channels, 0);
    
    if (!imageData) {
        fprintf(stderr, "Error: Failed to load image '%s'\n", inputPath);
        fprintf(stderr, "Reason: %s\n", stbi_failure_reason());
        return EXIT_FAILURE;
    }
    
    printf("  Resolution: %dx%d\n", width, height);
    printf("  Channels: %d (%s)\n", channels, 
           channels == 1 ? "Grayscale" :
           channels == 3 ? "RGB" :
           channels == 4 ? "RGBA" : "Unknown");
    printf("  Size: %.2f MB\n", (width * height * channels) / (1024.0f * 1024.0f));
    
    // ========================================================================
    // Handle RGBA (convert to RGB for simplicity)
    // ========================================================================
    
    unsigned char* processImage;
    int processChannels = channels;
    
    if (channels == 4) {
        printf("\nConverting RGBA to RGB...\n");
        processImage = convertRGBAtoRGB(imageData, width, height);
        processChannels = 3;
    } else {
        processImage = imageData;
    }
    
    printf("\nProcessing %s image...\n", 
           processChannels == 1 ? "grayscale" : "color");
    
    // ========================================================================
    // Query GPU information
    // ========================================================================
    
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        stbi_image_free(imageData);
        if (processImage != imageData) free(processImage);
        return EXIT_FAILURE;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("\nGPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    
    // ========================================================================
    // Allocate GPU memory
    // ========================================================================
    
    printf("Allocating GPU memory...\n");
    
    size_t imageSize = width * height * processChannels * sizeof(unsigned char);
    unsigned char* d_input;
    unsigned char* d_output;
    unsigned char* h_output = (unsigned char*)malloc(imageSize);
    
    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    
    // ========================================================================
    // Copy input to GPU
    // ========================================================================
    
    printf("Copying image to GPU...\n");
    CUDA_CHECK(cudaMemcpy(d_input, processImage, imageSize, cudaMemcpyHostToDevice));
    
    // ========================================================================
    // Run Gaussian blur
    // ========================================================================
    
    // Choose optimization level (can be changed to SHARED_MEMORY)
    OptimizationLevel level = (argc > 5) ? 
        (strcmp(argv[5], "shared") == 0 ? SHARED_MEMORY : NAIVE) : SHARED_MEMORY;
    
    const char* levelName = (level == NAIVE) ? "Level 1: Naive" : "Level 2: Shared Memory";
    printf("\n=== Running Gaussian Blur (%s) ===\n", levelName);
    
    PerformanceMetrics metrics;
    cudaError_t err = gaussianBlur(
        d_input,
        d_output,
        width,
        height,
        processChannels,
        sigma,
        radius,
        level,
        &metrics
    );
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Gaussian blur failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_output);
        stbi_image_free(imageData);
        if (processImage != imageData) free(processImage);
        return EXIT_FAILURE;
    }
    
    // ========================================================================
    // Copy result back
    // ========================================================================
    
    printf("\nCopying result from GPU...\n");
    CUDA_CHECK(cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    // ========================================================================
    // Save output image
    // ========================================================================
    
    printf("Saving output image to '%s'...\n", outputPath);
    
    int saved = 0;
    // Determine output format from extension
    const char* ext = strrchr(outputPath, '.');
    if (ext) {
        if (strcmp(ext, ".png") == 0 || strcmp(ext, ".PNG") == 0) {
            saved = stbi_write_png(outputPath, width, height, processChannels, h_output, width * processChannels);
        } else if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".JPG") == 0 ||
                   strcmp(ext, ".jpeg") == 0 || strcmp(ext, ".JPEG") == 0) {
            saved = stbi_write_jpg(outputPath, width, height, processChannels, h_output, 90);
        } else if (strcmp(ext, ".bmp") == 0 || strcmp(ext, ".BMP") == 0) {
            saved = stbi_write_bmp(outputPath, width, height, processChannels, h_output);
        } else if (strcmp(ext, ".tga") == 0 || strcmp(ext, ".TGA") == 0) {
            saved = stbi_write_tga(outputPath, width, height, processChannels, h_output);
        } else {
            // Default to PNG
            saved = stbi_write_png(outputPath, width, height, processChannels, h_output, width * processChannels);
        }
    } else {
        // No extension, default to PNG
        saved = stbi_write_png(outputPath, width, height, processChannels, h_output, width * processChannels);
    }
    
    if (!saved) {
        fprintf(stderr, "Error: Failed to save output image\n");
    } else {
        printf("✓ Output saved successfully!\n");
    }
    
    // ========================================================================
    // Display results
    // ========================================================================
    
    printf("\n=== Performance Results ===\n");
    printf("Resolution: %dx%d (%.1f MP)\n", width, height, (width * height) / 1e6);
    printf("Execution time: %.3f ms\n", metrics.time_ms);
    printf("Memory bandwidth: %.2f GB/s\n", metrics.bandwidth_gbps);
    printf("Throughput: %.2f FPS\n", metrics.fps);
    
    // Calculate if it meets real-time targets
    if (metrics.fps >= 60.0f) {
        printf("✓ Exceeds 60 FPS real-time target!\n");
    } else if (metrics.fps >= 30.0f) {
        printf("✓ Exceeds 30 FPS real-time target\n");
    } else {
        printf("⚠ Below 30 FPS (not real-time)\n");
    }
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);
    stbi_image_free(imageData);
    if (processImage != imageData) free(processImage);
    
    printf("\n=== Complete ===\n");
    printf("View your blurred image: %s\n", outputPath);
    
    return EXIT_SUCCESS;
}

