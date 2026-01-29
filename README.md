# GPU Image Processing Library

A high-performance CUDA-accelerated image processing library with a web-based interface for real-time filter comparison and performance analysis.

## Features

- **Three Image Filters**: Gaussian Blur, Box Blur, and Sobel Edge Detection
- **Multiple Optimization Levels**: Compare naive vs optimized implementations
- **Web Interface**: Drag-and-drop image processing with real-time performance metrics
- **REST API**: FastAPI backend with comprehensive endpoints
- **Performance Metrics**: Execution time, memory bandwidth, and FPS calculations
- **Full Color Support**: Grayscale, RGB, and RGBA image processing

## Implemented Filters

### 1. Gaussian Blur
Smooth blur using weighted averaging with a bell curve distribution.

- **Level 1 (Naive)**: Global memory only, direct reads
- **Level 2 (Texture Memory)**: Hardware texture caching, constant memory for kernel weights, vectorized access
- **Performance**: 23.24× speedup (Level 2 vs Level 1) on RTX 4050

### 2. Box Blur
Simple average blur - all pixels within kernel radius have equal weight.

- **Level 1 (Naive)**: Global memory only
- **Level 2 (Shared Memory)**: Tile-based with cooperative loading and halo regions

### 3. Sobel Edge Detection
Detects edges using gradient magnitude calculation with 3×3 Sobel kernels.

- **Level 1 (Naive)**: Global memory only
- **Level 2 (Shared Memory)**: Tile-based with pre-computed grayscale conversion
- **Performance**: 34.74× speedup (Level 2 vs Level 1) on RTX 4050

## Project Structure

```
gpu_image_processing/
├── cuda_lib/                    # Core CUDA library
│   ├── include/
│   │   └── image_filters.h      # Public API header
│   ├── src/
│   │   └── image_filters.cu     # CUDA kernel implementations
│   └── CMakeLists.txt
├── backend/                     # Python FastAPI server
│   ├── app.py                   # REST API endpoints
│   ├── cuda_bindings/           # pybind11 Python bindings
│   │   ├── bindings.cpp         # C++ ↔ Python bridge
│   │   └── CMakeLists.txt
│   ├── profiling/               # Nsight Compute integration
│   │   └── ncu_profiler.py
│   ├── requirements.txt
│   └── venv/                    # Python virtual environment
├── frontend/                    # Web interface
│   ├── index.html               # Main page
│   ├── js/
│   │   └── app.js               # Frontend logic
│   └── css/
│       └── styles.css           # Styling
├── tests/                       # Test programs
│   ├── test_gaussian_blur.cu
│   ├── test_box_blur.cu
│   ├── test_comparison.cu
│   └── test_real_image.cu
├── build/                       # Compiled binaries
├── CMakeLists.txt               # Main build config
├── start_servers.sh             # Start backend + frontend
└── stop_servers.sh              # Stop servers
```

## Building the Project

### Prerequisites

- **CUDA Toolkit** (11.0+)
- **CMake** (3.18+)
- **Python** (3.8+)
- **NVIDIA GPU** with compute capability 7.0+

### Step 1: Build CUDA Library

```bash
# Configure and build
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# This creates: build/cuda_lib/libgpu_image_filters.a
```

### Step 2: Build Python Bindings

```bash
cd backend/cuda_bindings

# Configure and build
cmake .
make

# This creates: backend/gpu_filters.cpython-*.so
```

### Step 3: Setup Python Environment

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Test Python module
cd backend
source venv/bin/activate
python3 -c "import gpu_filters; print('GPU filters loaded successfully!')"
```

## Running the Application

### Quick Start (Recommended)

```bash
# Start both backend and frontend servers
./start_servers.sh
```

This will:
- Build CUDA bindings if needed
- Start FastAPI backend on `http://localhost:8000`
- Start frontend web server on `http://localhost:8080`

### Manual Start

**Backend:**
```bash
cd backend
source venv/bin/activate
python app.py
```

**Frontend:**
```bash
cd frontend
python3 -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

## Web Interface

The web interface provides:

- **Drag & Drop Upload**: Easy image input
- **Filter Selection**: Choose from Gaussian, Box, or Sobel
- **Optimization Level Comparison**: Process with multiple levels simultaneously
- **Performance Metrics**: Real-time execution time, bandwidth, and FPS
- **Side-by-Side Comparison**: Visual comparison of results
- **Interactive Charts**: Performance visualization with Chart.js

## API Endpoints

### GET `/`
API information and status

### GET `/api/health`
Health check with GPU availability status

### GET `/api/filters`
List all available filters and their parameters

### POST `/api/process`
Process image with selected filter and optimization level

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "filter": "gaussian",
  "level": 2,
  "sigma": 2.0,
  "radius": 3
}
```

**Response:**
```json
{
  "processed_image": "data:image/png;base64,iVBORw0KGgo...",
  "metrics": {
    "time_ms": 0.293,
    "bandwidth_gbps": 39.99,
    "fps": 3415.67
  },
  "info": {
    "filter": "gaussian",
    "level": "texture_memory",
    "width": 1024,
    "height": 1023,
    "channels": 3
  }
}
```

### POST `/api/process-all`
Process image with ALL optimization levels for comparison

### POST `/api/upload`
Upload image file and get base64 encoded version

**API Documentation**: Visit `http://localhost:8000/docs` for interactive Swagger UI

## Performance Benchmarks

### RTX 4050 Laptop GPU

**Gaussian Blur (3239×2146 RGB, σ=2.0, radius=3):**
| Level | Time | Speedup |
|-------|------|---------|
| Level 1 (Naive) | 22.157 ms | 1× |
| Level 2 (Texture) | 0.953 ms | **23.24×** |

*Real kernel execution times from CUDA events (without profiling overhead)*

**Box Blur (3239×2146 RGB, radius=5):**
| Level | Time | Speedup |
|-------|------|---------|
| Level 1 (Naive) | 12.311 ms | 1× |
| Level 2 (Shared) | 2.766 ms | **4.45×** |

*Real kernel execution times from CUDA events (without profiling overhead)*

**Sobel Edge Detection (3239×2146 RGB):**
| Level | Time | Speedup |
|-------|------|---------|
| Level 1 (Naive) | 18.339 ms | 1× |
| Level 2 (Shared) | 0.528 ms | **34.74×** |

*Real kernel execution times from CUDA events (without profiling overhead)*

## Key CUDA Concepts Demonstrated

### Memory Hierarchy
- **Global Memory**: High latency (~400 cycles), high bandwidth
- **Shared Memory**: Low latency (~5 cycles), limited size (48KB per SM)
- **Constant Memory**: Read-only, cached, optimal for kernel weights
- **Texture Memory**: Hardware caching with automatic boundary handling

### Optimization Techniques
- **Separable Convolution**: 2D filter → two 1D passes (O(n²) → O(2n))
- **Cooperative Loading**: Threads work together to load shared memory tiles
- **Halo/Apron Pixels**: Border threads load extra pixels for neighbor access
- **Thread Synchronization**: `__syncthreads()` for shared memory consistency
- **Texture Objects**: Hardware-accelerated 2D spatial caching

### Performance Measurement
- **CUDA Events**: Precise GPU timing without CPU overhead
- **Memory Bandwidth**: Bytes transferred / execution time
- **Nsight Compute**: Optional detailed profiling metrics

## Testing

### CUDA Test Programs

```bash
cd build

# Test Gaussian blur
./test_gaussian_blur

# Test Box blur
./test_box_blur

# Test comparison (multiple filters)
./test_comparison

# Test with real images
./test_real_image path/to/image.jpg
```

### Python API Testing

```bash
cd backend
source venv/bin/activate
python test_client.py path/to/image.jpg
```

## Technologies Used

- **CUDA**: GPU-accelerated image processing kernels
- **C++/pybind11**: Python bindings for CUDA library
- **FastAPI**: Modern Python web framework
- **HTML/CSS/JavaScript**: Frontend web interface
- **Chart.js**: Performance visualization
- **CMake**: Build system
- **STB Image**: Image I/O library

## Future Enhancements

- **Additional Filters**: Canny Edge Detection, Bilateral Filter, Median Filter
- **Level 3 & 4 Optimizations**: Advanced techniques for all filters
- **Batch Processing**: Process multiple images simultaneously

## License

This project is provided as-is for educational and research purposes.

---

**Built with CUDA** 🚀 | **Performance-focused** ⚡ | **Web-enabled** 🌐
