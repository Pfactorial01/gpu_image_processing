"""
FastAPI Backend for GPU Image Processing
Exposes CUDA-accelerated image filters via REST API
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import base64
import io
import numpy as np
from PIL import Image
import sys
import os

# Add cuda_bindings to path (will be available after building)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import gpu_filters
    GPU_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GPU filters module not found: {e}")
    print("Build the module first: cd backend/cuda_bindings && cmake . && make")
    GPU_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="GPU Image Processing API",
    description="High-performance CUDA-accelerated image processing filters",
    version="1.0.0"
)

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class FilterRequest(BaseModel):
    image: str  # base64 encoded
    filter: str  # "gaussian", "box", or "sobel"
    level: int = 1  # 1=naive, 2=optimized (varies by filter: texture_memory for gaussian, running_sum for box, shared_memory for sobel)
    sigma: Optional[float] = 2.0  # for Gaussian blur
    radius: Optional[int] = 3  # for gaussian and box filters
    enable_profiling: bool = False  # Enable Nsight Compute profiling

class FilterResponse(BaseModel):
    processed_image: str  # base64 encoded
    metrics: Dict[str, Any]  # Allow Any type for metrics (includes floats, lists, dicts from profiling)
    info: Dict[str, Any]

class AllLevelsResponse(BaseModel):
    original_image: str  # base64 encoded
    results: Dict[str, FilterResponse]  # Key: "level_1", "level_2", etc.
    image_info: Dict[str, Any]
    profiling_available: bool = False  # Whether profiling data is available

# Helper functions
def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 string to NumPy array"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_str)
        
        # Open with PIL
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode == 'L':
            image = image.convert('RGB')  # Convert grayscale to RGB for consistency
        
        # Convert to NumPy array (height, width, channels)
        img_array = np.array(image)
        
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

def encode_image_to_base64(img_array: np.ndarray) -> str:
    """Encode NumPy array to base64 string"""
    try:
        # Convert NumPy array to PIL Image
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
        
        image = Image.fromarray(img_array)
        
        # Encode to PNG
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Encode to base64
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{base64_str}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to encode image: {str(e)}")

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "GPU Image Processing API",
        "version": "1.0.0",
        "status": "running",
        "gpu_available": GPU_AVAILABLE,
        "endpoints": {
            "GET /": "This message",
            "GET /api/filters": "List available filters",
            "POST /api/process": "Process image with filter",
            "GET /api/health": "Health check"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gpu_available": GPU_AVAILABLE
    }

@app.get("/api/filters")
async def list_filters():
    """List available filters and their parameters"""
    filters = {
        "gaussian": {
            "name": "Gaussian Blur",
            "description": "Smooth blur with weighted averaging (bell curve)",
            "parameters": {
                "sigma": {"type": "float", "default": 2.0, "range": [0.5, 20.0]},
                "radius": {"type": "int", "default": 3, "range": [1, 15]},
                "level": {"type": "int", "default": 1, "options": [1, 2]}
            },
            "optimization_levels": {
                "1": "Naive (global memory)",
                "2": "Texture + Constant + Vectorized (optimized)"
            }
        },
        "box": {
            "name": "Box Blur",
            "description": "Simple average blur (faster than Gaussian)",
            "parameters": {
                "radius": {"type": "int", "default": 3, "range": [1, 15]},
                "level": {"type": "int", "default": 1, "options": [1, 2]}
            },
            "optimization_levels": {
                "1": "Naive (global memory)",
                "2": "Shared memory tiling"
            }
        },
        "sobel": {
            "name": "Sobel Edge Detection",
            "description": "Detect edges using gradient magnitude (Gx, Gy)",
            "parameters": {
                "level": {"type": "int", "default": 2, "options": [1, 2]}
            },
            "optimization_levels": {
                "1": "Naive (global memory)",
                "2": "Shared memory (18x+ faster)"
            }
        }
    }
    
    return {
        "filters": filters,
        "gpu_available": GPU_AVAILABLE
    }

@app.post("/api/process", response_model=FilterResponse)
async def process_image(request: FilterRequest):
    """Process image with selected filter"""
    
    if not GPU_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="GPU filters module not available. Build it first."
        )
    
    # Validate filter type
    if request.filter not in ["gaussian", "box", "sobel"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filter: {request.filter}. Must be 'gaussian', 'box', or 'sobel'"
        )
    
    # Validate level
    # Validate level based on filter type
    if request.filter == "gaussian":
        if request.level not in [1, 2]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid level: {request.level}. Gaussian blur supports levels 1 (naive) or 2 (texture_memory)"
            )
    elif request.filter == "box":
        if request.level not in [1, 2]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid level: {request.level}. Box blur supports levels 1 (naive) or 2 (shared_memory)"
            )
    elif request.filter == "sobel":
        if request.level not in [1, 2]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid level: {request.level}. Sobel edge detection supports levels 1 (naive) or 2 (shared_memory)"
            )
    
    try:
        # Decode input image
        img_array = decode_base64_image(request.image)
        height, width, channels = img_array.shape
        
        # Process with selected filter
        if request.filter == "gaussian":
            result = gpu_filters.gaussian_blur(
                img_array,
                sigma=request.sigma,
                radius=request.radius,
                level=request.level
            )
        elif request.filter == "box":
            result = gpu_filters.box_blur(
                img_array,
                radius=request.radius,
                level=request.level
            )
        else:  # sobel
            result = gpu_filters.sobel_edge_detection(
                img_array,
                level=request.level
            )
        
        # Extract results
        processed_image = result["image"]
        
        # Encode output image
        output_base64 = encode_image_to_base64(processed_image)
        
        # Prepare response
        if request.filter == "gaussian":
            level_name = "naive" if request.level == 1 else "texture_memory"
        elif request.filter == "box":
            level_name = "naive" if request.level == 1 else "shared_memory"
        else:  # sobel
            level_name = "naive" if request.level == 1 else "shared_memory"
        
        return FilterResponse(
            processed_image=output_base64,
            metrics={
                "time_ms": float(result["time_ms"]),
                "bandwidth_gbps": float(result["bandwidth_gbps"]),
                "fps": float(result["fps"])
            },
            info={
                "filter": request.filter,
                "level": level_name,
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
                "parameters": {
                    "sigma": request.sigma if request.filter == "gaussian" else None,
                    "radius": request.radius if request.filter in ["gaussian", "box"] else None
                }
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/api/process-all", response_model=AllLevelsResponse)
async def process_all_levels(request: FilterRequest):
    """Process image with ALL available optimization levels for comparison"""
    
    if not GPU_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="GPU filters module not available. Build it first."
        )
    
    # Validate filter type
    if request.filter not in ["gaussian", "box", "sobel"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filter: {request.filter}. Must be 'gaussian', 'box', or 'sobel'"
        )
    
    try:
        # Decode input image once
        img_array = decode_base64_image(request.image)
        height, width, channels = img_array.shape
        
        # Get original image as base64
        original_base64 = encode_image_to_base64(img_array)
        
        # Check if profiling is requested and available
        profiling_available = False
        if request.enable_profiling:
            try:
                from profiling.ncu_profiler import check_ncu_available
                profiling_available = check_ncu_available()
                if not profiling_available:
                    print("Warning: ncu not available, profiling disabled")
            except ImportError:
                print("Warning: Profiling module not available")
        
        # Process with all available levels
        results = {}
        # Determine available levels based on filter type
        if request.filter == "gaussian":
            available_levels = [1, 2]
        elif request.filter == "box":
            available_levels = [1, 2]  # Naive and shared memory
        else:  # sobel
            available_levels = [1, 2]
        
        for level in available_levels:
            try:
                # Process with selected filter
                if request.filter == "gaussian":
                    result = gpu_filters.gaussian_blur(
                        img_array.copy(),  # Use copy to avoid modifying original
                        sigma=request.sigma,
                        radius=request.radius,
                        level=level
                    )
                elif request.filter == "box":
                    result = gpu_filters.box_blur(
                        img_array.copy(),  # Use copy to avoid modifying original
                        radius=request.radius,
                        level=level
                    )
                else:  # sobel
                    result = gpu_filters.sobel_edge_detection(
                        img_array.copy(),  # Use copy to avoid modifying original
                        level=level
                    )
                
                # Extract results
                processed_image = result["image"]
                
                # Encode output image
                output_base64 = encode_image_to_base64(processed_image)
                
                # Prepare base metrics
                # time_ms is the REAL execution time from CUDA events (without profiling overhead)
                # This is the actual kernel execution time, not profiled time
                base_metrics = {
                    "time_ms": float(result["time_ms"]),  # Real execution time from CUDA events
                    "bandwidth_gbps": float(result["bandwidth_gbps"]),
                    "fps": float(result["fps"])
                }
                
                # Add profiling metrics if requested and available
                if request.enable_profiling and profiling_available:
                    try:
                        from profiling.ncu_profiler import profile_kernel_with_ncu, get_common_ncu_metrics
                        
                        # Profile the kernel
                        ncu_metrics = profile_kernel_with_ncu(
                            img_array.copy(),
                            request.filter,
                            level,
                            request.sigma if request.filter == "gaussian" else None,
                            request.radius if request.filter in ["gaussian", "box"] else None
                        )
                        
                        # Only process metrics if we got valid data
                        if ncu_metrics and isinstance(ncu_metrics, dict):
                            # Extract common metrics (pass ncu_metrics as ncu_data for duration extraction)
                            common_metrics = get_common_ncu_metrics(ncu_metrics, ncu_data=ncu_metrics)
                        else:
                            print(f"Warning: Invalid ncu_metrics for level {level}, skipping profiling data")
                            common_metrics = {}
                        
                        # Store profiled time separately - keep real execution time as primary
                        # The real execution time from CUDA events is already in base_metrics["time_ms"]
                        # Store profiled time separately for reference
                        if common_metrics and "time_ms" in common_metrics and common_metrics["time_ms"] > 0:
                            # Store profiled time separately
                            profiled_time = common_metrics["time_ms"]
                            total_kernels = common_metrics.get("total_kernels", 0)
                            
                            if total_kernels == 1:
                                # For single-kernel filters (like Sobel), use the profiled time directly
                                # For multi-kernel filters, we'd need to sum durations
                                pass
                            
                            # Store profiled time for reference (but don't override real time)
                            base_metrics["ncu_profiled_time_ms"] = profiled_time
                            # Keep real execution time as time_ms (from CUDA events)
                            # Real execution time is already in base_metrics["time_ms"] from line 362
                        
                        if common_metrics:
                            # Filter out non-serializable types and convert where needed
                            # IMPORTANT: Skip 'time_ms' to preserve real execution time from CUDA events
                            # The profiled time includes overhead and should not override real time
                            for key, value in common_metrics.items():
                                # Skip time_ms - we want to keep the real execution time, not profiled time
                                if key == 'time_ms':
                                    continue
                                if isinstance(value, (int, float, str, bool, type(None))):
                                    base_metrics[key] = value
                                elif isinstance(value, list):
                                    # Keep lists as-is (e.g., kernel_durations, kernels_profiled)
                                    base_metrics[key] = value
                                elif isinstance(value, dict):
                                    # Keep dicts as-is (they'll be serialized to JSON)
                                    base_metrics[key] = value
                                else:
                                    # Convert other types to string
                                    base_metrics[key] = str(value)
                        
                        if ncu_metrics:
                            base_metrics["ncu_data"] = ncu_metrics  # Include full data
                        
                    except Exception as e:
                        print(f"Warning: Profiling failed for level {level}: {str(e)}")
                        base_metrics["profiling_error"] = str(e)
                
                # Prepare response for this level
                if request.filter == "gaussian":
                    level_name = "naive" if level == 1 else "texture_memory"
                elif request.filter == "box":
                    level_name = "naive" if level == 1 else "shared_memory"
                else:  # sobel
                    level_name = "naive" if level == 1 else "shared_memory"
                level_key = f"level_{level}"
                
                results[level_key] = FilterResponse(
                    processed_image=output_base64,
                    metrics=base_metrics,
                    info={
                        "filter": request.filter,
                        "level": level_name,
                        "level_number": level,
                        "width": int(width),
                        "height": int(height),
                        "channels": int(channels),
                        "parameters": {
                            "sigma": request.sigma if request.filter == "gaussian" else None,
                            "radius": request.radius if request.filter in ["gaussian", "box"] else None
                        }
                    }
                )
            except Exception as e:
                # Log error but continue with other levels
                import traceback
                print(f"Error processing level {level}: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                continue
        
        if not results:
            error_msg = "Failed to process image with any optimization level"
            print(f"ERROR: {error_msg}")
            print(f"Attempted levels: {available_levels}")
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )
        
        return AllLevelsResponse(
            original_image=original_base64,
            results=results,
            image_info={
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
                "filter": request.filter,
                "parameters": {
                    "sigma": request.sigma if request.filter == "gaussian" else None,
                    "radius": request.radius if request.filter in ["gaussian", "box"] else None
                }
            },
            profiling_available=profiling_available
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload image and return base64 encoded version"""
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Open with PIL
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB
        if image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')
        
        # Convert to NumPy array
        img_array = np.array(image)
        
        # Encode to base64
        base64_str = encode_image_to_base64(img_array)
        
        return {
            "base64_image": base64_str,
            "width": image.width,
            "height": image.height,
            "channels": len(img_array.shape) if len(img_array.shape) == 2 else img_array.shape[2]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("GPU Image Processing API Server")
    print("=" * 70)
    print(f"GPU Available: {GPU_AVAILABLE}")
    
    if not GPU_AVAILABLE:
        print("\n⚠️  WARNING: GPU module not loaded!")
        print("Build it with: cd backend/cuda_bindings && cmake . && make")
        print("Then rebuild this backend\n")
    
    print("\nStarting server on http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    print("=" * 70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

