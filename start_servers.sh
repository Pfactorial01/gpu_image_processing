#!/bin/bash

# GPU Image Processing - Start Backend and Frontend Servers
# This script starts both the FastAPI backend and frontend web server

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKEND_PORT=8000
FRONTEND_PORT=8080
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

# PID files for cleanup
BACKEND_PID_FILE="/tmp/gpu_image_processing_backend.pid"
FRONTEND_PID_FILE="/tmp/gpu_image_processing_frontend.pid"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down servers...${NC}"
    
    if [ -f "$BACKEND_PID_FILE" ]; then
        BACKEND_PID=$(cat "$BACKEND_PID_FILE")
        if kill -0 "$BACKEND_PID" 2>/dev/null; then
            echo -e "${BLUE}Stopping backend server (PID: $BACKEND_PID)...${NC}"
            kill "$BACKEND_PID" 2>/dev/null || true
        fi
        rm -f "$BACKEND_PID_FILE"
    fi
    
    if [ -f "$FRONTEND_PID_FILE" ]; then
        FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
        if kill -0 "$FRONTEND_PID" 2>/dev/null; then
            echo -e "${BLUE}Stopping frontend server (PID: $FRONTEND_PID)...${NC}"
            kill "$FRONTEND_PID" 2>/dev/null || true
        fi
        rm -f "$FRONTEND_PID_FILE"
    fi
    
    echo -e "${GREEN}Servers stopped.${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if ports are already in use
check_port() {
    local port=$1
    local name=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${RED}Error: Port $port is already in use ($name)${NC}"
        echo -e "${YELLOW}Please stop the service using port $port or change the port in this script.${NC}"
        exit 1
    fi
}

# Check Python availability
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: python3 not found${NC}"
        exit 1
    fi
}

# Check if backend dependencies are installed
check_backend() {
    if [ ! -d "$BACKEND_DIR" ]; then
        echo -e "${RED}Error: Backend directory not found: $BACKEND_DIR${NC}"
        exit 1
    fi
    
    if [ ! -f "$BACKEND_DIR/app.py" ]; then
        echo -e "${RED}Error: Backend app.py not found${NC}"
        exit 1
    fi
}

# Check if frontend exists
check_frontend() {
    if [ ! -d "$FRONTEND_DIR" ]; then
        echo -e "${RED}Error: Frontend directory not found: $FRONTEND_DIR${NC}"
        exit 1
    fi
    
    if [ ! -f "$FRONTEND_DIR/index.html" ]; then
        echo -e "${RED}Error: Frontend index.html not found${NC}"
        exit 1
    fi
}

# Build CUDA bindings
build_bindings() {
    echo -e "${BLUE}Checking CUDA bindings...${NC}"
    
    local bindings_file="$BACKEND_DIR/gpu_filters.cpython-*.so"
    local cuda_lib_file="$PROJECT_ROOT/build/cuda_lib/libgpu_image_filters.a"
    local bindings_dir="$BACKEND_DIR/cuda_bindings"
    local build_dir="$PROJECT_ROOT/build"
    
    # Check if cmake is available
    if ! command -v cmake &> /dev/null; then
        echo -e "${YELLOW}Warning: cmake not found. Skipping bindings build.${NC}"
        echo -e "${YELLOW}Install cmake to enable automatic bindings build.${NC}"
        return 1
    fi
    
    # Check if make is available
    if ! command -v make &> /dev/null; then
        echo -e "${YELLOW}Warning: make not found. Skipping bindings build.${NC}"
        echo -e "${YELLOW}Install make to enable automatic bindings build.${NC}"
        return 1
    fi
    
    # Check if CUDA compiler is available
    if ! command -v nvcc &> /dev/null; then
        echo -e "${YELLOW}Warning: nvcc (CUDA compiler) not found. Skipping bindings build.${NC}"
        echo -e "${YELLOW}Install CUDA Toolkit to enable automatic bindings build.${NC}"
        return 1
    fi
    
    # Check if bindings already exist and are recent
    if ls $bindings_file 1> /dev/null 2>&1; then
        local bindings_age=$(find $BACKEND_DIR -name "gpu_filters.cpython-*.so" -type f -mmin -5 2>/dev/null | wc -l)
        if [ "$bindings_age" -gt 0 ]; then
            echo -e "${GREEN}✓ CUDA bindings are up to date${NC}"
            return 0
        fi
    fi
    
    echo -e "${BLUE}Building CUDA bindings...${NC}"
    
    # Step 1: Build the main CUDA library
    echo -e "${BLUE}  → Building main CUDA library...${NC}"
    cd "$PROJECT_ROOT"
    if [ ! -d "$build_dir" ]; then
        mkdir -p "$build_dir"
    fi
    
    if cmake -B "$build_dir" > /tmp/cuda_lib_build.log 2>&1 && \
       cmake --build "$build_dir" --target gpu_image_filters >> /tmp/cuda_lib_build.log 2>&1; then
        echo -e "${GREEN}  ✓ Main CUDA library built${NC}"
    else
        echo -e "${RED}  ✗ Failed to build main CUDA library${NC}"
        echo -e "${YELLOW}  Check logs: cat /tmp/cuda_lib_build.log${NC}"
        return 1
    fi
    
    # Step 2: Build Python bindings
    echo -e "${BLUE}  → Building Python bindings...${NC}"
    local original_dir=$(pwd)
    cd "$bindings_dir"
    
    if cmake . > /tmp/bindings_build.log 2>&1 && \
       make >> /tmp/bindings_build.log 2>&1; then
        echo -e "${GREEN}  ✓ Python bindings built${NC}"
        
        # Return to original directory
        cd "$original_dir"
        
        # Verify the module was created
        if ls $bindings_file 1> /dev/null 2>&1; then
            echo -e "${GREEN}✓ CUDA bindings build complete${NC}"
            return 0
        else
            echo -e "${YELLOW}Warning: Build completed but bindings file not found${NC}"
            return 1
        fi
    else
        # Return to original directory even on error
        cd "$original_dir"
        echo -e "${RED}  ✗ Failed to build Python bindings${NC}"
        echo -e "${YELLOW}  Check logs: cat /tmp/bindings_build.log${NC}"
        return 1
    fi
}

# Setup and activate virtual environment
setup_venv() {
    local venv_path="$BACKEND_DIR/venv"
    
    # Check if venv exists, create if not
    if [ ! -d "$venv_path" ]; then
        echo -e "${BLUE}Creating virtual environment...${NC}"
        python3 -m venv "$venv_path" 2>/dev/null || {
            echo -e "${YELLOW}Warning: Could not create venv with python3 -m venv${NC}"
            echo -e "${YELLOW}Creating venv structure manually...${NC}"
            mkdir -p "$venv_path/bin" "$venv_path/lib/python3.12/site-packages" "$venv_path/include"
            PYTHON_PATH=$(which python3)
            ln -sf "$PYTHON_PATH" "$venv_path/bin/python3"
            ln -sf "$PYTHON_PATH" "$venv_path/bin/python"
        }
    fi
    
    # Activate virtual environment
    if [ -f "$venv_path/bin/activate" ]; then
        source "$venv_path/bin/activate"
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
        
        # Check if pip is available in venv
        if ! command -v pip &> /dev/null && ! "$venv_path/bin/python3" -m pip --version &> /dev/null; then
            echo -e "${BLUE}Installing pip in virtual environment...${NC}"
            "$venv_path/bin/python3" -m ensurepip --upgrade 2>/dev/null || {
                echo -e "${YELLOW}Warning: Could not install pip automatically${NC}"
                echo -e "${YELLOW}Run: cd backend && ./setup_venv.sh to set up venv properly${NC}"
            }
        fi
        
        # Install/upgrade requirements if needed
        local pip_cmd="pip"
        if ! command -v pip &> /dev/null; then
            pip_cmd="$venv_path/bin/python3 -m pip"
        fi
        
        if [ -f "$BACKEND_DIR/requirements.txt" ]; then
            echo -e "${BLUE}Checking/installing backend requirements...${NC}"
            $pip_cmd install --quiet --upgrade pip 2>/dev/null || true
            $pip_cmd install --quiet -r "$BACKEND_DIR/requirements.txt" 2>/dev/null || {
                echo -e "${YELLOW}Warning: Some requirements may not be installed${NC}"
                echo -e "${YELLOW}Run: cd backend && source venv/bin/activate && pip install -r requirements.txt${NC}"
            }
        fi
    else
        echo -e "${YELLOW}Warning: venv activate script not found, using system Python${NC}"
    fi
}

# Start backend server
start_backend() {
    echo -e "${BLUE}Starting backend server...${NC}"
    cd "$BACKEND_DIR"
    
    # Setup and activate virtual environment
    setup_venv
    
    # Build CUDA bindings if needed
    if ! build_bindings; then
        echo -e "${YELLOW}Warning: CUDA bindings build failed or skipped.${NC}"
        echo -e "${YELLOW}The backend will start but GPU processing may not work.${NC}"
    fi
    
    # Check if gpu_filters module exists (add backend directory to path)
    # Note: app.py adds the backend directory to sys.path, so this check mimics that
    if ! python3 -c "import sys; import os; sys.path.insert(0, os.path.abspath('.')); import gpu_filters" 2>/dev/null; then
        # Check if the .so file exists at least
        if ls "$BACKEND_DIR"/gpu_filters.cpython-*.so 1> /dev/null 2>&1; then
            echo -e "${GREEN}✓ CUDA bindings file found (module should load when server starts)${NC}"
        else
            echo -e "${YELLOW}Warning: gpu_filters module not found.${NC}"
            echo -e "${YELLOW}The backend will start but GPU processing may not work.${NC}"
            echo -e "${YELLOW}Try building manually: cd backend/cuda_bindings && cmake . && make${NC}"
        fi
    else
        echo -e "${GREEN}✓ gpu_filters module is available${NC}"
    fi
    
    # Determine Python command (use venv python if available and activated)
    local python_cmd="python3"
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -f "$VIRTUAL_ENV/bin/python3" ]; then
        python_cmd="$VIRTUAL_ENV/bin/python3"
    elif [ -f "$BACKEND_DIR/venv/bin/python3" ]; then
        python_cmd="$BACKEND_DIR/venv/bin/python3"
    fi
    
    # Ensure we're in the backend directory
    cd "$BACKEND_DIR"
    
    # Start backend in background (with venv activated)
    $python_cmd app.py > /tmp/gpu_backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > "$BACKEND_PID_FILE"
    
    # Wait a moment for server to start
    sleep 2
    
    # Check if server started successfully
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "${RED}Error: Backend server failed to start${NC}"
        echo -e "${YELLOW}Check logs: cat /tmp/gpu_backend.log${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Backend server started (PID: $BACKEND_PID)${NC}"
    echo -e "${GREEN}  API: http://localhost:$BACKEND_PORT${NC}"
    echo -e "${GREEN}  Docs: http://localhost:$BACKEND_PORT/docs${NC}"
}

# Start frontend server
start_frontend() {
    echo -e "${BLUE}Starting frontend server...${NC}"
    cd "$FRONTEND_DIR"
    
    # Start frontend in background
    python3 -m http.server $FRONTEND_PORT > /tmp/gpu_frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$FRONTEND_PID_FILE"
    
    # Wait a moment for server to start
    sleep 1
    
    # Check if server started successfully
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "${RED}Error: Frontend server failed to start${NC}"
        echo -e "${YELLOW}Check logs: cat /tmp/gpu_frontend.log${NC}"
        cleanup
        exit 1
    fi
    
    echo -e "${GREEN}✓ Frontend server started (PID: $FRONTEND_PID)${NC}"
    echo -e "${GREEN}  Web UI: http://localhost:$FRONTEND_PORT${NC}"
}

# Main execution
main() {
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║     GPU Image Processing - Starting Servers                 ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Pre-flight checks
    check_python
    check_backend
    check_frontend
    check_port $BACKEND_PORT "Backend"
    check_port $FRONTEND_PORT "Frontend"
    
    echo ""
    
    # Start servers
    start_backend
    start_frontend
    
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✓ Both servers are running!                               ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Backend API:${NC}  http://localhost:$BACKEND_PORT"
    echo -e "${BLUE}API Docs:${NC}     http://localhost:$BACKEND_PORT/docs"
    echo -e "${BLUE}Frontend UI:${NC}  http://localhost:$FRONTEND_PORT"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop both servers${NC}"
    echo ""
    
    # Show initial logs
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Backend logs (tailing in real-time):${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Show last few lines if log file exists
    if [ -f /tmp/gpu_backend.log ]; then
        tail -n 5 /tmp/gpu_backend.log 2>/dev/null
    fi
    echo ""
    echo -e "${GREEN}>>> Following backend logs (new entries will appear below) <<<${NC}"
    echo ""
    
    # Function to check if processes are still running
    check_processes() {
        # Check if backend is still running
        if [ -f "$BACKEND_PID_FILE" ]; then
            BACKEND_PID=$(cat "$BACKEND_PID_FILE")
            if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
                echo -e "${RED}Backend server stopped unexpectedly!${NC}"
                cleanup
                exit 1
            fi
        fi
        
        # Check if frontend is still running
        if [ -f "$FRONTEND_PID_FILE" ]; then
            FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
            if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
                echo -e "${RED}Frontend server stopped unexpectedly!${NC}"
                cleanup
                exit 1
            fi
        fi
    }
    
    # Initialize PIDs (will be set below)
    TAIL_PID=""
    MONITOR_PID=""
    
    # Cleanup function for this scope
    cleanup_logs() {
        if [ -n "$TAIL_PID" ]; then
            kill $TAIL_PID 2>/dev/null || true
        fi
        if [ -n "$MONITOR_PID" ]; then
            kill $MONITOR_PID 2>/dev/null || true
        fi
    }
    
    # Trap Ctrl+C to cleanup (override the earlier trap)
    trap 'echo ""; echo -e "${YELLOW}Stopping servers...${NC}"; cleanup_logs; cleanup; exit 0' INT TERM
    
    # Monitor processes in background
    (
        while true; do
            check_processes
            sleep 2
        done
    ) &
    MONITOR_PID=$!
    
    # Tail the backend log file in real-time (this will block until Ctrl+C)
    tail -f /tmp/gpu_backend.log 2>/dev/null &
    TAIL_PID=$!
    
    # Wait for tail to finish (will be killed by trap on Ctrl+C)
    wait $TAIL_PID 2>/dev/null || true
    
    # Cleanup if tail exits unexpectedly
    cleanup_logs
}

# Run main function
main

