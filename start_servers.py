#!/usr/bin/env python3
"""
GPU Image Processing - Start Backend and Frontend Servers
This script starts both the FastAPI backend and frontend web server
"""

import os
import sys
import time
import signal
import subprocess
import shutil
from pathlib import Path

# Configuration
BACKEND_PORT = 8000
FRONTEND_PORT = 8080

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

# Global process references
backend_process = None
frontend_process = None

def cleanup(signum=None, frame=None):
    """Cleanup function to stop both servers"""
    print(f"\n{Colors.YELLOW}Shutting down servers...{Colors.NC}")
    
    if backend_process:
        print(f"{Colors.BLUE}Stopping backend server (PID: {backend_process.pid})...{Colors.NC}")
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
    
    if frontend_process:
        print(f"{Colors.BLUE}Stopping frontend server (PID: {frontend_process.pid})...{Colors.NC}")
        frontend_process.terminate()
        try:
            frontend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            frontend_process.kill()
    
    print(f"{Colors.GREEN}Servers stopped.{Colors.NC}")
    sys.exit(0)

def check_port(port, name):
    """Check if a port is already in use"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    
    if result == 0:
        print(f"{Colors.RED}Error: Port {port} is already in use ({name}){Colors.NC}")
        print(f"{Colors.YELLOW}Please stop the service using port {port} or change the port in this script.{Colors.NC}")
        sys.exit(1)

def check_dependencies():
    """Check if required dependencies are available"""
    if not shutil.which('python3'):
        print(f"{Colors.RED}Error: python3 not found{Colors.NC}")
        sys.exit(1)
    
    # Check if uvicorn is available (for backend)
    try:
        import uvicorn
    except ImportError:
        print(f"{Colors.YELLOW}Warning: uvicorn not found. Install with: pip install uvicorn{Colors.NC}")

def start_backend():
    """Start the FastAPI backend server"""
    global backend_process
    
    project_root = Path(__file__).parent
    backend_dir = project_root / "backend"
    
    if not backend_dir.exists():
        print(f"{Colors.RED}Error: Backend directory not found: {backend_dir}{Colors.NC}")
        sys.exit(1)
    
    if not (backend_dir / "app.py").exists():
        print(f"{Colors.RED}Error: Backend app.py not found{Colors.NC}")
        sys.exit(1)
    
    print(f"{Colors.BLUE}Starting backend server...{Colors.NC}")
    
    # Check if gpu_filters module exists
    try:
        import gpu_filters
    except ImportError:
        print(f"{Colors.YELLOW}Warning: gpu_filters module not found.{Colors.NC}")
        print(f"{Colors.YELLOW}The backend will start but GPU processing may not work.{Colors.NC}")
        print(f"{Colors.YELLOW}Build it with: cd backend/cuda_bindings && cmake . && make{Colors.NC}")
    
    # Start backend
    os.chdir(backend_dir)
    backend_process = subprocess.Popen(
        [sys.executable, "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Wait a moment for server to start
    time.sleep(2)
    
    # Check if process is still running
    if backend_process.poll() is not None:
        print(f"{Colors.RED}Error: Backend server failed to start{Colors.NC}")
        output = backend_process.stdout.read() if backend_process.stdout else "No output"
        print(f"{Colors.YELLOW}Output: {output}{Colors.NC}")
        sys.exit(1)
    
    print(f"{Colors.GREEN}✓ Backend server started (PID: {backend_process.pid}){Colors.NC}")
    print(f"{Colors.GREEN}  API: http://localhost:{BACKEND_PORT}{Colors.NC}")
    print(f"{Colors.GREEN}  Docs: http://localhost:{BACKEND_PORT}/docs{Colors.NC}")

def start_frontend():
    """Start the frontend web server"""
    global frontend_process
    
    project_root = Path(__file__).parent
    frontend_dir = project_root / "frontend"
    
    if not frontend_dir.exists():
        print(f"{Colors.RED}Error: Frontend directory not found: {frontend_dir}{Colors.NC}")
        cleanup()
        sys.exit(1)
    
    if not (frontend_dir / "index.html").exists():
        print(f"{Colors.RED}Error: Frontend index.html not found{Colors.NC}")
        cleanup()
        sys.exit(1)
    
    print(f"{Colors.BLUE}Starting frontend server...{Colors.NC}")
    
    # Start frontend HTTP server
    os.chdir(frontend_dir)
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(FRONTEND_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Wait a moment for server to start
    time.sleep(1)
    
    # Check if process is still running
    if frontend_process.poll() is not None:
        print(f"{Colors.RED}Error: Frontend server failed to start{Colors.NC}")
        output = frontend_process.stdout.read() if frontend_process.stdout else "No output"
        print(f"{Colors.YELLOW}Output: {output}{Colors.NC}")
        cleanup()
        sys.exit(1)
    
    print(f"{Colors.GREEN}✓ Frontend server started (PID: {frontend_process.pid}){Colors.NC}")
    print(f"{Colors.GREEN}  Web UI: http://localhost:{FRONTEND_PORT}{Colors.NC}")

def main():
    """Main function"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    print(f"{Colors.GREEN}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     GPU Image Processing - Starting Servers                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.NC}")
    
    # Pre-flight checks
    check_dependencies()
    check_port(BACKEND_PORT, "Backend")
    check_port(FRONTEND_PORT, "Frontend")
    
    print()
    
    # Start servers
    start_backend()
    start_frontend()
    
    print()
    print(f"{Colors.GREEN}╔══════════════════════════════════════════════════════════════╗{Colors.NC}")
    print(f"{Colors.GREEN}║  ✓ Both servers are running!                               ║{Colors.NC}")
    print(f"{Colors.GREEN}╚══════════════════════════════════════════════════════════════╝{Colors.NC}")
    print()
    print(f"{Colors.BLUE}Backend API:{Colors.NC}  http://localhost:{BACKEND_PORT}")
    print(f"{Colors.BLUE}API Docs:{Colors.NC}     http://localhost:{BACKEND_PORT}/docs")
    print(f"{Colors.BLUE}Frontend UI:{Colors.NC}  http://localhost:{FRONTEND_PORT}")
    print()
    print(f"{Colors.YELLOW}Press Ctrl+C to stop both servers{Colors.NC}")
    print()
    
    # Monitor processes
    try:
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process and backend_process.poll() is not None:
                print(f"{Colors.RED}Backend server stopped unexpectedly!{Colors.NC}")
                cleanup()
            
            if frontend_process and frontend_process.poll() is not None:
                print(f"{Colors.RED}Frontend server stopped unexpectedly!{Colors.NC}")
                cleanup()
    
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()

