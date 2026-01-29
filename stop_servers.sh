#!/bin/bash

# GPU Image Processing - Stop Backend and Frontend Servers
# This script stops servers started by start_servers.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# PID files
BACKEND_PID_FILE="/tmp/gpu_image_processing_backend.pid"
FRONTEND_PID_FILE="/tmp/gpu_image_processing_frontend.pid"

echo -e "${BLUE}Stopping GPU Image Processing servers...${NC}"
echo ""

# Stop backend
if [ -f "$BACKEND_PID_FILE" ]; then
    BACKEND_PID=$(cat "$BACKEND_PID_FILE")
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "${BLUE}Stopping backend server (PID: $BACKEND_PID)...${NC}"
        kill "$BACKEND_PID" 2>/dev/null
        echo -e "${GREEN}✓ Backend server stopped${NC}"
    else
        echo -e "${YELLOW}Backend server not running${NC}"
    fi
    rm -f "$BACKEND_PID_FILE"
else
    echo -e "${YELLOW}No backend PID file found${NC}"
fi

# Stop frontend
if [ -f "$FRONTEND_PID_FILE" ]; then
    FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
    if kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "${BLUE}Stopping frontend server (PID: $FRONTEND_PID)...${NC}"
        kill "$FRONTEND_PID" 2>/dev/null
        echo -e "${GREEN}✓ Frontend server stopped${NC}"
    else
        echo -e "${YELLOW}Frontend server not running${NC}"
    fi
    rm -f "$FRONTEND_PID_FILE"
else
    echo -e "${YELLOW}No frontend PID file found${NC}"
fi

# Also try to kill by port (in case PID files are missing)
BACKEND_PORT=8000
FRONTEND_PORT=8080

# Kill processes on backend port
BACKEND_PID_BY_PORT=$(lsof -ti:$BACKEND_PORT 2>/dev/null)
if [ ! -z "$BACKEND_PID_BY_PORT" ]; then
    echo -e "${BLUE}Found process on port $BACKEND_PORT (PID: $BACKEND_PID_BY_PORT), stopping...${NC}"
    kill "$BACKEND_PID_BY_PORT" 2>/dev/null
    echo -e "${GREEN}✓ Process on port $BACKEND_PORT stopped${NC}"
fi

# Kill processes on frontend port
FRONTEND_PID_BY_PORT=$(lsof -ti:$FRONTEND_PORT 2>/dev/null)
if [ ! -z "$FRONTEND_PID_BY_PORT" ]; then
    echo -e "${BLUE}Found process on port $FRONTEND_PORT (PID: $FRONTEND_PID_BY_PORT), stopping...${NC}"
    kill "$FRONTEND_PID_BY_PORT" 2>/dev/null
    echo -e "${GREEN}✓ Process on port $FRONTEND_PORT stopped${NC}"
fi

echo ""
echo -e "${GREEN}All servers stopped.${NC}"

