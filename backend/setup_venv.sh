#!/bin/bash
# Setup script to create and configure the virtual environment

set -e

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$BACKEND_DIR/venv"

echo "Setting up virtual environment..."

# Create venv if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH" 2>/dev/null || {
        echo "Warning: python3 -m venv failed, creating structure manually..."
        mkdir -p "$VENV_PATH/bin" "$VENV_PATH/lib/python3.12/site-packages" "$VENV_PATH/include"
        PYTHON_PATH=$(which python3)
        ln -sf "$PYTHON_PATH" "$VENV_PATH/bin/python3"
        ln -sf "$PYTHON_PATH" "$VENV_PATH/bin/python"
    }
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# Install/upgrade pip
echo "Installing/upgrading pip..."
python3 -m ensurepip --upgrade 2>/dev/null || {
    # Try using get-pip.py if ensurepip fails
    if command -v wget &> /dev/null; then
        wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py
        python3 /tmp/get-pip.py
        rm /tmp/get-pip.py
    elif command -v python3 -m pip --version &> /dev/null 2>&1; then
        echo "Using system pip to install into venv..."
        python3 -m pip install --target "$VENV_PATH/lib/python3.12/site-packages" pip
    else
        echo "Warning: Could not install pip automatically"
        echo "Please install pip manually: sudo apt install python3-pip"
        exit 1
    fi
}

# Install requirements
if [ -f "$BACKEND_DIR/requirements.txt" ]; then
    echo "Installing requirements..."
    pip install --upgrade pip
    pip install -r "$BACKEND_DIR/requirements.txt"
    echo "✓ Requirements installed"
else
    echo "Warning: requirements.txt not found"
fi

echo "✓ Virtual environment setup complete!"

