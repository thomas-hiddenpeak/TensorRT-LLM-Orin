#!/bin/bash
# Script to build TensorRT Python bindings for Python 3.12 on Jetson
# This is needed because NVIDIA only provides pre-built bindings for Python 3.10

set -e

# Configuration
PYTHON_MAJOR=3
PYTHON_MINOR=12
TARGET_ARCH=aarch64
TRT_OSS_PATH="${TRT_OSS_PATH:-/home/rm01/TensorRT-OSS}"
EXT_PATH="${EXT_PATH:-/home/rm01/external}"

# TensorRT library paths (from dpkg installation)
export TRT_LIBPATH="/usr/lib/aarch64-linux-gnu"
export TRT_INCPATH="/usr/include/aarch64-linux-gnu"

echo "=========================================="
echo "Building TensorRT Python bindings"
echo "  Python: ${PYTHON_MAJOR}.${PYTHON_MINOR}"
echo "  Architecture: ${TARGET_ARCH}"
echo "  TRT_OSS_PATH: ${TRT_OSS_PATH}"
echo "  TRT_LIBPATH: ${TRT_LIBPATH}"
echo "=========================================="

# Check TensorRT OSS directory exists
if [ ! -d "$TRT_OSS_PATH/python" ]; then
    echo "ERROR: TensorRT OSS not found at $TRT_OSS_PATH"
    echo "Please clone it first:"
    echo "  GIT_SSL_NO_VERIFY=1 git clone --depth 1 --branch v10.7.0 https://github.com/NVIDIA/TensorRT.git $TRT_OSS_PATH"
    exit 1
fi

# Create external directory
mkdir -p "$EXT_PATH"
cd "$EXT_PATH"

# Download pybind11 if needed
if [ ! -d "$EXT_PATH/pybind11" ]; then
    echo "Downloading pybind11..."
    GIT_SSL_NO_VERIFY=1 git clone --depth 1 https://github.com/pybind/pybind11.git
fi

# Download Python headers if needed
PYTHON_DIR="$EXT_PATH/python${PYTHON_MAJOR}.${PYTHON_MINOR}"
if [ ! -d "$PYTHON_DIR/include" ]; then
    echo "Downloading Python ${PYTHON_MAJOR}.${PYTHON_MINOR} headers..."
    mkdir -p "$PYTHON_DIR/include"
    
    # Download Python source for headers
    PYTHON_VERSION="${PYTHON_MAJOR}.${PYTHON_MINOR}.0"
    wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" -O python.tgz || {
        # Try alternative version
        PYTHON_VERSION="${PYTHON_MAJOR}.${PYTHON_MINOR}.1"
        wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" -O python.tgz
    }
    tar -xf python.tgz
    cp -r Python-${PYTHON_VERSION}/Include/* "$PYTHON_DIR/include/"
    rm -rf python.tgz Python-${PYTHON_VERSION}
    
    # Get pyconfig.h from conda environment
    CONDA_PYCONFIG=$(find /home/rm01/miniconda3/envs/build4all/include -name "pyconfig.h" 2>/dev/null | head -1)
    if [ -n "$CONDA_PYCONFIG" ]; then
        echo "Using pyconfig.h from conda environment"
        cp "$CONDA_PYCONFIG" "$PYTHON_DIR/include/"
    else
        echo "WARNING: pyconfig.h not found in conda environment"
        echo "You may need to manually copy it from your Python installation"
    fi
fi

# Set environment for build
export TRT_OSSPATH="$TRT_OSS_PATH"
export EXT_PATH="$EXT_PATH"

# Build the Python bindings
cd "$TRT_OSS_PATH/python"

echo "Building TensorRT Python bindings..."
TENSORRT_MODULE=tensorrt \
PYTHON_MAJOR_VERSION=$PYTHON_MAJOR \
PYTHON_MINOR_VERSION=$PYTHON_MINOR \
TARGET_ARCHITECTURE=$TARGET_ARCH \
./build.sh

# Find and report the built wheel
WHEEL_PATH=$(find "$TRT_OSS_PATH/python/build" -name "tensorrt*.whl" 2>/dev/null | head -1)
if [ -n "$WHEEL_PATH" ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS! Built wheel at:"
    echo "  $WHEEL_PATH"
    echo ""
    echo "Install with:"
    echo "  conda activate build4all"
    echo "  pip install $WHEEL_PATH"
    echo "=========================================="
else
    echo "ERROR: Could not find built wheel"
    exit 1
fi
