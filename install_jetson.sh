#!/bin/bash
# ============================================================================
# TensorRT-LLM Installation Script for Jetson AGX Orin
# ============================================================================
# This script installs TensorRT-LLM and all required dependencies on Jetson
# AGX Orin running JetPack with CUDA 12.6 and TensorRT 10.7.0.
#
# Prerequisites:
#   - JetPack 6.x with CUDA 12.6 and TensorRT 10.7.0
#   - Python 3.12 environment (conda recommended)
#   - Local wheel files in WHEELS_DIR
#
# Usage:
#   ./install_jetson.sh [OPTIONS]
#
# Options:
#   -w, --wheels-dir DIR    Directory containing local wheel files (default: ~/build/wheels)
#   -e, --env NAME          Conda environment name (default: trt-llm)
#   -h, --help              Show this help message
# ============================================================================

set -e

# Default values
WHEELS_DIR="${HOME}/build/wheels"
CONDA_ENV="trt-llm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--wheels-dir)
            WHEELS_DIR="$2"
            shift 2
            ;;
        -e|--env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -h|--help)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}TensorRT-LLM Installation for Jetson${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Wheels directory: ${WHEELS_DIR}"
echo "Conda environment: ${CONDA_ENV}"
echo ""

# Check if running on Jetson
if [[ ! -f /etc/nv_tegra_release ]] && [[ ! -f /proc/device-tree/model ]]; then
    echo -e "${YELLOW}Warning: This doesn't appear to be a Jetson device${NC}"
fi

# Check architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" ]]; then
    echo -e "${RED}Error: This script is for aarch64 architecture only (got: ${ARCH})${NC}"
    exit 1
fi

# Check wheels directory
if [[ ! -d "$WHEELS_DIR" ]]; then
    echo -e "${RED}Error: Wheels directory not found: ${WHEELS_DIR}${NC}"
    exit 1
fi

# Find required wheel files
find_wheel() {
    local pattern="$1"
    local wheel=$(find "$WHEELS_DIR" -name "${pattern}" -type f 2>/dev/null | head -1)
    if [[ -z "$wheel" ]]; then
        echo ""
    else
        echo "$wheel"
    fi
}

TORCH_WHEEL=$(find_wheel "torch-2.9*.whl")
TENSORRT_WHEEL=$(find_wheel "tensorrt-10.7*.whl")
TRITON_WHEEL=$(find_wheel "triton-3.5*.whl")
FLASHINFER_WHEEL=$(find_wheel "flashinfer_python*.whl")
TRTLLM_WHEEL=$(find_wheel "tensorrt_llm*.whl")

# Also check parent directory and TensorRT-OSS location
if [[ -z "$TENSORRT_WHEEL" ]]; then
    TENSORRT_WHEEL=$(find "${HOME}/TensorRT-OSS" -name "tensorrt-10.7*.whl" -type f 2>/dev/null | head -1)
fi
if [[ -z "$FLASHINFER_WHEEL" ]]; then
    FLASHINFER_WHEEL=$(find "${HOME}/build/flashinfer" -name "flashinfer_python*.whl" -type f 2>/dev/null | head -1)
fi
if [[ -z "$TRTLLM_WHEEL" ]]; then
    TRTLLM_WHEEL=$(find "${SCRIPT_DIR}/dist" -name "tensorrt_llm*cp312*aarch64.whl" -type f 2>/dev/null | head -1)
    if [[ -z "$TRTLLM_WHEEL" ]]; then
        TRTLLM_WHEEL=$(find "${HOME}" -maxdepth 1 -name "tensorrt_llm*cp312*aarch64.whl" -type f 2>/dev/null | head -1)
    fi
fi

echo "Found wheels:"
echo "  torch:        ${TORCH_WHEEL:-NOT FOUND}"
echo "  tensorrt:     ${TENSORRT_WHEEL:-NOT FOUND}"
echo "  triton:       ${TRITON_WHEEL:-NOT FOUND}"
echo "  flashinfer:   ${FLASHINFER_WHEEL:-NOT FOUND}"
echo "  tensorrt_llm: ${TRTLLM_WHEEL:-NOT FOUND}"
echo ""

# Check all required wheels exist
MISSING=""
[[ -z "$TORCH_WHEEL" ]] && MISSING="${MISSING} torch"
[[ -z "$TENSORRT_WHEEL" ]] && MISSING="${MISSING} tensorrt"
[[ -z "$TRITON_WHEEL" ]] && MISSING="${MISSING} triton"
[[ -z "$FLASHINFER_WHEEL" ]] && MISSING="${MISSING} flashinfer"
[[ -z "$TRTLLM_WHEEL" ]] && MISSING="${MISSING} tensorrt_llm"

if [[ -n "$MISSING" ]]; then
    echo -e "${RED}Error: Missing required wheel files:${MISSING}${NC}"
    echo ""
    echo "Please ensure the following wheels are available:"
    echo "  - torch-2.9.x-cp312-cp312-linux_aarch64.whl"
    echo "  - tensorrt-10.7.x-cp312-none-linux_aarch64.whl"
    echo "  - triton-3.5.x-cp312-cp312-linux_aarch64.whl"
    echo "  - flashinfer_python-x.x.x-py3-none-any.whl"
    echo "  - tensorrt_llm-x.x.x-cp312-cp312-linux_aarch64.whl"
    exit 1
fi

# Check conda environment
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo -e "${YELLOW}Conda environment '${CONDA_ENV}' not found. Creating...${NC}"
    conda create -n "${CONDA_ENV}" python=3.12 -y
fi

echo -e "${GREEN}Installing packages into conda environment: ${CONDA_ENV}${NC}"
echo ""

# Install function
pip_install() {
    local pkg="$1"
    echo -e "${GREEN}Installing: ${pkg}${NC}"
    conda run -n "${CONDA_ENV}" pip install "$pkg" --no-deps 2>&1 | tail -3
}

pip_install_deps() {
    local pkg="$1"
    echo -e "${GREEN}Installing with deps: ${pkg}${NC}"
    conda run -n "${CONDA_ENV}" pip install "$pkg" 2>&1 | tail -5
}

# Step 1: Install local wheels (no deps to avoid conflicts)
echo ""
echo -e "${GREEN}Step 1: Installing local wheel files...${NC}"
pip_install "$TORCH_WHEEL"
pip_install "$TENSORRT_WHEEL"
pip_install "$TRITON_WHEEL"

# flashinfer needs some dependencies first
echo -e "${GREEN}Installing flashinfer dependencies...${NC}"
conda run -n "${CONDA_ENV}" pip install apache-tvm-ffi nvidia-cudnn-frontend nvidia-cutlass-dsl tabulate 2>&1 | tail -3
pip_install "$FLASHINFER_WHEEL"

# Step 2: Install tensorrt_llm (this will pull in all other dependencies)
echo ""
echo -e "${GREEN}Step 2: Installing tensorrt_llm and dependencies...${NC}"
pip_install_deps "$TRTLLM_WHEEL"

# Step 3: Verify installation
echo ""
echo -e "${GREEN}Step 3: Verifying installation...${NC}"
conda run -n "${CONDA_ENV}" python -c "
import tensorrt_llm
print('TensorRT-LLM version:', tensorrt_llm.__version__)

import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device:', torch.cuda.get_device_name(0))

import tensorrt
print('TensorRT version:', tensorrt.__version__)
"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Installation complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "To use TensorRT-LLM, activate the environment:"
echo "  conda activate ${CONDA_ENV}"
echo ""
