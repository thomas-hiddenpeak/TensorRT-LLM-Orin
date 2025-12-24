# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..logger import logger

IS_CUBLASLT_AVAILABLE = False

# Check cuBLASLt availability
try:
    import torch

    # Check if CublasLtFP4GemmRunner (used by nvfp4_gemm_cublaslt) is available
    # Use try/except to handle the case where the class doesn't exist
    if hasattr(torch.classes, 'trtllm'):
        try:
            # Try to access the class - this will raise if it doesn't exist
            _ = torch.classes.trtllm.CublasLtFP4GemmRunner
            logger.info(f"cuBLASLt FP4 GEMM is available")
            IS_CUBLASLT_AVAILABLE = True
        except (RuntimeError, AttributeError):
            # Class not registered - FP4 is not available
            pass
except ImportError:
    pass
