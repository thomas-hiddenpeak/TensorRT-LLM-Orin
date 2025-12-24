/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// FP4 compatibility header for CUDA < 12.8
// Provides stub types when real FP4 types are not available

#include <cuda_fp8.h>

// Check if we have native FP4 support (CUDA 12.8+)
#if defined(__CUDA_FP4_TYPES_EXIST__) || (defined(CUDART_VERSION) && CUDART_VERSION >= 12080)
#include <cuda_fp4.h>
#define TRTLLM_HAS_NATIVE_FP4 1
#else
#define TRTLLM_HAS_NATIVE_FP4 0

// Stub FP4 type definition for CUDA < 12.8
// This allows code to compile but FP4 functionality will not be available at runtime
struct __nv_fp4_e2m1
{
    unsigned char __x;
    
    __nv_fp4_e2m1() = default;
    __host__ __device__ explicit __nv_fp4_e2m1(float val) : __x(0) {}
    __host__ __device__ explicit __nv_fp4_e2m1(double val) : __x(0) {}
    __host__ __device__ explicit __nv_fp4_e2m1(__half val) : __x(0) {}
    __host__ __device__ explicit __nv_fp4_e2m1(__nv_bfloat16 val) : __x(0) {}
    
    __host__ __device__ explicit operator float() const { return 0.0f; }
    __host__ __device__ explicit operator double() const { return 0.0; }
};

// Stub E8M0 type for CUDA < 12.8
struct __nv_fp8_e8m0
{
    unsigned char __x;
    
    __nv_fp8_e8m0() = default;
    __host__ __device__ explicit __nv_fp8_e8m0(float val) : __x(0) {}
};

// Stub conversion function
__host__ __device__ inline unsigned char __nv_cvt_float_to_e8m0(float val, int satfinite, int roundMode)
{
    return 0;
}

// Define saturation mode for compatibility
#ifndef __NV_SATFINITE
#define __NV_SATFINITE 0
#endif

// Stub round mode
#ifndef cudaRoundPosInf
#define cudaRoundPosInf 0
#endif

#endif // Native FP4 check

// TensorRT DataType::kFP4 compatibility
// This is handled separately in dataType.h via ENABLE_FP4 macro
