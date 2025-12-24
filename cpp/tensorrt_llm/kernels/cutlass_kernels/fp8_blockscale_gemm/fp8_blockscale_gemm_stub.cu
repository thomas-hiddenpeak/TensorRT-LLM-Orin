/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stub implementation for CutlassFp8BlockScaleGemmRunner when CUDA < 12.8
 * This file provides empty implementations to satisfy linker requirements
 * from the prebuilt internal_cutlass_kernels library.
 */

#include "fp8_blockscale_gemm.h"
#include "tensorrt_llm/common/assert.h"

#ifndef BUILD_FP8_BLOCKSCALE_GEMM

TRTLLM_NAMESPACE_BEGIN

namespace kernels::fp8_blockscale_gemm
{

// Stub implementations for CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>
// These are needed because the prebuilt internal_cutlass_kernels library references these symbols

template <>
void CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::gemm(
    void* mat_d, void const* mat_a, void const* mat_b, int shape_m, int shape_n, int shape_k,
    cudaStream_t stream, float const* scales_a, float const* scales_b)
{
    TLLM_THROW("FP8 Block Scale GEMM is not available (requires CUDA 12.8+)");
}

template <>
void CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::gemm(
    __nv_fp8_e4m3 const* mat_a, int ld_a, __nv_fp8_e4m3 const* mat_b, int ld_b, __nv_bfloat16* mat_d,
    int ld_d, int shape_m, int shape_n, int shape_k, float const* scales_a, float const* scales_b,
    cudaStream_t stream)
{
    TLLM_THROW("FP8 Block Scale GEMM is not available (requires CUDA 12.8+)");
}

template <>
void CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::moeGemm(
    void* mat_d, void const* mat_a, void const* mat_b, int64_t const* problem_m_offsets,
    size_t num_problems, size_t expected_m, size_t shape_n, size_t shape_k, cudaStream_t stream,
    float const* scales_a, float const* scales_b)
{
    TLLM_THROW("FP8 Block Scale GEMM is not available (requires CUDA 12.8+)");
}

template <>
void CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::moeGemm(
    void* mat_d, void const* mat_a, void const* mat_b, int64_t const* problem_m_offsets,
    size_t num_problems, size_t shape_n, size_t shape_k, cudaStream_t stream,
    float const* scales_a, float const* scales_b)
{
    TLLM_THROW("FP8 Block Scale GEMM is not available (requires CUDA 12.8+)");
}

template <>
void CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::strideBatchGemm(
    __nv_bfloat16* mat_d, int ld_d, int stride_d, __nv_fp8_e4m3* mat_a, int ld_a,
    int stride_a, __nv_fp8_e4m3* mat_b, int ld_b, int stride_b, int num_problems, int shape_m, int shape_n,
    int shape_k, cudaStream_t stream, float* scales_a, int stride_scales_a, float* scales_b)
{
    TLLM_THROW("FP8 Block Scale GEMM is not available (requires CUDA 12.8+)");
}

template <>
void CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::fp8CS1x128(
    __nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y,
    cudaStream_t stream)
{
    TLLM_THROW("FP8 Block Scale GEMM is not available (requires CUDA 12.8+)");
}

template <>
void CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::fp8CS1x128(
    __nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x, int shape_y,
    cudaStream_t stream, bool use_ue8m0)
{
    TLLM_THROW("FP8 Block Scale GEMM is not available (requires CUDA 12.8+)");
}

template <>
void CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::fp8CS1x128Reshape(
    __nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x,
    int shape_h, int shape_y, int stride_x, cudaStream_t stream)
{
    TLLM_THROW("FP8 Block Scale GEMM is not available (requires CUDA 12.8+)");
}

template <>
void CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::fp8CS128x128(
    __nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x,
    int shape_y, cudaStream_t stream)
{
    TLLM_THROW("FP8 Block Scale GEMM is not available (requires CUDA 12.8+)");
}

template <>
size_t CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::getWorkspaceSizeBase(
    size_t max_shape_m, size_t shape_n, size_t shape_k, size_t num_problems)
{
    return 0;
}

template <>
size_t CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::getWorkspaceSize(
    size_t shape_m, size_t shape_n, size_t shape_k, size_t top_k, size_t num_problems)
{
    return 0;
}

template <>
size_t CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::getFP8DataSize(
    int shape_m, int shape_n, bool is_act)
{
    return 0;
}

template <>
size_t CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::getActScaleSize(
    int shape_m, int shape_k)
{
    return 0;
}

template <>
size_t CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::getWeightScaleSize(
    int shape_n, int shape_k)
{
    return 0;
}

template <>
size_t CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::getActWorkspaceSize(
    int shape_m, int shape_k)
{
    return 0;
}

template <>
size_t CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>::getWeightWorkspaceSize(
    int shape_n, int shape_k)
{
    return 0;
}

} // namespace kernels::fp8_blockscale_gemm

TRTLLM_NAMESPACE_END

#endif // !BUILD_FP8_BLOCKSCALE_GEMM
