/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Stub implementation for SM87 builds where SM90-specific FusedGatedGemm kernels are not available
// This provides the template instantiation to satisfy linker requirements,
// but throws an error at runtime if actually used.

#include "fused_gated_gemm.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_fp8.h>
#include <stdexcept>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace cutlass_kernels
{

template <typename T>
CutlassFusedGatedGemmRunner<T>::CutlassFusedGatedGemmRunner()
{
    mSm = tensorrt_llm::common::getSMVersion();
}

template <typename T>
CutlassFusedGatedGemmRunner<T>::~CutlassFusedGatedGemmRunner()
{
}

template <typename T>
void CutlassFusedGatedGemmRunner<T>::gemm(void* D, void const* A, void const* B, void const* C_bias,
    tk::QuantMode quantOption, int m, int n, int k, float scale_d0, float scale_d1, float scale_output,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy)
{
    throw std::runtime_error(
        "[TensorRT LLM Error][CutlassFusedGatedGemmRunner] FusedGatedGemm requires SM90 (Hopper) or later. "
        "Current SM version (" + std::to_string(mSm) + ") is not supported.");
}

template <typename T>
size_t CutlassFusedGatedGemmRunner<T>::getWorkspaceSize(int const m, int const n, int const k)
{
    // Return 0 as this is a stub
    return 0;
}

template <typename T>
std::vector<tkc::CutlassGemmConfig> CutlassFusedGatedGemmRunner<T>::getConfigs() const
{
    // Return empty configs as this is a stub
    return {};
}

template <typename T>
size_t CutlassFusedGatedGemmRunner<T>::dispatchToArch(void* D, void const* A, void const* B, void const* C_bias,
    tk::QuantMode quantOption, int m, int n, int k, float scale_d0, float scale_d1, float scale_output,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy)
{
    throw std::runtime_error(
        "[TensorRT LLM Error][CutlassFusedGatedGemmRunner] FusedGatedGemm requires SM90 (Hopper) or later.");
    return 0;
}

template <typename T>
size_t CutlassFusedGatedGemmRunner<T>::getWorkspaceSizeImpl(int const m, int const n, int const k)
{
    return 0;
}

// Explicit instantiation for FP8
template class CutlassFusedGatedGemmRunner<__nv_fp8_e4m3>;

} // namespace cutlass_kernels
} // namespace kernels

TRTLLM_NAMESPACE_END
