/*
 * inkling_scodes: FlashInfer 0.7.4's SM120 NVFP4 grouped GEMM (include/flashinfer/gemm/
 * group_gemm_nvfp4_groupwise_sm120.cuh, Apache-2.0, Copyright (c) 2026 by FlashInfer team), reduced to the
 * swap_ab=true, tile_m=128, BF16-output, no-alpha configuration SGLang's inkling2 SM120 path runs, with the
 * collective mainloop swapped for the scale-code one (scodes_sm120_mma_array.hpp). Everything else (builder
 * choices, schedule, epilogue, argument kernel, launch with PDL) is the stock code path.
 */
#pragma once

#include <flashinfer/allocator.h>
#include <flashinfer/cutlass_utils.cuh>

#include "scodes_args.h"
#include "scodes_sm120_mma_array.hpp"

namespace inkling_scodes {

using namespace cute;

template <class T>
struct WithScodes;

// Same collective as the builder's, with TransformA replaced by the tag that selects the scale-code mainloop.
template <class DP, class TS, class EPA, class SPA, class EPB, class SPB, class TM, class GA, class SLA, class SCA,
          class TA, class GB, class SLB, class SCB, class TB>
struct WithScodes<cutlass::gemm::collective::CollectiveMma<DP, TS, EPA, SPA, EPB, SPB, TM, GA, SLA, SCA, TA, GB, SLB,
                                                           SCB, TB>> {
  static_assert(cute::is_same_v<TA, cute::identity>, "unexpected TransformA from the SM120 builder");
  using type = cutlass::gemm::collective::CollectiveMma<DP, TS, EPA, SPA, EPB, SPB, TM, GA, SLA, SCA,
                                                        ScodesTransformA, GB, SLB, SCB, TB>;
};

template <typename T>
__host__ __device__ __forceinline__ constexpr T* safe_inc_ptr(T* ptr, size_t offset) {
  constexpr int adjustment = (cutlass::sizeof_bits<T>::value < 8) ? (8 / cutlass::sizeof_bits<T>::value) : 1;
  return ptr + offset / adjustment;
}

// FlashInfer's compute_sm120_cutlass_nvfp4_group_gemm_args, SwapAB branch (weights are the kernel's A operand).
template <int ScaleGranularity, typename ScaleConfig, typename ElementA, typename ElementB, typename ElementSFA,
          typename ElementSFB, typename ElementD, typename ProblemShape, typename StrideA, typename StrideB,
          typename StrideD, typename LayoutSFA, typename LayoutSFB>
__global__ void scodes_group_gemm_args(ElementA* A, ElementB* B, ElementSFA* SFA, ElementSFB* SFB, ElementD* D,
                                       int* m_indptr, int n, int k, int num_groups, ProblemShape* problem_sizes,
                                       const ElementA** A_ptr, const ElementB** B_ptr, const ElementSFA** SFA_ptr,
                                       const ElementSFB** SFB_ptr, ElementD** D_ptr, StrideA* stride_A,
                                       StrideB* stride_B, StrideD* stride_D, LayoutSFA* layout_SFA,
                                       LayoutSFB* layout_SFB) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_groups) {
    return;
  }
  constexpr size_t alignment_swizzled_mn = 128;
  constexpr size_t alignment_swizzled_k = static_cast<size_t>(ScaleGranularity) * 4;
  size_t sf_n = (static_cast<size_t>(n) + alignment_swizzled_mn - 1) / alignment_swizzled_mn * alignment_swizzled_mn;
  size_t swizzled_k =
      (static_cast<size_t>(k) + alignment_swizzled_k - 1) / alignment_swizzled_k * alignment_swizzled_k;
  size_t sf_k = swizzled_k / static_cast<size_t>(ScaleGranularity);
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
  asm volatile("griddepcontrol.launch_dependents;");
#endif
  int m_offset = m_indptr[i];
  int m_offset_next = m_indptr[i + 1];
  size_t m = static_cast<size_t>(m_offset_next) - static_cast<size_t>(m_offset);
  size_t sf_m_offset =
      (static_cast<size_t>(m_offset) + static_cast<size_t>(i) * (alignment_swizzled_mn - 1)) /
      alignment_swizzled_mn * alignment_swizzled_mn;

  problem_sizes[i] = ProblemShape(n, m, k);
  stride_A[i] = cutlass::make_cute_packed_stride(StrideA{}, {n, k, 1});
  stride_B[i] = cutlass::make_cute_packed_stride(StrideB{}, {m, k, 1});
  stride_D[i] = cutlass::make_cute_packed_stride(StrideD{}, {n, m, 1});
  B_ptr[i] = safe_inc_ptr(B, static_cast<size_t>(m_offset) * static_cast<size_t>(k));
  A_ptr[i] = safe_inc_ptr(A, static_cast<size_t>(i) * static_cast<size_t>(n) * static_cast<size_t>(k));
  D_ptr[i] = safe_inc_ptr(D, static_cast<size_t>(m_offset) * static_cast<size_t>(n));
  layout_SFA[i] = ScaleConfig::tile_atom_to_shape_SFA(
      make_shape(static_cast<int>(sf_n), static_cast<int>(m), static_cast<int>(swizzled_k), 1));
  // Stock weight scales: the SFA tensormap is never used for a copy; escape words are read from here.
  SFA_ptr[i] = safe_inc_ptr(SFA, static_cast<size_t>(i) * sf_n * sf_k);
  layout_SFB[i] = ScaleConfig::tile_atom_to_shape_SFB(
      make_shape(static_cast<int>(sf_n), static_cast<int>(m), static_cast<int>(swizzled_k), 1));
  SFB_ptr[i] = safe_inc_ptr(SFB, sf_m_offset * sf_k);
}



template <int TileN, int TileK>
cudaError_t ScodesGroupGemm(ScodesArgs const& a) {
  constexpr int TileM = 128;
  using DTypeIn = cutlass::float_e2m1_t;
  using ElementA = DTypeIn;
  using ElementB = DTypeIn;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementD = cutlass::bfloat16_t;
  using ElementC = void;
  using LayoutC = void;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  if (a.num_groups == 0) {
    return cudaSuccess;
  }
  if (a.k % std::max(AlignmentA, AlignmentB) != 0 || a.n % AlignmentD != 0 || a.k % 128 != 0 || a.n % 128 != 0) {
    return cudaErrorInvalidValue;
  }
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementAMainloop = cutlass::nv_float4_t<ElementA>;
  using ElementBMainloop = cutlass::nv_float4_t<ElementB>;
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;  // SwapAB
  using ClusterShape = Shape<_1, _1, _1>;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using ThreadBlockShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ThreadBlockShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC, LayoutD*,
      AlignmentD, ElementD, LayoutD*, AlignmentD, EpilogueSchedule>::CollectiveOp;
  using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using StockMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementAMainloop, LayoutA*, AlignmentA,
      ElementBMainloop, LayoutB*, AlignmentB, ElementAccumulator, ThreadBlockShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;
  using CollectiveMainloop = typename WithScodes<StockMainloop>::type;
  static_assert(sizeof(typename CollectiveMainloop::SharedStorage) <= sizeof(typename StockMainloop::SharedStorage),
                "scodes mainloop must not need more shared memory than the stock one");
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  constexpr int ScaleGranularity = Gemm::GemmKernel::CollectiveMainloop::TiledMma::SFVecSize;
  static_assert(ScaleGranularity == 16, "NVFP4 scale granularity must be 16");

  int const num_groups = a.num_groups;
  flashinfer::AlignedAllocator allocator(a.int_buffer, a.int_buffer_bytes);
  auto problem_sizes = allocator.aligned_alloc<typename ProblemShape::UnderlyingProblemShape>(
      num_groups * sizeof(typename ProblemShape::UnderlyingProblemShape), 16, "scodes_problem_sizes");
  auto A_ptr = allocator.aligned_alloc<const typename Gemm::ElementA*>(
      num_groups * sizeof(const typename Gemm::ElementA*), 16, "scodes_A_ptr");
  auto B_ptr = allocator.aligned_alloc<const typename Gemm::ElementB*>(
      num_groups * sizeof(const typename Gemm::ElementB*), 16, "scodes_B_ptr");
  auto D_ptr = allocator.aligned_alloc<typename Gemm::EpilogueOutputOp::ElementOutput*>(
      num_groups * sizeof(typename Gemm::EpilogueOutputOp::ElementOutput*), 16, "scodes_D_ptr");
  auto SFA_ptr = allocator.aligned_alloc<const ElementSFA*>(num_groups * sizeof(const ElementSFA*), 16,
                                                           "scodes_SFA_ptr");
  auto SFB_ptr = allocator.aligned_alloc<const ElementSFB*>(num_groups * sizeof(const ElementSFB*), 16,
                                                           "scodes_SFB_ptr");
  auto stride_A = allocator.aligned_alloc<StrideA>(num_groups * sizeof(StrideA), 16, "scodes_stride_A");
  auto stride_B = allocator.aligned_alloc<StrideB>(num_groups * sizeof(StrideB), 16, "scodes_stride_B");
  auto stride_D = allocator.aligned_alloc<StrideD>(num_groups * sizeof(StrideD), 16, "scodes_stride_D");
  auto layout_SFA = allocator.aligned_alloc<LayoutSFA>(num_groups * sizeof(LayoutSFA), 16, "scodes_layout_SFA");
  auto layout_SFB = allocator.aligned_alloc<LayoutSFB>(num_groups * sizeof(LayoutSFB), 16, "scodes_layout_SFB");

  int num_threads = std::min(num_groups, 1024);
  int num_blocks = (num_groups + num_threads - 1) / num_threads;
  cudaLaunchConfig_t config;
  config.gridDim = num_blocks;
  config.blockDim = num_threads;
  config.dynamicSmemBytes = 0;
  config.stream = a.stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = true;
  config.numAttrs = 1;
  config.attrs = attrs;
  auto prepare_args_kernel =
      scodes_group_gemm_args<ScaleGranularity, ScaleConfig, ElementA, ElementB, ElementSFA, ElementSFB, ElementD,
                             typename ProblemShape::UnderlyingProblemShape, StrideA, StrideB, StrideD, LayoutSFA,
                             LayoutSFB>;
  cudaError_t err = cudaLaunchKernelEx(
      &config, prepare_args_kernel, static_cast<ElementA*>(a.weight), static_cast<ElementB*>(a.act),
      static_cast<ElementSFA*>(a.weight_scales), static_cast<ElementSFB*>(a.act_scales),
      static_cast<ElementD*>(a.out), a.m_indptr, a.n, a.k, num_groups, problem_sizes, A_ptr, B_ptr, SFA_ptr,
      SFB_ptr, D_ptr, stride_A, stride_B, stride_D, layout_SFA, layout_SFB);
  if (err != cudaSuccess) {
    return err;
  }

  thread_local int last_device_id = -1;
  thread_local int sm_count = 0;
  if (last_device_id != a.device_id) {
    last_device_id = a.device_id;
    sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  }
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a.device_id;
  hw_info.sm_count = sm_count;
  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                                     {num_groups, problem_sizes, /*problem_sizes_host=*/nullptr},
                                     {
                                         A_ptr,
                                         stride_A,
                                         B_ptr,
                                         stride_B,
                                         SFA_ptr,
                                         layout_SFA,
                                         SFB_ptr,
                                         layout_SFB,
                                     },
                                     {
                                         {},
                                         nullptr,
                                         nullptr,
                                         D_ptr,
                                         stride_D,
                                     },
                                     hw_info};
  arguments.mainloop.scodes_codes = a.codes;
  arguments.mainloop.scodes_bases = a.bases;
  arguments.mainloop.scodes_expert_stride = a.expert_stride;
  arguments.mainloop.scodes_chunks = a.k / 128;
  arguments.mainloop.scodes_rows = a.n;
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  flashinfer::AlignedAllocator float_allocator(a.float_buffer, a.float_buffer_bytes);
  auto workspace_ptr = float_allocator.aligned_alloc<void>(workspace_size, 16, "scodes_float_workspace");
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }
  if (gemm.initialize(arguments, workspace_ptr) != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  if (gemm.run(a.stream, /*cuda_adapter=*/nullptr, /*launch_with_pdl=*/true) != cutlass::Status::kSuccess) {
    return cudaErrorLaunchFailure;
  }
  return cudaSuccess;
}

}  // namespace inkling_scodes

#define INKLING_SCODES_INSTANTIATE(TN, TK) \
  template cudaError_t inkling_scodes::ScodesGroupGemm<TN, TK>(inkling_scodes::ScodesArgs const&);
