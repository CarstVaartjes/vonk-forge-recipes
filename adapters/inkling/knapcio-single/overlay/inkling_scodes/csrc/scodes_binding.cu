// inkling_scodes: TVM-FFI entry of the scale-code NVFP4 grouped GEMM (swap_ab=true, tile_m=128, BF16 out).
#include <cstdint>

#include "tvm_ffi_utils.h"

#include "scodes_args.h"

// A: activations [cum_m, k/2] u8; B: weights [E, n, k/2] u8; SFA: activation scales (swizzled, u8);
// SFB: stock swizzled weight scales [E, n * k/16] u8 (escape words are read from them; validated in kernel.py);
// D: [cum_m, n] bf16; m_indptr: [E + 1] int32; codes / bases: encode.py layout.
void ScodesGroupGemmNVFP4(TensorView int_workspace_buffer, TensorView float_workspace_buffer, TensorView A,
                          TensorView B, TensorView SFA, TensorView SFB, TensorView D, TensorView m_indptr,
                          TensorView codes, TensorView bases, int64_t n, int64_t k, int64_t tile_n,
                          int64_t tile_k, int64_t expert_stride) {
  int device_id = float_workspace_buffer.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  auto stream = get_stream(float_workspace_buffer.device());
  int num_groups = m_indptr.size(0) - 1;
  TVM_FFI_ICHECK(B.size(0) == num_groups) << "weights have " << B.size(0) << " experts, m_indptr " << num_groups;
  TVM_FFI_ICHECK(bases.numel() == static_cast<int64_t>(num_groups) * n) << "bases must be [experts, n]";
  TVM_FFI_ICHECK(SFB.numel() * get_element_size(SFB) == static_cast<int64_t>(num_groups) * n * (k / 16))
      << "stock weight scales must be [experts, n * k/16] bytes";
  TVM_FFI_ICHECK(codes.numel() >= static_cast<int64_t>(num_groups) * expert_stride + 640)
      << "codes buffer too small (needs experts * expert_stride + 640 tail bytes)";
  TVM_FFI_ICHECK(expert_stride == (n / 128) * (k / 128) * 640) << "bad expert_stride";
  inkling_scodes::ScodesArgs a{};
  a.int_buffer = int_workspace_buffer.data_ptr();
  a.int_buffer_bytes = get_element_size(int_workspace_buffer) * int_workspace_buffer.size(0);
  a.float_buffer = float_workspace_buffer.data_ptr();
  a.float_buffer_bytes = get_element_size(float_workspace_buffer) * float_workspace_buffer.size(0);
  a.weight = B.data_ptr();
  a.act = A.data_ptr();
  a.weight_scales = SFB.data_ptr();
  a.act_scales = SFA.data_ptr();
  a.out = D.data_ptr();
  a.m_indptr = static_cast<int*>(m_indptr.data_ptr());
  a.codes = static_cast<uint8_t const*>(codes.data_ptr());
  a.bases = static_cast<uint8_t const*>(bases.data_ptr());
  a.expert_stride = expert_stride;
  a.n = static_cast<int>(n);
  a.k = static_cast<int>(k);
  a.num_groups = num_groups;
  a.stream = stream;
  a.device_id = device_id;
  cudaError_t err = cudaErrorInvalidValue;
#define INKLING_SCODES_CASE(TN, TK) \
  if (tile_n == TN && tile_k == TK) err = inkling_scodes::ScodesGroupGemm<TN, TK>(a);
  INKLING_SCODES_CASE(32, 128)
  INKLING_SCODES_CASE(32, 256)
  INKLING_SCODES_CASE(64, 128)
  INKLING_SCODES_CASE(64, 256)
  INKLING_SCODES_CASE(128, 128)
#undef INKLING_SCODES_CASE
  TVM_FFI_ICHECK(err == cudaSuccess) << "scodes grouped GEMM failed (tile " << tile_n << "x" << tile_k
                                     << "): " << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(scodes_group_gemm_nvfp4, ScodesGroupGemmNVFP4);
