// inkling_scodes: launch arguments shared by the binding and the per-tile instantiations.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace inkling_scodes {

struct ScodesArgs {
  void* int_buffer;
  size_t int_buffer_bytes;
  void* float_buffer;
  size_t float_buffer_bytes;
  void* weight;          // [experts][n][k/2] packed NVFP4 (kernel A)
  void* act;             // [cum_m][k/2] packed NVFP4 (kernel B)
  void* weight_scales;   // stock swizzled UE4M3 weight scales (escape words read them; SFA tensormap unused)
  void* act_scales;      // swizzled UE4M3 activation scales
  void* out;             // [cum_m][n] BF16
  int* m_indptr;         // [experts + 1]
  uint8_t const* codes;  // scale codes
  uint8_t const* bases;  // row bases [experts][n]
  int64_t expert_stride; // bytes of codes per expert
  int n;
  int k;
  int num_groups;
  cudaStream_t stream;
  int device_id;
};

template <int TileN, int TileK>
cudaError_t ScodesGroupGemm(ScodesArgs const& a);

}  // namespace inkling_scodes
