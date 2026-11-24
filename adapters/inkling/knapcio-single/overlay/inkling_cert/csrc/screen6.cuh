#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace screen6 {

constexpr int kK = 2304;
constexpr int kStep = 64;
constexpr int kSteps = kK / kStep;
constexpr int kGroup = 192;
constexpr int kGroups = kK / kGroup;
constexpr int kStepsPerGroup = kGroup / kStep;
constexpr int kTileBytes = 384;
constexpr int kRowsCta = 16;
constexpr float kInfl = 1.0f + 1.0f / 65536.0f;
constexpr float kXAbs = 0x1p-58f;

// q + 32 in [1, 63]. Form BF16(128 + q + 32) and subtract 160 exactly.
__device__ __forceinline__ uint32_t six_pair(uint32_t nib, uint32_t hi, int p) {
  const uint32_t lo = (nib >> (4 * (p & 3))) & 0x000F000Fu;
  const uint32_t h2 = ((hi >> (2 * p)) & 0x00030003u) << 4;
  uint32_t v = lo | h2 | 0x43004300u;
  const __nv_bfloat162 off = __floats2bfloat162_rn(160.f, 160.f);
  __nv_bfloat162 r = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&v), off);
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ float half_to_float_u16(uint16_t h) { return __half2float(__ushort_as_half(h)); }

}  // namespace screen6
