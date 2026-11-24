#pragma once

namespace certwide {

using namespace screen6;

constexpr int kMaxM = 192;
constexpr int kWarps = 8;       // eight n8 warps, or four candidate tiles per CTA
constexpr int kCols = 8 * kWarps;
constexpr int kThreads = 32 * kWarps;
constexpr int kRowBytes = 2 * kStep;
constexpr float kMaxX = 0x1p20f;
constexpr float kAbsSlop = 0x1p-78f;

enum Reason { kCertified, kUnsafe, kTooMany, kOff };

// One block per activation row. Sum squares before screening so each vocab CTA
// can use the same upper bound on ||x||. Infl covers the serial and tree sums.
__global__ void k_norm(const __nv_bfloat16* x, float* norms, int* state, const int* on) {
  if (!*on) {
    if (threadIdx.x == 0 && blockIdx.x == 0) state[0] = kOff;
    return;
  }
  __shared__ float warp_sums[8];
  const auto* row = x + (size_t)blockIdx.x * kK;
  float sum = 0.f;
  for (int k = threadIdx.x; k < kK; k += blockDim.x) {
    const float a = __bfloat162float(row[k]);
    sum = fmaf(a, a, sum);
  }
  for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? warp_sums[threadIdx.x] : 0.f;
    for (int d = 4; d; d >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, d);
    if (threadIdx.x == 0) {
      const float norm = sqrtf(sum) * kInfl + 4.f * kXAbs;
      norms[blockIdx.x] = norm;
      if (!(norm <= kMaxX)) atomicOr(state, kUnsafe);
    }
  }
}

__device__ __forceinline__ void copy_x(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
               "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

// Stream one 64-vocab CTA of Q. Each step is loaded by one warp per n8
// tile, decoded once in that warp, and used for every m16 fragment before
// advancing K. Shared stages hold only activations.
template <int MR, int Stages>
__global__ void __launch_bounds__(kThreads) k_screen_wide(
    const __nv_bfloat16* x, const uint8_t* q, const uint16_t* sc, const float* radius,
    const float* norms, int M, int V, int v_real, float* lower, float* upper, const int* state,
    float* debug_s, float* debug_lo, float* debug_hi) {
  if (state[0] != kCertified) return;
  constexpr int kStage = MR * 16 * kRowBytes;
  extern __shared__ __align__(16) uint8_t work[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int nt = blockIdx.x * kWarps + warp, n = nt * 8 + 2 * t;
  const int m16 = (M + 15) / 16;
  const bool live_tile = nt < V / 8;
  const uint8_t* qw = q + (size_t)(live_tile ? nt : 0) * kSteps * kTileBytes;
  const float r0 = live_tile ? __ldg(radius + n) : 0.f;
  const float r1 = live_tile ? __ldg(radius + n + 1) : 0.f;

  auto issue_x = [&](int step) {
    auto* dst = work + (step % Stages) * kStage;
    if (step < kSteps) {
      for (int c = threadIdx.x; c < m16 * 16 * 8; c += kThreads) {
        const int row = c / 8, ch = c % 8;
        copy_x(dst + row * kRowBytes + (ch ^ ((row & 1) << 2)) * 16,
               x + (size_t)(row < M ? row : 0) * kK + step * kStep + ch * 8, row < M);
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };

  float sum[MR][4] = {};
  issue_x(0);
  if constexpr (Stages == 3) issue_x(1);
  for (int group = 0; group < kGroups; ++group) {
    const float s0 = live_tile ? half_to_float_u16(__ldg(sc + (size_t)group * V + n)) : 0.f;
    const float s1 = live_tile ? half_to_float_u16(__ldg(sc + (size_t)group * V + n + 1)) : 0.f;
    float partial[MR][4] = {};
    for (int u = 0; u < kStepsPerGroup; ++u) {
      const int step = group * kStepsPerGroup + u;
      const uint8_t* base = qw + step * kTileBytes;
      const uint2 nib = live_tile ? dsk::ldg_stream64(base + 8 * lane) : make_uint2(0, 0);
      const uint32_t high = live_tile ? dsk::ldg_stream32(base + 256 + 4 * lane) : 0;
      asm volatile("cp.async.wait_group %0;" :: "n"(Stages - 2) : "memory");
      __syncthreads();
      // This stage was last read by the previous step, which has now
      // finished in every warp of the CTA.
      issue_x(step + Stages - 1);
      if (live_tile) {
        uint32_t b[8];
#pragma unroll
        for (int p = 0; p < 8; ++p) b[p] = six_pair(p < 4 ? nib.x : nib.y, high, p);
        const uint8_t* xb = work + (step % Stages) * kStage;
        const int off = ((2 * t) ^ ((g & 1) << 2)) * 16;
#pragma unroll
        for (int j = 0; j < MR; ++j) {
          if (j >= m16) break;
          const auto* a = reinterpret_cast<const uint4*>(xb + (j * 16 + g) * kRowBytes + off);
          const auto* a2 = reinterpret_cast<const uint4*>(xb + (j * 16 + g + 8) * kRowBytes + off);
          const uint4 a0 = a[0], a1 = a[1], a20 = a2[0], a21 = a2[1];
          dsk::mma_bf16(partial[j], a0.x, a20.x, a0.y, a20.y, b[0], b[1]);
          dsk::mma_bf16(partial[j], a0.z, a20.z, a0.w, a20.w, b[2], b[3]);
          dsk::mma_bf16(partial[j], a1.x, a21.x, a1.y, a21.y, b[4], b[5]);
          dsk::mma_bf16(partial[j], a1.z, a21.z, a1.w, a21.w, b[6], b[7]);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < MR; ++j) {
      sum[j][0] = fmaf(partial[j][0], s0, sum[j][0]);
      sum[j][1] = fmaf(partial[j][1], s1, sum[j][1]);
      sum[j][2] = fmaf(partial[j][2], s0, sum[j][2]);
      sum[j][3] = fmaf(partial[j][3], s1, sum[j][3]);
    }
  }
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();

  // Reuse the activation storage for maxima. Four lanes handle an n8 row.
  auto* lo8 = reinterpret_cast<float*>(work);
  auto* hi8 = lo8 + kWarps * MR * 16;
  auto epilogue = [&](int row, float s0, float s1) {
    if (row >= M) return;
    const float norm = __ldg(norms + row);
    const float b0 = __fmul_ru(r0, norm), b1 = __fmul_ru(r1, norm);
    const float e0 = (fabsf(s0) + b0) * 0x1p-20f + kAbsSlop;
    const float e1 = (fabsf(s1) + b1) * 0x1p-20f + kAbsSlop;
    const float lo0 = n < v_real ? s0 - b0 - e0 : -INFINITY;
    const float lo1 = n + 1 < v_real ? s1 - b1 - e1 : -INFINITY;
    const float hi0 = n < v_real ? s0 + b0 + e0 : -INFINITY;
    const float hi1 = n + 1 < v_real ? s1 + b1 + e1 : -INFINITY;
    if (debug_s && live_tile) {
      reinterpret_cast<float2*>(debug_s + (size_t)row * V + n)[0] = make_float2(s0, s1);
      reinterpret_cast<float2*>(debug_lo + (size_t)row * V + n)[0] = make_float2(lo0, lo1);
      reinterpret_cast<float2*>(debug_hi + (size_t)row * V + n)[0] = make_float2(hi0, hi1);
    }
    float lo = fmaxf(lo0, lo1), hi = fmaxf(hi0, hi1);
    const unsigned lanes = 0xfu << (4 * g);
    lo = fmaxf(lo, __shfl_xor_sync(lanes, lo, 1));
    lo = fmaxf(lo, __shfl_xor_sync(lanes, lo, 2));
    hi = fmaxf(hi, __shfl_xor_sync(lanes, hi, 1));
    hi = fmaxf(hi, __shfl_xor_sync(lanes, hi, 2));
    if (t == 0) {
      lo8[warp * MR * 16 + row] = lo;
      hi8[warp * MR * 16 + row] = hi;
    }
  };
#pragma unroll
  for (int j = 0; j < MR; ++j) {
    epilogue(16 * j + g, sum[j][0], sum[j][1]);
    epilogue(16 * j + g + 8, sum[j][2], sum[j][3]);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < M * (kCols / 16); i += kThreads) {
    const int row = i / (kCols / 16), pair = i % (kCols / 16);
    const int tile = blockIdx.x * (kCols / 16) + pair;
    if (tile >= V / 16) continue;
    const int a = 2 * pair * MR * 16 + row, b = a + MR * 16;
    const size_t addr = (size_t)row * (V / 16) + tile;
    lower[addr] = fmaxf(lo8[a], lo8[b]);
    upper[addr] = fmaxf(hi8[a], hi8[b]);
  }
}

// (1 - z / 2^20)^(2^20) <= exp(-z). Use downward rounding at each
// operation on this nonnegative base, including the final FP32 conversion.
__device__ float derivative_lower(float L, float cap) {
  const double z = __ddiv_ru(2.0 * (double)L, (double)cap);
  if (!(z <= 32.0)) return 0.f;
  double lower = __dsub_rd(1.0, z * 0x1p-20);
  for (int i = 0; i < 20; ++i) lower = __dmul_rd(lower, lower);
  return __double2float_rd(lower);
}

__global__ void k_threshold(const float* lower, const float* upper, int ntiles,
                            float* threshold, int* state, float cap, float two_e, float dmax) {
  // An earlier range failure can also suppress this block. The candidate
  // pass ignores every threshold whenever any row has forced fallback.
  __shared__ int reason;
  if (threadIdx.x == 0) reason = atomicAdd(state, 0);
  __syncthreads();
  if (reason != kCertified) return;
  __shared__ float los[8], his[8];
  const int row = blockIdx.x;
  float lo = -INFINITY, hi = -INFINITY;
  const size_t base = (size_t)row * ntiles;
  for (int i = threadIdx.x; i < ntiles; i += blockDim.x) {
    lo = fmaxf(lo, lower[base + i]);
    hi = fmaxf(hi, upper[base + i]);
  }
  lo = dsk::warp_max(lo);
  hi = dsk::warp_max(hi);
  if ((threadIdx.x & 31) == 0) {
    los[threadIdx.x >> 5] = lo;
    his[threadIdx.x >> 5] = hi;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    lo = dsk::warp_max(threadIdx.x < 8 ? los[threadIdx.x] : -INFINITY);
    hi = dsk::warp_max(threadIdx.x < 8 ? his[threadIdx.x] : -INFINITY);
    if (threadIdx.x == 0) {
      const float L = fmaxf(__fadd_ru(fabsf(lo), dmax), fabsf(hi));
      const float slope = derivative_lower(L, cap);
      const float delta = __fdiv_ru(2.f * two_e, slope);
      if (!(isfinite(lo) && isfinite(hi) && delta <= dmax))
        atomicOr(state, kUnsafe);
      threshold[row] = __fsub_rd(lo, delta);
    }
  }
}

// Compact the union once. The mask retains the m16 groups in which at least
// one row selected the tile. One reservation per warp with live candidates.
__global__ void k_candidates(const float* upper, const float* threshold, int M, int ntiles,
                             int* state, int* tiles, int* groups, int capacity) {
  if (state[0] != kCertified) return;
  __shared__ float thr[kMaxM];
  if (threadIdx.x < M) thr[threadIdx.x] = threshold[threadIdx.x];
  __syncthreads();
  const int tile = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned group_mask = 0;
  if (tile < ntiles) {
    for (int m = 0; m < M; ++m) {
      if (upper[(size_t)m * ntiles + tile] >= thr[m]) group_mask |= 1u << (m / 16);
    }
  }
  const unsigned selected = __ballot_sync(0xffffffffu, group_mask != 0);
  if (!selected) return;
  int start = 0;
  const int lane = threadIdx.x & 31;
  if (lane == 0) start = atomicAdd(state + 1, __popc(selected));
  start = __shfl_sync(0xffffffffu, start, 0);
  if (group_mask) {
    const int slot = start + __popc(selected & ((1u << lane) - 1));
    if (slot < capacity) {
      tiles[slot] = tile;
      groups[slot] = group_mask;
    }
  }
}

__global__ void k_finish(int* state, int capacity, int ntiles) {
  if (state[0] == kCertified && (state[1] > capacity || state[1] == ntiles || state[1] == 0))
    state[0] = kTooMany;
}

}  // namespace certwide
