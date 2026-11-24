// inkling_cert: certified greedy argmax of the target LM head (FP32 logits) on GB10. Ours, except the exact pass:
// it runs pico's coded-head GEMV body (gemv16_body, kBF16X; pico by Andy Shu, MIT, vendored by inkling_pico12 in
// third_party/pico and compiled into this extension from that same copy) on the CTAs a device mask selects.
//
// Per step, for the M <= 16 verify rows x [M][K] (K = 2304) and the V-row head (V % 16 == 0):
//   screen  s = x . D^T from a 6-bit copy D of the head (q in [-31, 31], one FP16 scale per row and 192-value group),
//           the bound B = sum_g Rt[g][v] ||x_g|| + CN * Nv[v] ||x|| (Rt: residual group norms, Nv: max(||W_v||, ||D_v||),
//           both rounded up), lo / hi = s -+ (B + slop); per pico CTA tile of 16 vocab rows and per row m: the max of lo
//           and of hi over the tile's real-vocab rows (lt / ht); a non-finite s or B sets flags[1].
//   rowmax  T[m] = max lo, H[m] = max hi over all tiles.
//   select  per tile: mask = 1 if the step is not certifiable (on == 0, flags[1], a row outside the softcap range),
//           else OR over rows of (ht[m][tile] >= T[m] - delta[m]); counters.
//   exact   pico's head GEMV for the masked tiles (its own arithmetic, the bits of the full call); every other tile's
//           outputs are set to -inf.
// No host sync, no allocation, fixed grids: CUDA-graph safe. The bound and its proof: ../README.md, ../cert_math.py.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>
#include <type_traits>

#include "kernels/gemv16.cu"  // pico (MIT): gemv16_body, mma_bf16, streaming loads (this TU's own copy)

#include "screen6.cuh"
#include "certwide_screen.cuh"

namespace cert {

using namespace screen6;

template <int RB>
__global__ void __launch_bounds__(32 * kGroups) k_screen(const __nv_bfloat16* __restrict__ x, int M,
                                                       const uint8_t* __restrict__ Q, const uint16_t* __restrict__ SC,
                                                       const uint16_t* __restrict__ RT, const float* __restrict__ NV,
                                                       int V, int v_real, float cn, float* __restrict__ lt,
                                                       float* __restrict__ ht, int ncta, int* __restrict__ flags,
                                                       const int* __restrict__ on, float* __restrict__ sdbg,
                                                       float* __restrict__ bdbg) {
  constexpr bool kHi = RB > 8;
  if (*on == 0) return;  // not eligible: select computes every tile, lt / ht are not read
  __shared__ float red[kGroups][RB][kRowsCta];
  __shared__ float xg2[kGroups][RB];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const bool live = g < M, live2 = kHi && g + 8 < M;
  // one 16-row tile per CTA (a persistent grid-stride variant with x loaded once measured 140-200 GB/s: the x
  // loads then precede the weight stream)
  const int cta = blockIdx.x, n_cta = cta * kRowsCta;

  // weights first: 2 tiles x 3 steps, 12 bytes each per lane
  uint2 wn[2][kStepsPerGroup];
  uint32_t wh[2][kStepsPerGroup];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const size_t tile = (size_t)(n_cta / 8 + i) * kSteps;
#pragma unroll
    for (int u = 0; u < kStepsPerGroup; ++u) {
      const uint8_t* base = Q + (tile + warp * kStepsPerGroup + u) * kTileBytes;
      wn[i][u] = dsk::ldg_stream64(base + 8 * lane);
      wh[i][u] = dsk::ldg_stream32(base + 256 + 4 * lane);
    }
  }
  // activations: row g (and g + 8), the lane's 16 values of each step
  const int k0 = warp * kGroup + 16 * t;
  uint4 xa[kStepsPerGroup][2], xb[kStepsPerGroup][2];
#pragma unroll
  for (int u = 0; u < kStepsPerGroup; ++u) {
    const __nv_bfloat16* xr = x + (size_t)(live ? g : 0) * kK + k0 + u * kStep;
    xa[u][0] = live ? __ldg(reinterpret_cast<const uint4*>(xr)) : make_uint4(0, 0, 0, 0);
    xa[u][1] = live ? __ldg(reinterpret_cast<const uint4*>(xr + 8)) : make_uint4(0, 0, 0, 0);
    if constexpr (kHi) {
      const __nv_bfloat16* xr2 = x + (size_t)(live2 ? g + 8 : 0) * kK + k0 + u * kStep;
      xb[u][0] = live2 ? __ldg(reinterpret_cast<const uint4*>(xr2)) : make_uint4(0, 0, 0, 0);
      xb[u][1] = live2 ? __ldg(reinterpret_cast<const uint4*>(xr2 + 8)) : make_uint4(0, 0, 0, 0);
    } else {
      xb[u][0] = xb[u][1] = make_uint4(0, 0, 0, 0);
    }
  }
  // ||x_g||^2 of this warp's group (sum of the lane's squares, then over the 4 lanes of a row)
  float sq = 0.f, sq2 = 0.f;
#pragma unroll
  for (int u = 0; u < kStepsPerGroup; ++u)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t* a = &xa[u][h].x;
      const uint32_t* b = &xb[u][h].x;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float a0 = dsk::bf16_lo(a[j]), a1 = dsk::bf16_hi(a[j]);
        sq = fmaf(a0, a0, fmaf(a1, a1, sq));
        if constexpr (kHi) {
          const float b0 = dsk::bf16_lo(b[j]), b1 = dsk::bf16_hi(b[j]);
          sq2 = fmaf(b0, b0, fmaf(b1, b1, sq2));
        }
      }
    }
  sq += __shfl_xor_sync(0xffffffffu, sq, 1);
  sq += __shfl_xor_sync(0xffffffffu, sq, 2);
  if constexpr (kHi) {
    sq2 += __shfl_xor_sync(0xffffffffu, sq2, 1);
    sq2 += __shfl_xor_sync(0xffffffffu, sq2, 2);
  }
  if (t == 0) {
    if (g < RB) xg2[warp][g] = live ? sq : 0.f;
    if constexpr (kHi) xg2[warp][g + 8] = live2 ? sq2 : 0.f;
  }

  float c[2][4];
#pragma unroll
  for (int i = 0; i < 2; ++i) c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f;
#pragma unroll
  for (int u = 0; u < kStepsPerGroup; ++u)
#pragma unroll
    for (int j4 = 0; j4 < 4; ++j4) {
      const uint4& av = xa[u][j4 >> 1];
      const uint4& bv = xb[u][j4 >> 1];
      const uint32_t a0 = (j4 & 1) ? av.z : av.x, a2 = (j4 & 1) ? av.w : av.y;
      const uint32_t a1 = (j4 & 1) ? bv.z : bv.x, a3 = (j4 & 1) ? bv.w : bv.y;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint32_t nib = (j4 < 2) ? wn[i][u].x : wn[i][u].y;
        const uint32_t b0 = six_pair(nib, wh[i][u], 2 * j4), b1 = six_pair(nib, wh[i][u], 2 * j4 + 1);
        dsk::mma_bf16(c[i], a0, a1, a2, a3, b0, b1);
      }
    }
  // group scale (fp16, [kGroups][V]) of the lane's two columns per tile, then the partials to shared memory
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int n = n_cta + 8 * i + 2 * t;
    const uint32_t s2 = __ldg(reinterpret_cast<const uint32_t*>(SC + (size_t)warp * V + n));
    const float s0 = half_to_float_u16((uint16_t)(s2 & 0xFFFFu)), s1 = half_to_float_u16((uint16_t)(s2 >> 16));
    const int r = 8 * i + 2 * t;
    if (g < RB) {
      red[warp][g][r] = c[i][0] * s0;
      red[warp][g][r + 1] = c[i][1] * s1;
    }
    if constexpr (kHi) {
      red[warp][g + 8][r] = c[i][2] * s0;
      red[warp][g + 8][r + 1] = c[i][3] * s1;
    }
  }
  __syncthreads();

  // epilogue: one thread per (row m, column r) of the CTA, m < RB
  const int e = threadIdx.x;
  if (e < RB * kRowsCta) {
    const int m = e / kRowsCta, r = e % kRowsCta, n = n_cta + r;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < kGroups; ++w) s += red[w][m][r];
    float b = 0.f, xx = 0.f;
#pragma unroll
    for (int w = 0; w < kGroups; ++w) {
      const float q2 = xg2[w][m];
      xx += q2;
      // >= ||x_g||: fp32 sum of 192 squares (relative error < 2^-16 after sqrt), sqrt rounding, and squares that
      // underflow (each < 2^-126 is at worst lost: 192 * 2^-126 < 2^-116, sqrt < 2^-58)
      const float xg = sqrtf(q2) * kInfl + kXAbs;
      b = fmaf(half_to_float_u16(__ldg(RT + (size_t)w * V + n)), xg, b);
    }
    const float xn = sqrtf(xx) * kInfl + 4.f * kXAbs;
    b = (b * kInfl + cn * __ldg(NV + n) * xn) * kInfl;
    const float slop = (fabsf(s) + b) * (1.0f / 1048576.0f) + 1e-30f;
    float lo = s - b - slop, hi = s + b + slop;
    const bool real = m < M && n < v_real;
    if (sdbg != nullptr && m < M) sdbg[(size_t)m * V + n] = s, bdbg[(size_t)m * V + n] = b;  // GPU check only
    // outside the supported domain (non-finite, or ||x|| > 2^20 where the absolute underflow allowance in slop would
    // no longer cover flushed subnormal products): the whole step takes the full head
    if (m < M && n < V && !(isfinite(s) && isfinite(b) && xn <= 1048576.f)) atomicOr(flags + 1, 1);
    lo = real ? lo : -INFINITY;
    hi = real ? hi : -INFINITY;
    // max over the 16 columns of row m: 16 consecutive lanes
#pragma unroll
    for (int o = 8; o >= 1; o >>= 1) {
      lo = fmaxf(lo, __shfl_xor_sync(0xffffffffu, lo, o));
      hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, o));
    }
    if (r == 0 && m < M) {
      lt[(size_t)m * ncta + cta] = lo;
      ht[(size_t)m * ncta + cta] = hi;
    }
  }
}

// T[m] = max lo, H[m] = max hi over the tiles (grid M, 1024 threads)
__global__ void __launch_bounds__(1024) k_rowmax(const float* __restrict__ lt, const float* __restrict__ ht, int ncta,
                                                 float* __restrict__ tm, float* __restrict__ hm,
                                                 const int* __restrict__ on) {
  __shared__ float sl[32], sh[32];
  const int m = blockIdx.x;
  if (*on == 0) return;
  float lo = -INFINITY, hi = -INFINITY;
  for (int i = threadIdx.x; i < ncta; i += blockDim.x) {
    lo = fmaxf(lo, lt[(size_t)m * ncta + i]);
    hi = fmaxf(hi, ht[(size_t)m * ncta + i]);
  }
  lo = dsk::warp_max(lo);
  hi = dsk::warp_max(hi);
  if ((threadIdx.x & 31) == 0) sl[threadIdx.x >> 5] = lo, sh[threadIdx.x >> 5] = hi;
  __syncthreads();
  if (threadIdx.x < 32) {
    lo = threadIdx.x < (int)(blockDim.x >> 5) ? sl[threadIdx.x] : -INFINITY;
    hi = threadIdx.x < (int)(blockDim.x >> 5) ? sh[threadIdx.x] : -INFINITY;
    lo = dsk::warp_max(lo);
    hi = dsk::warp_max(hi);
    if (threadIdx.x == 0) tm[m] = lo, hm[m] = hi;
  }
}

// Softcap margin. The stock path applies f(l) = fl(fl(tanh(fl(l / c))) * c) (Triton: div.full, libdevice tanhf),
// |f(l) - c tanh(l / c)| <= E. With every relevant logit in [T - DMAX, H] and |T| + DMAX, |H| <= L:
//   f(a) >= f(b), a < b  =>  b - a <= 2E / min sech^2 <= 2E / exp(-2L / c)   (sech^2 y >= exp(-2|y|));
//   we use delta = 2 * 2E / exp(-2L / c) (2x margin for __expf), and require delta <= DMAX;
// and the -inf fill (f = -c) stays below every candidate maximum when c (1 - tanh(L / c)) >= c exp(-2L / c) > E;
// we require c exp(-2L / c) > 4 * 2E.
__global__ void __launch_bounds__(256) k_select(const float* __restrict__ ht, int ncta, int M, const float* __restrict__ tm,
                                                const float* __restrict__ hm, const int* __restrict__ on,
                                                int* __restrict__ flags, uint8_t* __restrict__ mask,
                                                unsigned long long* __restrict__ stats, float cap, float two_e, float dmax) {
  __shared__ float thr[16];
  __shared__ int full_s;
  __shared__ int cnt;
  if (threadIdx.x == 0) {
    int full = (*on == 0) || (flags[1] != 0);
    cnt = 0;
    for (int m = 0; m < M && !full; ++m) {
      const float T = tm[m], H = hm[m];
      const float L = fmaxf(fabsf(T) + dmax, fabsf(H));
      const float ex = __expf(-2.f * L / cap);  // sech^2(L / c) >= ex
      const float delta = 2.f * two_e / fmaxf(ex, 1e-30f);  // 2x the bound 2E / ex
      const bool ok = isfinite(T) && isfinite(H) && L < 1e4f && delta <= dmax && cap * ex > 4.f * two_e;
      if (!ok) full = 1;
      thr[m] = T - (ok ? delta : 0.f);
    }
    full_s = full;
  }
  __syncthreads();
  const int full = full_s;
  const int tile = blockIdx.x * blockDim.x + threadIdx.x;
  int set = 0;
  if (tile < ncta) {
    if (full) {
      set = 1;
    } else {
      for (int m = 0; m < M; ++m) set |= ht[(size_t)m * ncta + tile] >= thr[m];
    }
    mask[tile] = (uint8_t)set;
  }
  if (!full && set) atomicAdd(&cnt, 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (!full && cnt) atomicAdd(stats + 0, (unsigned long long)cnt);
    if (blockIdx.x == 0) {
      if (*on == 0) atomicAdd(stats + 4, 1ULL);
      else if (full) atomicAdd(stats + 2, 1ULL);
      else atomicAdd(stats + 1, 1ULL);
      flags[2] = (*on != 0) && !full;  // certified step (read by the check mode)
    }
  }
}

// pico's coded-head GEMV (gemv_bf16x, mt 2, wn 1) on the masked CTAs; -inf for the others. Same body, template
// arguments and launch bounds as pico's k_gemv16 (RB 8) / k_gemv16_r16 (RB 16).
template <int RB>
__device__ __forceinline__ void exact_cta(const uint8_t* __restrict__ mask, const __nv_bfloat16* __restrict__ x, int M,
                                          const uint8_t* __restrict__ X, int N, int K, float* __restrict__ yf, int wk,
                                          int* __restrict__ flags) {
  if (blockIdx.x == 0 && threadIdx.x == 0) flags[1] = 0;  // the screen's non-finite flag (select has read it)
  if (!mask[blockIdx.x]) {
    const int n_cta = blockIdx.x * kRowsCta;
    for (int idx = threadIdx.x; idx < M * kRowsCta; idx += blockDim.x)
      yf[(size_t)(idx / kRowsCta) * N + n_cta + idx % kRowsCta] = -INFINITY;
    return;
  }
  dsk::gemv16_body<dsk::kBF16X, 2, false, false, RB>(x, M, X, nullptr, N, K, N, yf, nullptr, nullptr, nullptr, 1, wk,
                                                     dsk::NormIn{});
}

__global__ void __launch_bounds__(512) k_exact8(const uint8_t* __restrict__ mask, const __nv_bfloat16* __restrict__ x,
                                                int M, const uint8_t* __restrict__ X, int N, int K,
                                                float* __restrict__ yf, int wk, int* __restrict__ flags) {
  exact_cta<dsk::kLowRows>(mask, x, M, X, N, K, yf, wk, flags);
}

__global__ void __launch_bounds__(512, 1) k_exact16(const uint8_t* __restrict__ mask, const __nv_bfloat16* __restrict__ x,
                                                    int M, const uint8_t* __restrict__ X, int N, int K,
                                                    float* __restrict__ yf, int wk, int* __restrict__ flags) {
  exact_cta<dsk::kMaxRows>(mask, x, M, X, N, K, yf, wk, flags);
}

}  // namespace cert

namespace {

void check_x(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == cert::kK,
              "inkling_cert: x must be CUDA bf16 [M, 2304]");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 16, "inkling_cert: rows ", x.size(0), " not in [1, 16]");
  TORCH_CHECK(x.stride(1) == 1 && (x.size(0) == 1 || x.stride(0) == cert::kK), "inkling_cert: x rows not contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "inkling_cert: x not 16-byte aligned");
}

void check_dev(const torch::Tensor& t, const torch::Tensor& like, at::ScalarType dt, int64_t numel, const char* what) {
  TORCH_CHECK(t.is_cuda() && t.device() == like.device() && t.scalar_type() == dt && t.is_contiguous() &&
                  t.numel() >= numel,
              "inkling_cert: bad ", what);
}

// screen + rowmax + select (the candidate mask for exact())
void screen(const torch::Tensor& x, const torch::Tensor& q, const torch::Tensor& sc, const torch::Tensor& rt,
            const torch::Tensor& nv, int64_t V, int64_t v_real, double cn, const torch::Tensor& lt,
            const torch::Tensor& ht, const torch::Tensor& tm, const torch::Tensor& hm, const torch::Tensor& on,
            const torch::Tensor& flags, const torch::Tensor& mask, const torch::Tensor& stats, double cap,
            double two_e, double dmax, const torch::Tensor& sdbg, const torch::Tensor& bdbg) {
  check_x(x);
  const int64_t M = x.size(0), ncta = V / cert::kRowsCta;
  TORCH_CHECK(V % cert::kRowsCta == 0 && v_real <= V && v_real > 0, "inkling_cert: V=", V, " v_real=", v_real);
  check_dev(q, x, at::kByte, V * cert::kK * 3 / 4, "q");
  check_dev(sc, x, at::kHalf, cert::kGroups * V, "sc");
  check_dev(rt, x, at::kHalf, cert::kGroups * V, "rt");
  check_dev(nv, x, at::kFloat, V, "nv");
  check_dev(lt, x, at::kFloat, M * ncta, "lt");
  check_dev(ht, x, at::kFloat, M * ncta, "ht");
  check_dev(tm, x, at::kFloat, M, "tm");
  check_dev(hm, x, at::kFloat, M, "hm");
  check_dev(on, x, at::kInt, 1, "on");
  check_dev(flags, x, at::kInt, 4, "flags");
  check_dev(mask, x, at::kByte, ncta, "mask");
  check_dev(stats, x, at::kLong, 8, "stats");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto st = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* scp = reinterpret_cast<const uint16_t*>(sc.data_ptr());
  const auto* rtp = reinterpret_cast<const uint16_t*>(rt.data_ptr());
  float* sd = nullptr;
  float* bd = nullptr;
  if (sdbg.numel() > 0) {
    check_dev(sdbg, x, at::kFloat, M * V, "sdbg");
    check_dev(bdbg, x, at::kFloat, M * V, "bdbg");
    sd = sdbg.data_ptr<float>();
    bd = bdbg.data_ptr<float>();
  }
  if (M <= 8)
    cert::k_screen<8><<<ncta, 32 * cert::kGroups, 0, st>>>(xp, (int)M, q.data_ptr<uint8_t>(), scp, rtp,
                                                          nv.data_ptr<float>(), (int)V, (int)v_real, (float)cn,
                                                          lt.data_ptr<float>(), ht.data_ptr<float>(), (int)ncta,
                                                          flags.data_ptr<int>(), on.data_ptr<int>(), sd, bd);
  else
    cert::k_screen<16><<<ncta, 32 * cert::kGroups, 0, st>>>(xp, (int)M, q.data_ptr<uint8_t>(), scp, rtp,
                                                           nv.data_ptr<float>(), (int)V, (int)v_real, (float)cn,
                                                           lt.data_ptr<float>(), ht.data_ptr<float>(), (int)ncta,
                                                           flags.data_ptr<int>(), on.data_ptr<int>(), sd, bd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cert::k_rowmax<<<M, 1024, 0, st>>>(lt.data_ptr<float>(), ht.data_ptr<float>(), (int)ncta, tm.data_ptr<float>(),
                                     hm.data_ptr<float>(), on.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cert::k_select<<<(ncta + 255) / 256, 256, 0, st>>>(ht.data_ptr<float>(), (int)ncta, (int)M, tm.data_ptr<float>(),
                                                     hm.data_ptr<float>(), on.data_ptr<int>(), flags.data_ptr<int>(),
                                                     mask.data_ptr<uint8_t>(), reinterpret_cast<unsigned long long*>(stats.data_ptr<int64_t>()), (float)cap,
                                                     (float)two_e, (float)dmax);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// out[M, N] = pico's coded head GEMV on the masked CTAs, -inf elsewhere
void exact(const torch::Tensor& x, const torch::Tensor& packed, int64_t N, int64_t K, const torch::Tensor& mask,
           const torch::Tensor& out, int64_t wk, const torch::Tensor& flags) {
  check_x(x);
  TORCH_CHECK(K == cert::kK && N % cert::kRowsCta == 0, "inkling_cert: exact N=", N, " K=", K);
  TORCH_CHECK(wk >= 1 && wk <= 16 && wk <= K / 32, "inkling_cert: wk=", wk);
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == at::kByte && packed.is_contiguous(), "inkling_cert: packed");
  check_dev(mask, x, at::kByte, N / cert::kRowsCta, "mask");
  check_dev(flags, x, at::kInt, 4, "flags");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.dim() == 2 && out.size(0) == x.size(0) &&
                  out.size(1) == N && out.stride(1) == 1 && (out.size(0) == 1 || out.stride(0) == N),
              "inkling_cert: out must be fp32 [M, N] with contiguous rows");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto st = at::cuda::getCurrentCUDAStream();
  const int M = (int)x.size(0);
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const int ncta = (int)(N / cert::kRowsCta);
  if (M <= dsk::kLowRows) {
    const size_t smem = (size_t)wk * dsk::kLowRows * cert::kRowsCta * sizeof(float);
    cert::k_exact8<<<ncta, 32 * wk, smem, st>>>(mask.data_ptr<uint8_t>(), xp, M,
                                                             packed.data_ptr<uint8_t>(), (int)N, (int)K,
                                                             out.data_ptr<float>(), (int)wk, flags.data_ptr<int>());
  } else {
    const size_t smem = (size_t)wk * dsk::kMaxRows * cert::kRowsCta * sizeof(float);
    cert::k_exact16<<<ncta, 32 * wk, smem, st>>>(mask.data_ptr<uint8_t>(), xp, M,
                                                             packed.data_ptr<uint8_t>(), (int)N, (int)K,
                                                             out.data_ptr<float>(), (int)wk, flags.data_ptr<int>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int MR, int Stages>
void prepare_screen_wide() {
  C10_CUDA_CHECK(cudaFuncSetAttribute(certwide::k_screen_wide<MR, Stages>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      Stages * MR * 16 * certwide::kRowBytes));
}

void prepare_certwide(int64_t device) {
  const c10::cuda::CUDAGuard guard((c10::DeviceIndex)device);
  prepare_screen_wide<4, 3>();
  prepare_screen_wide<8, 3>();
  prepare_screen_wide<12, 2>();
  prepare_screen_wide<12, 3>();
}

template <typename Fn>
auto dispatch_screen_wide(int M, bool three_stages, Fn launch) {
  using Three = std::integral_constant<int, 3>;
  if (M <= 64) return launch(std::integral_constant<int, 4>{}, Three{});
  if (M <= 128) return launch(std::integral_constant<int, 8>{}, Three{});
  if (three_stages) return launch(std::integral_constant<int, 12>{}, Three{});
  return launch(std::integral_constant<int, 12>{}, std::integral_constant<int, 2>{});
}

std::tuple<int, size_t, int, int> screen_wide_info(int64_t device, int M, bool three_stages) {
  using namespace certwide;
  TORCH_CHECK(M >= 17 && M <= kMaxM, "inkling_certwide: unsupported screen rows");
  const c10::cuda::CUDAGuard guard((c10::DeviceIndex)device);
  return dispatch_screen_wide(M, three_stages, [](auto bucket, auto stages) {
    constexpr int MR = decltype(bucket)::value, Stages = decltype(stages)::value;
    constexpr int smem = Stages * MR * 16 * kRowBytes;
    auto kernel = k_screen_wide<MR, Stages>;
    cudaFuncAttributes attr{};
    int blocks = 0;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, kThreads, smem));
    return std::make_tuple(attr.numRegs, attr.localSizeBytes, smem, blocks);
  });
}

void select_wide(const torch::Tensor& lower, const torch::Tensor& upper, const torch::Tensor& threshold,
                 const torch::Tensor& state, const torch::Tensor& tiles, const torch::Tensor& groups,
                 int M, int ntiles, float cap, float two_e, float dmax, cudaStream_t st) {
  using namespace certwide;
  auto* sp = state.data_ptr<int>();
  k_threshold<<<M, 256, 0, st>>>(lower.data_ptr<float>(), upper.data_ptr<float>(), ntiles,
       threshold.data_ptr<float>(), sp, cap, two_e, dmax);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int capacity = tiles.numel();
  k_candidates<<<(ntiles + 255) / 256, 256, 0, st>>>(upper.data_ptr<float>(), threshold.data_ptr<float>(), M,
       ntiles, sp, tiles.data_ptr<int>(), groups.data_ptr<int>(), capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  k_finish<<<1, 1, 0, st>>>(sp, capacity, ntiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void debug_select_wide(const torch::Tensor& lower, const torch::Tensor& upper, const torch::Tensor& threshold,
                       const torch::Tensor& state, const torch::Tensor& tiles, const torch::Tensor& groups,
                       double cap, double two_e, double dmax) {
  TORCH_CHECK(lower.dim() == 2 && lower.size(0) > 0 && lower.size(0) <= certwide::kMaxM &&
                lower.size(1) > 0 && lower.size(1) <= INT32_MAX / 16 && lower.sizes() == upper.sizes() &&
                tiles.numel() > 0 && tiles.numel() <= lower.size(1), "inkling_certwide: bad debug maxima shape");
  const int M = lower.size(0), ntiles = lower.size(1);
  check_dev(lower, lower, at::kFloat, lower.numel(), "lower");
  check_dev(upper, lower, at::kFloat, lower.numel(), "upper");
  check_dev(threshold, lower, at::kFloat, M, "threshold");
  check_dev(state, lower, at::kInt, 2, "state");
  check_dev(tiles, lower, at::kInt, tiles.numel(), "tiles");
  check_dev(groups, lower, at::kInt, tiles.numel(), "groups");
  const c10::cuda::CUDAGuard guard(lower.device());
  const auto st = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(state.data_ptr<int>(), 0, 2 * sizeof(int), st));
  select_wide(lower, upper, threshold, state, tiles, groups, M, ntiles, cap, two_e, dmax, st);
}

void screen_wide(const torch::Tensor& x, const torch::Tensor& q, const torch::Tensor& sc,
                 const torch::Tensor& radius, int64_t V, int64_t v_real, const torch::Tensor& norms,
                 const torch::Tensor& lower, const torch::Tensor& upper, const torch::Tensor& threshold,
                 const torch::Tensor& on, const torch::Tensor& state, const torch::Tensor& tiles,
                 const torch::Tensor& groups, const torch::Tensor& dbg_s, const torch::Tensor& dbg_lo,
                 const torch::Tensor& dbg_hi, double cap, double two_e, double dmax, bool three_stages) {
  using namespace certwide;
  TORCH_CHECK(x.dim() == 2 && x.size(0) >= 17 && x.size(0) <= kMaxM && x.size(1) == kK,
              "inkling_certwide: x must be [17..192, 2304]");
  const int M = (int)x.size(0);
  const int64_t ntiles = V / 16, capacity = tiles.numel();
  TORCH_CHECK(V > 0 && V <= INT32_MAX && V % 16 == 0 && v_real > 0 && v_real <= V &&
                capacity > 0 && capacity <= ntiles, "inkling_certwide: bad vocab or capacity");
  check_dev(x, x, at::kBFloat16, M * kK, "x");
  check_dev(q, x, at::kByte, V * kK * 3 / 4, "q");
  check_dev(sc, x, at::kHalf, kGroups * V, "sc");
  check_dev(radius, x, at::kFloat, V, "radius");
  check_dev(norms, x, at::kFloat, M, "norms");
  check_dev(lower, x, at::kFloat, M * ntiles, "lower");
  check_dev(upper, x, at::kFloat, M * ntiles, "upper");
  check_dev(threshold, x, at::kFloat, M, "threshold");
  check_dev(on, x, at::kInt, 1, "on");
  check_dev(state, x, at::kInt, 2, "state");
  check_dev(tiles, x, at::kInt, capacity, "tiles");
  check_dev(groups, x, at::kInt, capacity, "groups");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(q.data_ptr()) % 8 == 0, "inkling_certwide: misaligned screen input");
  float* sd = nullptr;
  float* ld = nullptr;
  float* hd = nullptr;
  if (dbg_s.numel() || dbg_lo.numel() || dbg_hi.numel()) {
    check_dev(dbg_s, x, at::kFloat, M * V, "debug screen logits");
    check_dev(dbg_lo, x, at::kFloat, M * V, "debug lower endpoints");
    check_dev(dbg_hi, x, at::kFloat, M * V, "debug upper endpoints");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(dbg_s.data_ptr()) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(dbg_lo.data_ptr()) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(dbg_hi.data_ptr()) % 8 == 0,
                "inkling_certwide: misaligned float2 debug output");
    sd = dbg_s.data_ptr<float>();
    ld = dbg_lo.data_ptr<float>();
    hd = dbg_hi.data_ptr<float>();
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const auto st = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* scp = reinterpret_cast<const uint16_t*>(sc.data_ptr());
  auto* sp = state.data_ptr<int>();
  C10_CUDA_CHECK(cudaMemsetAsync(sp, 0, 2 * sizeof(int), st));
  k_norm<<<M, 256, 0, st>>>(xp, norms.data_ptr<float>(), sp, on.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto launch = [&](auto bucket, auto stages) {
    constexpr int MR = decltype(bucket)::value;
    constexpr int Stages = decltype(stages)::value;
    k_screen_wide<MR, Stages><<<(V + kCols - 1) / kCols, kThreads, Stages * MR * 16 * kRowBytes, st>>>(
        xp, q.data_ptr<uint8_t>(), scp, radius.data_ptr<float>(), norms.data_ptr<float>(), M, (int)V, (int)v_real,
        lower.data_ptr<float>(), upper.data_ptr<float>(), sp, sd, ld, hd);
  };
  dispatch_screen_wide(M, three_stages, launch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  select_wide(lower, upper, threshold, state, tiles, groups, M, (int)ntiles, cap, two_e, dmax, st);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "inkling_cert: certified greedy LM-head argmax (6-bit screen + pico's exact head GEMV on candidates)";
  m.def("screen", &screen, "6-bit screen, row maxima and candidate mask");
  m.def("screen_wide", &screen_wide, "batched 6-bit screen and candidate compaction");
  m.def("screen_wide_info", &screen_wide_info, "registers, local bytes, dynamic shared bytes and resident blocks per SM");
  m.def("debug_select_wide", &debug_select_wide, "select candidates from supplied tile maxima");
  m.def("prepare_certwide", &prepare_certwide, "prepare batched screen kernels");
  m.def("exact", &exact, "pico's coded head GEMV on the masked CTAs, -inf elsewhere");
  m.attr("K") = cert::kK;
  m.attr("GROUP") = cert::kGroup;
  m.attr("ROWS_CTA") = cert::kRowsCta;
}
