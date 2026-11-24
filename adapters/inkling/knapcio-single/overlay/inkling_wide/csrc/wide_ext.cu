// inkling_wide: weight-streaming BF16 GEMM for 17..256 activation rows (DSpark verify batches), on raw BF16
// weights (natural [N][K] row-major, the layer's own storage) or on pico's exactly coded `bf16x` copy (built by
// inkling_pico12; reader in bf16x_format.cuh, pico by Andy Shu, MIT).
//
//   out[m, n] = sum_k x[m, k] W[n, k]      BF16 operands, FP32 accumulation (mma.sync m16n8k16), FP32 or BF16 out
//
// Each weight byte is read from DRAM once per call. A CTA owns kBN = 128 weight rows (16 tiles of 8 rows) and a
// K slice (split-K over gridDim.y CTAs, fixed-order reduction by the last CTA of a column block). Its warps own
// MT tiles each (MT = 2 up to 128 rows, 1 above) and stream them straight into registers (16 bytes per lane and
// tile step, one batch of 2 steps ahead), decode them in registers (bf16x) and multiply them with ALL M rows of
// the activations: the weights are the MMA's B operand (n8 = 8 weight rows), the activations its A operand
// (m16 = 16 rows), read from shared memory, where the CTA stages x[0:M, 64-value chunk] with cp.async (3 stages).
//
// A tile step's lane layout (lane (g, t): weight row g, values 8t .. 8t+7 of the step's 32) is the layout of the
// mma B fragment with the k order permuted (pair j of the lane's 8 values is k position 2t + 8*(j&1) of MMA
// (j>>1)); the A fragment is loaded with the same permutation (16 contiguous bytes of row g and of row g + 8), so
// each MMA computes a correct partial dot product. Raw and coded weights give the same B registers after the
// decode, so the two formats give the same bits.
//
// Stacked q/k/v (Inkling2Attention._project_sources with three sources): the CTA's column block picks its
// activation source (columns [rb1, rb2): x1, [rb2, rb3): x2, else x0) and its output row offset (roff), so the
// three sources share one read of the weight. Region bounds must be multiples of kBN (checked on the host).
//
// CUDA graphs: the launch allocates nothing and does not synchronise; split-K uses a caller-owned FP32 workspace
// and a counter per column block that the reducing CTA resets to zero.
// PDL: the first weight batch is loaded before griddepcontrol.wait; x, out, ws and cnt only after it. The weights
// must therefore not be written by the kernel launched just before (they are load-time constants here).
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <tuple>

#include "bf16x_format.cuh"

namespace wide {

constexpr int kBN = 128;                         // weight rows (output columns) per CTA
constexpr int kTilesCta = kBN / 8;               // 16 tiles of 8 rows
constexpr int kStepK = 32;                       // k values per tile step
constexpr int kBatch = 2;                        // steps per pipeline batch
constexpr int kStages = 3;                       // activation stages in shared memory
constexpr int kRowB = kBatch * kStepK * 2;       // 128 bytes of a staged activation row
constexpr int kMaxM = 256;

enum Fmt { kRaw = 0, kBF16X = 2 };

struct Params {
  const __nv_bfloat16* x0;
  const __nv_bfloat16* x1;
  const __nv_bfloat16* x2;
  long long ldx;  // row stride of every source (elements)
  int M;
  int rb1, rb2, rb3;      // column regions: [rb1, rb2) -> x1, [rb2, rb3) -> x2, else x0
  long long roff1, roff2; // output row offset of sources 1 and 2 (source 0: 0)
  const uint8_t* W;
  int N, K;
  void* out;
  long long ldo;  // output row stride (elements)
  int f32;        // 1: FP32 out, 0: BF16 out
  int S;          // split-K slices (gridDim.y)
  int pfd;        // L2 prefetch distance in batches (0: none)
  float* ws;      // [S][M][N] partials (S > 1)
  int* cnt;       // [gridDim.x] arrival counters, zero between calls (S > 1)
  const int* cert_state;  // optional device gate [fallback reason, union size]
  const int* cert_tiles;  // sparse slot -> original 16-column tile
  const int* cert_groups; // sparse slot -> bitmask of candidate m16 groups
};

__device__ __forceinline__ void mma_bf16(float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldg_stream128(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((uint32_t)__cvta_generic_to_shared(smem)), "l"(gmem)
               : "memory");
}
// 16 bytes, or 16 zero bytes when !live (src-size 0; src must still be a valid address)
__device__ __forceinline__ void cp_async16_zfill(void* smem, const void* gmem, bool live) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"((uint32_t)__cvta_generic_to_shared(smem)),
               "l"(gmem), "r"(live ? 16 : 0)
               : "memory");
}
__device__ __forceinline__ int atomic_add_acq_rel_gpu(int* p, int v) {
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// MT: tiles per warp. 2 up to 128 rows; above, 1 (16 warps, 64 accumulators a lane) or 2 (8 warps, 128
// accumulators a lane, half the shared-memory traffic per MMA).
constexpr int default_mt(int MI) { return MI <= 8 ? 2 : 1; }

// MG: warp groups along M. 1: every warp multiplies its tiles with all rows; 2 (buckets 6 and 8): two warps load
// the same tiles and take half the rows each (16 warps a CTA, half the accumulators a lane; the second load of a
// tile is an L2 hit).
template <int MI, int MT_, int MG_ = 1>
struct Cfg {
  static constexpr int MT = MT_;
  static constexpr int MG = MG_;
  static constexpr int MIW = MI / MG;             // m16 tiles per warp
  static constexpr int NWN = kTilesCta / MT;      // warps along N
  static constexpr int NW = NWN * MG;             // warps per CTA
  static constexpr int NT = NW * 32;              // threads per CTA
  static constexpr int MINB = (MI <= 4 && MG == 1) ? 2 : 1;  // CTAs per SM the register budget aims at
  static constexpr int STAGE = MI * 16 * kRowB;   // bytes of one activation stage
  static constexpr int SMEM = kStages * STAGE;
};

template <int FMT, int MI, int MT_, int MG_, bool SPARSE = false>
__global__ void __launch_bounds__(Cfg<MI, MT_, MG_>::NT, Cfg<MI, MT_, MG_>::MINB) k_wide(const Params p) {
  using C = Cfg<MI, MT_, MG_>;
  static_assert(MI % MG_ == 0, "rows split evenly between the warp groups");
  constexpr int MT = C::MT;
  constexpr int MIW = C::MIW;
  constexpr int NT = C::NT;
  constexpr int STAGE = C::STAGE;
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ int s_last;
  __shared__ unsigned active_m16;
  // The certificate is a preceding stream dependency; wait before using its
  // device-produced gate or indices. Regular wide still preloads weights first.
  if (p.cert_state) {
    pdl_wait();
    const bool fallback = p.cert_state[0] != 0;
    if constexpr (SPARSE) {
      if (fallback || blockIdx.x * (kBN / 16) >= p.cert_state[1]) return;
    } else {
      if (!fallback) return;
    }
  }

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wn = warp % C::NWN, mg = warp / C::NWN;  // the warp's tiles, its row group
  if constexpr (SPARSE) {
    if (warp == 0) {
      constexpr int slots_per_cta = kBN / 16;
      const int slot = blockIdx.x * slots_per_cta + lane;
      const unsigned mask = lane < slots_per_cta && slot < p.cert_state[1] ? p.cert_groups[slot] : 0u;
      const unsigned all_groups = __reduce_or_sync(0xffffffffu, mask);
      if (lane == 0) active_m16 = all_groups;
    }
    __syncthreads();
  }

  const int M = p.M, N = p.N, K = p.K;
  const int P = K / kStepK;    // tile steps along K
  const int NB = P / kBatch;   // batches along K
  const int S = p.S, s = blockIdx.y;
  const int b0 = (int)((long long)NB * s / S), b1 = (int)((long long)NB * (s + 1) / S);
  const int n_cta = blockIdx.x * kBN;
  const int ntiles = N / 8;
  const int src = (n_cta >= p.rb1 && n_cta < p.rb2) ? 1 : (n_cta >= p.rb2 && n_cta < p.rb3) ? 2 : 0;
  const __nv_bfloat16* __restrict__ xs = src == 1 ? p.x1 : src == 2 ? p.x2 : p.x0;
  const long long roff = src == 1 ? p.roff1 : src == 2 ? p.roff2 : 0;

  // ---- this lane's weight pointers
  int nt[MT];
  bool valid[MT];
  unsigned groups[MT];
  unsigned warp_groups = 0;
  const uint8_t* wptr[MT];  // raw: row 8*nt+g at value 8t; bf16x: tile (nt, 0)'s sign-mantissa bytes of the lane
  const uint8_t* eptr[MT];  // bf16x: exponent offsets of tile (nt, 0) of the lane
  const uint32_t* hptr[MT]; // bf16x: header of tile (nt, 0)
  const size_t T = (size_t)ntiles * P;
  // bf16x: the lane's bytes of raw tile 0, less 2^40: raw tile r (header h = 2^31 + r) at rawq + 512 h
  const uint64_t rawq =
      FMT == kBF16X ? reinterpret_cast<uint64_t>(p.W + bf16x_raw_off(T) + 16 * lane) - (1ull << 40) : 0;
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    const int slot = blockIdx.x * kTilesCta + wn * MT + i;
    if constexpr (SPARSE) {
      valid[i] = slot / 2 < p.cert_state[1];
      nt[i] = valid[i] ? 2 * p.cert_tiles[slot / 2] + slot % 2 : 0;
      groups[i] = valid[i] ? p.cert_groups[slot / 2] : 0;
      warp_groups |= groups[i];
    } else {
      nt[i] = slot;
      valid[i] = nt[i] < ntiles;
    }
    const size_t ti = (size_t)(valid[i] ? nt[i] : 0) * P;
    if constexpr (FMT == kBF16X) {
      wptr[i] = p.W + ti * 256 + 8 * lane;
      eptr[i] = p.W + T * 256 + ti * 128 + 4 * lane;
      hptr[i] = reinterpret_cast<const uint32_t*>(p.W + T * 384) + ti;
    } else {
      wptr[i] = p.W + ((size_t)(valid[i] ? nt[i] : 0) * 8 + g) * (size_t)K * 2 + 16 * t;
      eptr[i] = nullptr;
      hptr[i] = nullptr;
    }
  }

  // bf16x: the headers of the warp's tiles in registers, loaded once (before the PDL wait: weights only): lane l
  // holds steps l, l + 32, l + 64 (P <= 96, checked on the host); a step's header is a shuffle from lane ps % 32.
  uint32_t hreg[MT][3];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int ps = lane + 32 * j;
      hreg[i][j] = (FMT == kBF16X && valid[i] && ps < P) ? __ldg(hptr[i] + ps) : 0u;
    }
  auto hdr = [&](int i, int ps) -> uint32_t {
    const int j = ps >> 5;  // warp-uniform
    const uint32_t v = j == 0 ? hreg[i][0] : j == 1 ? hreg[i][1] : hreg[i][2];
    return __shfl_sync(0xffffffffu, v, ps & 31);
  };

  // L2 prefetch of batch b's weight lines (fire and forget; group 0 only): lanes 16i .. 16i+15 for tile i.
  auto prefetch = [&](int b) {
    if (b >= b1 || mg != 0) return;
    const int i = MT == 2 ? (lane >> 4) : 0, li = lane & 15;
    if (MT == 1 && lane >= 16) return;
    const bool vi = MT == 2 ? (i == 0 ? valid[0] : valid[MT - 1]) : valid[0];
    if (!vi) return;
    const int nti = MT == 2 ? (i == 0 ? nt[0] : nt[MT - 1]) : nt[0];
    const uint8_t* a = nullptr;
    if constexpr (FMT == kBF16X) {
      const size_t ti = (size_t)nti * P;
      if (li < 4) a = p.W + ti * 256 + (size_t)b * 512 + li * 128;
      else if (li < 6) a = p.W + T * 256 + ti * 128 + (size_t)b * 256 + (li - 4) * 128;
    } else {
      if (li < 8) a = p.W + (((size_t)nti * 8 + li) * (size_t)K + (size_t)b * 64) * 2;
    }
    if (a) asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
  };

  float acc[MT][MIW][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < MIW; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  uint4 wa[kBatch][MT], wb[kBatch][MT];
  uint32_t ha[kBatch][MT], hb[kBatch][MT];
#pragma unroll
  for (int u = 0; u < kBatch; ++u)
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      wa[u][i] = wb[u][i] = make_uint4(0, 0, 0, 0);
      ha[u][i] = hb[u][i] = 0;
    }

  auto load = [&](uint4 (&w)[kBatch][MT], uint32_t (&h)[kBatch][MT], int b) {
#pragma unroll
    for (int u = 0; u < kBatch; ++u) {
      const int ps = b * kBatch + u;
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        if (!valid[i]) continue;
        if constexpr (FMT == kBF16X) {
          const uint32_t hh = hdr(i, ps);
          h[u][i] = hh;
          w[u][i] = ldg_bf16x(hh, reinterpret_cast<const uint8_t*>(rawq + (uint64_t)hh * 512),
                              wptr[i] + (size_t)ps * 256, eptr[i] + (size_t)ps * 128);
        } else {
          w[u][i] = ldg_stream128(wptr[i] + (size_t)ps * 64);
        }
      }
    }
  };

  // activation stage of batch b -> buffer `buf`: rows < M copied, rows M .. Mpad-1 (the last m16 tile's padding)
  // zero-filled; 8 chunks of 16 bytes a row; chunk c of row r lands at chunk c ^ 4 when r is odd, so that the two
  // rows (16mi+g, g = 2j and 2j+1) one 8-lane phase of a 128-bit load reads fall into different bank halves.
  const int Mpad = (M + 15) & ~15;
  auto issue_x = [&](int b, int buf) {
    uint8_t* dst = smem + buf * STAGE;
    const __nv_bfloat16* srcp = xs + (size_t)b * (kBatch * kStepK);
    for (int c = threadIdx.x; c < Mpad * 8; c += NT) {
      const int r = c >> 3, ch = c & 7;
      if constexpr (SPARSE) {
        if (!(active_m16 & (1u << (r / 16)))) continue;
      }
      const bool live = r < M;
      cp_async16_zfill(dst + r * kRowB + ((ch ^ ((r & 1) << 2)) << 4), srcp + (size_t)(live ? r : 0) * p.ldx + ch * 8,
                       live);
    }
  };

  auto comp = [&](uint4 (&w)[kBatch][MT], const uint32_t (&h)[kBatch][MT], int buf) {
    const uint8_t* xb = smem + buf * STAGE;
#pragma unroll
    for (int u = 0; u < kBatch; ++u) {
      uint4 wd[MT];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        wd[i] = w[u][i];
        if constexpr (FMT == kBF16X) {
          if ((int)h[u][i] >= 0) wd[i] = bf16x_dec(wd[i].x, wd[i].y, wd[i].z, h[u][i]);
        }
      }
      const int off = ((u * 4 + t) ^ ((g & 1) << 2)) << 4;  // rows 16mi+g and 16mi+g+8 have g's parity
#pragma unroll
      for (int j = 0; j < MIW; ++j) {
        const int mi = mg * MIW + j;
        if (mi * 16 >= M) break;
        if constexpr (SPARSE) {
          if (!(warp_groups & (1u << mi))) continue;
        }
        const int r = mi * 16 + g;
        const uint4 a = *reinterpret_cast<const uint4*>(xb + r * kRowB + off);
        const uint4 a2 = *reinterpret_cast<const uint4*>(xb + (r + 8) * kRowB + off);
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          if constexpr (SPARSE) {
            if (!(groups[i] & (1u << mi))) continue;
          }
          mma_bf16(acc[i][j], a.x, a2.x, a.y, a2.y, wd[i].x, wd[i].y);
          mma_bf16(acc[i][j], a.z, a2.z, a.w, a2.w, wd[i].z, wd[i].w);
        }
      }
    }
  };

  load(wa, ha, b0);
  for (int d = 1; d <= p.pfd; ++d) prefetch(b0 + d);
  pdl_launch_dependents();
  pdl_wait();
  issue_x(b0, 0);
  cp_async_commit();
  if (b0 + 1 < b1) issue_x(b0 + 1, 1);
  cp_async_commit();
  int stage = 0;
  for (int b = b0; b < b1; b += 2) {
    if (b + 1 < b1) load(wb, hb, b + 1);
    if (p.pfd) prefetch(b + 1 + p.pfd);
    cp_async_wait<1>();
    __syncthreads();
    if (b + 2 < b1) issue_x(b + 2, stage == 0 ? 2 : stage - 1);
    cp_async_commit();
    comp(wa, ha, stage);
    stage = stage == 2 ? 0 : stage + 1;
    if (b + 1 >= b1) break;
    if (b + 2 < b1) load(wa, ha, b + 2);
    if (p.pfd) prefetch(b + 2 + p.pfd);
    cp_async_wait<1>();
    __syncthreads();
    if (b + 3 < b1) issue_x(b + 3, stage == 0 ? 2 : stage - 1);
    cp_async_commit();
    comp(wb, hb, stage);
    stage = stage == 2 ? 0 : stage + 1;
  }
  cp_async_wait<0>();

  // ---- epilogue. acc[i][mi]: c0, c1 = (row 16mi+g, columns 8nt+2t, +1), c2, c3 = (row 16mi+g+8, same columns)
  auto store2 = [&](int m, int n, float v0, float v1) {
    const size_t o = (size_t)(roff + m) * (size_t)p.ldo + n;
    if (p.f32) {
      *reinterpret_cast<float2*>(reinterpret_cast<float*>(p.out) + o) = make_float2(v0, v1);
    } else {
      *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(p.out) + o) = __floats2bfloat162_rn(v0, v1);
    }
  };
  if (S == 1) {
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      if (!valid[i]) continue;
      const int out_tile = SPARSE ? blockIdx.x * kTilesCta + wn * MT + i : nt[i];
      const int n = out_tile * 8 + 2 * t;
#pragma unroll
      for (int j = 0; j < MIW; ++j) {
        const int mi = mg * MIW + j;
        if constexpr (SPARSE) {
          if (!(groups[i] & (1u << mi))) continue;
        }
        const int r = mi * 16 + g;
        if (r < M) store2(r, n, acc[i][j][0], acc[i][j][1]);
        if (r + 8 < M) store2(r + 8, n, acc[i][j][2], acc[i][j][3]);
      }
    }
    return;
  }
  if constexpr (SPARSE) return;  // certified heads use S = 1

  // split-K: every slice writes its partials; the last CTA of the column block sums slices 0..S-1 in order
  float* wsl = p.ws + (size_t)s * M * N;
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    if (!valid[i]) continue;
    const int n = nt[i] * 8 + 2 * t;
#pragma unroll
    for (int j = 0; j < MIW; ++j) {
      const int r = (mg * MIW + j) * 16 + g;
      if (r < M) __stcg(reinterpret_cast<float2*>(wsl + (size_t)r * N + n), make_float2(acc[i][j][0], acc[i][j][1]));
      if (r + 8 < M)
        __stcg(reinterpret_cast<float2*>(wsl + (size_t)(r + 8) * N + n), make_float2(acc[i][j][2], acc[i][j][3]));
    }
  }
  // publication: every writer fences, the CTA barrier orders the writers before the leader's device-scope acq_rel
  // arrival (release: the CTA's partials; acquire, in the last CTA: every other CTA's), and the second barrier
  // orders the leader's acquire before the readers' loads.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) s_last = atomic_add_acq_rel_gpu(p.cnt + blockIdx.x, 1) == S - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    if (!valid[i]) continue;
    const int n = nt[i] * 8 + 2 * t;
#pragma unroll
    for (int jm = 0; jm < MIW; ++jm) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int r = (mg * MIW + jm) * 16 + g + 8 * hf;
        if (r >= M) continue;
        float2 v = __ldcg(reinterpret_cast<const float2*>(p.ws + (size_t)r * N + n));
        for (int j = 1; j < S; ++j) {
          const float2 q = __ldcg(reinterpret_cast<const float2*>(p.ws + ((size_t)j * M + r) * N + n));
          v.x += q.x;
          v.y += q.y;
        }
        store2(r, n, v.x, v.y);
      }
    }
  }
  if (threadIdx.x == 0) p.cnt[blockIdx.x] = 0;  // ready for the next call (stream order)
}

// coded (bf16x) -> natural [N][K] BF16, through the same reader as k_wide: the GPU check compares it with the
// layer's weight bit for bit. One warp per tile.
__global__ void k_decode(const uint8_t* __restrict__ X, int N, int K, __nv_bfloat16* __restrict__ out) {
  const int P = K / kStepK;
  const size_t T = (size_t)(N / 8) * P;
  const int lane = threadIdx.x & 31;
  const uint64_t rawq = reinterpret_cast<uint64_t>(X + bf16x_raw_off(T) + 16 * lane) - (1ull << 40);
  for (size_t tile = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5; tile < T;
       tile += ((size_t)gridDim.x * blockDim.x) >> 5) {
    const uint32_t h = reinterpret_cast<const uint32_t*>(X + T * 384)[tile];
    uint4 v = ldg_bf16x(h, reinterpret_cast<const uint8_t*>(rawq + (uint64_t)h * 512), X + tile * 256 + 8 * lane,
                        X + T * 256 + tile * 128 + 4 * lane);
    if ((int)h >= 0) v = bf16x_dec(v.x, v.y, v.z, h);
    const size_t n8 = tile / P, ps = tile % P;
    const size_t row = n8 * 8 + (lane >> 2), col = ps * kStepK + 8 * (lane & 3);
    *reinterpret_cast<uint4*>(out + row * K + col) = v;
  }
}

template <int FMT, int MI, int MT, int MG, bool SPARSE = false>
void launch_mi(const Params& p, int nblk, bool pdl, cudaStream_t st) {
  using C = Cfg<MI, MT, MG>;
  auto kern = k_wide<FMT, MI, MT, MG, SPARSE>;  // its shared-memory attribute is set by prepare() at load time
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(nblk, p.S);
  cfg.blockDim = dim3(C::NT);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, p));
}

// The m16-tile buckets: rows up to 16, 32, 48, 64, 96, 128, 192, 256.
constexpr int kBuckets[] = {1, 2, 3, 4, 6, 8, 12, 16};

int bucket_of(int M) {
  const int need = (M + 15) / 16;
  for (int b : kBuckets)
    if (b >= need) return b;
  return -1;
}

// The kernel of a bucket: mt 0 = default_mt; buckets 12 and 16 also exist with the other MT; buckets 6 and 8 also
// with two warp groups along M (mg 2).
int mt_of(int MI, int mt) { return MI >= 12 && (mt == 1 || mt == 2) ? mt : default_mt(MI); }
int mg_of(int MI, int mg) { return (MI == 6 || MI == 8) && mg == 2 ? 2 : 1; }
int key_of(int MI, int mt, int mg) { return MI * 16 + mt_of(MI, mt) * 4 + mg_of(MI, mg); }
constexpr int kkey(int MI, int MT, int MG) { return MI * 16 + MT * 4 + MG; }

#define WIDE_KERNELS(X, FMT) \
  X(FMT, 1, 2, 1) X(FMT, 2, 2, 1) X(FMT, 3, 2, 1) X(FMT, 4, 2, 1) X(FMT, 6, 2, 1) X(FMT, 6, 2, 2) X(FMT, 8, 2, 1) \
  X(FMT, 8, 2, 2) X(FMT, 12, 1, 1) X(FMT, 12, 2, 1) X(FMT, 16, 1, 1) X(FMT, 16, 2, 1)

template <int FMT>
void launch_fmt(const Params& p, int nblk, bool pdl, int mt, int mg, cudaStream_t st) {
  const int MI = bucket_of(p.M);
  TORCH_CHECK(MI > 0, "inkling_wide: M=", p.M, " not in [1, ", kMaxM, "]");
  const int key = key_of(MI, mt, mg);
#define WIDE_LAUNCH(F, MI_, MT_, MG_) \
  if (key == kkey(MI_, MT_, MG_)) return launch_mi<F, MI_, MT_, MG_>(p, nblk, pdl, st);
  WIDE_KERNELS(WIDE_LAUNCH, FMT)
#undef WIDE_LAUNCH
  TORCH_CHECK(false, "inkling_wide: no kernel for M=", p.M);
}

template <int FMT>
void prepare_fmt() {
  // cudaFuncSetAttribute for every instantiation at load time (outside any graph capture)
#define WIDE_ATTR(F, MI, MT, MG)                                                                    \
  C10_CUDA_CHECK(cudaFuncSetAttribute(k_wide<F, MI, MT, MG>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                      Cfg<MI, MT, MG>::SMEM));
  WIDE_KERNELS(WIDE_ATTR, FMT)
#undef WIDE_ATTR
}

// ---- bindings

void check_x(const torch::Tensor& x, int64_t M, int64_t K, const torch::Tensor& ref) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2, "inkling_wide: x must be CUDA bf16 [M, K]");
  TORCH_CHECK(x.device() == ref.device(), "inkling_wide: x on another device");
  TORCH_CHECK(x.size(0) >= M && x.size(1) == K && x.stride(1) == 1, "inkling_wide: x must be [>= M, K] with unit column stride");
  TORCH_CHECK(x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "inkling_wide: x rows must be 16-byte aligned");
}

// out[roff(n) + m, n] = sum_k x_src(n)[m, k] W[n, k] for m < M, every column n; see Params.
// fmt 0: W is the BF16 weight [N, K] (contiguous); fmt 2: W is the bf16x coded uint8 buffer of [N, K].
void gemm(const torch::Tensor& x0, const torch::Tensor& x1, const torch::Tensor& x2, int64_t M, const torch::Tensor& W,
          int64_t fmt, int64_t N, int64_t K, int64_t rb1, int64_t rb2, int64_t rb3, int64_t roff1, int64_t roff2,
          const torch::Tensor& out, int64_t S, const torch::Tensor& ws, const torch::Tensor& cnt, bool pdl, int64_t mt, int64_t mg,
          int64_t pfd) {
  TORCH_CHECK(M >= 1 && M <= kMaxM, "inkling_wide: M=", M, " not in [1, 256]");
  TORCH_CHECK(N % 8 == 0 && N > 0 && K % 64 == 0 && K > 0 && N <= INT32_MAX && K <= INT32_MAX,
              "inkling_wide: N=", N, " K=", K, " (N a multiple of 8, K of 64)");
  check_x(x0, M, K, W);
  check_x(x1, M, K, W);
  check_x(x2, M, K, W);
  TORCH_CHECK(x0.stride(0) == x1.stride(0) && x0.stride(0) == x2.stride(0), "inkling_wide: sources differ in row stride");
  TORCH_CHECK(W.is_cuda() && W.is_contiguous() && W.device() == x0.device(), "inkling_wide: bad weight");
  if (fmt == kRaw) {
    TORCH_CHECK(W.scalar_type() == at::kBFloat16 && W.dim() == 2 && W.size(0) == N && W.size(1) == K,
                "inkling_wide: raw weight must be bf16 [N, K]");
  } else {
    TORCH_CHECK(fmt == kBF16X && W.scalar_type() == at::kByte && W.dim() == 1, "inkling_wide: fmt ", fmt);
    const size_t T = (size_t)(N / 8) * (K / 32);
    TORCH_CHECK(T < (1ull << 31) && (size_t)W.numel() >= bf16x_raw_off(T), "inkling_wide: coded buffer too small");
    TORCH_CHECK(K <= 96 * 32, "inkling_wide: bf16x needs K <= 3072 (headers held in 3 registers a lane)");
  }
  TORCH_CHECK(reinterpret_cast<uintptr_t>(W.data_ptr()) % 16 == 0, "inkling_wide: weight not 16-byte aligned");
  const bool f32 = out.scalar_type() == at::kFloat;
  TORCH_CHECK(out.is_cuda() && out.device() == x0.device() && out.dim() == 2 &&
                  (f32 || out.scalar_type() == at::kBFloat16),
              "inkling_wide: out must be CUDA fp32 / bf16 [rows, N]");
  TORCH_CHECK(out.size(1) == N && out.stride(1) == 1 && out.stride(0) >= N && out.stride(0) % 2 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % (f32 ? 8 : 4) == 0,
              "inkling_wide: out rows must hold N contiguous, aligned values");
  TORCH_CHECK(roff1 >= 0 && roff2 >= 0 && out.size(0) >= M && roff1 <= out.size(0) - M && roff2 <= out.size(0) - M,
              "inkling_wide: out has ", out.size(0), " rows, row offsets ", roff1, ", ", roff2, " with M=", M);
  auto overlaps = [](const torch::Tensor& a, const torch::Tensor& b) {
    const auto a0 = reinterpret_cast<uintptr_t>(a.data_ptr()), b0 = reinterpret_cast<uintptr_t>(b.data_ptr());
    const auto a1 = a0 + (uintptr_t)((a.size(0) - 1) * a.stride(0) + a.size(1)) * a.element_size();
    const auto b1 = b0 + (uintptr_t)((b.size(0) - 1) * b.stride(0) + b.size(1)) * b.element_size();
    return a0 < b1 && b0 < a1;
  };
  TORCH_CHECK(!overlaps(out, x0) && !overlaps(out, x1) && !overlaps(out, x2), "inkling_wide: out overlaps x");
  auto bound_ok = [&](int64_t b) { return b >= N || (b >= 0 && b % kBN == 0); };
  TORCH_CHECK(bound_ok(rb1) && bound_ok(rb2) && bound_ok(rb3) && rb1 <= rb2 && rb2 <= rb3,
              "inkling_wide: column regions must be multiples of ", kBN, " (or >= N)");
  const int nblk = (int)((N + kBN - 1) / kBN);
  const int64_t NB = K / 64;
  TORCH_CHECK(S >= 1 && S <= NB && S <= 64, "inkling_wide: split-K S=", S, " not in [1, ", NB, "]");
  if (S > 1) {
    TORCH_CHECK(ws.is_cuda() && ws.device() == x0.device() && ws.scalar_type() == at::kFloat && ws.is_contiguous() &&
                    ws.numel() >= S * M * N && reinterpret_cast<uintptr_t>(ws.data_ptr()) % 16 == 0,
                "inkling_wide: split-K workspace too small, misaligned or on another device");
    TORCH_CHECK(cnt.is_cuda() && cnt.device() == x0.device() && cnt.scalar_type() == at::kInt && cnt.is_contiguous() &&
                    cnt.numel() >= nblk,
                "inkling_wide: split-K counters too few or on another device");
  }
  const c10::cuda::CUDAGuard guard(x0.device());
  Params p{};
  p.x0 = reinterpret_cast<const __nv_bfloat16*>(x0.data_ptr());
  p.x1 = reinterpret_cast<const __nv_bfloat16*>(x1.data_ptr());
  p.x2 = reinterpret_cast<const __nv_bfloat16*>(x2.data_ptr());
  p.ldx = x0.stride(0);
  p.M = (int)M;
  p.rb1 = (int)std::min<int64_t>(rb1, INT32_MAX);
  p.rb2 = (int)std::min<int64_t>(rb2, INT32_MAX);
  p.rb3 = (int)std::min<int64_t>(rb3, INT32_MAX);
  p.roff1 = roff1;
  p.roff2 = roff2;
  p.W = reinterpret_cast<const uint8_t*>(W.data_ptr());
  p.N = (int)N;
  p.K = (int)K;
  p.out = out.data_ptr();
  p.ldo = out.stride(0);
  p.f32 = f32 ? 1 : 0;
  p.S = (int)S;
  p.ws = S > 1 ? ws.data_ptr<float>() : nullptr;
  p.cnt = S > 1 ? cnt.data_ptr<int>() : nullptr;
  TORCH_CHECK(pfd >= 0 && pfd <= 64, "inkling_wide: prefetch distance ", pfd);
  p.pfd = (int)pfd;
  const auto st = at::cuda::getCurrentCUDAStream();
  if (fmt == kRaw)
    launch_fmt<kRaw>(p, nblk, pdl, (int)mt, (int)mg, st);
  else
    launch_fmt<kBF16X>(p, nblk, pdl, (int)mt, (int)mg, st);
}

// S=1 head only. Use exactly the same weight decoder, fragment order, and
// accumulation loop as the dense launch, with a compact column map.
#define CERT_WIDE_BUCKETS(X) \
  X(2, 2) X(3, 2) X(4, 2) X(6, 2) X(8, 2) X(12, 1)

void prepare_certwide(int64_t device) {
  const c10::cuda::CUDAGuard guard((c10::DeviceIndex)device);
#define CERT_ATTR(MI, MT) \
  C10_CUDA_CHECK(cudaFuncSetAttribute(k_wide<kBF16X, MI, MT, 1, true>, \
      cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<MI, MT, 1>::SMEM));
  CERT_WIDE_BUCKETS(CERT_ATTR)
#undef CERT_ATTR
}

// Both exact branches and the Triton reducer read this state after validation.
// The caller keeps the state and workspace exclusive through reduction.
__global__ void k_validate_certwide(int* state, const int* tiles, const int* groups,
                                    int slots, int ntiles, int live_tiles, unsigned row_groups) {
  constexpr int kInvalid = 4;
  __shared__ int count;
  __shared__ unsigned seen;
  if (threadIdx.x == 0) {
    const int reason = state[0], n = state[1];
    count = reason == 0 && n > 0 && n <= slots && n < ntiles ? n : 0;
    if ((!count && reason == 0) || (unsigned)reason > kInvalid) state[0] = kInvalid;
    seen = 0;
  }
  __syncthreads();
  if (!count) return;
  unsigned mask = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const unsigned group = groups[i];
    if ((unsigned)tiles[i] >= (unsigned)live_tiles || !group || (group & ~row_groups)) mask |= 1u << 31;
    mask |= group;
  }
  mask = __reduce_or_sync(0xffffffffu, mask);
  if ((threadIdx.x & 31) == 0) atomicOr(&seen, mask);
  __syncthreads();
  if (threadIdx.x == 0 && seen != row_groups) state[0] = kInvalid;
}

void certwide_exact(const torch::Tensor& x, const torch::Tensor& W,
                    const torch::Tensor& dense, const torch::Tensor& compact,
                    const torch::Tensor& state, const torch::Tensor& tiles,
                    const torch::Tensor& groups, int64_t N, int64_t v_real, int64_t mt, int64_t mg, int64_t pfd) {
  constexpr int K = 2304;
  TORCH_CHECK(x.dim() == 2, "inkling_certwide: x must be [M, K]");
  const int64_t M = x.size(0), slots = tiles.numel();
  TORCH_CHECK(M >= 17 && M <= 192 && N > 0 && N % 16 == 0 && N <= INT32_MAX && v_real > 0 && v_real <= N,
              "inkling_certwide: bad head shape");
  check_x(x, M, K, W);
  auto same = [&](const torch::Tensor& t, at::ScalarType dt) {
    return t.is_cuda() && t.device() == x.device() && t.is_contiguous() && t.scalar_type() == dt;
  };
  const size_t T = (size_t)(N / 8) * (K / 32);
  TORCH_CHECK(same(W, at::kByte) && W.dim() == 1 && T < (1ull << 31) &&
                (size_t)W.numel() >= bf16x_raw_off(T) && reinterpret_cast<uintptr_t>(W.data_ptr()) % 16 == 0,
              "inkling_certwide: bad packed weight");
  TORCH_CHECK(same(state, at::kInt) && state.numel() == 2 &&
                same(tiles, at::kInt) && same(groups, at::kInt) && groups.numel() == slots &&
                slots > 0 && slots <= N / 16 &&
                same(dense, at::kFloat) && dense.dim() == 2 && dense.size(0) == M && dense.size(1) == N &&
                same(compact, at::kFloat) && compact.dim() == 2 && compact.size(0) == M && compact.size(1) == slots * 16 &&
                reinterpret_cast<uintptr_t>(dense.data_ptr()) % 8 == 0 &&
                reinterpret_cast<uintptr_t>(compact.data_ptr()) % 8 == 0,
              "inkling_certwide: bad exact workspace");
  TORCH_CHECK(mt >= 0 && mt <= 2 && mg >= 1 && mg <= 2 && pfd >= 0 && pfd <= 64,
              "inkling_certwide: bad kernel settings");
  // Disjoint address ranges, even when the caller takes views of one storage.
  const torch::Tensor* buffers[] = {&x, &W, &dense, &compact, &state, &tiles, &groups};
  TORCH_CHECK(x.is_contiguous(), "inkling_certwide: x must be contiguous");
  for (int i = 0; i < 7; ++i) {
    const auto a = reinterpret_cast<uintptr_t>(buffers[i]->data_ptr());
    for (int j = 0; j < i; ++j) {
      const auto b = reinterpret_cast<uintptr_t>(buffers[j]->data_ptr());
      TORCH_CHECK(a + buffers[i]->nbytes() <= b || b + buffers[j]->nbytes() <= a,
                  "inkling_certwide: aliased exact buffers");
    }
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const auto st = at::cuda::getCurrentCUDAStream();
  k_validate_certwide<<<1, 256, 0, st>>>(state.data_ptr<int>(), tiles.data_ptr<int>(), groups.data_ptr<int>(),
                                       slots, N / 16, (v_real + 15) / 16, (1u << ((M + 15) / 16)) - 1u);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  Params p{};
  p.x0 = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.ldx = x.stride(0);
  p.M = (int)M;
  p.rb1 = p.rb2 = p.rb3 = INT32_MAX;
  p.W = W.data_ptr<uint8_t>();
  p.N = (int)N;
  p.K = K;
  p.f32 = p.S = 1;
  p.pfd = (int)pfd;
  p.cert_state = state.data_ptr<int>();
  p.out = dense.data_ptr<float>();
  p.ldo = N;
  launch_fmt<kBF16X>(p, (N + kBN - 1) / kBN, false, (int)mt, (int)mg, st);

  p.out = compact.data_ptr<float>();
  p.ldo = slots * 16;
  p.cert_tiles = tiles.data_ptr<int>();
  p.cert_groups = groups.data_ptr<int>();
  const int nblk = (p.ldo + kBN - 1) / kBN;
  const int mi = bucket_of(p.M);
#define CERT_LAUNCH(MI, MT) \
  if (mi == MI) return launch_mi<kBF16X, MI, MT, 1, true>(p, nblk, false, st);
  CERT_WIDE_BUCKETS(CERT_LAUNCH)
#undef CERT_LAUNCH
  TORCH_CHECK(false, "inkling_certwide: unsupported M=", M);
}
#undef CERT_WIDE_BUCKETS

torch::Tensor decode(const torch::Tensor& X, int64_t N, int64_t K) {
  TORCH_CHECK(X.is_cuda() && X.scalar_type() == at::kByte && X.is_contiguous(), "inkling_wide: bad coded buffer");
  TORCH_CHECK(N > 0 && K > 0 && N % 8 == 0 && K % 32 == 0 && N <= INT32_MAX && K <= INT32_MAX, "inkling_wide: N=", N,
              " K=", K);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(X.data_ptr()) % 16 == 0, "inkling_wide: coded buffer not 16-byte aligned");
  const size_t T = (size_t)(N / 8) * (K / 32);
  TORCH_CHECK(T < (1ull << 31) && (size_t)X.numel() >= bf16x_raw_off(T), "inkling_wide: coded buffer too small");
  const c10::cuda::CUDAGuard guard(X.device());
  auto out = torch::empty({N, K}, X.options().dtype(at::kBFloat16));
  k_decode<<<1024, 256, 0, at::cuda::getCurrentCUDAStream()>>>(X.data_ptr<uint8_t>(), (int)N, (int)K,
                                                               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void prepare(int64_t device) {
  const c10::cuda::CUDAGuard guard((c10::DeviceIndex)device);
  prepare_fmt<kRaw>();
  prepare_fmt<kBF16X>();
}

// (registers, local bytes, max threads) of the kernel at fmt / bucket, for the build log
std::tuple<int64_t, int64_t, int64_t> kernel_info(int64_t fmt, int64_t M, int64_t mt, int64_t mg) {
  cudaFuncAttributes a{};
  const int MI = bucket_of((int)M);
  TORCH_CHECK(MI > 0, "inkling_wide: M=", M);
  const int key = key_of(MI, (int)mt, (int)mg);
  const void* f = nullptr;
#define WIDE_INFO(F, MI_, MT_, MG_) \
  if (key == kkey(MI_, MT_, MG_)) f = (const void*)k_wide<F, MI_, MT_, MG_>;
  if (fmt == kRaw) {
    WIDE_KERNELS(WIDE_INFO, kRaw)
  } else {
    WIDE_KERNELS(WIDE_INFO, kBF16X)
  }
#undef WIDE_INFO
  TORCH_CHECK(f != nullptr, "inkling_wide: no kernel for M=", M, " mt=", mt, " mg=", mg);
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, f));
  return {a.numRegs, (int64_t)a.localSizeBytes, a.maxThreadsPerBlock};
}

}  // namespace wide

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "inkling_wide: weight-streaming BF16 GEMM for 17..256 rows on raw or pico-coded (bf16x) weights";
  m.def("gemm", &wide::gemm, "out[roff + m, n] = x_src[m] . W[n] (see wide_ext.cu)");
  m.def("decode", &wide::decode, "bf16x coded buffer -> natural [N, K] bf16 (the kernel's reader)");
  m.def("prepare_certwide", &wide::prepare_certwide, "prepare the sparse packed head kernels");
  m.def("certwide_exact", &wide::certwide_exact, "guarded dense or compact packed head with FP32 outputs");
  m.def("prepare", &wide::prepare, "set the kernels' shared-memory attribute on a device (load time)");
  m.def("bucket", &wide::bucket_of, "m16 tiles of the kernel taking M rows");
  m.def("kernel_info", &wide::kernel_info, "(regs, local bytes, max threads) of a kernel");
  m.attr("max_rows") = wide::kMaxM;
  m.attr("block_n") = wide::kBN;
}
