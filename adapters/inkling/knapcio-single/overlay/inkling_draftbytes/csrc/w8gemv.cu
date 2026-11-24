// inkling_draftbytes `w8gemv`: a weight-streaming GEMV for the MXFP8 draft-linear packs (draftmx8:mx) at 1-16 rows.
//
//   y[m, n] = bf16( sum_k x[m, k] * bf16(q[n, k] * 2^(s[n, k / 32] - 127)) (+ bias[n]) ),  FP32 accumulation
//
// q: FP8 e4m3 [N][K] row-major (the pack's own tensor, not re-laid), s: e8m0 [N][K/32], x: bf16 [M][K] rows of ldx,
// y: bf16 [M][N] rows of ldy. The per-output arithmetic is that of _triton._w8_mm_kernel (SCALE == 2) at <= 16
// rows: the dequantised weight is formed exactly in FP32 and rounded to BF16 (exact: FP8 times a power of two), every
// output is one chain of m16n8k16 BF16 MMAs with FP32 accumulation over k in ascending 16-wide chunks (Triton's
// MMAv2 dot chains its accumulator the same way across and within its K tiles), with Triton's operand order: its dot
// operands here are kWidth = 4 (the TTGIR of _w8_mm_kernel, Triton 3.8), so per 32-wide k tile at u = 32j + 4t the
// first MMA holds k (u, u+1) and (u+16, u+17), the second (u+2, u+3) and (u+18, u+19) (MMAv2.cpp's largeK order);
// then + bias in FP32 and one rounding to BF16. No split-K.
//
// Layout of the work: a warp owns MT n-tiles of 8 weight rows over the whole K; a CTA holds WPC warps. Per 64-byte
// step of a row a lane loads 16 contiguous bytes (lane group g = lane / 4: row n0 + g; t = lane % 4: chunk t, bytes
// [16t, 16t + 16)); four shuffles inside the lane quad (a 4x4 word transpose) give each lane bytes 4t .. 4t + 3 of
// every chunk. Weight loads run D steps ahead (a register ring), the activations (8-byte loads) XD steps ahead.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ uint4 ldg_stream128(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ uint32_t ldg_u16(const uint8_t* p) {
  unsigned short v;
  asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ uint2 ldg_u64(const __nv_bfloat16* p) {
  uint2 v;
  asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// two e4m3 bytes (bits 15:0 of u) -> bf16x2 of value * scale, formed in FP32 and rounded once (exact)
__device__ __forceinline__ uint32_t dq2(uint32_t u, float scale) {
  const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(u & 0xFFFFu), __NV_E4M3);
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
  const __nv_bfloat162 b = __floats2bfloat162_rn(f.x * scale, f.y * scale);
  return *reinterpret_cast<const uint32_t*>(&b);
}

// 2^(e - 127) exactly (e = 0: the FP32 subnormal 2^-127)
__device__ __forceinline__ float e8m0_scale(uint32_t e) {
  return __uint_as_float(e ? e << 23 : 0x00400000u);
}

template <int MT, int WPC, int D, int XD, bool HI>
__global__ void __launch_bounds__(32 * WPC) k_w8gemv(const __nv_bfloat16* __restrict__ x, int M, int ldx,
                                                     const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                                     const __nv_bfloat16* __restrict__ bias,
                                                     __nv_bfloat16* __restrict__ y, int ldy, int N, int K) {
  static_assert(D % XD == 0, "the activation ring slot of step s0 + j is j % XD (s0 a multiple of D)");
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int n_base = (blockIdx.x * WPC + (threadIdx.x >> 5)) * 8 * MT;
  if (n_base >= N) return;  // whole warp (N is a multiple of 8 * MT)
  const int steps = K >> 6;
  const int quad = lane & ~3;
  const bool live = g < M, live2 = HI && g + 8 < M;  // HI: a launch for 9..16 rows
  // lane's activations of chunk c of step s: x[row][64 s + 16 c + 4 t .. + 4) (Triton's kWidth-4 operand order)
  const __nv_bfloat16* xr = x + (size_t)(live ? g : 0) * ldx + 4 * t;
  const __nv_bfloat16* xr2 = x + (size_t)(live2 ? g + 8 : 0) * ldx + 4 * t;
  const uint8_t* wrow[MT];
  const uint8_t* srow[MT];
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    wrow[i] = W + (size_t)(n_base + 8 * i + g) * K + 16 * t;
    srow[i] = S + (size_t)(n_base + 8 * i + g) * (K >> 5);
  }
  float c[MT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i) c[i][0] = c[i][1] = c[i][2] = c[i][3] = 0.f;

  uint4 ring[D][MT];
  uint32_t sring[D][MT];
#pragma unroll
  for (int j = 0; j < D; ++j)
    if (j < steps)
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        ring[j][i] = ldg_stream128(wrow[i] + 64 * j);
        sring[j][i] = ldg_u16(srow[i] + 2 * j);
      }
  // activations XD steps ahead: xa[q][ch] (row g), xb[q][ch] (row g + 8, HI only) for the steps s with s % XD == q
  uint2 xa[XD][4], xb[HI ? XD : 1][4];
  auto load_x = [&](int q, int s) {
#pragma unroll
    for (int ch = 0; ch < 4; ++ch) {
      xa[q][ch] = live ? ldg_u64(xr + 64 * s + 16 * ch) : make_uint2(0u, 0u);
      if constexpr (HI) xb[q][ch] = live2 ? ldg_u64(xr2 + 64 * s + 16 * ch) : make_uint2(0u, 0u);
    }
  };
#pragma unroll
  for (int q = 0; q < XD; ++q)
    if (q < steps) load_x(q, q);

  for (int s0 = 0; s0 < steps; s0 += D) {
#pragma unroll
    for (int j = 0; j < D; ++j) {
      const int s = s0 + j;
      if (s >= steps) break;
      const int xq = j % XD;  // s0 is a multiple of D, hence of XD: the activation slot of s
      uint4 w[MT];
      uint32_t sc[MT];
#pragma unroll
      for (int i = 0; i < MT; ++i) w[i] = ring[j][i], sc[i] = sring[j][i];
      if (s + D < steps)
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          ring[j][i] = ldg_stream128(wrow[i] + 64 * (s + D));
          sring[j][i] = ldg_u16(srow[i] + 2 * (s + D));
        }
      uint2 xs[4], xs2[4];
#pragma unroll
      for (int ch = 0; ch < 4; ++ch) {
        xs[ch] = xa[xq][ch];
        if constexpr (HI) xs2[ch] = xb[xq][ch];
        else xs2[ch] = make_uint2(0u, 0u);
      }
      if (s + XD < steps) load_x(xq, s + XD);
      // the quad exchange (a 4x4 transpose of 32-bit words): lane t loaded chunk t (bytes [16t, 16t + 16) of the
      // step) and needs word t (bytes 4t .. 4t + 3) of every chunk; in round r it receives chunk (t + r) % 4's
      uint32_t v[MT][4];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int d = (t - r) & 3;  // the lane this lane's word goes to in round r
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          const uint32_t lo = (d & 1) ? w[i].y : w[i].x;
          const uint32_t hi = (d & 1) ? w[i].w : w[i].z;
          v[i][r] = __shfl_sync(0xffffffffu, (d & 2) ? hi : lo, quad | ((t + r) & 3));
        }
      }
      // Triton's kWidth-4 MMAv2 order, per 32-wide k tile T of the step (chunks 2T, 2T + 1; u = 32T + 4t): the first
      // MMA takes k (u, u+1) in slots (2t, 2t+1) and (u+16, u+17) in (2t+8, 2t+9), the second (u+2, u+3) and
      // (u+18, u+19); A and B alike. One MMA chain per output, tiles in ascending k.
#pragma unroll
      for (int T = 0; T < 2; ++T) {
        const int i0 = (2 * T - t) & 3, i1 = (2 * T + 1 - t) & 3;  // rounds in which chunks 2T, 2T + 1 arrived
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          const uint32_t q0 = (i0 & 2) ? ((i0 & 1) ? v[i][3] : v[i][2]) : ((i0 & 1) ? v[i][1] : v[i][0]);
          const uint32_t q1 = (i1 & 2) ? ((i1 & 1) ? v[i][3] : v[i][2]) : ((i1 & 1) ? v[i][1] : v[i][0]);
          const float scale = e8m0_scale((sc[i] >> (8 * T)) & 0xFFu);  // MX block 2s + T covers both chunks
          mma_bf16(c[i], xs[2 * T].x, xs2[2 * T].x, xs[2 * T + 1].x, xs2[2 * T + 1].x, dq2(q0, scale), dq2(q1, scale));
          mma_bf16(c[i], xs[2 * T].y, xs2[2 * T].y, xs[2 * T + 1].y, xs2[2 * T + 1].y, dq2(q0 >> 16, scale),
                   dq2(q1 >> 16, scale));
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    const int n = n_base + 8 * i + 2 * t;
    float b0 = 0.f, b1 = 0.f;
    const bool hb = bias != nullptr;
    if (hb) b0 = __bfloat162float(bias[n]), b1 = __bfloat162float(bias[n + 1]);
    if (live) {
      const __nv_bfloat162 o = __floats2bfloat162_rn(hb ? c[i][0] + b0 : c[i][0], hb ? c[i][1] + b1 : c[i][1]);
      *reinterpret_cast<__nv_bfloat162*>(y + (size_t)g * ldy + n) = o;
    }
    if (live2) {
      const __nv_bfloat162 o = __floats2bfloat162_rn(hb ? c[i][2] + b0 : c[i][2], hb ? c[i][3] + b1 : c[i][3]);
      *reinterpret_cast<__nv_bfloat162*>(y + (size_t)(g + 8) * ldy + n) = o;
    }
  }
}

template <int MT, int WPC, int D, int XD>
void launch(const torch::Tensor& x, const torch::Tensor& q, const torch::Tensor& s, const __nv_bfloat16* bias,
            const torch::Tensor& y, cudaStream_t st) {
  const int N = (int)q.size(0), K = (int)q.size(1);
  const int warps = N / (8 * MT);
  const int ctas = (warps + WPC - 1) / WPC;
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* qp = reinterpret_cast<const uint8_t*>(q.data_ptr());
  auto* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
  if (x.size(0) > 8)
    k_w8gemv<MT, WPC, D, XD, true><<<ctas, 32 * WPC, 0, st>>>(xp, (int)x.size(0), (int)x.stride(0), qp,
                                                              s.data_ptr<uint8_t>(), bias, yp, (int)y.stride(0), N, K);
  else
    k_w8gemv<MT, WPC, D, XD, false><<<ctas, 32 * WPC, 0, st>>>(xp, (int)x.size(0), (int)x.stride(0), qp,
                                                               s.data_ptr<uint8_t>(), bias, yp, (int)y.stride(0), N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// cfg = MT * 1000 + WPC * 100 + D * 10 + XD: the compiled variants (w8gemv.VARIANTS); each has a <= 8-row and a
// 9..16-row kernel
#define W8G_VARIANTS(X) \
  X(1, 1, 4, 2) X(1, 1, 4, 4) X(1, 1, 8, 2) X(1, 1, 8, 4) X(1, 2, 4, 2) X(1, 2, 4, 4) X(1, 2, 8, 2) X(1, 2, 8, 4) \
  X(1, 4, 4, 2) X(1, 4, 4, 4) X(1, 4, 8, 2) X(1, 4, 8, 4)

void w8gemv(const torch::Tensor& x, const torch::Tensor& q, const torch::Tensor& s, const c10::optional<torch::Tensor>& bias,
            const torch::Tensor& y, int64_t cfg) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1,
              "w8gemv: x must be CUDA bf16 [M, K] with contiguous rows");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 16, "w8gemv: rows ", x.size(0), " not in [1, 16]");
  TORCH_CHECK(q.is_cuda() && q.dim() == 2 && q.is_contiguous() && q.device() == x.device() &&
                  (q.scalar_type() == at::kFloat8_e4m3fn || q.scalar_type() == at::kByte),
              "w8gemv: q must be a contiguous CUDA [N, K] float8_e4m3fn (or its uint8 view) matrix");
  const int64_t N = q.size(0), K = q.size(1);
  TORCH_CHECK(N > 0 && N <= INT32_MAX / 2, "w8gemv: N=", N);
  TORCH_CHECK(x.stride(0) <= INT32_MAX / 32 && y.stride(0) <= INT32_MAX / 32, "w8gemv: row strides too large");
  TORCH_CHECK(x.size(1) == K && K > 0 && K % 64 == 0 && K <= INT32_MAX / 2, "w8gemv: K=", K, " (x has ", x.size(1), ")");
  TORCH_CHECK(x.size(0) == 1 || x.stride(0) >= K, "w8gemv: x rows overlap");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0 && (x.size(0) == 1 || x.stride(0) % 4 == 0),
              "w8gemv: x rows must be 8-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0, "w8gemv: q not 16-byte aligned");
  TORCH_CHECK(s.is_cuda() && s.scalar_type() == at::kByte && s.dim() == 2 && s.is_contiguous() && s.size(0) == N &&
                  s.size(1) == K / 32 && s.device() == x.device() && reinterpret_cast<uintptr_t>(s.data_ptr()) % 2 == 0,
              "w8gemv: s must be contiguous, 2-byte aligned uint8 [N, K / 32]");
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::kBFloat16 && y.dim() == 2 && y.size(0) == x.size(0) &&
                  y.size(1) == N && y.stride(1) == 1 && (y.size(0) == 1 || y.stride(0) >= N) &&
                  (y.size(0) == 1 || y.stride(0) % 2 == 0) && reinterpret_cast<uintptr_t>(y.data_ptr()) % 4 == 0 &&
                  y.device() == x.device(),
              "w8gemv: y must be CUDA bf16 [M, N] with contiguous, 4-byte aligned rows");
  const __nv_bfloat16* bp = nullptr;
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == at::kBFloat16 && bias->dim() == 1 && bias->size(0) == N &&
                    bias->is_contiguous() && bias->device() == x.device(),
                "w8gemv: bias must be contiguous CUDA bf16 [N]");
    bp = reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr());
  }
  const c10::cuda::CUDAGuard guard(x.device());
  const auto st = at::cuda::getCurrentCUDAStream();
#define W8G_CASE(MT_, WPC_, D_, XD_)                                                  \
  if (cfg == MT_ * 1000 + WPC_ * 100 + D_ * 10 + XD_) {                               \
    TORCH_CHECK(N % (8 * MT_) == 0, "w8gemv: N=", N, " not a multiple of ", 8 * MT_); \
    launch<MT_, WPC_, D_, XD_>(x, q, s, bp, y, st);                                   \
    return;                                                                           \
  }
  W8G_VARIANTS(W8G_CASE)
#undef W8G_CASE
  TORCH_CHECK(false, "w8gemv: no compiled variant cfg=", cfg);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "inkling_draftbytes w8gemv: MXFP8 weight-streaming GEMV (1-16 rows), Triton w8_mm's per-output arithmetic";
  m.def("w8gemv", &w8gemv, "y[M, N] = x[M, K] @ dequant(q, s)^T (+ bias), bf16 out");
}
