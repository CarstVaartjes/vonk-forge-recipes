// inkling_pico12 `stack1`: pico's coded GEMV (gemv_bf16x) restricted to a range of the weight's output rows, so
// that the stacked q/k/v projection reads every coded byte of qkvr once: the q rows over the columns outside the k/v
// rows, then the k and v rows over the k/v rows only, both straight from the one coded copy of the whole matrix.
//
// This file is ours. It includes pico's gemv16.cu (third_party/pico, MIT, Copyright (c) 2026 Andy Shu) verbatim, in
// its own extension (inkling_pico12_range_ext, built only when `stack1` is on), to reach gemv16_body, the kernel body
// template every pico GEMV kernel runs. The kernels below are k_gemv16 / k_gemv16_r16 (the kernels gemv_bf16x
// launches) with the same template arguments, launch bounds, block, shared memory and PDL launch, plus one test in
// front: a CTA whose 8 * MT output rows are not in the asked range returns at once (it reads no weight and writes no
// output). A CTA that runs executes gemv16_body exactly as gemv_bf16x's kernel does, so each output it writes has the
// bits of gemv_bf16x's output for that (row, column).
//
// Range semantics (rows_cta = 8 * MT output rows per CTA; lo and hi multiples of rows_cta, 0 <= lo < hi <= N):
//   inside = 1: CTAs [lo / rows_cta, hi / rows_cta) run; the grid is hi / rows_cta CTAs.
//   inside = 0: every CTA but those runs; the grid is N / rows_cta CTAs (N a multiple of rows_cta).
// y is [M][N] bf16 with rows of N contiguous values (the full width: the kernel indexes it as gemv_bf16x does);
// outputs of skipped CTAs are left as they were.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/gemv16.cu"

namespace dsk {
namespace {

__device__ __forceinline__ bool range_runs(int c_lo, int c_hi, int inside) {
  const int b = (int)blockIdx.x;
  return (b >= c_lo && b < c_hi) == (inside != 0);
}

// up to 8 rows: k_gemv16<kBF16X, MT, false, false>'s body
template <int MT>
__global__ void __launch_bounds__(512) k_gemv16x_range(int c_lo, int c_hi, int inside,
                                                        const __nv_bfloat16* __restrict__ x, int M,
                                                        const uint8_t* __restrict__ W, int N, int K,
                                                        __nv_bfloat16* __restrict__ yb, int wk) {
  if (!range_runs(c_lo, c_hi, inside)) return;
  gemv16_body<kBF16X, MT, false, false, kLowRows>(x, M, W, nullptr, N, K, N, nullptr, yb, nullptr, nullptr, 1, wk,
                                                  NormIn{});
}

// 9..16 rows: k_gemv16_r16<kBF16X, MT, false>'s body, with its (512, 1) launch bounds
template <int MT>
__global__ void __launch_bounds__(512, 1) k_gemv16x_range_r16(int c_lo, int c_hi, int inside,
                                                               const __nv_bfloat16* __restrict__ x, int M,
                                                               const uint8_t* __restrict__ W, int N, int K,
                                                               __nv_bfloat16* __restrict__ yb, int wk) {
  if (!range_runs(c_lo, c_hi, inside)) return;
  gemv16_body<kBF16X, MT, false, false, kMaxRows>(x, M, W, nullptr, N, K, N, nullptr, yb, nullptr, nullptr, 1, wk,
                                                  NormIn{});
}

// The whole stacked call in one launch (M <= 8 rows a source; mt 1; grid N / 8): CTAs in [c_lo, c_hi) run the k and
// v rows (x + M K, 2M rows: k_gemv16_r16's body from 9 rows, k_gemv16's up to 8) into y + M N, the others the q rows
// (x, M rows: k_gemv16's body) into y. Each CTA's arithmetic is that of the separate range calls.
__global__ void __launch_bounds__(512, 1) k_gemv16x_stacked(int c_lo, int c_hi, const __nv_bfloat16* __restrict__ x,
                                                             int M, const uint8_t* __restrict__ W, int N, int K,
                                                             __nv_bfloat16* __restrict__ yb, int wk) {
  const int b = (int)blockIdx.x;
  if (b >= c_lo && b < c_hi) {
    if (2 * M > kLowRows)
      gemv16_body<kBF16X, 1, false, false, kMaxRows>(x + (size_t)M * K, 2 * M, W, nullptr, N, K, N, nullptr,
                                                     yb + (size_t)M * N, nullptr, nullptr, 1, wk, NormIn{});
    else
      gemv16_body<kBF16X, 1, false, false, kLowRows>(x + (size_t)M * K, 2 * M, W, nullptr, N, K, N, nullptr,
                                                     yb + (size_t)M * N, nullptr, nullptr, 1, wk, NormIn{});
  } else {
    gemv16_body<kBF16X, 1, false, false, kLowRows>(x, M, W, nullptr, N, K, N, nullptr, yb, nullptr, nullptr, 1, wk,
                                                   NormIn{});
  }
}

template <int MT>
void launch_range(const __nv_bfloat16* x, int M, const uint8_t* W, int N, int K, __nv_bfloat16* yb, int wk, bool pdl,
                  int lo, int hi, bool inside, cudaStream_t st) {
  constexpr int rows_cta = 8 * MT;
  const int c_lo = lo / rows_cta, c_hi = hi / rows_cta;
  const int grid = inside ? c_hi : N / rows_cta;
  const int rb = M > kLowRows ? kMaxRows : kLowRows;
  const size_t smem = (size_t)wk * rb * rows_cta * sizeof(float);  // launch16's (no NORM): <= 16 KB
  if (M > kLowRows)
    launch_pdl(k_gemv16x_range_r16<MT>, dim3(grid), dim3(32 * wk), smem, st, pdl, c_lo, c_hi, (int)inside, x, M, W, N,
               K, yb, wk);
  else
    launch_pdl(k_gemv16x_range<MT>, dim3(grid), dim3(32 * wk), smem, st, pdl, c_lo, c_hi, (int)inside, x, M, W, N, K,
               yb, wk);
}

}  // namespace
}  // namespace dsk

namespace {

bool overlaps(const torch::Tensor& a, const torch::Tensor& b) {
  const auto lo_a = reinterpret_cast<uintptr_t>(a.data_ptr()), lo_b = reinterpret_cast<uintptr_t>(b.data_ptr());
  const auto span = [](const torch::Tensor& t) {
    return (uintptr_t)(((t.size(0) - 1) * t.stride(0) + t.size(1)) * t.element_size());
  };
  return lo_a < lo_b + span(b) && lo_b < lo_a + span(a);
}

// raw_tiles: the pack's raw tile count (encode's second result), so that the buffer is known to hold every raw tile
// a header can point to
void check_common(const torch::Tensor& x, const torch::Tensor& packed, int64_t N, int64_t K, const torch::Tensor& out,
                  int64_t wk, int64_t raw_tiles) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2, "pico12 range: x must be CUDA bf16 [M, K]");
  TORCH_CHECK(x.size(1) == K && x.stride(1) == 1 && (x.size(0) == 1 || x.stride(0) == K),
              "pico12 range: x must have rows of K contiguous values");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "pico12 range: x not 16-byte aligned");
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == at::kByte && packed.is_contiguous() &&
                  packed.device() == x.device(),
              "pico12 range: bad weight buffer");
  TORCH_CHECK(out.is_cuda() && out.device() == x.device() && out.dim() == 2 && out.scalar_type() == at::kBFloat16,
              "pico12 range: out must be CUDA bf16 [M, N]");
  TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == N && out.stride(1) == 1 &&
                  (out.size(0) == 1 || out.stride(0) == N),
              "pico12 range: out must be [M, N] with rows of N contiguous values");
  TORCH_CHECK(N > 0 && N % 8 == 0 && N <= INT32_MAX && K > 0 && K <= INT32_MAX / 2, "pico12 range: N=", N, " K=", K);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(packed.data_ptr()) % 16 == 0, "pico12 range: coded buffer not 16-byte aligned");
  TORCH_CHECK(!overlaps(x, out), "pico12 range: out overlaps x");
  TORCH_CHECK(K % 32 == 0 && wk >= 1 && wk <= 16 && wk <= K / 32, "pico12 range: K=", K, " wk=", wk);
  const int64_t tiles = (N / 8) * (2 * K / 64);
  TORCH_CHECK(tiles < (int64_t(1) << 31), "pico12 range: too many tiles");
  TORCH_CHECK(raw_tiles >= 0 && raw_tiles <= tiles &&
                  packed.numel() >= (int64_t)dsk::bf16x_raw_off((size_t)tiles) + raw_tiles * 512,
              "pico12 range: coded buffer smaller than the [", N, ", ", K, "] layout with ", raw_tiles, " raw tiles");
}

// out[M, N] (bf16, rows of N contiguous values) for the output columns in [lo, hi) (inside) or outside it,
// x[M, K] bf16 (rows contiguous), packed: the coded weight (inkling_pico12's encode) of an [N, K] matrix.
void gemv_range(const torch::Tensor& x, const torch::Tensor& packed, int64_t N, int64_t K, const torch::Tensor& out,
                int64_t raw_tiles, int64_t mt, int64_t wk, bool pdl, int64_t lo, int64_t hi, bool inside) {
  check_common(x, packed, N, K, out, wk, raw_tiles);
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= dsk::kMaxRows, "pico12 range: rows ", x.size(0), " not in [1, 16]");
  TORCH_CHECK(mt == 1 || mt == 2, "pico12 range: mt=", mt, " (1 or 2)");
  const int64_t rows_cta = 8 * mt;
  TORCH_CHECK(0 <= lo && lo < hi && hi <= N && lo % rows_cta == 0 && hi % rows_cta == 0,
              "pico12 range: [", lo, ", ", hi, ") not a range of whole ", rows_cta, "-row CTAs in [0, ", N, ")");
  TORCH_CHECK(inside || N % rows_cta == 0, "pico12 range: outside a range needs N=", N, " a multiple of ", rows_cta);
  const c10::cuda::CUDAGuard guard(x.device());
  const auto st = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* yp = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  if (mt == 1)
    dsk::launch_range<1>(xp, (int)x.size(0), packed.data_ptr<uint8_t>(), (int)N, (int)K, yp, (int)wk, pdl, (int)lo,
                         (int)hi, inside, st);
  else
    dsk::launch_range<2>(xp, (int)x.size(0), packed.data_ptr<uint8_t>(), (int)N, (int)K, yp, (int)wk, pdl, (int)lo,
                         (int)hi, inside, st);
}

// The stacked q/k/v call in one launch: x [3M, K] (q rows, then k rows, then v rows), out [3M, N]; M <= 8.
// out[:M] gets the columns outside [lo, hi), out[M:] the columns inside it (lo, hi multiples of 8).
void gemv_stacked(const torch::Tensor& x, const torch::Tensor& packed, int64_t N, int64_t K, const torch::Tensor& out,
                  int64_t raw_tiles, int64_t M, int64_t wk, bool pdl, int64_t lo, int64_t hi) {
  check_common(x, packed, N, K, out, wk, raw_tiles);
  TORCH_CHECK(M >= 1 && M <= dsk::kLowRows && x.size(0) == 3 * M, "pico12 stacked: x must be [3M, K] with M in [1, 8]");
  TORCH_CHECK(0 <= lo && lo < hi && hi <= N && lo % 8 == 0 && hi % 8 == 0, "pico12 stacked: bad range [", lo, ", ", hi,
              ")");
  const c10::cuda::CUDAGuard guard(x.device());
  const int rb = 2 * M > dsk::kLowRows ? dsk::kMaxRows : dsk::kLowRows;
  const size_t smem = (size_t)wk * rb * 8 * sizeof(float);  // the larger of the two bodies' (mt 1): <= 8 KB
  dsk::launch_pdl(dsk::k_gemv16x_stacked, dim3((unsigned)(N / 8)), dim3((unsigned)(32 * wk)), smem,
                  at::cuda::getCurrentCUDAStream(), pdl, (int)(lo / 8), (int)(hi / 8),
                  reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)M, packed.data_ptr<uint8_t>(), (int)N,
                  (int)K, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (int)wk);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "inkling_pico12 stack1: pico's coded GEMV over a range of output rows (pico by Andy Shu, MIT)";
  m.def("gemv_range", &gemv_range, "out[:, cols] = x @ W[cols]^T for the columns in (or outside) [lo, hi), coded W");
  m.def("gemv_stacked", &gemv_stacked, "the stacked q/k/v call (M <= 8 a source) in one launch, coded W");
  m.attr("max_rows") = dsk::kMaxRows;
}
