// inkling_pico12: PyTorch bindings for pico's packed 12-bit BF16 GEMV (third_party/pico, MIT, Copyright (c) 2026
// Andy Shu). This file is ours; it only calls pico's public API from kernels/dsk.h:
//   bf16x_encode / bf16x_check   exact 12-bit coding of a BF16 weight in pico's tiled layout, and its decode check
//   gemv_bf16x                   the GEMV on the coded weight (M <= 16 activation rows), fp32 or bf16 out
//   gemv_bf16                    the same GEMV on the raw tiled weight (for the bit-equality check only)
// plus one kernel of our own (k_tile_bf16) that re-lays a natural [N][K] BF16 weight into the tiled byte layout
// dsk.h documents for tile_fp8 with 2K bytes per row: [N/8][2K/64][8 rows][64 B].
//
// Every launch goes to PyTorch's current CUDA stream. gemv / gemv_raw do no allocation, no host sync and no
// host<->device copy, so they can be captured into CUDA graphs. tile / encode / check synchronise and allocate:
// load time only.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>

#include "kernels/dsk.h"

namespace {

// natural [N][K] bf16 (rowb = 2K bytes a row) -> [N/8][rowb/64][8][64 B]; 16 bytes a thread
__global__ void k_tile_bf16(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int64_t N, int64_t rowb) {
  const int64_t pieces = N * rowb / 16;
  for (int64_t c = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; c < pieces; c += (int64_t)gridDim.x * blockDim.x) {
    const int64_t o = c * 16, n = o / rowb, b = o % rowb;
    const int64_t t = (n / 8) * 8 * rowb + (b / 64) * 512 + (n % 8) * 64 + b % 64;
    *reinterpret_cast<uint4*>(dst + t) = *reinterpret_cast<const uint4*>(src + o);
  }
}

void check_weight(const torch::Tensor& w) {
  TORCH_CHECK(w.is_cuda(), "pico12: weight must be a CUDA tensor");
  TORCH_CHECK(w.scalar_type() == at::kBFloat16, "pico12: weight must be bfloat16");
  TORCH_CHECK(w.dim() == 2 && w.is_contiguous(), "pico12: weight must be a contiguous [N, K] matrix");
  TORCH_CHECK(w.size(0) % 8 == 0 && w.size(1) % 32 == 0, "pico12: weight [", w.size(0), ", ", w.size(1),
              "]: N must be a multiple of 8 and K of 32");
  TORCH_CHECK(w.size(0) <= INT32_MAX && w.size(1) <= INT32_MAX, "pico12: weight too large");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "pico12: weight not 16-byte aligned");
}

// [N, K] bf16 -> the tiled byte layout, uint8 [N * K * 2]
torch::Tensor tile(const torch::Tensor& w) {
  check_weight(w);
  const c10::cuda::CUDAGuard guard(w.device());
  const int64_t N = w.size(0), rowb = 2 * w.size(1);
  auto out = torch::empty({N * rowb}, w.options().dtype(at::kByte));
  const auto st = at::cuda::getCurrentCUDAStream();
  k_tile_bf16<<<2048, 256, 0, st>>>(reinterpret_cast<const uint8_t*>(w.data_ptr()), out.data_ptr<uint8_t>(), N, rowb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void check_tiled(const torch::Tensor& t, int64_t N, int64_t K) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kByte && t.dim() == 1 && t.is_contiguous(),
              "pico12: tiled weight must be a contiguous CUDA uint8 vector");
  TORCH_CHECK(t.numel() == N * K * 2, "pico12: tiled weight has ", t.numel(), " bytes, expected ", N * K * 2);
}

// Coded copy of w: (packed uint8 tensor, raw tiles, total tiles). pico's bf16x_encode allocates the coded buffer
// with cudaMalloc; the returned tensor owns it (cudaFree on release). Fails if the coded copy does not decode
// back to w bit for bit (bf16x_check).
std::tuple<torch::Tensor, int64_t, int64_t> encode(const torch::Tensor& w) {
  check_weight(w);
  const c10::cuda::CUDAGuard guard(w.device());
  const int64_t N = w.size(0), K = w.size(1);
  torch::Tensor tiled = tile(w);
  const auto st = at::cuda::getCurrentCUDAStream();
  size_t bytes = 0;
  int raw = 0;
  uint8_t* X = dsk::bf16x_encode(tiled.data_ptr<uint8_t>(), (int)N, (int)K, &bytes, &raw, st);
  auto packed = torch::from_blob(
      X, {(int64_t)bytes}, [](void* p) { cudaFree(p); }, w.options().dtype(at::kByte));
  const uint64_t bad = dsk::bf16x_check(X, tiled.data_ptr<uint8_t>(), (int)N, (int)K, st);
  TORCH_CHECK(bad == 0, "pico12: coded weight [", N, ", ", K, "] does not decode exactly (", bad, " pieces differ)");
  return {packed, (int64_t)raw, (N / 8) * (2 * K / 64)};
}

// Count of 16-byte pieces of the tiled weight that the coded one does not decode to (0: exact).
int64_t check(const torch::Tensor& packed, const torch::Tensor& tiled, int64_t N, int64_t K) {
  check_tiled(tiled, N, K);
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == at::kByte && packed.is_contiguous(), "pico12: bad packed");
  const c10::cuda::CUDAGuard guard(packed.device());
  return (int64_t)dsk::bf16x_check(packed.data_ptr<uint8_t>(), tiled.data_ptr<uint8_t>(), (int)N, (int)K,
                                   at::cuda::getCurrentCUDAStream());
}

void check_call(const torch::Tensor& x, const torch::Tensor& w, int64_t N, int64_t K, const torch::Tensor& out,
                int64_t mt, int64_t wk) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2, "pico12: x must be CUDA bf16 [M, K]");
  TORCH_CHECK(x.size(1) == K && x.stride(1) == 1 && (x.size(0) == 1 || x.stride(0) == K),
              "pico12: x must have rows of K contiguous values");
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= dsk::kMaxRows, "pico12: rows ", x.size(0), " not in [1, 16]");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "pico12: x not 16-byte aligned");
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kByte && w.device() == x.device(), "pico12: bad weight buffer");
  TORCH_CHECK(out.is_cuda() && out.device() == x.device() && out.dim() == 2, "pico12: out must be CUDA [M, N]");
  TORCH_CHECK(out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16, "pico12: out fp32 or bf16");
  TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == N && out.stride(1) == 1 &&
                  (out.size(0) == 1 || out.stride(0) == N),
              "pico12: out must be [M, N] with rows of N contiguous values");
  TORCH_CHECK((mt == 1 || mt == 2) && N % (8 * mt) == 0, "pico12: N=", N, " mt=", mt);
  TORCH_CHECK(K % 32 == 0 && wk >= 1 && wk <= 16 && wk <= K / 32, "pico12: K=", K, " wk=", wk);
}

dsk::GemvCfg cfg_of(int64_t mt, int64_t wk, bool pdl) {
  dsk::GemvCfg c{};
  c.mt = (int)mt;
  c.wn = 1;
  c.wk = (int)wk;
  c.pdl = pdl;
  c.ring = -1;  // the one-tile-per-CTA kernel (what gemv_bf16x always runs)
  return c;
}

// out[M, N] = x[M, K] W^T, W coded (encode); fp32 math, out fp32 or rounded to bf16.
void gemv(const torch::Tensor& x, const torch::Tensor& packed, int64_t N, int64_t K, const torch::Tensor& out, int64_t mt,
          int64_t wk, bool pdl) {
  check_call(x, packed, N, K, out, mt, wk);
  const c10::cuda::CUDAGuard guard(x.device());
  const bool f32 = out.scalar_type() == at::kFloat;
  dsk::gemv_bf16x(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)x.size(0), packed.data_ptr<uint8_t>(),
                  (int)N, (int)K, f32 ? out.data_ptr<float>() : nullptr,
                  f32 ? nullptr : reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), cfg_of(mt, wk, pdl),
                  at::cuda::getCurrentCUDAStream());
}

// The same on the raw tiled weight (tile): pico's gemv_bf16, whose bits gemv_bf16x reproduces.
void gemv_raw(const torch::Tensor& x, const torch::Tensor& tiled, int64_t N, int64_t K, const torch::Tensor& out, int64_t mt,
              int64_t wk, bool pdl) {
  check_call(x, tiled, N, K, out, mt, wk);
  check_tiled(tiled, N, K);
  const c10::cuda::CUDAGuard guard(x.device());
  const bool f32 = out.scalar_type() == at::kFloat;
  dsk::gemv_bf16(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)x.size(0), tiled.data_ptr<uint8_t>(),
                 (int)N, (int)K, f32 ? out.data_ptr<float>() : nullptr,
                 f32 ? nullptr : reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), cfg_of(mt, wk, pdl), false,
                 at::cuda::getCurrentCUDAStream());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "inkling_pico12: pico's packed 12-bit BF16 GEMV (pico by Andy Shu, MIT)";
  m.def("tile", &tile, "natural [N, K] bf16 -> pico's tiled byte layout");
  m.def("encode", &encode, "bf16 [N, K] -> (coded uint8 buffer, raw tiles, tiles); verified exact");
  m.def("check", &check, "pieces of the tiled weight the coded one does not decode to");
  m.def("gemv", &gemv, "out[M, N] = x[M, K] @ W^T on the coded weight (M <= 16)");
  m.def("gemv_raw", &gemv_raw, "out[M, N] = x[M, K] @ W^T on the raw tiled weight (M <= 16)");
  m.attr("max_rows") = dsk::kMaxRows;
}
