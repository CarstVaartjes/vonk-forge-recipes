// Reader of pico's exactly coded BF16 weight format (`bf16x`). The layout description, bf16x_raw_off, mad_imm,
// bf16x_dec and ldg_bf16x below are copied from pico (dsgx10 ds41f-4xgx10-switchless/engine/kernels/gemv16.cu,
// commit f2117916), MIT License, Copyright (c) 2026 Andy Shu; full licence text in third_party/pico/LICENSE.
// Only the reader is copied; the coding itself (bf16x_encode / bf16x_check) stays in inkling_pico12, whose
// packed buffers this package reads.
//
// pico's tiled layout of a BF16 weight W [N][K]: tile (n8, p) = rows 8*n8 .. 8*n8+7, values 32p .. 32p+31
// (64 bytes of each row); lane l of a warp owns row l/4, values 32p + 8(l%4) .. +7 (16 bytes). With
// T = N/8 * K/32 tiles, tile index i = n8 * (K/32) + p, the coded buffer holds:
//   [T][256 B]  sign and mantissa bytes (sign << 7 | mantissa): lane l's 8 values at 8l
//   [T][128 B]  exponent offsets: lane l's u32 at 4l, nibble j (bits 4j) value 2j, nibble 4 + j value 2j + 1
//   [T] u32     header: (b << 23) | (b << 7), b the tile's least exponent; raw tile r: 0x80000000 | r
//   [R][512 B]  the raw tiles as they were, from the first multiple of 256 bytes after the headers
// A raw tile's 512 bytes are the tile's 8 rows x 64 bytes, lane l's 16 bytes at 16l.
#pragma once
#include <cstddef>
#include <cstdint>

namespace wide {

__host__ __device__ constexpr size_t bf16x_raw_off(size_t T) { return (T * 388 + 255) & ~(size_t)255; }

// a * M + c as one multiply-add
template <uint32_t M>
__device__ __forceinline__ uint32_t mad_imm(uint32_t a, uint32_t c) {
  uint32_t d;
  asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "n"(M), "r"(c));
  return d;
}

// A lane's 8 values of a coded tile (s0, s1: their sign-mantissa bytes; e: their exponent offsets; hb: the
// tile's header) -> the raw tile's 16 bytes of that lane (4 words of bf16 pairs)
__device__ __forceinline__ uint4 bf16x_dec(uint32_t s0, uint32_t s1, uint32_t e, uint32_t hb) {
  const uint32_t e2 = e >> 8;
  uint4 o;
  o.x = (__byte_perm(s0, 0, 0x1100) & 0x807F807Fu) | mad_imm<128>(e & 0x000F000Fu, hb);
  o.y = (__byte_perm(s0, 0, 0x3322) & 0x807F807Fu) | mad_imm<8>(e & 0x00F000F0u, hb);
  o.z = (__byte_perm(s1, 0, 0x1100) & 0x807F807Fu) | mad_imm<128>(e2 & 0x000F000Fu, hb);
  o.w = (__byte_perm(s1, 0, 0x3322) & 0x807F807Fu) | mad_imm<8>(e2 & 0x00F000F0u, hb);
  return o;
}

// A lane's load of one tile by the tile's header h: a raw tile's 16 bytes at r, or a coded one's 8 at s and 4 at
// e (x, y; z). Predicated rather than branched (h is warp-uniform). Streamed past L1.
__device__ __forceinline__ uint4 ldg_bf16x(uint32_t h, const uint8_t* r, const uint8_t* s, const uint8_t* e) {
  uint4 v;
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.lt.s32 p, %4, 0;\n\t"
      "@p ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%5];\n\t"
      "@!p ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%6];\n\t"
      "@!p ld.global.nc.L1::no_allocate.u32 %2, [%7];\n\t}"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "r"(h), "l"(r), "l"(s), "l"(e));
  return v;
}

}  // namespace wide
