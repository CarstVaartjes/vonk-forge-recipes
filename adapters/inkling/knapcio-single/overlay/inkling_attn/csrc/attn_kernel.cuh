// inkling_attn: prefill/extend attention for Inkling-2-Mini on GB10 (sm_121) with the relative-position bias
// computed inside the kernel.
//
// Math (the same as SGLang's FA4 path with rel_bias, flash_attn/cute/interface.py + fa4_sm120/flash_fwd.py):
//   for request b, query row i (0 <= i < sq), absolute position p = i + (sk - sq), key j (0 <= j < sk):
//     visible   j <= p, and for sliding-window layers also j >= p - window_left
//     d       = p - j                       (relative distance)
//     bias    = bf16( sum_e r[t, h, e] * proj[e, d] )  if d < rel_extent, else 0   (FP32 dot, BF16 round:
//               the stock cuBLAS einsum thd,de->the writes a BF16 tensor)
//     s       = qk * softmax_scale + bias   (FP32)
//     out     = softmax_j(s) @ V   (P rounded to BF16 before the PV MMA, as FA4 does), FP32 [T, H, 128]
//     lse     = ln(sum_j exp(s))   FP32 [H, T]  (-inf when a row sees no key)
//
// Instead of reading a materialised [T, H, rel_extent] tensor (2 bytes per score from DRAM), each warp
// computes the bias of its 16 query rows x 64 keys tile from r (a 16 x 16 A fragment held in registers) and
// proj (10 B fragments, L1-cached loads) with 10 extra m16n8k16 MMAs, stages the 16 x 80 BF16 result in shared
// memory and reads it back sheared (element (a, b) of the score tile is T[a][a + 63 - b]).
//
// FA2-style: 4 warps x 16 query rows (BM = 64), BN = 64 keys a tile, Q fragments in registers, K and V single
// buffered with cp.async (the K load of the next tile overlaps PV, the V load overlaps QK + softmax), ldmatrix
// on 128-byte XOR-swizzled rows, online softmax in FP32. Paged KV (page size a multiple of 64), GQA by head index.
#pragma once

#include <cuda_bf16.h>
#include <cstdint>

namespace inkling_attn {

constexpr int kD = 128;          // head dim
constexpr int kBN = 64;          // keys a tile
constexpr int kDRel = 16;        // r / proj inner dim
constexpr int kTCols = 16 + kBN; // staged bias columns a warp tile (79 used, +1 for the odd-row shift)
constexpr int kTStride = 88;     // BF16 elements a staged row (bank-conflict check: <= 2-way)
constexpr float kLog2e = 1.4426950408889634f;
constexpr int tstage_bytes(int warps) { return warps * 16 * kTStride * 2; }

struct Params {
  const __nv_bfloat16* q;  // [T, H, 128]
  long long q_st, q_sh;
  const __nv_bfloat16* k;  // [pages, page_size, H_kv, 128]
  long long k_sp, k_ss, k_sh;
  const __nv_bfloat16* v;
  long long v_sp, v_ss, v_sh;
  // Packed planes share the byte stride of the allocator's page envelope.
  const uint8_t *sm, *ex;
  const int *esc, *off, *overflow_ref, *flags;
  const uint16_t *staging, *overflow;
  long long packed_stride;
  int kv_heads, stage_slots, overflow_slots;
  const int* page_table;   // [B, max_pages]
  long long pt_s;
  const int* cu_q;         // [n_req + 1]
  const int* seq_k;        // [>= n_req]
  int n_req;
  const __nv_bfloat16* r;  // [T, H, 16]
  long long r_st, r_sh;
  const __nv_bfloat16* proj;  // [16, rel_extent] contiguous
  const __nv_bfloat16* projp; // optional packed copy [rel_extent + 1, 16] (pack_proj), row rel_extent = 0
  int rel_extent;
  float* out;              // [T, H, 128]
  long long o_st, o_sh;
  float* lse;              // lse[h * lse_sh + t]
  long long lse_sh;
  int heads, group, page_size;
  int total_q;             // rows of q (guards cu_seqlens_q)
  int pt_cols;             // page_table columns (guards seqused_k)
  int num_pages;           // pages of the KV cache (guards page ids)
  float scale;
  int window_left;         // sliding-window layers only
  // request selection (global layers, two launches of different warp counts over one call, see ops.W8_FROM):
  // 0 = every request; 1 = requests whose prefix (seqused_k - new rows) is < sel_keys; 2 = prefix >= sel_keys.
  int sel_mode, sel_keys;
  // gated epilogue (kGate, sliding-window BF16 path): y[t, h*128 + d] = bf16(x * sigmoid(gate) * sigmoid(lse - sink))
  // with x = O / l, exactly SGLang's gate_sink kernel (inkling2/gate_sink.cuh) on the FP32 out and LSE this kernel
  // would have written; out is then not written (lse still is).
  const __nv_bfloat16* gate;  // [T, H * 128], row stride g_st (16-byte aligned rows: staged with 16-byte loads)
  long long g_st;
  const __nv_bfloat16* sink;  // [T, H], row stride s_st
  long long s_st;
  __nv_bfloat16* y;           // [T, H * 128], row stride y_st
  long long y_st;
};

namespace dev {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
  const int n = valid ? 16 : 0;  // 0: zero-fill, nothing read
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr)
               : "memory");
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr)
               : "memory");
}

// D += A * B, m16n8k16, BF16 inputs, FP32 accumulate
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ uint32_t ld_u32(const __nv_bfloat16* p) {
  return __ldg(reinterpret_cast<const unsigned int*>(p));
}

__device__ __forceinline__ uint32_t ld_u16(const __nv_bfloat16* p) {
  return static_cast<uint32_t>(__ldg(reinterpret_cast<const unsigned short*>(p)));
}

}  // namespace dev

// proj [16, E] -> projp [E + 1, 16]: row d = {P(0,d),P(1,d),P(8,d),P(9,d), P(2,d),P(3,d),P(10,d),P(11,d), ...},
// i.e. lane qd's B-fragment words {P(2qd,d),P(2qd+1,d)} and {P(2qd+8,d),P(2qd+9,d)} are 8 contiguous bytes;
// row E is zero (distances beyond the extent).
__global__ void pack_proj(const __nv_bfloat16* __restrict__ proj, __nv_bfloat16* __restrict__ projp, int E) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;  // one output element
  if (i >= (E + 1) * kDRel) return;
  const int d = i / kDRel, slot = i % kDRel;
  const int qd = slot / 4, w = slot % 4;
  const int e = 2 * qd + (w & 1) + 8 * (w >> 1);
  projp[i] = d < E ? proj[e * E + d] : __float2bfloat16_rn(0.f);
}

// Eight BF16 words for one swizzled shared-memory vector. No FP conversion.
__device__ __forceinline__ uint4 packed_vector(const Params& p, int page, int kind,
                                              int head, int row, int dim, bool valid) {
  const uint4 zero = make_uint4(0, 0, 0, 0);
  if (!valid || page <= 0 || page >= p.num_pages) return zero;
  const long long start = static_cast<long long>(page) * p.packed_stride;
  const int flag = p.flags[start / sizeof(int)];
  if (flag <= 0) return zero;
  const int element = ((kind * 128 + row) * p.kv_heads + head) * kD + dim;
  if (flag >= 2) {
    // The low byte stores valid rows plus one. The high bits store the slot.
    const int slot = flag >> 8;
    const int count = (flag & 255) - 1;
    if (slot >= p.stage_slots || row >= count || count > 128) return zero;
    const auto* src = p.staging + static_cast<long long>(slot) * 2 * 128 * p.kv_heads * kD;
    return *reinterpret_cast<const uint4*>(src + element);
  }
  const int lane = head * 2 + kind;
  const int ref = p.overflow_ref[start / sizeof(int) + lane];
  if (ref) {
    if (ref < 0 || ref > p.overflow_slots) return zero;
    const auto* src = p.overflow + static_cast<long long>(ref - 1) * 128 * kD;
    return *reinterpret_cast<const uint4*>(src + row * kD + dim);
  }
  const auto sm = *reinterpret_cast<const uint2*>(p.sm + start + element);
  const uint32_t ex = *reinterpret_cast<const uint32_t*>(p.ex + start + element / 2);
  uint32_t words[4] = {};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t bits = ((i < 4 ? sm.x : sm.y) >> (8 * (i & 3))) & 255;
    const uint32_t code = (ex >> (4 * i)) & 15;
    const uint32_t exponent = code == 15 ? 0 : code + 116;
    const uint32_t word = ((bits & 128) << 8) | (exponent << 7) | (bits & 127);
    words[i / 2] |= word << (16 * (i & 1));
  }
  if (((ex & (ex >> 1) & (ex >> 2) & (ex >> 3)) & 0x11111111u) != 0) {
    const int* offsets = p.off + start / sizeof(int) + lane * 5 + row / 32;
    const int* entries = p.esc + start / sizeof(int) + lane * 64;
    const int begin = offsets[0], end = offsets[1];
    const int limit = begin >= 0 && end >= begin && end <= 64 ? end : begin;
    for (int i = begin; i < limit; ++i) {
      const unsigned entry = entries[i];
      const int column = (entry >> 16) & 127;
      if (((entry >> 23) & 127) == row && column >= dim && column < dim + 8) {
        const int index = column - dim, shift = (index & 1) * 16;
        words[index / 2] = (words[index / 2] & ~(0xffffu << shift)) | ((entry & 0xffffu) << shift);
      }
    }
  }
  return make_uint4(words[0], words[1], words[2], words[3]);
}

// kW warps x 16 query rows a CTA (BM = 16 kW): 4 warps -> 2 CTAs / SM, 8 warps -> 1 CTA / SM (half the K/V
// shared-memory fills per FLOP). Dynamic shared memory: tstage_bytes(kW).
// gate_sink.cuh's sigmoid: ex2.approx / rcp.approx with round-to-nearest multiply and add
__device__ __forceinline__ float gs_sigmoid(float x) {
  float e, r;
  float const t = __fmul_rn(-x, 1.4426950408889634f);
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
  asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(__fadd_rn(e, 1.0f)));
  return r;
}

template <bool kLocal, bool kPacked, int kW, bool kPackedKV = false, bool kGate = false>
__global__ void __launch_bounds__(kW * 32, 8 / kW) prefill_fwd(const Params p) {
  constexpr int kThr = kW * 32;
  constexpr int kBMw = 16 * kW;
  using namespace dev;
  __shared__ __align__(128) __nv_bfloat16 sK[kBN * kD];
  __shared__ __align__(128) __nv_bfloat16 sV[kBN * kD];
  extern __shared__ __align__(16) __nv_bfloat16 sT[];  // kW * 16 * kTStride (dynamic)

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, qd = lane & 3;

  // ---- work item: heads fastest (the 24 heads of one query block share K/V in L2), the last query blocks
  // (the most keys on causal global layers) first.
  const int lin = static_cast<int>(gridDim.x - 1 - blockIdx.x);
  const int head = lin % p.heads;
  int tile = lin / p.heads;
  int b = -1, start = 0, sq = 0;
  int prev_end = 0;
  for (int i = 0; i < p.n_req; ++i) {
    // metadata guards (CTA-uniform): rows inside q and never shared by two requests, keys inside the page table
    const int s = max(p.cu_q[i], prev_end);
    const int e = min(p.cu_q[i + 1], p.total_q);
    const int len = e > s ? e - s : 0;
    prev_end = max(prev_end, e);
    const int nb = (len + kBMw - 1) / kBMw;
    if (tile < nb) {
      b = i;
      start = s;
      sq = len;
      break;
    }
    tile -= nb;
  }
  if (b < 0) return;

  const int sk = min(max(p.seq_k[b], 0), p.pt_cols * p.page_size);
  const int off = sk - sq;  // prefix length
  if (p.sel_mode != 0) {  // the other launch takes request b (decided on the unclamped prefix)
    const long long pre = static_cast<long long>(p.seq_k[b]) - sq;
    if ((pre >= p.sel_keys) != (p.sel_mode == 2)) return;
  }
  const int m0 = tile * kBMw;
  const int kvh = head / p.group;
  const int wl = p.window_left;
  const int lo_key = kLocal ? max(0, m0 + off - wl) : 0;
  const int hi_key = static_cast<int>(min(static_cast<long long>(sk), static_cast<long long>(m0) + kBMw + off));
  const int nb_lo = lo_key / kBN;
  const int nb_hi = hi_key > lo_key ? hi_key / kBN + (hi_key % kBN != 0) : nb_lo;  // exclusive

  const int wrow0 = m0 + warp * 16;  // request-relative first row of this warp
  const bool warp_rows = wrow0 < sq;
  const int pw0 = wrow0 + off;        // its absolute position

  // ---- Q and r fragments (A operands), rows wrow0 + g and + 8
  uint32_t qa[8][4];
  uint32_t ra[4];
  {
    const int r0 = wrow0 + g, r1 = r0 + 8;
    const bool v0 = r0 < sq, v1 = r1 < sq;
    const long long t0 = start + (v0 ? r0 : 0), t1 = start + (v1 ? r1 : 0);
    const __nv_bfloat16* q0 = p.q + t0 * p.q_st + static_cast<long long>(head) * p.q_sh;
    const __nv_bfloat16* q1 = p.q + t1 * p.q_st + static_cast<long long>(head) * p.q_sh;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const int c = 16 * kk + 2 * qd;
      qa[kk][0] = v0 ? ld_u32(q0 + c) : 0u;
      qa[kk][1] = v1 ? ld_u32(q1 + c) : 0u;
      qa[kk][2] = v0 ? ld_u32(q0 + c + 8) : 0u;
      qa[kk][3] = v1 ? ld_u32(q1 + c + 8) : 0u;
    }
    const __nv_bfloat16* rr0 = p.r + t0 * p.r_st + static_cast<long long>(head) * p.r_sh;
    const __nv_bfloat16* rr1 = p.r + t1 * p.r_st + static_cast<long long>(head) * p.r_sh;
    ra[0] = v0 ? ld_u32(rr0 + 2 * qd) : 0u;
    ra[1] = v1 ? ld_u32(rr1 + 2 * qd) : 0u;
    ra[2] = v0 ? ld_u32(rr0 + 2 * qd + 8) : 0u;
    ra[3] = v1 ? ld_u32(rr1 + 2 * qd + 8) : 0u;
  }

  float o[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float mrow[2] = {-INFINITY, -INFINITY};
  float lrow[2] = {0.f, 0.f};

  // ---- one K or V tile of 64 keys -> shared memory (row = key, 16 chunks of 16 B, chunk ^= row & 7);
  // keys outside [lo_key, sk) are zero-filled (never read from memory).
  const long long pt_row = static_cast<long long>(b) * p.pt_s;
  auto load_tile = [&](__nv_bfloat16* dst, const __nv_bfloat16* base, long long sp, long long ss, long long sh,
                       int nb, int kind) {
    const int n0 = nb * kBN;
    const int col = n0 / p.page_size;
    const int page = col >= 0 && col < p.pt_cols ? p.page_table[pt_row + col] : -1;
    const bool page_ok = page >= 0 && page < p.num_pages;        // a bad page id reads nothing (zero tile)
    const int slot0 = n0 % p.page_size;
    const int chunk = tid & 15;
#pragma unroll
    for (int i = 0; i < kBN * 16 / kThr; ++i) {
      const int row = (tid >> 4) + i * (kThr / 16);
      const int j = n0 + row;
      const bool ok = page_ok && j >= lo_key && j < sk;
      auto* target = dst + row * kD + ((chunk ^ (row & 7)) * 8);
      if constexpr (kPackedKV) {
        *reinterpret_cast<uint4*>(target) = packed_vector(p, page, kind, kvh, slot0 + row, chunk * 8, ok);
      } else {
        const auto* src0 = base + static_cast<long long>(page_ok ? page : 0) * sp + static_cast<long long>(kvh) * sh;
        const auto* src = ok ? src0 + static_cast<long long>(slot0 + row) * ss + chunk * 8 : src0;
        cp_async16(smem_u32(target), src, ok);
      }
    }
  };

  __nv_bfloat16* tw = sT + warp * 16 * kTStride;
  const int par = g & 1;  // odd rows are staged one column to the right so sheared reads are 4-byte aligned
  const __nv_bfloat16* proj = p.proj;
  const int ext = p.rel_extent;

  if (nb_hi > nb_lo) load_tile(sK, p.k, p.k_sp, p.k_ss, p.k_sh, nb_hi - 1, 0);
  cp_commit();

  for (int nb = nb_hi - 1; nb >= nb_lo; --nb) {
    const int n0 = nb * kBN;
    cp_wait_all();
    __syncthreads();  // K(nb) visible; every warp is done with V(nb + 1)
    load_tile(sV, p.v, p.v_sp, p.v_ss, p.v_sh, nb, 1);
    cp_commit();

    // this warp's rows see keys [pw0 - wl, pw0 + 15]
    const bool active = warp_rows && n0 <= pw0 + 15 && (!kLocal || n0 + kBN - 1 >= pw0 - wl);
    float s[8][4];
    if (active) {
#pragma unroll
      for (int i = 0; i < 8; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
      // S = Q K^T
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          const int key = 8 * (2 * np + (lane >> 4)) + (lane & 7);
          const int chunk = 2 * kk + ((lane >> 3) & 1);
          uint32_t b0, b1, b2, b3;
          ldsm_x4(b0, b1, b2, b3, smem_u32(sK + key * kD + ((chunk ^ (key & 7)) * 8)));
          mma(s[2 * np], qa[kk], b0, b1);
          mma(s[2 * np + 1], qa[kk], b2, b3);
        }
      }
      // relative bias: d(a, b) = pw0 + a - (n0 + b) = dbase + c, c = a + 63 - b in [0, 78]
      const int dbase = pw0 - n0 - (kBN - 1);
      if (dbase < ext) {
        float t[10][4];
#pragma unroll
        for (int ct = 0; ct < 10; ++ct) {
          t[ct][0] = t[ct][1] = t[ct][2] = t[ct][3] = 0.f;
          const int d = dbase + 8 * ct + g;
          const bool in = d >= 0 && d < ext;  // d < 0: masked (causal); d >= ext: bias 0
          if constexpr (kPacked) {
            const int row = in ? d : ext;  // the zero row
            const uint2 w = __ldg(reinterpret_cast<const uint2*>(p.projp + static_cast<long long>(row) * kDRel) + qd);
            mma(t[ct], ra, w.x, w.y);
          } else {
            const __nv_bfloat16* pc = proj + (in ? d : 0);
            const uint32_t k0 = in ? ld_u16(pc + (2 * qd) * ext) : 0u;
            const uint32_t k1 = in ? ld_u16(pc + (2 * qd + 1) * ext) : 0u;
            const uint32_t k8 = in ? ld_u16(pc + (2 * qd + 8) * ext) : 0u;
            const uint32_t k9 = in ? ld_u16(pc + (2 * qd + 9) * ext) : 0u;
            mma(t[ct], ra, k0 | (k1 << 16), k8 | (k9 << 16));
          }
        }
#pragma unroll
        for (int ct = 0; ct < 10; ++ct) {
          const int col = 8 * ct + 2 * qd + par;
          tw[g * kTStride + col] = __float2bfloat16_rn(t[ct][0]);
          tw[g * kTStride + col + 1] = __float2bfloat16_rn(t[ct][1]);
          tw[(g + 8) * kTStride + col] = __float2bfloat16_rn(t[ct][2]);
          tw[(g + 8) * kTStride + col + 1] = __float2bfloat16_rn(t[ct][3]);
        }
        __syncwarp();
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int a = g + 8 * h;
            const int s1 = a + (kBN - 2) - 8 * nt - 2 * qd + par;  // even
            const uint32_t w = *reinterpret_cast<const uint32_t*>(tw + a * kTStride + s1);
            const float b_e0 = __uint_as_float(w & 0xffff0000u);  // column c   -> key n0 + 8nt + 2qd
            const float b_e1 = __uint_as_float(w << 16);          // column c-1 -> key n0 + 8nt + 2qd + 1
            s[nt][2 * h] = fmaf(s[nt][2 * h], p.scale, b_e0);
            s[nt][2 * h + 1] = fmaf(s[nt][2 * h + 1], p.scale, b_e1);
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          s[i][0] *= p.scale;
          s[i][1] *= p.scale;
          s[i][2] *= p.scale;
          s[i][3] *= p.scale;
        }
      }
      // mask: causal diagonal, window left edge (rows past sq see zero-filled keys and are never stored)
      const bool need_mask = (n0 + kBN - 1 > pw0) || (kLocal && n0 < pw0 + 15 - wl);
      if (need_mask) {
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int a = g + 8 * (e >> 1);
            const int j = n0 + 8 * nt + 2 * qd + (e & 1);
            const int pr = pw0 + a;
            if (j > pr || (kLocal && j < pr - wl)) s[nt][e] = -INFINITY;
          }
        }
      }
      // online softmax (natural-log units, exp2 with log2(e) like FA4)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float mx = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) mx = fmaxf(mx, fmaxf(s[nt][2 * h], s[nt][2 * h + 1]));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        const float mnew = fmaxf(mrow[h], mx);
        const float muse = mnew == -INFINITY ? 0.f : mnew;
        const float alpha = ex2((mrow[h] - muse) * kLog2e);
        mrow[h] = mnew;
        float sum = 0.f;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const float p0 = ex2((s[nt][2 * h] - muse) * kLog2e);
          const float p1 = ex2((s[nt][2 * h + 1] - muse) * kLog2e);
          s[nt][2 * h] = p0;
          s[nt][2 * h + 1] = p1;
          sum += p0 + p1;
        }
        lrow[h] = lrow[h] * alpha + sum;
#pragma unroll
        for (int dn = 0; dn < 16; ++dn) {
          o[dn][2 * h] *= alpha;
          o[dn][2 * h + 1] *= alpha;
        }
      }
    }

    cp_wait_all();
    __syncthreads();  // V(nb) visible; every warp is done with K(nb)
    if (nb - 1 >= nb_lo) load_tile(sK, p.k, p.k_sp, p.k_ss, p.k_sh, nb - 1, 0);
    cp_commit();

    if (active) {
      // O += P V
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        uint32_t pa[4];
        pa[0] = pack_bf16(s[2 * kk][0], s[2 * kk][1]);
        pa[1] = pack_bf16(s[2 * kk][2], s[2 * kk][3]);
        pa[2] = pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]);
        pa[3] = pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
        for (int dp = 0; dp < 8; ++dp) {
          const int i4 = lane >> 3;
          const int key = 16 * kk + 8 * (i4 & 1) + (lane & 7);
          const int chunk = 2 * dp + (i4 >> 1);
          uint32_t v0, v1, v2, v3;
          ldsm_x4_t(v0, v1, v2, v3, smem_u32(sV + key * kD + ((chunk ^ (key & 7)) * 8)));
          mma(o[2 * dp], pa, v0, v1);
          mma(o[2 * dp + 1], pa, v2, v3);
        }
      }
    }
  }
  cp_wait_all();

  // ---- epilogue: O / l (FP32), LSE = m + ln(l)
  // Gated (kGate): the warp's 16 x 128 gate tile is staged in shared memory (sK / sV are free once every warp left the
  // mainloop) with coalesced 16-byte loads, y is computed in place in the MMA layout, then stored with coalesced
  // 16-byte stores. 16-byte chunks are XOR-swizzled by row so the MMA-layout accesses are bank-conflict free.
  __nv_bfloat16* wbuf = nullptr;
  if constexpr (kGate) {
    __syncthreads();
    wbuf = (warp < 4 ? sK + warp * 16 * kD : sV + (warp - 4) * 16 * kD);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int rr = 2 * i + (lane >> 4), cc = (lane & 15) * 8;
      if (wrow0 + rr < sq) {
        const long long t = start + wrow0 + rr;
        *reinterpret_cast<uint4*>(wbuf + rr * kD + (((cc >> 3) ^ (rr & 7)) * 8)) =
            *reinterpret_cast<const uint4*>(p.gate + t * p.g_st + static_cast<long long>(head) * kD + cc);
      }
    }
    __syncwarp();
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float l = lrow[h];
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    const int row = wrow0 + g + 8 * h;
    if (row < sq) {
      const bool ok = l > 0.f && l == l;
      const float inv = ok ? 1.f / l : 0.f;
      const long long t = start + row;
      if constexpr (kGate) {
        const float lse_v = ok ? mrow[h] + logf(l) : -INFINITY;
        if (qd == 0) p.lse[static_cast<long long>(head) * p.lse_sh + t] = lse_v;
        const float sc = gs_sigmoid(__fsub_rn(lse_v, __bfloat162float(p.sink[t * p.s_st + head])));
        const int rl = g + 8 * h;  // row within the warp's staged tile
#pragma unroll
        for (int dn = 0; dn < 16; ++dn) {
          __nv_bfloat162* cell = reinterpret_cast<__nv_bfloat162*>(wbuf + rl * kD + ((dn ^ (rl & 7)) * 8) + 2 * qd);
          const __nv_bfloat162 gg = *cell;
          const float x0 = o[dn][2 * h] * inv, x1 = o[dn][2 * h + 1] * inv;
          __nv_bfloat162 yy;
          yy.x = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x0, gs_sigmoid(__low2float(gg))), sc));
          yy.y = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x1, gs_sigmoid(__high2float(gg))), sc));
          *cell = yy;  // in place: each element is read and written by this thread only
        }
      } else {
        float* orow = p.out + t * p.o_st + static_cast<long long>(head) * p.o_sh;
#pragma unroll
        for (int dn = 0; dn < 16; ++dn) {
          *reinterpret_cast<float2*>(orow + 8 * dn + 2 * qd) = make_float2(o[dn][2 * h] * inv, o[dn][2 * h + 1] * inv);
        }
        if (qd == 0) p.lse[static_cast<long long>(head) * p.lse_sh + t] = ok ? mrow[h] + logf(l) : -INFINITY;
      }
    }
  }
  if constexpr (kGate) {
    __syncwarp();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int rr = 2 * i + (lane >> 4), cc = (lane & 15) * 8;
      if (wrow0 + rr < sq) {
        const long long t = start + wrow0 + rr;
        *reinterpret_cast<uint4*>(p.y + t * p.y_st + static_cast<long long>(head) * kD + cc) =
            *reinterpret_cast<const uint4*>(wbuf + rr * kD + (((cc >> 3) ^ (rr & 7)) * 8));
      }
    }
  }
}

}  // namespace inkling_attn
