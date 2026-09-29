#!/usr/bin/env python3
"""Stub AVX CPU targets so ExLlamaV3's extension compiles on aarch64/GB10."""

from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/exllamav3/exllamav3/exllamav3_ext")
(root / "avx2_target.cpp").write_text(
    '#include "avx2_target.h"\n'
    "bool is_avx2_supported() { return false; }\n"
    "bool is_f16c_supported() { return false; }\n"
)
(root / "avx512_target.cpp").write_text(
    '#include "avx512_target.h"\nbool is_avx512_supported() { return false; }\n'
)
(root / "parallel/all_reduce_cpu_avx2.cpp").write_text(
    """#include "all_reduce_cpu_avx2.h"
#include "all_reduce_cpu_avx512.h"
#include <cstdlib>
void enable_fast_fp() {}
void enable_fast_fp_avx2() {}
void perform_cpu_reduce(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }
void perform_cpu_reduce_avx2(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }
void cpu_reduce_parallel(
    void (*)(uint16_t*, const uint16_t*, const uint16_t*, size_t),
    void (*)(uint16_t*, const uint16_t*, size_t),
    uint16_t*, const uint16_t*, const uint16_t*, size_t, int) { std::abort(); }
"""
)
(root / "parallel/all_reduce_cpu_avx512.cpp").write_text(
    """#include "all_reduce_cpu_avx512.h"
#include <cstdlib>
void enable_fast_fp_avx512() {}
void bf16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) {}
void bf16_add_twosrc_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t) { std::abort(); }
void fp16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) {}
void fp16_add_twosrc_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t) { std::abort(); }
void perform_cpu_reduce_avx512(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }
"""
)
for hdr, extras in (
    ("avx2_target.h", "avx2"),
    ("avx512_target.h", "avx512"),
):
    name = extras
    guard = name.upper()
    extra_decls = "bool is_f16c_supported();\n" if name == "avx2" else ""
    extra_macros = f"#define {guard}_F16C_TARGET\n" if name == "avx2" else ""
    (root / hdr).write_text(
        "#pragma once\n"
        f"bool is_{name}_supported();\n"
        f"{extra_decls}"
        f"#define {guard}_TARGET\n"
        f"{extra_macros}"
        f"#define {guard}_TARGET_OPTIONAL\n"
    )

# v1.4.x CPU MoE (mul1) is AVX2/AVX-512 only and includes <immintrin.h>.
# GB10 is aarch64; GPU fused exl3_moe is the decode path. Stub the CPU GEMM.
moe_cpu = root / "cpu" / "moe_mul1.cpp"
if moe_cpu.is_file():
    moe_cpu.write_text(
        """#include "moe_mul1.h"
#include <stdexcept>

static void unsupported() {
    throw std::runtime_error("exl3_moe_cpu is x86-only; this image is aarch64/GB10");
}

int64_t exl3_moe_cpu_make_layer(
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    int64_t, double, int64_t)
{
    unsupported();
    return -1;
}
void exl3_moe_cpu_free_layer(int64_t) {}
void exl3_moe_cpu_forward(int64_t, const at::Tensor&, const at::Tensor&,
                          const at::Tensor&, at::Tensor&, int64_t) { unsupported(); }
void exl3_moe_cpu_forward_raw(int64_t, const at::Half*, const int32_t*,
                              const at::Half*, float*, int, int, int) { unsupported(); }
void exl3_moe_cpu_stage_experts(int64_t, const uint32_t*, int, uint8_t*, int) { unsupported(); }
void exl3_moe_cpu_set_prof(bool) {}
bool exl3_moe_cpu_has_avx2() { return false; }
bool exl3_moe_cpu_has_avx512_vnni() { return false; }
bool exl3_moe_cpu_has_avx512_vbmi() { return false; }
"""
    )
    print(f"aarch64 stubbed {moe_cpu}")

# v1.4.x CPU MoE handoff spins with x86 PAUSE.
for path in root.rglob("*"):
    if path.suffix not in {".cu", ".cpp", ".cuh", ".h", ".c"}:
        continue
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        continue
    if "__builtin_ia32_pause" not in text:
        continue
    path.write_text(text.replace("__builtin_ia32_pause()", "((void)0)"))
    print(f"aarch64 replaced ia32_pause in {path}")

print(f"aarch64 EXL3 CPU-target stubs written in {root}")
