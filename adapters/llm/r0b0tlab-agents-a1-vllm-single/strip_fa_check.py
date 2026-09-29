from pathlib import Path
p = Path('/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/__init__.py')
s = p.read_text()
old = '''if not (FA2_AVAILABLE or FA3_AVAILABLE):
    raise ImportError(
        "vllm.vllm_flash_attn requires the CUDA flash attention extensions "
        "(_vllm_fa2_C or _vllm_fa3_C). On ROCm, use upstream flash_attn."
    )
'''
new = '''if not (FA2_AVAILABLE or FA3_AVAILABLE):
    # SM121 NVFP4 text-only images intentionally omit FA2/FA3 fallback
    # extension artifacts. FlashInfer is selected as the active attention
    # backend; leaving this importable lets model modules that reference
    # fa_utils load without activating FA2/FA3 kernels.
    pass
'''
assert old in s
p.write_text(s.replace(old, new))
