#!/usr/bin/env python3
"""Quantize WeiboAI/VibeThinker-3B to NVFP4 with NVIDIA Model Optimizer.

Simple dense Qwen2-3B — no MoE, no multimodal, no GDN.
NVFP4_DEFAULT_CFG (full W4A4). tie_word_embeddings=true (no lm_head pitfall).
"""
import torch
import modelopt.torch.quantization as mtq
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from modelopt.torch.export import export_hf_checkpoint
import os, shutil

MODEL_DIR = "/home/r0b0tdgx/models/WeiboAI-VibeThinker-3B"
OUTPUT_DIR = "/home/r0b0tdgx/vibethinker-3b-nvfp4/vibethinker-3b-nvfp4"
CALIB_SAMPLES = 512

print("=" * 60)
print("VibeThinker-3B → NVFP4 Quantization")
print(f"Model: {MODEL_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Calibration: {CALIB_SAMPLES} samples, cnn_dailymail")
print("=" * 60)

# Step 1: Load model to CPU (GB10 unified, then move to CUDA)
print("\n[1/4] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
print(f"  Loaded {sum(p.numel() for p in model.parameters())/1e9:.2f}B params on CPU")

# Move to CUDA (unified memory on GB10 — this is fast)
for name, param in model.named_parameters():
    param.data = param.data.to("cuda")
for name, buf in model.named_buffers():
    buf.data = buf.data.to("cuda")
print(f"  Moved to CUDA")

# Step 2: Load calibration data
print(f"\n[2/4] Loading calibration data ({CALIB_SAMPLES} samples)...")
calib_data = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"train[:{CALIB_SAMPLES}]")
print(f"  Loaded {len(calib_data)} samples")

def forward_loop(model):
    for i in range(0, len(calib_data), 16):
        batch = calib_data[i:i+16]["article"]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=1024).to("cuda")
        model(**inputs)

# Step 3: Quantize
print(f"\n[3/4] Quantizing with NVFP4_DEFAULT_CFG...")
mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop)
print(f"  Quantization complete")

# Step 4: Export
print(f"\n[4/4] Exporting checkpoint...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with torch.inference_mode():
    export_hf_checkpoint(model, export_dir=OUTPUT_DIR)
print(f"  Exported to {OUTPUT_DIR}")

# Post-export: copy tokenizer/config files
print("\nPost-export: copying tokenizer/config files...")
for fname in [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "generation_config.json", "vocab.json", "added_tokens.json",
]:
    src = os.path.join(MODEL_DIR, fname)
    dst = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied {fname}")

# Verify output
print("\nVerification:")
hf_quant = os.path.join(OUTPUT_DIR, "hf_quant_config.json")
if os.path.exists(hf_quant):
    import json
    with open(hf_quant) as f:
        cfg = json.load(f)
    print(f"  hf_quant_config.json: {cfg.get('quant_method', 'MISSING')}")
else:
    print(f"  ❌ hf_quant_config.json NOT FOUND!")

total = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f))
           for f in os.listdir(OUTPUT_DIR)
           if os.path.isfile(os.path.join(OUTPUT_DIR, f)))
print(f"  Total output: {total/1e9:.2f} GB")
print("\n✅ Done!")
