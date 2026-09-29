# PLAN: VibeThinker-3B NVFP4 Quantization + SM121 Optimization

## Model Summary

| Field | Value |
|-------|-------|
| Source | WeiboAI/VibeThinker-3B |
| Base | Qwen2.5-Coder-3B (from Qwen2.5-3B) |
| Architecture | Qwen2ForCausalLM (standard dense transformer) |
| Parameters | 3.09B |
| Precision | BF16 |
| hidden_size | 2048 |
| intermediate_size | 11008 |
| num_hidden_layers | 36 |
| num_attention_heads | 16 |
| num_key_value_heads | 2 (GQA) |
| vocab_size | 151936 (divisible by 16) |
| tie_word_embeddings | **true** (no lm_head pitfall) |
| max_position_embeddings | 131072 |
| License | MIT |
| Est. NVFP4 size | ~1.8 GB (3.09B × 0.6) |

This is the simplest NVFP4 quantization case: no MoE, no multimodal, no
GDN/Mamba, no vision tower. Standard Qwen2 dense architecture. Tie_word_embeddings
eliminates the lm_head export pitfall entirely.

---

## Phase 1: Pre-Flight Checks (30 min)

### 1.1 Matmul Early Rejection Test — CRITICAL

VibeThinker-3B has hidden=2048, intermediate=11008. This is smaller than
FastContext-4B (hidden=2560, inter=9728) which showed 2.8-4.5× NVFP4 speedup.
Promising, but must be tested per skill `nvfp4-llm-modelopt`.

**Action:** Run the matmul benchmark comparing BF16 vs NVFP4 on the model's
two largest linear layers:
- MLP gate/up: [11008, 2048]
- MLP down: [2048, 11008]
- Attention QKV: [2048+512, 2048] (fused QKV with GQA)
- Attention O: [2048, 2048]

**Decision rule:** If ALL layers show ratio < 1.0 (NVFP4 slower), proceed
only for memory savings (3× compression) and document honestly. If ANY
layer shows ratio > 1.0, NVFP4 throughput will benefit.

**Prior data points on GB10 SM121:**
| Model | hidden | inter | Matmul ratio | Verdict |
|-------|--------|-------|-------------|---------|
| FastContext (Qwen3-4B) | 2560 | 9728 | 2.8-4.5× | NVFP4 faster |
| HiDream-O1 (Qwen3VL 8B) | 4096 | 12288 | 0.87× | NVFP4 slower |
| **VibeThinker-3B** | **2048** | **11008** | **?** | **Test** |

### 1.2 Model Download

```bash
hf download WeiboAI/VibeThinker-3B --local-dir /home/r0b0tdgx/models/WeiboAI-VibeThinker-3B
```

### 1.3 Config Verification

- [x] architectures: ["Qwen2ForCausalLM"] — set, no fix needed
- [x] tie_word_embeddings: true — no lm_head pitfall
- [x] vocab_size: 151936 — divisible by 16, embed_tokens can be quantized
- [x] No MoE, no vision, no audio — no exclusion list needed
- [x] transformers_version: 4.51.3 — Qwen2 is stable across all versions

---

## Phase 2: Quantization (1-2 hours)

### 2.1 Environment Setup

```bash
uv venv ~/.venvs/modelopt --python 3.12
source ~/.venvs/modelopt/bin/activate

# Install order: torch FIRST, modelopt LAST
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install "transformers>=5.4" safetensors accelerate datasets
uv pip install "nvidia-modelopt[hf]>=0.44.0"
```

### 2.2 Quantization Script

Config: `NVFP4_DEFAULT_CFG` (full W4A4). This is a simple dense model —
no exclusion list, no MoE unfuse, no multimodal preservation.

```python
import torch
import modelopt.torch.quantization as mtq
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from modelopt.torch.export import export_hf_checkpoint

model = AutoModelForCausalLM.from_pretrained(
    "/home/r0b0tdgx/models/WeiboAI-VibeThinker-3B",
    torch_dtype=torch.bfloat16,
    device_map="cpu",         # CPU load, then .cuda() (GB10 unified)
    low_cpu_mem_usage=True,
)
for name, param in model.named_parameters():
    param.data = param.data.to("cuda")
for name, buf in model.named_buffers():
    buf.data = buf.data.to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    "/home/r0b0tdgx/models/WeiboAI-VibeThinker-3B"
)

# Calibration: 512 samples, cnn_dailymail (full namespace!)
calib_data = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train[:512]")

def forward_loop(model):
    for i in range(0, len(calib_data), 16):
        batch = calib_data[i:i+16]["article"]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=1024).to("cuda")
        model(**inputs)

mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop)

with torch.inference_mode():
    export_hf_checkpoint(model, export_dir="./vibethinker-3b-nvfp4")
```

### 2.3 Post-Export Fixes

- [ ] Copy tokenizer files from source: `tokenizer.json`, `tokenizer_config.json`,
      `chat_template.jinja` (if exists), `special_tokens_map.json`, `generation_config.json`
- [ ] Verify `hf_quant_config.json` exists in output
- [ ] Verify `quant_method: modelopt_fp4` in hf_quant_config.json
- [ ] Verify no lm_head issue (tie_word_embeddings=true → should be clean)
- [ ] Verify model.safetensors.index.json weight_map is complete

### 2.4 Quality Smoke Test

Run 4-5 reasoning prompts through both BF16 and NVFP4. Compare:
- Math problem (basic arithmetic)
- Code generation (simple function)
- General knowledge question
- Chain-of-thought reasoning

**Pass criteria:** NVFP4 output matches BF16 in correctness and coherence.

---

## Phase 3: vLLM Serving + SM121 Optimization (1 hour)

### 3.1 vLLM Launch

```bash
# Option A: vLLM pip wheel (simplest for small models)
uv pip install vllm  # v0.23.0 with native SM121 support

vllm serve /path/to/vibethinker-3b-nvfp4 \
    --quantization modelopt_fp4 \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code
```

No MoE backend flag needed (dense model). No reasoning parser needed
(unless model outputs <think> tags — check chat template).

### 3.2 Verify Backend Selection

Check vLLM startup logs for:
- `FlashInferCutlassNvFp4LinearKernel` — native NVFP4 GEMM (good)
- NOT `MARLIN` (broken on SM121 for some cases)
- NOT `EMULATION` (fallback, no hardware acceleration)

### 3.3 Optimization Sweep

Test these configs (c=1 first per user pref):
- [ ] c1 throughput (primary metric)
- [ ] c1 with CUDA graphs vs enforce-eager
- [ ] c1 with FP8 KV cache vs default
- [ ] BF16 baseline on same model (for comparison)
- [ ] c2, c4, c8 concurrency ramp
- [ ] Depth sweep (512, 4096, 8192, 16384 tokens)
- [ ] Long generation (512+ output tokens)
- [ ] GPU utilization, power draw, thermals

---

## Phase 4: Benchmarking (2 hours)

### 4.1 Benchmark Suite

Run `vllm bench serve` (or equivalent):

| Test | Config | Purpose |
|------|--------|---------|
| pp2048 | prefill 2048 tokens | Prefill throughput |
| tg128 | generate 128 tokens | Decode throughput (primary) |
| tg512 | generate 512 tokens | Long generation |
| pp2048 @ d4096 | prefill at 4K depth | Depth stability |
| tg128 @ d4096 | decode at 4K depth | KV cache performance |
| tg128 @ d8192 | decode at 8K depth | Deep context |
| tg128 @ d16384 | decode at 16K depth | Long context |
| c1-c8 ramp | concurrent requests | Aggregate throughput |

### 4.2 BF16 Baseline

Run the SAME benchmark suite on the BF16 model for before/after comparison.

### 4.3 Results Capture

For each run:
- [ ] JSON results saved to `benchmarks/results/`
- [ ] GPU utilization (≥0.70 target per user pref)
- [ ] GPU power draw + thermals
- [ ] Tokens/sec, TTFT, latency
- [ ] Config c1-c5 minimum coverage

### 4.4 HTML Report

Generate `benchmarks/reports/benchmark_final.html` with:
- [ ] BF16 vs NVFP4 before/after comparison table
- [ ] Concurrency ramp chart
- [ ] Depth sweep chart
- [ ] GPU utilization/power/thermal data
- [ ] Flat dark background, no gradients (per user pref)
- [ ] Mobile-compatible layout
- [ ] Overflow handling (clamp/text-overflow/overflow-x:auto)

### 4.5 Real-World Benchmark (Bonus)

Run the VibeThinker reasoning prompt from the model card (math/coding problem)
and visually capture the output for a demo.

---

## Phase 5: Container + Publish (2 hours)

### 5.1 Dockerfile

Lightweight container based on `vllm/vllm-openai:v0.22.0-aarch64-ubuntu2404`:
- [ ] HF CLI for auto-download
- [ ] Startup script with health check
- [ ] All vLLM flags env-var overridable
- [ ] NO baked weights (image stays ~10 GB, model downloads from HF)

### 5.2 HuggingFace Upload

- [ ] Create `r0b0tlab/VibeThinker-3B-NVFP4` (public)
- [ ] Upload NVFP4 checkpoint
- [ ] Professional model card with:
      - YAML frontmatter (base_model: WeiboAI/VibeThinker-3B, tags, license)
      - Credits section (WeiboAI, NVIDIA ModelOpt, vLLM, calibration data authors)
      - Quantization recipe (ModelOpt NVFP4_DEFAULT_CFG, W4A4, group_size=16)
      - Exact hf_quant_config.json
      - "How to verify" section
      - Honest limitations
      - BibTeX citations
- [ ] base_model linking for HF quantizations section

### 5.3 GitHub Repo

- [ ] Create `github.com/r0b0tlab/vibethinker-3b-nvfp4` via API
- [ ] README with setup instructions
- [ ] Reproducible quantization script
- [ ] Docker-compose
- [ ] Link to HF model
- [ ] Benchmark results embedded
- [ ] GHCR container push

### 5.4 Deliverables

- [ ] HF model: `r0b0tlab/VibeThinker-3B-NVFP4`
- [ ] GitHub repo: `r0b0tlab/vibethinker-3b-nvfp4`
- [ ] GHCR container: `ghcr.io/r0b0tlab/vibethinker-3b-nvfp4`
- [ ] HTML benchmark report
- [ ] Deliver to Telegram (7650220336)

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| NVFP4 slower than BF16 at 2048 hidden | Medium | Low | Matmul test Phase 1.1; if slower, still publish for 3× memory savings, document honestly |
| Model uses <think> reasoning tags | Medium | Low | Check chat template; add --reasoning-parser if needed |
| CUDA graph deadlock | Low | Low | Use --enforce-eager (already in default flags) |
| Quality regression on math/coding | Low | Medium | Quality smoke test Phase 2.4; fallback to NVFP4_MLP_ONLY_CFG if needed |

---

## Key Design Decisions

1. **NVFP4_DEFAULT_CFG** (not MLP-only) — simple dense model, no special handling
2. **No exclusion list** — text-only, no multimodal components
3. **CPU load then .cuda()** — GB10 unified memory pattern (pitfall #16 from skill)
4. **enforce-eager** — stability for single-model serving
5. **FP8 KV cache** — standard for NVFP4 serving on SM121
6. **vLLM pip wheel** — no Docker needed for quantization/serving of 3B model
7. **Container is for distribution** — lets others reproduce without setup

---

*Model: WeiboAI/VibeThinker-3B (3B dense, Qwen2 architecture)*
*Quantization: NVIDIA ModelOpt NVFP4 (W4A4, group_size=16)*
*Hardware: NVIDIA DGX Spark (GB10, SM121, 128GB unified)*
*Publish target: r0b0tlab on GitHub + HF*
