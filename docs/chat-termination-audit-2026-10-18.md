# Chat termination audit, 2026-10-18

This repair addresses base weights served through a chat endpoint. The sweep
was read only: `~/.local/state/vonk-sweep/sweep-state/results.jsonl` and all
three `results.jsonl.*` siblings present on 2026-10-18. Repeated records can
replay an earlier response; they are not independent inference measurements.

## DeepSeek-V2-Lite: base checkpoint served as chat

`recipes/deepseek-v2-lite-nvidia-sglang-single.json` selected
`deepseek-ai/DeepSeek-V2-Lite` at
`604d5664dddd88a0433dbae533b7fe9472482de0`, mounted at `/models`.
Its launcher simply runs `python3 -m sglang.launch_server` with the recipe
arguments, without a chat-template override, stop override or `ignore-eos`.

The A323 response on 2026-10-17 15:32Z and 2026-10-18 15:38Z answers the
arithmetic correctly, then invents repeated `User:` / `Assistant:` turns until
256 completion tokens, with no matched stop. Current `results.jsonl` lines
2, 33, 100 and 103 contain this evidence (two response IDs, with replays).

The [pinned NVIDIA playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/98b3a473f30916ec05eef3e7a01b781aaf4daa8a/nvidia/sglang/README.md#step-5-run-the-inference-server)
launches the base checkpoint. DeepSeek's
[model card](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/85864749cd611b4353ce1decdb286193298f64c7/README.md)
distinguishes base text completion from chat completion: chat uses the SFT
`DeepSeek-V2-Lite-Chat` checkpoint, `tokenizer.apply_chat_template`, and EOS.

Both checkpoints contain the **same** `tokenizer_config.json` chat template:
BOS, `User: ...\n\nAssistant:`, and EOS after completed assistant messages.
Both config and generation config declare EOS token **100001**,
`<｜end▁of▁sentence｜>`. This is not a missing-template or missing-EOS defect:
the base weights were never SFT-trained to follow the chat turn boundary.

The pinned image is labelled SGLang v0.5.20, build `94602c9c` (the recipe's
separate release metadata says 0.5.24). In that
[source's template manager](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/parser/template_manager.py#L143),
an explicit template wins; otherwise model-path detection falls back to the
Hugging Face tokenizer's template. `/models` does not name a built-in chat
template. Its
[stop check](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/sglang/srt/managers/schedule_batch.py#L1711)
recognizes the tokenizer/model EOS and request stop tokens. The fix changes
the selected weights to the Chat checkpoint at
`85864749cd611b4353ce1decdb286193298f64c7`, retaining the upstream template,
EOS, engine image, alias and launch arguments. It adds no output processing.

## All recipes with length-limited smoke evidence

All identifiers below have publisher `vonk-forge`. `R` means current
`results.jsonl`; `O` means `results.jsonl.20261015T094957281645Z`; `V` means
`results.jsonl.20261017T150119474154Z`. The remaining rotated sibling,
`results.jsonl.20261015T174221928735Z`, has no length-limited smoke evidence.

| Recipe | Evidence lines | Class and action |
| --- | --- | --- |
| `deepseek-v2-lite-nvidia-sglang-single` | R:2,33,100,103, A323 | Correct answer followed by invented turns; switch base weights to Chat. |
| `phi-4-reasoning-plus-nvfp4-nvidia-sglang-single` | O:30,85,139, A323 / A323_UNPARSED | Reasoning repeats inside `<think>` until the budget; unchanged. |
| `phi-4-reasoning-plus-nvfp4-nvidia-trtllm-single` | O:140, A323_UNPARSED | Reasoning budget exhaustion; unchanged. |
| `phi-4-reasoning-plus-nvfp4-nvidia-vllm-single` | O:138, A323_UNPARSED | Reasoning budget exhaustion; unchanged. |
| `phi-4-reasoning-plus-fp8-nvidia-sglang-single` | O:849, A323_UNPARSED | Reasoning budget exhaustion; unchanged. |
| `phi-4-reasoning-plus-fp8-nvidia-trtllm-single` | O:852, A323_UNPARSED | Reasoning budget exhaustion; unchanged. |
| `phi-4-reasoning-plus-fp8-nvidia-vllm-single` | O:853, A323_UNPARSED | Reasoning budget exhaustion; unchanged. |
| `muse-glimmer-30b-nvfp4-r0b0tlab-vllm-single` | V:165; R:43, V7 | `content: null`, digit deliberation in `reasoning_content`; unchanged. |
| `muse-glimmer-30b-nvfp4-vllm-mia-single` | V:170; R:44, V7 | `content: null`, digit deliberation in `reasoning_content`; unchanged. |

## Regression boundary

`tests/test_chat_configuration.py` checks every recipe declaring an
`openai.chat` validation: it must declare a runtime chat template, select a
`chat_template.jinja`/`chat_template.json` file, or select a served Model with
the `chat` capability. The guard checks the served Model before companions
or drafters. An embedded template in a tokenizer configuration does not by
itself declare the weights chat-capable. The new test rejects the original
DeepSeek recipe on main; a fault-injection test also removes chat capability
and checks rejection and explicit-template acceptance.

This is a catalog configuration regression check. Successful generation and
EOS termination on Spark remain unverified until a separately authorized run;
this task changes neither production nor the hardware sweep.
