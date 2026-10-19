# Creator runtime intake, 19 October 2026

The previous intake searched recently created repositories. That missed new model
and runtime profiles added inside older repositories. This follow-up inventories
1,557 public repositories from the 15 tracked accounts and requests the READMEs
of 621 relevant/non-fork repositories: 594 returned content and 27 had no readable
README. This is discovery coverage, not proof that every branch or file in every
repository has been reviewed. Missing README responses remain unverified.

## Additions

| Recipe | Exact source and behavior |
| --- | --- |
| `deepseek-v4-1-flash-1017-exl3-2-9bpw-tonyd2wild-tensorfold-dual` | Tony `9fa230d3eb119f588e690bf9b87ce41ed902db60`; native TensorFold TP2, 2.9bpw, 1M context, four streams. |
| `deepseek-v4-1-flash-1017-exl3-3-5bpw-tonyd2wild-tensorfold-four` | Same Tony commit; native TensorFold TP4, 3.5bpw, 1M context, sixteen streams and 2M shared pool. |
| `deepseek-v4-1-flash-1017-mia-sglang-triple` | Mia `64b02dea8681fb1d8e3a659ea432a8e9e9ba3995`; native MXFP4, TP3/EP1, DSpark 5, 262K context. |
| `deepseek-v4-1-flash-1017-mia-sglang-four` | Same Mia commit; supported base-image profile, TP4/EP2, DSpark 5, 1M context. This is the base profile, not the optional canary RoCEnante/EP1 branch. |
| `minimax-music3-drowzeys-tensorfold-single` | Studio `1e8677e7c9151836de375e3cc6306ed6e4170c65`, native CUDA engine `6b714531e2d60f91d26e98e414547ad94ad771bf`; caption/optional lyrics to bounded stereo WAV, int8 autoregressive weights generated at load. |

DeepSeek weights are the native 1017 checkpoint at
`1bae0e1088993b9b82de08f59de5be5a34e42170`, EXL3 2.9bpw at
`9a85740a8dce89e858c2113c4d04b6f8fcd33999`, and EXL3 3.5bpw at
`cc6a7ef5bdf61d64417806edfd49a48b3d9fda35`. Music3 uses only the three
required Comfy-Org files at `9e364ccd290a401561091fb441ed19cd7de92550`.
The original MiniMax-Music3 Community License is preserved despite the repack's
Apache-2.0 card metadata.

The adapters use Controller-managed local model mounts. TensorFold derives
rank-local caches and the RoPE/vision kit under the managed writable cache;
completed work is reused and interrupted/malformed generated work is retried.
The SGLang row store reads native Engram shards directly from local storage.
No host NFS setup, SSH launch, global cache drop, or display-memory reservation
is included. Upstream measurements are estimates, not physical qualification.

## Other discoveries and concrete remaining gaps

- **Tony native and full-EXL3 vLLM 1017 profiles:** the required base
  `vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c`
  returns `manifest unknown`. The creator's later overlay images are host-local
  builds, including GPU prewarming. Replacing the engine requires validating its
  specialized overlays; these profiles have not been added as runnable recipes.
- **Mia MiMo V2.6 Flash SGLang:** its required
  `lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5` also returns
  `manifest unknown`. Existing MiMo vLLM and r0b0tlab NVFP4 SGLang recipes remain
  distinct alternatives, not proof that this native MXFP4/EAGLE profile is covered.
- **DeepSeek 1017 uncensored TensorFold overlay:** provider access is gated at
  `08442c1cc5e0cf611c9d9c55ea97ba875c6e0aa0`. The file tree masks LFS hashes and
  `patch_rank_cache_wob.py` returns HTTP 401 anonymously. Tony publishes the
  sidecar hash, but the required rank-slicing implementation has not been verified.
- **Fish Drama3 CUDA:** weights are public, but the pinned converter imports
  `drama3_ref.build_codec` from an external `ORACLE_DIR`; neither the inspected
  engine commit `81173e6c612739a3427844fd8131cf4905e0a9c4`, studio tree nor checkpoint
  provides that module. A reproducible codec conversion must be closed first.
- **Studio TensorFold H3:** this is a ComfyUI transformer-block plugin. ComfyUI
  still owns text encoding, sampling, projections and decoding. It needs a complete
  pinned ComfyUI workflow and companion-model mapping; adding only the shared
  library would not create an executable video recipe. This integration remains
  outstanding; existing H3 recipes do not establish this acceleration variant.
- **Amarjeet Qwen TensorFold:** a port of Mia's already represented upstream
  recipe; it is not a newly discovered model or engine. Mac-only MLX ports,
  dashboards, benchmarks and client applications are not Spark runtime additions.

## Evidence boundaries

Pinned source comparisons, producer checks, deterministic packages and hosted
ARM64 image builds establish repository/publication evidence. They do not prove
that the local Controller has downloaded these assets or that any physical Spark
has run the models. This intake does not deploy or qualify workloads on the fleet.
