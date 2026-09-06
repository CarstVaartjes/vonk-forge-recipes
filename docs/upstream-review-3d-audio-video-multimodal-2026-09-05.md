# Upstream review: 3D, audio, video, and multimodal jobs

Checked at 2026-09-05T22:55:00+02:00 against the 84-row inventory in
`/private/tmp/vonk-all-recipe-upstream-review-20260905/inventory.json`; the
live source audit was checked at `2026-09-05T20:50:27Z` and is retained at
`/private/tmp/vonk-all-recipe-upstream-review-20260905/live-source-audit/upstream-drift.json`.
The recipe checkout is the exact requested main57 baseline at
`57f61176510938542ad50d1d7d86ed8d052c7369`. This review covers the 21
non-ComfyUI/non-Diffusers-owner rows whose executable engine is
`pytorch-pipeline`; it does not regenerate packages or catalog indexes.

## Exact assigned set

The assigned set is exactly 21 unique inventory rows (no rows are omitted or
duplicated):

```text
hunyuan-video-foley-xl-pytorch-single
hunyuan-video-foley-xxl-pytorch-single
hunyuan3d-omni-pytorch-single
hunyuanocr-1-5-vllm-dflash-single
ltx-2-19b-dev-bf16-diffusers-single
ltx-2-19b-dev-fp4-pytorch-single
ltx-2-19b-distilled-diffusers-single
ltx-2-19b-distilled-fp8-diffusers-single
ltx-2-3-22b-distilled-1-1-diffusers-single
moss-vl-realtime-11b-pytorch-single
mova-360p-diffusers-single
mova-720p-diffusers-single
pixal3d-pytorch-single
skintokens-pytorch-single
step1x-3d-geometry-pytorch-single
step1x-3d-label-geometry-pytorch-single
step1x-3d-texture-pytorch-single
trellis-2-4b-pytorch-single
triposg-pytorch-single
wan-dancer-14b-disk-offload-pytorch-single
wan-dancer-14b-pytorch-single
```

## Source evidence and decisions

The exact immutable Hugging Face heads below came from the Hugging Face model
API. For every pinned model that is not marked as updated, the selected
checkpoint files and companion files remain byte-identical at the current
head, or the recipe deliberately retains an older single-mode closure. The
four GitHub source pages were also checked. GitHub's API/clone endpoint reset
in this environment, so the current branch page is recorded where a full
current SHA could not be observed; the exact pinned source SHA remains in the
recipe and is the executable source bundle provenance.

| recipe ID | engine | primary and companion model versions | pinned source / current authoritative ref | decision |
|---|---|---|---|---|
| `hunyuan-video-foley-xl-pytorch-single` | pytorch-pipeline | `hunyuan-video-foley-xl`; SigLIP2 `a89f5c50`; CLAP `ada0c23a` | HF `tencent/HunyuanVideo-Foley` `3abd4e833b95b8db0fc9c687afc52483a48e9a97` = current head | already current |
| `hunyuan-video-foley-xxl-pytorch-single` | pytorch-pipeline | `hunyuan-video-foley-xxl`; SigLIP2 `a89f5c50`; CLAP `ada0c23a` | HF `tencent/HunyuanVideo-Foley` `3abd4e833b95b8db0fc9c687afc52483a48e9a97` = current head | already current |
| `hunyuan3d-omni-pytorch-single` | pytorch-pipeline | `hunyuan3d-omni`; DINOv2-large `47b73eef` | HF `tencent/Hunyuan3D-Omni` `70e803bfb4e127d534049d8ab8c8cb511780d485` = current head | already current |
| `hunyuanocr-1-5-vllm-dflash-single` | pytorch-pipeline | `hunyuanocr-1-5-47644ecc` | HF `tencent/HunyuanOCR` `47644ecc4fc854efa4f505155158831f36773ee4` = current head | already current |
| `ltx-2-19b-dev-bf16-diffusers-single` | pytorch-pipeline | `ltx-2-19b-dev-bf16`; Gemma encoder `ltx-2-gemma3-text-encoder-dfcc2108` | HF `Lightricks/LTX-2` `dfcc2108383fe1aaa0584bdf55d368a4bdadd90c` = current head | already current |
| `ltx-2-19b-dev-fp4-pytorch-single` | pytorch-pipeline | `ltx-2-19b-dev-fp4-dfcc2108` | HF `Lightricks/LTX-2` `dfcc2108383fe1aaa0584bdf55d368a4bdadd90c` = current head | already current |
| `ltx-2-19b-distilled-diffusers-single` | pytorch-pipeline | `ltx-2-19b-distilled`; Gemma encoder `ltx-2-gemma3-text-encoder-dfcc2108` | HF `Lightricks/LTX-2` `dfcc2108383fe1aaa0584bdf55d368a4bdadd90c` = current head | already current |
| `ltx-2-19b-distilled-fp8-diffusers-single` | pytorch-pipeline | `ltx-2-19b-distilled-fp8`; Gemma encoder `ltx-2-gemma3-text-encoder-dfcc2108` | HF `Lightricks/LTX-2` `dfcc2108383fe1aaa0584bdf55d368a4bdadd90c` = current head | already current |
| `ltx-2-3-22b-distilled-1-1-diffusers-single` | pytorch-pipeline | `ltx-2-3-22b-distilled-1-1`; Gemma encoder `ltx-2-gemma3-text-encoder-dfcc2108` | HF pinned `6b5a83e3045eaf8e46cfa0acce512412aa2b9cce`; current head `5948be4ced3a4493d1f836df64378ff136ddb770` | intentional variant: required distilled and x2-1.1 blobs have identical LFS OIDs; current head only adds other checkpoints and changes README, while this historical single disk-offload closure is self-contained |
| `moss-vl-realtime-11b-pytorch-single` | pytorch-pipeline | `moss-vl-realtime-11b-bf16-25e81cb9` | HF `OpenMOSS-Team/MOSS-VL-Realtime` `25e81cb952d5f353a5690f2c1ea09a725815df80` = current head | already current on main57; the compatible remote-code refresh is already published in baseline commit `9ac07dd` |
| `mova-360p-diffusers-single` | pytorch-pipeline | `mova-360p` | HF `OpenMOSS-Team/MOVA-360p` `eab4aa91d6d5eb515e259a0c8533c90062b117a5` = current head | already current |
| `mova-720p-diffusers-single` | pytorch-pipeline | `mova-720p` | HF `OpenMOSS-Team/MOVA-720p` `8462bea9a317a3591279faeedfd3ca01b055cb59` = current head | already current |
| `pixal3d-pytorch-single` | pytorch-pipeline | `pixal3d`; DINOv3 `3c276edd` and NAF `c096c1ab` | HF pinned `0b31f9160aa400719af409098bff7936a932f726`; current `b0cb2e1b794cab9aa0ac38a95d794a4d9337437f`; GitHub pinned `cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af`, observed current `f7cf38429b0bd264f1995f0f8743a88b1c728b94` | intentional variant: current HF adds multi-view files and pipeline metadata, while the selected single-image files have unchanged OIDs and the package's `trellis2-native` closure emits one GLB from one image |
| `skintokens-pytorch-single` | pytorch-pipeline | `skintokens` | HF `VAST-AI/SkinTokens` `79736cad0fd84de384d5eede659b4ebd24effe33` = current head | already current |
| `step1x-3d-geometry-pytorch-single` | pytorch-pipeline | `step1x-3d-geometry`; DINOv2-registers `e4c89a4e` | HF `stepfun-ai/Step1X-3D` `bf7084495b3a72222f36549b7942948aa4d9daa7` = current head | already current |
| `step1x-3d-label-geometry-pytorch-single` | pytorch-pipeline | `step1x-3d-label-geometry`; DINOv2-registers `e4c89a4e`; CLIP `32bd6428` | HF `stepfun-ai/Step1X-3D` `bf7084495b3a72222f36549b7942948aa4d9daa7` = current head | already current |
| `step1x-3d-texture-pytorch-single` | pytorch-pipeline | `step1x-3d-texture`; SDXL `46216598`; SDXL VAE `207b116d` | HF `stepfun-ai/Step1X-3D` `bf7084495b3a72222f36549b7942948aa4d9daa7` = current head | already current |
| `trellis-2-4b-pytorch-single` | pytorch-pipeline | `trellis-2-4b`; sparse decoder files from `TRELLIS-image-large` `25e0d31f`; DINOv3 `3c276edd` | HF `microsoft/TRELLIS.2-4B` `af44b45f2e35a493886929c6d786e563ec68364d` = current head; GitHub `microsoft/TRELLIS.2` pinned/current `75fbf0183001ed9876c8dbb35de6b68552ee08bd` | intentional variant: model snapshot `2a2c5ab0`, CUDA ARM64 digest `0e1392f4`, and `trellis2-native` one-image GLB fixture form a bounded executable closure; upstream multi-image/portability changes are outside this job |
| `triposg-pytorch-single` | pytorch-pipeline | `triposg`; RMBG `2ceba5a5` | HF `VAST-AI/TripoSG` `2c1c516d22d58db486a058d98d31bb6177344e06` = current head | already current |
| `wan-dancer-14b-disk-offload-pytorch-single` | pytorch-pipeline | `wan-dancer-14b` | HF `Wan-AI/Wan-Dancer-14B` `85ce88dd8d025459dcf0fe93982d6da8b9002957` = current head; DiffSynth pinned `84f93fc4907b6c193be5501bab0b5c37f383033c`, observed current `4dbf980d4d0eb34eda136300dd0d72014cff8965` | intentional variant: model snapshot `01430f10`, `wan-dancer-diffsynth-disk` plus `patch-runtime.py`, and prompt/reference/music/controls -> MP4 fixture define the self-contained CPU/disk staging contract; upstream launcher/NFS changes are not imported |
| `wan-dancer-14b-pytorch-single` | pytorch-pipeline | `wan-dancer-14b` | HF `Wan-AI/Wan-Dancer-14B` `85ce88dd8d025459dcf0fe93982d6da8b9002957` = current head; Wan-Dancer source pinned `e6c87a94ec733230dac15b924c015f6e6501e618`, current repository page checked | intentional variant: the native Spark adapter owns its bounded music/reference/control fixture and does not checkout or launch the upstream shell orchestration |

The following disposition ledger makes the per-ID outcome explicit. `none`
under applied commit means the reviewed executable inputs remain unchanged;
the MOSS baseline commit is cited to show why its earlier refresh is not
reapplied here. Every row has the same checked-at value shown in the heading
and source-audit evidence above.

| recipe ID | current authoritative ref | decision | applied commit | blocker |
|---|---|---|---|---|
| `hunyuan-video-foley-xl-pytorch-single` | HF `tencent/HunyuanVideo-Foley@3abd4e83` | already current | none (review only) | none |
| `hunyuan-video-foley-xxl-pytorch-single` | HF `tencent/HunyuanVideo-Foley@3abd4e83` | already current | none (review only) | none |
| `hunyuan3d-omni-pytorch-single` | HF `tencent/Hunyuan3D-Omni@70e803bf` | already current | none (review only) | none |
| `hunyuanocr-1-5-vllm-dflash-single` | HF `tencent/HunyuanOCR@47644ecc` | already current | none (review only) | none |
| `ltx-2-19b-dev-bf16-diffusers-single` | HF `Lightricks/LTX-2@dfcc2108` | already current | none (review only) | none |
| `ltx-2-19b-dev-fp4-pytorch-single` | HF `Lightricks/LTX-2@dfcc2108` | already current | none (review only) | none |
| `ltx-2-19b-distilled-diffusers-single` | HF `Lightricks/LTX-2@dfcc2108` | already current | none (review only) | none |
| `ltx-2-19b-distilled-fp8-diffusers-single` | HF `Lightricks/LTX-2@dfcc2108` | already current | none (review only) | none |
| `ltx-2-3-22b-distilled-1-1-diffusers-single` | HF `Lightricks/LTX-2.3@5948be4c` | intentional single-closure variant | none (review only) | none; selected LFS OIDs unchanged |
| `moss-vl-realtime-11b-pytorch-single` | HF `OpenMOSS-Team/MOSS-VL-Realtime@25e81cb9` | already current on main57 | `9ac07dd` (baseline) | none |
| `mova-360p-diffusers-single` | HF `OpenMOSS-Team/MOVA-360p@eab4aa91` | already current | none (review only) | none |
| `mova-720p-diffusers-single` | HF `OpenMOSS-Team/MOVA-720p@8462bea9` | already current | none (review only) | none |
| `pixal3d-pytorch-single` | HF `TencentARC/Pixal3D@b0cb2e1b`; GitHub `@f7cf3842` | intentional single-image variant | none (review only) | none; selected files/OIDs unchanged |
| `skintokens-pytorch-single` | HF `VAST-AI/SkinTokens@79736cad` | already current | none (review only) | none |
| `step1x-3d-geometry-pytorch-single` | HF `stepfun-ai/Step1X-3D@bf708449` | already current | none (review only) | none |
| `step1x-3d-label-geometry-pytorch-single` | HF `stepfun-ai/Step1X-3D@bf708449` | already current | none (review only) | none |
| `step1x-3d-texture-pytorch-single` | HF `stepfun-ai/Step1X-3D@bf708449` | already current | none (review only) | none |
| `trellis-2-4b-pytorch-single` | HF `microsoft/TRELLIS.2-4B@af44b45f`; GitHub `@75fbf018` | intentional one-image ARM64 variant | none (review only) | none; native closure retained |
| `triposg-pytorch-single` | HF `VAST-AI/TripoSG@2c1c516d` | already current | none (review only) | none |
| `wan-dancer-14b-disk-offload-pytorch-single` | HF `Wan-AI/Wan-Dancer-14B@85ce88dd`; DiffSynth `@4dbf980d` current | intentional disk-offload variant | none (review only) | none; pinned fork remains executable |
| `wan-dancer-14b-pytorch-single` | HF `Wan-AI/Wan-Dancer-14B@85ce88dd`; Wan-Dancer `@e6c87a94` | intentional native Spark variant | none (review only) | none; specialized adapter retained |

## Executable contract inventory

All rows use the `pytorch-pipeline` engine and one Spark. The following table
records the build/source closure and typed serving fixture. The `base` value is
the digest-pinned `FROM` image in the named context; repeated contexts are
shared and were reviewed once.

| recipe ID | base image digest | build context (digest, bytes) | patch bundle | serving fixture |
|---|---|---|---|---|
| `hunyuan-video-foley-xl-pytorch-single`, `hunyuan-video-foley-xxl-pytorch-single` | CUDA 13.0.1 runtime `36050649ad1a` | `adapters/audio/hunyuan-video-foley-native` (`9beb6e1e`, 20,992,000) | none | prompt `text/plain` + video `mp4/quicktime/webm/matroska` -> WAV |
| `hunyuan3d-omni-pytorch-single` | CUDA 13.2.1 runtime `a52783d8` | `adapters/three-d/hunyuan3d-omni` (`f458c1fe`, 71,680) | none | JSON job + image/control mesh -> GLB |
| `hunyuanocr-1-5-vllm-dflash-single` | vLLM OpenAI `1c8e60a0` | `adapters/ocr/hunyuanocr-1-5-vllm-dflash` (`1233e601`, 20,480) | none | document image + JSON config -> ZIP |
| `ltx-2-19b-dev-bf16-diffusers-single`, `ltx-2-3-22b-distilled-1-1-diffusers-single` | CUDA 13.2.1 runtime `a52783d8` | `adapters/video/ltx23-sync-native-disk` (`ff999ce2`, 20,480) | none | text prompt -> MP4 |
| `ltx-2-19b-dev-fp4-pytorch-single` | CUDA 13.2.1 runtime `a52783d8` | `adapters/video/ltx2-pytorch` (`e6856007`, 20,480) | none | text prompt -> MP4 |
| `ltx-2-19b-distilled-diffusers-single`, `ltx-2-19b-distilled-fp8-diffusers-single` | CUDA 13.2.1 runtime `a52783d8` | `adapters/video/ltx2-sync-native` (`ad4f57a1`, 20,480) | none | text prompt -> MP4 |
| `moss-vl-realtime-11b-pytorch-single` | CUDA 13.0.1 runtime `36050649ad1a` | `adapters/video/moss-vl-realtime` (`af410409`, 102,400) | none | authenticated session JSON + JPEG/PNG/WebP frames -> MP4 + NDJSON |
| `mova-360p-diffusers-single`, `mova-720p-diffusers-single` | CUDA 13.2.1 runtime `a52783d8` | `adapters/video/mova-pytorch` (`5a929c3a`, 30,720) | `mova-inference-only-import` | text prompt + reference image -> MP4 |
| `pixal3d-pytorch-single`, `trellis-2-4b-pytorch-single` | CUDA 13.2.1 devel `0e1392f4` | `adapters/three-d/trellis2-native` (`b4a92759`, 81,920) | `trellis2-pixal3d-sdpa-arm64` | image -> GLB |
| `skintokens-pytorch-single` | CUDA 13.2.1 devel `0e1392f4` | `adapters/three-d/skintokens` (`cce641d7`, 71,680) | none | GLB mesh -> rigged GLB mesh |
| `step1x-3d-geometry-pytorch-single`, `step1x-3d-label-geometry-pytorch-single`, `step1x-3d-texture-pytorch-single` | CUDA 13.2.1 devel `0e1392f4` | `adapters/three-d/step1x-3d` (`64400115`, 81,920) | none | image, plus mesh for texture row -> GLB |
| `triposg-pytorch-single` | CUDA 13.2.1 devel `0e1392f4` | `adapters/three-d/triposg` (`5710927e`, 61,440) | none | image -> GLB |
| `wan-dancer-14b-disk-offload-pytorch-single` | CUDA 13.0.1 runtime `36050649ad1a` | `adapters/video/wan-dancer-diffsynth-disk` (`1f34b5d5`, 3,399,680) | none | prompt + reference image + music + JSON controls -> MP4 |
| `wan-dancer-14b-pytorch-single` | CUDA 13.0.1 runtime `36050649ad1a` | `adapters/video/wan-dancer-native` (`71b98bb6`, 32,081,920) | none | prompt + reference image + music + JSON controls -> MP4 |

## Applied update and validation

There is no new source, model, recipe, release, package, index, generator,
workflow, or upstream-watch change in this domain. The MOSS refresh described
above is already present in main57 via `9ac07dd`; the Pixal3D and all other
intentional single-mode closures remain unchanged because their executable
artifact evidence supports the recorded bounded variants. This commit adds
only this domain report, so it does not duplicate the MOSS/Pixal/PR69/PR70
changes already published before the audit branch.

Validation completed with the repository's current contract/resolver and
focused adapter checks using the configured control environment:

```
PYTHONPATH=contracts/src /opt/vonk-forge/control/.venv/bin/python -m pytest -q \
  tests/test_contracts.py tests/test_contract_examples.py \
  tests/test_recipe_executability.py tests/test_recipe_build_inputs.py \
  tests/test_recipe_package_closure.py tests/test_hunyuan_single_recipes.py \
  tests/test_hunyuanocr_adapter.py tests/test_moss_vl_realtime_job.py \
  tests/test_mova_adapter.py tests/test_three_d_native_adapters.py \
  tests/test_wan_dancer_adapter.py tests/test_wan_dancer_disk_offload_canary.py \
  tests/test_ltx_fp4_adapter.py tests/test_ltx_sync_adapter.py
104 passed, 128 subtests passed in 64.66s
```

The frozen recipe checkout has no standalone project environment, so the
configured platform control environment was used for this validation.
Catalog/index regeneration is intentionally left to the sole integrator. No
physical Spark, container build, or model-quality result is claimed here.
