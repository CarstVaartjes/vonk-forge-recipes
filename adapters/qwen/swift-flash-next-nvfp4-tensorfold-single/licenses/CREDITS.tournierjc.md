# Credits

- **[tournierjc/Swift-1.5-TensorFold-Single-DGX-Spark](https://github.com/tournierjc/Swift-1.5-TensorFold-Single-DGX-Spark)**
  (MIT, commit d956df5fea1db1fc9b5ffcd46f627d88a1be0b68): the single-Spark rig this recipe follows. It supplied the
  serve flags and their measured defaults (three lanes, the full window, `--vision`, `--kv-dtype int8`, `--mtp-confidence 0.60`,
  `TENSORFOLD_FACES_FP8=all`, `TENSORFOLD_VISION_WORKSPACE_MIB=1280`) and the finding that `--ple-on-ssd` is refused for an NVFP4 checkpoint. The repository's
  Dockerfile and scripts are not reused: the platform builds its own image from the vendored TensorFold source.
- **[tournierjc/TensorFold](https://github.com/tournierjc/TensorFold)** (Apache-2.0), branch `integration/0.6.1`, commit
  808767fd479c6bd8dbb2eb68f2a3537f75e6d520: the engine the rig's Dockerfile pins (`ARG TF_REPO`, `ARG TF_REF`), vendored
  unmodified. It is **[TensorFold](https://github.com/ashhart/TensorFold)** v0.6.1 by Ash Hart and the TensorFold
  contributors (Apache-2.0 from 0.6.0; earlier releases MIT) plus the rig's five changes: 8-bit decode faces, video
  input, the 80,014-id MTP draft vocabulary, PLE row prefetch and a dormant 12-bit decode-face prototype.
- **[UkisAI](https://ukisai.com)**: the Swift 1.5 Qwen3.8 Flash Next NVFP4 checkpoint
  (`ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4`), licensed under the Swift Open License v1.0 on top of the Qwen
  Community License 1.0 of the base model **Qwen3.8 Flash Next** by Qwen. The weights are not part of this image.
