# Credits

- **[tournierjc/Swift-1.5-TensorFold-Single-DGX-Spark](https://github.com/tournierjc/Swift-1.5-TensorFold-Single-DGX-Spark)**
  (MIT, commit b695b88f786977428246dcc3e0d97293caca2c34): the single-Spark rig this recipe follows. It supplied the
  serve flags and their measured defaults (`--mtp-drafts 5`, `--mtp-confidence`, `--no-thinking`, the context and
  `--parallel` guidance) and the finding that `--ple-on-ssd` is refused for an NVFP4 checkpoint. The repository's
  Dockerfile and scripts are not reused: the platform builds its own image from the vendored TensorFold source.
- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart and the TensorFold contributors (MIT): the
  engine, vendored unmodified at v0.4.0 (commit 7a00336b2f6d1a1a3e0ba49d1ef5b4b82e927759). The rig's own text pins
  0.3.6.3 (commit 191188075bca56a7c71074a79375eb4c1cb22e1c), which v0.4.0 contains.
- **[UkisAI](https://ukisai.com)**: the Swift 1.5 Qwen3.8 Flash Next NVFP4 checkpoint
  (`ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4`), licensed under the Swift Open License v1.0 on top of the Qwen
  Community License 1.0 of the base model **Qwen3.8 Flash Next** by Qwen. The weights are not part of this image.
