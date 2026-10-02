# Credits

- **[taussoe/tensorfold-recipes](https://github.com/taussoe/tensorfold-recipes)** (MIT, commit
  d8a7c09c893eb41be5e93dbd684ce7d4f127d7ac): the recipe this one follows for one DGX Spark
  (`dgx-spark/qwen3.8-27b`, `NODES=1`): the DFlash2 draft model, automatic context, `--max-tokens 32768`, optional
  `--parallel`, and the measurements quoted in the recipe. Its shell scripts and Dockerfile are not reused: the
  platform builds its own image from the vendored source. The kit builds the taussoe/TensorFold fork branch
  on a Spark by default, so this recipe follows the fork too.
- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart and the TensorFold contributors (MIT): the
  engine, vendored unmodified from the creator's fork branch glm-long-context (commit c32abe92411eced43565a63f5be7fe3595f8e22c), the engine the kit builds on a Spark by default.
- **[Vontra](https://huggingface.co/Vontra)**: the MLX 4-bit conversion of **Qwen3.8-27B** by Qwen (Apache-2.0).
- **[z-lab](https://huggingface.co/z-lab)** (mirror of incoai): the Qwen3.8-27B DFlash2 draft model (Apache-2.0 per
  its card).
