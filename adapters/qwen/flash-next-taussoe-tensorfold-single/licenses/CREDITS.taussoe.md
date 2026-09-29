# Credits

- **[taussoe/tensorfold-recipes](https://github.com/taussoe/tensorfold-recipes)** (MIT, commit
  d8a7c09c893eb41be5e93dbd684ce7d4f127d7ac): the recipe this one follows for one DGX Spark: the served settings
  (automatic context, `--max-tokens 32768`, optional `--parallel`) and the measurements quoted in the recipe.
  Its shell scripts and Dockerfile are not reused: the platform builds its own image from the vendored source.
- **[taussoe/TensorFold](https://github.com/taussoe/TensorFold)**, branch `glm-long-context` at
  c32abe92411eced43565a63f5be7fe3595f8e22c (MIT): a fork of TensorFold 0.3.6.3 with GLM long-context work and, for
  Flash Next, agent-turn resume. Vendored unmodified.
- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart and the TensorFold contributors (MIT): the
  engine the fork builds on.
- **[Vontra](https://huggingface.co/Vontra)**: the MLX 4-bit checkpoint with the preserved MTP head
  (`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`) of **Qwen3.8 Flash Next** by Qwen (Qwen Community License 1.0).
