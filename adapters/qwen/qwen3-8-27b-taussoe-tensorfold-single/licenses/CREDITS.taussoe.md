# Credits

- **[taussoe/tensorfold-recipes](https://github.com/taussoe/tensorfold-recipes)** (MIT, commit
  d8a7c09c893eb41be5e93dbd684ce7d4f127d7ac): the recipe this one follows for one DGX Spark
  (`dgx-spark/qwen3.8-27b`, `NODES=1`): the DFlash2 draft model, automatic context, `--max-tokens 32768`, optional
  `--parallel`, and the measurements quoted in the recipe. Its shell scripts and Dockerfile are not reused: the
  platform builds its own image from the vendored source. That recipe runs TensorFold 0.3.6.3's own 27B code, so this
  one pins the upstream release instead of the taussoe/TensorFold fork.
- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart and the TensorFold contributors (MIT): the
  engine, vendored unmodified at v0.3.6.3 (commit 191188075bca56a7c71074a79375eb4c1cb22e1c), the commit the kit pins.
- **[Vontra](https://huggingface.co/Vontra)**: the MLX 4-bit conversion of **Qwen3.8-27B** by Qwen (Apache-2.0).
- **[z-lab](https://huggingface.co/z-lab)** (mirror of incoai): the Qwen3.8-27B DFlash2 draft model (Apache-2.0 per
  its card).
