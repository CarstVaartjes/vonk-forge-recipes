# Upstream provenance

Source: https://github.com/r0b0tlab/glm53-flash-exl3-dflash2-sm121 at commit
2dfbdbe4840682ea1892bb1300e10d11d2ff4758 (2026-09-27), MIT.

This directory holds the upstream files the image build needs, unmodified:
`src/`, `csrc/`, `vendored/` (turboderp-org/exllamav3 v1.5.0 kernels, MIT),
`pyproject.toml`, `runtime.lock.json`, `LICENSE`,
`THIRD_PARTY_NOTICES.md`, `provenance.md`, `README.md`, and vLLM patches
`patches/0007`..`0012`. Patch `0006` is intentionally absent: upstream
documents it as a strict subset of `0009`, and applying it first makes `0009`
fail. Upstream ships no container image and measured on a host-native vLLM
v0.30.0rc1 (a00a3544b93e) source build; this recipe applies the same patches to
the digest-pinned vLLM v0.30.0 (ced6857afa0e) image. The patches apply cleanly
to that tree (checked with `git apply --check`).

Upstream's `tests/` are not shipped (they need the built plugin on the path).
