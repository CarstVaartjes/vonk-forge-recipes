# Agent notes

This repository serves the official DeepSeek-V4.1-Flash checkpoint on four GB10 nodes, TP4/EP4, with vision and a local file-backed Engram store. Do not substitute checkpoints or transfer scores between image/profile identities.

Read README.md and docs/REPRODUCIBILITY.md. Private topology belongs in ignored cluster.local.json, never source. All heavy lanes serialize; use a durable tmux owner and all-rank memory guards. The overlay-v2 campaign permits no further rebuild. Raw prompts/responses and host records stay private; publish sanitized score/timing ledgers.

All adapter code is written from scratch. Preserve upstream attribution and the no-collective Engram lookup design. Host-side publication portability changes do not change or rebuild the qualified runtime image.
