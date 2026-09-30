# Responsible Use & User Agreement

**WARNING:** The compiled stack uses **uncensored / abliterated** weights (AEON-7 Ultimate Uncensored) plus a **cybersecurity-unlock** default chat template. That is useful for red-teaming, security research, evaluation, and unfiltered assistant tasks — and it **removes guardrails you must therefore supply yourself**.

By cloning this repo, pulling the image, downloading linked weights, requesting Hugging Face access, or running the serve recipe, **you agree** to the terms below.

## Intended use

Typical allowed research / evaluation uses include:

- Red-teaming and safety evaluation of refusals / jailbreaks
- Security research and adversarial testing
- Offline evaluation and benchmarking
- Local / self-hosted unfiltered assistant experiments **with your own filters**

You must describe your **reason for intended use** when requesting Hugging Face access.

## Access request fields (Hugging Face gate)

| Field | Description |
|---|---|
| **Username** | Your name or handle |
| **Email** | Contact email |
| **Reason for intended use** | e.g. red-teaming, security research, evaluation, local assistant |

Plus all agreement checkboxes under **Prohibited uses**.

## Prohibited uses

1. **Anything involving the sexual exploitation or endangerment of minors.**
2. **You must be of age 18 years or older** to use and download this stack.
3. **You agree any information generated that can cause harm** in terms of generating recipe, knowledge to make any materials/substances **is your own input and responsibility**. You **will be accountable for any harm/damage caused by your action/input**.
4. **Content promoting self-harm or suicide.**
5. **Generation of material that is illegal in your jurisdiction**, or that targets real individuals for harassment, doxxing, or fraud.
6. **Any use prohibited by the upstream Qwen / Apache-2.0 licenses.**

You are responsible for adding appropriate **safety filtering**, **human review**, and **access controls** for your deployment. Everything here is provided **as-is, with no warranty**. Licenses are inherited from the upstream community projects — review and comply with them before use or redistribution.

This repository does **not** re-host the 20 GB NVFP4 body or the DFlash2 drafter. Those stay at their original Hugging Face repos. This repository is a **compile**: recipe, overlays, measured eval, and a retag of the community prebuilt runtime image.
