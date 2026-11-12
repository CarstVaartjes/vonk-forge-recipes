"""CPU smoke checks for the pinned ComfyUI source and Wan runtime APIs."""

import runpy
import sys

# model_management may query the current device unless ComfyUI is put in CPU mode.
sys.argv.insert(1, "--cpu")

import torch
import comfy.options

# ComfyUI intentionally parses command-line flags only when its entry point
# opts in. Enable that same behavior before importing cli_args or model modules.
comfy.options.enable_args_parsing()
from comfy.cli_args import args

assert args.cpu

from comfy.ldm.modules.attention import (
    AttentionTensorContainer,
    ComfyAttention,
    optimized_attention,
)
from comfy.ldm.wan.model import WanSelfAttention
import comfy.quant_ops

# Run ComfyUI's own RGBA channel-count regression from this pinned source tree.
runpy.run_path("tests-unit/comfy_test/qwen_vl_image_channels_test.py")[
    "test_extra_channels_are_dropped_before_patching"
]()

# Match the Wan path: tensor containers, preferred Comfy attention, and CK AdaLN.
q = torch.randn(1, 2, 4)
k = torch.randn(1, 2, 4)
v = torch.randn(1, 2, 4)
output = optimized_attention(
    AttentionTensorContainer(q),
    AttentionTensorContainer(k),
    AttentionTensorContainer(v),
    heads=1,
    preferred_attention=ComfyAttention(),
)
assert output.shape == q.shape
assert callable(comfy.quant_ops.ck.adaln)
assert WanSelfAttention is not None
