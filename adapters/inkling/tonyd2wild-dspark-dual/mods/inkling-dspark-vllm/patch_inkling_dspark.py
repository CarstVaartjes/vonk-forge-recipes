#!/usr/bin/env python3
"""Add vLLM DSpark auxiliary-hidden-state support to Inkling."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

MARKER = "# spark-vllm mod: inkling-dspark-vllm v1"

INTERFACE_IMPORT = """from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)"""

INTERFACE_IMPORT_PATCHED = f"""from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)

{MARKER}"""

MODEL_CLASS = "class InklingModel(nn.Module):"
MODEL_CLASS_PATCHED = "class InklingModel(nn.Module, EagleModelMixin):"

BASE_CLASS = "class _TmlForCausalLMBase(nn.Module, SupportsPP, SupportsLoRA):"
BASE_CLASS_PATCHED = (
    "class _TmlForCausalLMBase("
    "nn.Module, SupportsEagle3, SupportsPP, SupportsLoRA):"
)

LOOP_ANCHOR = """        pending: tuple[torch.Tensor | None, InklingShortConv] | None = None
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, pending = layer(
                positions,
                hidden_states,
                pending=pending,
                defer_mlp_add=True,
                attn_in=attn_in0,
                log_scaling=log_scaling,
            )
            attn_in0 = None
"""

LOOP_PATCHED = """        pending: tuple[torch.Tensor | None, InklingShortConv] | None = None
        aux_hidden_states: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer],
            start=self.start_layer,
        ):
            hidden_states, pending = layer(
                positions,
                hidden_states,
                pending=pending,
                defer_mlp_add=True,
                attn_in=attn_in0,
                log_scaling=log_scaling,
            )
            attn_in0 = None

            # Inkling normally carries the MLP short-convolution delta into
            # the next layer so the residual add and next RMSNorm can fuse.
            # At DSpark tap points, materialize the logical post-layer state
            # before publishing it to the draft model. Non-tap layers retain
            # the original fused path.
            if (layer_idx + 1) in self.aux_hidden_state_layers:
                assert pending is not None
                hidden_states = _sconv_add_norm(
                    pending[0], hidden_states, pending[1], None, positions
                )[1]
                pending = None
                aux_hidden_states.append(hidden_states)
"""

RETURN_ANCHOR = """        if pending is not None:
            # Final RS/sconv/AG + residual add fused with the final rmsnorm.
            norm_out = _sconv_add_norm(
                pending[0], hidden_states, pending[1], self.norm, positions
            )[0]
            assert norm_out is not None
            return norm_out
        return self.norm(hidden_states)
"""

RETURN_PATCHED = """        if pending is not None:
            # Final RS/sconv/AG + residual add fused with the final rmsnorm.
            norm_out = _sconv_add_norm(
                pending[0], hidden_states, pending[1], self.norm, positions
            )[0]
            assert norm_out is not None
        else:
            norm_out = self.norm(hidden_states)
        if aux_hidden_states:
            return norm_out, aux_hidden_states
        return norm_out
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected exactly one {label}; found {count}")
    return text.replace(old, new, 1)


def validate(text: str) -> None:
    tree = ast.parse(text)
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    for required in ("InklingModel", "_TmlForCausalLMBase"):
        if required not in classes:
            raise ValueError(f"unexpected Inkling module; missing {required}")
    compile(text, "<inkling-dspark-vllm>", "exec")


def patched_text(text: str) -> str:
    validate(text)
    if MARKER in text:
        required = (
            "EagleModelMixin",
            "SupportsEagle3",
            "aux_hidden_states.append(hidden_states)",
        )
        missing = [item for item in required if item not in text]
        if missing:
            raise ValueError(
                "mod marker exists but required changes are missing: "
                + ", ".join(missing)
            )
        return text

    text = replace_once(
        text, INTERFACE_IMPORT, INTERFACE_IMPORT_PATCHED, "interfaces import"
    )
    text = replace_once(text, MODEL_CLASS, MODEL_CLASS_PATCHED, "InklingModel class")
    text = replace_once(
        text, BASE_CLASS, BASE_CLASS_PATCHED, "causal LM base class"
    )
    text = replace_once(text, LOOP_ANCHOR, LOOP_PATCHED, "decoder loop")
    text = replace_once(text, RETURN_ANCHOR, RETURN_PATCHED, "final return block")
    validate(text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not args.target.is_file():
        print(f"[inkling-dspark-vllm ERROR] target not found: {args.target}", file=sys.stderr)
        return 1

    original = args.target.read_text()
    try:
        patched = patched_text(original)
    except (SyntaxError, ValueError) as exc:
        print(f"[inkling-dspark-vllm ERROR] {exc}", file=sys.stderr)
        return 1

    if args.check:
        state = "already patched" if patched == original else "compatible"
        print(f"[inkling-dspark-vllm] {args.target} is {state}.")
        return 0

    if patched == original:
        print("[inkling-dspark-vllm] already patched; skipping.")
        return 0

    temporary = args.target.with_suffix(args.target.suffix + ".dspark.tmp")
    temporary.write_text(patched)
    temporary.replace(args.target)
    print(f"[inkling-dspark-vllm] Patched {args.target}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
