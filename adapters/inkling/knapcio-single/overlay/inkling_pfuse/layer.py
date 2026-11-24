"""Patch of Inkling2DecoderLayer._forward for prefill / extend chunks.

Stock (sglang/srt/models/inkling2.py), per layer visit at an 8192-token chunk:
    attention input  attn_res(state) -> FP32 [Q, T, D]; passes 2-3 of the loop: attn_res(bank) -> FP32 [3, T, D],
                     message_controllers (norm -> BF16 [3, T, 2D], down, mix -> FP32 [3, T, D]); RMSNorm -> BF16
    mlp input        attn_res(state, partial + attention) -> FP32; passes 2-3: attn_res(bank), message_controllers;
                     RMSNorm -> BF16
Every FP32 [T, D] intermediate is 75 MB written and read back at least once.

Levers (INKLING_PFUSE):
  resnorm  visits without loop messages: attn_res + RMSNorm in one kernel (mix.resnorm); the FP32 mix is never stored
  loopmix  visits with loop messages (core layers, passes 2-3): mix.loop_mix; the attention side keeps its final
           RMSNorm inside the mix kernel (NORM=True); the mlp side stores the FP32 mix output and runs the stock
           mlp_norm on it (fewer bytes than recomputing the mix twice for one controller)

Everything else in _forward is the stock code. The patched method serves only EXTEND / MIXED batches with at least
`min_rows` rows and the expected dtypes / layouts; every other call (decode, target verify, small chunks, anything
unexpected) runs the stock method. Python-level branching only, on shapes and modes that are fixed per captured
CUDA graph.
"""

import torch

TARGET_MODULE = "sglang.srt.models.inkling2"
MARK = "_inkling_pfuse_forward"

CFG = {"levers": (), "min_rows": 512, "loop_norm": "fused"}
STATS = {"fused_calls": 0, "stock_calls": 0}


_MODES = ()


def _eligible(layer, state, forward_batch):
    mode = forward_batch.forward_mode
    if mode not in _MODES:
        return False
    partial = state.partial
    if partial.dim() != 2 or partial.shape[0] < CFG["min_rows"] or partial.dtype != torch.bfloat16:
        return False
    if partial.shape[1] != 2304 or not state.blocks:
        return False
    norms = [layer.mlp_norm] + (
        [layer.attn_norm] if layer.is_local else [layer.q_input_norm, layer.k_input_norm, layer.v_input_norm]
    )
    return all(
        n.pre_weight is None
        and n.weight is not None
        and n.weight.shape == (2304,)
        and n.weight.is_contiguous()
        and n.weight.device == partial.device
        and n.output_dtype == torch.bfloat16
        and n.eps == norms[0].eps
        for n in norms
    )


def _state_query(attn_res, kind, layer_idx):
    if kind == "attn":
        raw = attn_res.attn_query(layer_idx)
    else:
        raw = attn_res.mlp_queries[layer_idx]
    scaled = attn_res.scaled_query(kind, layer_idx)
    q = scaled.view(raw.shape) if scaled is not None else raw.float() * (raw.shape[-1] ** -0.5)
    return (q.unsqueeze(0) if q.ndim == 1 else q).contiguous()


def _fused_attention_input(self, state, messages, mix):
    """Stock state.attention_input + _attention_input for a visit with blocks; None = not handled."""
    attn_res = state.attn_res
    sources = [*state.blocks, state.partial]
    queries = _state_query(attn_res, "attn", self.layer_id)
    if not mix._check_sources(sources):
        return None
    if messages is None:
        if "resnorm" not in CFG["levers"]:
            return None
        norms = (self.attn_norm,) if self.is_local else (self.q_input_norm, self.k_input_norm, self.v_input_norm)
        out = mix.resnorm(sources, None, queries, [n.weight for n in norms], norms[0].eps,
                          attn_res.normalize_values)
        if self.is_local:
            return out[0]
        return out
    if "loopmix" not in CFG["levers"]:
        return None
    layer = self.layer_id - self.config.loop_core_layers.start
    down, gain, forget = (w[0:3] for w in self.controller_weights)
    bank = messages.blocks
    if len(bank) != mix.NBP or not mix._check_sources(bank) or bank[0].shape != state.partial.shape:
        return None
    norms = (self.attn_norm,) * 3 if self.is_local else (self.q_input_norm, self.k_input_norm, self.v_input_norm)
    return mix.loop_mix(sources, None, queries, attn_res.normalize_values, bank,
                        messages.scaled_queries[layer, 0:3], down, gain, forget,
                        norm_weights=[n.weight for n in norms], eps_norm=norms[0].eps,
                        fuse_norm=CFG["loop_norm"] == "fused")


def _fused_mlp_input(self, state, attention, messages, mix):
    """Stock state.mlp_input (+ messages.mix) + mlp_norm; None = not handled."""
    attn_res = state.attn_res
    if attn_res.is_block_boundary(self.layer_id):
        sources, addend = [*state.blocks, state.partial, attention], None
    else:
        sources, addend = [*state.blocks, state.partial], attention
    if not mix._check_sources(sources, (addend,)):
        return None
    queries = _state_query(attn_res, "mlp", self.layer_id)
    if messages is None:
        if "resnorm" not in CFG["levers"]:
            return None
        return mix.resnorm(sources, addend, queries, [self.mlp_norm.weight], self.mlp_norm.eps,
                           attn_res.normalize_values)[0]
    if "loopmix" not in CFG["levers"]:
        return None
    layer = self.layer_id - self.config.loop_core_layers.start
    down, gain, forget = (w[3:4] for w in self.controller_weights)
    bank = messages.blocks
    if len(bank) != mix.NBP or not mix._check_sources(bank) or bank[0].shape != state.partial.shape:
        return None
    mixed = mix.loop_mix(sources, addend, queries, attn_res.normalize_values, bank,
                         messages.scaled_queries[layer, 3:4], down, gain, forget, norm_weights=None)
    return self.mlp_norm(mixed[0])


def _sconv_residual_ok(sconv, mlp, attention, state, mix):
    return (
        not sconv.is_dynamic
        and sconv.use_residual
        and isinstance(mlp, torch.Tensor)
        and mlp.dim() == 2
        and mlp.shape[0] >= mix.sconv_min_rows()
        and mlp.dtype == attention.dtype == state.partial.dtype == torch.bfloat16
        and mlp.stride(-1) == 1
        and mlp.shape == attention.shape == state.partial.shape
        and attention.is_contiguous()
        and state.partial.is_contiguous()
    )


def _sconv_residual(self, state, attention, mlp, forward_batch, mix, update_sconv_cache):
    """Stock ShortConvolution.forward (extend, static weights) + Inkling2AttnResState.finalize_layer, with the conv
    output and the residual update in one kernel."""
    sconv = self.mlp_sconv
    meta = sconv._conv_state(forward_batch)
    cache = sconv._sconv_cache()
    sconv._prepare_extend_sconv_cache(forward_batch, cache, mlp, meta.track_conv_indices)
    boundary = state.attn_res.is_block_boundary(self.layer_id)
    new = mix.sconv_residual(mlp, sconv._weight_2d(), cache, meta.precomputed, attention,
                             None if boundary else state.partial)
    update_sconv_cache(
        x=mlp,
        sconv_cache=cache,
        cache_indices=meta.cache_indices,
        has_initial_state=meta.has_initial_state,
        query_start_loc=meta.query_start_loc,
    )
    if boundary:
        state.blocks.append(state.partial)
    state.partial = new


def make_forward(module, stock_forward):
    from sglang.srt.model_executor.forward_mode import ForwardMode

    from . import mix

    global _MODES
    _MODES = (ForwardMode.EXTEND, ForwardMode.MIXED)

    routed_output = module.RoutedOutput
    verify_max = module.VERIFY_ROUTED_EPILOGUE_MAX_TOKENS

    def _forward(self, state, positions, forward_batch, pass_idx, messages):
        if not _eligible(self, state, forward_batch):
            STATS["stock_calls"] += 1
            return stock_forward(self, state, positions, forward_batch, pass_idx, messages)
        STATS["fused_calls"] += 1
        # ---- stock: attn_input = self._attention_input(state.attention_input(self.layer_id), messages)
        attn_input = _fused_attention_input(self, state, messages, mix)
        if attn_input is None:
            attn_input = self._attention_input(state.attention_input(self.layer_id), messages)
        elif not isinstance(attn_input, torch.Tensor) or attn_input.dim() == 3:
            attn_input = module.AttentionSources(*attn_input.unbind(0))
        attention = self.attn(attn_input, positions, forward_batch, pass_idx=pass_idx)
        attention = self.attn_sconv(attention, positions, forward_batch)
        # ---- stock: mlp_input = state.mlp_input(...); messages.mix(...); self.mlp_norm(...)
        mlp_input = _fused_mlp_input(self, state, attention, messages, mix)
        if mlp_input is None:
            mlp_input = state.mlp_input(self.layer_id, attention)
            if messages is not None:
                mlp_input = messages.mix(
                    self.layer_id - self.config.loop_core_layers.start,
                    (mlp_input,),
                    self.controller_weights,
                    "mlp",
                )[0]
            mlp_input = self.mlp_norm(mlp_input)
        # ---- stock from here on
        routed = forward_batch.forward_mode.is_decode() or (
            forward_batch.forward_mode.is_target_verify()
            and attention.shape[0] <= verify_max
        )
        mlp = self.mlp(mlp_input, forward_batch, combine=not routed)
        if isinstance(mlp, torch.Tensor) and (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        ):
            mlp = routed_output(mlp, None, 0)
        if isinstance(mlp, routed_output):
            state.finalize_layer_with(
                self.layer_id,
                lambda partial: self.mlp_sconv.routed_decode_epilogue(
                    mlp, attention, partial, forward_batch
                ),
            )
            return
        if "sconvres" in CFG["levers"] and _sconv_residual_ok(self.mlp_sconv, mlp, attention, state, mix):
            _sconv_residual(self, state, attention, mlp, forward_batch, mix, update_sconv_cache)
            return
        mlp = self.mlp_sconv(mlp, positions, forward_batch)
        state.finalize_layer(self.layer_id, attention, mlp)

    from sglang.srt.models.inkling_common.kernels.sconv import update_sconv_cache

    _forward.__wrapped__ = stock_forward
    setattr(_forward, MARK, True)
    return _forward


def patch(module):
    cls = getattr(module, "Inkling2DecoderLayer", None)
    if cls is None:
        raise RuntimeError("inkling_pfuse: Inkling2DecoderLayer is missing")
    for name in ("RoutedOutput", "VERIFY_ROUTED_EPILOGUE_MAX_TOKENS", "AttentionSources"):
        if not hasattr(module, name):
            raise RuntimeError(f"inkling_pfuse: {TARGET_MODULE}.{name} is missing")
    stock = cls.__dict__.get("_forward")
    if stock is None or getattr(stock, MARK, False):
        raise RuntimeError("inkling_pfuse: Inkling2DecoderLayer._forward is missing or already patched")
    cls._forward = make_forward(module, stock)
