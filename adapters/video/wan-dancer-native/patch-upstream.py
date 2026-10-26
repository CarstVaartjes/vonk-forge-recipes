"""Fail-closed build adaptation of the official Wan-Dancer inference source."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    value = path.read_text(encoding="utf-8")
    count = value.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one patch target, found {count}: {old!r}")
    path.write_text(value.replace(old, new), encoding="utf-8")


def replace_all(path: Path, old: str, new: str, expected: int) -> None:
    value = path.read_text(encoding="utf-8")
    count = value.count(old)
    if count != expected:
        raise SystemExit(
            f"{path}: expected {expected} patch targets, found {count}: {old!r}"
        )
    path.write_text(value.replace(old, new), encoding="utf-8")


def patch_stage(path: Path, model_name: str) -> None:
    replace_once(
        path,
        "from diffsynth import save_video",
        "from diffsynth.data.video import save_video",
    )
    replace_once(
        path, '    assert world_size == 8, "WORLD_SIZE must be 8"\n', ""
    ) if model_name == "global_model.safetensors" else None
    replace_all(path, "        use_usp=True,", "        use_usp=False,", 1)
    replace_all(
        path,
        "    if dist.get_rank() == 0:",
        "    if True:",
        2 if model_name == "global_model.safetensors" else 3,
    )
    barrier_count = 1 if model_name == "global_model.safetensors" else 3
    replace_all(
        path,
        "    dist.barrier(device_ids=[dist.get_rank()])",
        "    # Single-Spark execution does not initialize a process group.",
        barrier_count,
    )

    replacements = {
        f'ModelConfig(\n                model_id="Wan-AI/Wan-Dancer-14B",\n                origin_file_pattern="{model_name}",\n                offload_device="cpu",\n            )': f'ModelConfig(path="/models/{model_name}", offload_device="cuda")',
        'ModelConfig(\n                model_id="Wan-AI/Wan-Dancer-14B",\n                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",\n                offload_device="cpu",\n            )': 'ModelConfig(path="/models/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cuda")',
        'ModelConfig(\n                model_id="Wan-AI/Wan-Dancer-14B",\n                origin_file_pattern="Wan2.1_VAE.pth",\n                offload_device="cpu",\n            )': 'ModelConfig(path="/models/Wan2.1_VAE.pth", offload_device="cuda")',
        'ModelConfig(\n                model_id="Wan-AI/Wan-Dancer-14B",\n                origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",\n                offload_device="cpu",\n            )': 'ModelConfig(path="/models/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cuda")',
        'ModelConfig(\n            model_id="Wan-AI/Wan-Dancer-14B", origin_file_pattern="google/umt5-xxl/"\n        )': 'ModelConfig(path="/models/google/umt5-xxl")',
    }
    if model_name == "local_model.safetensors":
        replacements = {
            'ModelConfig(model_id="Wan-AI/Wan-Dancer-14B", \n                        origin_file_pattern="local_model.safetensors", \n                        offload_device="cpu")': 'ModelConfig(path="/models/local_model.safetensors", offload_device="cuda")',
            'ModelConfig(model_id="Wan-AI/Wan-Dancer-14B", \n                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", \n                        offload_device="cpu")': 'ModelConfig(path="/models/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cuda")',
            'ModelConfig(model_id="Wan-AI/Wan-Dancer-14B", \n                        origin_file_pattern="Wan2.1_VAE.pth", \n                        offload_device="cpu")': 'ModelConfig(path="/models/Wan2.1_VAE.pth", offload_device="cuda")',
            'ModelConfig(model_id="Wan-AI/Wan-Dancer-14B", \n                        origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", \n                        offload_device="cpu")': 'ModelConfig(path="/models/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cuda")',
            'tokenizer_config=ModelConfig(model_id="Wan-AI/Wan-Dancer-14B",origin_file_pattern="google/umt5-xxl/")': 'tokenizer_config=ModelConfig(path="/models/google/umt5-xxl")',
        }
    for old, new in replacements.items():
        replace_once(path, old, new)
    replace_once(path, "    pipe.enable_vram_management()\n", "")

    replace_once(
        path,
        '    music_folder = "outputs/tmp_results/" + final_name + "_" + str(time_name)',
        '    music_folder = os.path.join(os.environ["VONK_WORK_DIR"], "tmp_results", final_name + "_" + str(time_name))',
    ) if model_name == "global_model.safetensors" else replace_once(
        path,
        "    music_folder = 'outputs/tmp_results/' + final_name + '_' + str(time_name)",
        "    music_folder = os.path.join(os.environ['VONK_WORK_DIR'], 'tmp_results', final_name + '_' + str(time_name))",
    )


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    if not (root / "LICENSE").is_file():
        raise SystemExit("Wan-Dancer source root is incomplete")

    # Import only the model-specific modules. Upstream's aggregate imports load
    # unrelated pipelines and make optional dependencies mandatory.
    (root / "diffsynth/__init__.py").write_text(
        '"""Wan-Dancer runtime package."""\n', encoding="utf-8"
    )
    (root / "diffsynth/pipelines/__init__.py").write_text(
        '"""Wan-Dancer pipelines."""\n', encoding="utf-8"
    )
    (root / "diffsynth/prompters/__init__.py").write_text(
        "from .wan_prompter import WanPrompter\n", encoding="utf-8"
    )

    pipeline = root / "diffsynth/pipelines/wan_video_new.py"
    replace_once(
        pipeline,
        "from modelscope import snapshot_download",
        "def snapshot_download(*args, **kwargs):\n    raise RuntimeError('network model loading is disabled')",
    )
    replace_once(
        pipeline,
        "import xfuser\nfrom xfuser.core.distributed import get_sp_group",
        "xfuser = None\n\ndef get_sp_group():\n    raise RuntimeError('sequence parallelism is disabled for the single-Spark runtime')",
    )
    # model_fn_wan_video imports get_sequence_parallel_world_size only when
    # use_unified_sequence_parallel is set, so the name is local to the function
    # and its one unguarded read (the music-injection length) raises
    # UnboundLocalError on every run without sequence parallelism. Without it
    # the world size is 1.
    replace_once(
        pipeline,
        "            N = int(float(frame_num * 8) / get_sequence_parallel_world_size()) * get_sequence_parallel_world_size()\n",
        "            N = frame_num * 8\n"
        "            if use_unified_sequence_parallel:\n"
        "                N = int(float(N) / get_sequence_parallel_world_size()) * get_sequence_parallel_world_size()\n",
    )

    dit = root / "diffsynth/models/wan_video_dit.py"
    replace_once(
        dit,
        "from xfuser.core.distributed import (get_sequence_parallel_rank,\n                                     get_sequence_parallel_world_size,\n                                     get_sp_group)\nfrom yunchang import LongContextAttention",
        "def _single_spark_only(*args, **kwargs):\n    raise RuntimeError('sequence parallelism is disabled for the single-Spark runtime')\n\nget_sequence_parallel_rank = _single_spark_only\nget_sequence_parallel_world_size = _single_spark_only\nget_sp_group = _single_spark_only\nLongContextAttention = _single_spark_only",
    )

    # The music encoder's attention is the only DiT layer pinned to CUDA at
    # construction. That made the WanModel constructor the process's first
    # CUDA use; the model is moved to its device after its weights load.
    replace_once(
        dit,
        "                        rotary=rotary,\n                        device='cuda',\n",
        "                        rotary=rotary,\n",
    )

    # Upstream builds every model as real FP32 tensors on CPU (about 69 GB for
    # the 17.2 B-parameter DiT) while its BF16 state dict is also resident, all
    # in GB10 unified memory. Build on meta and take every parameter from the
    # checkpoint; a parameter the checkpoint lacks fails with its name instead
    # of loading with random initialization (the loader is strict=False).
    manager = root / "diffsynth/models/model_manager.py"
    replace_once(
        manager,
        '        with init_weights_on_device(device = torch.device("cpu")):',
        '        with init_weights_on_device(device=torch.device("meta")):',
    )
    replace_once(
        manager,
        "        model.load_state_dict(model_state_dict, assign=True, strict=False)\n"
        "        model = model.to(dtype=torch_dtype, device=device)\n",
        "        model.load_state_dict(model_state_dict, assign=True, strict=False)\n"
        "        unloaded = [name for name, parameter in model.named_parameters() if parameter.is_meta]\n"
        "        if unloaded:\n"
        "            raise RuntimeError(f'{model_name}: checkpoint has no tensor for parameters {unloaded}')\n"
        "        model = model.to(dtype=torch_dtype, device=device)\n",
    )

    downloader = root / "diffsynth/models/downloader.py"
    replace_once(
        downloader,
        "from modelscope import snapshot_download",
        "def snapshot_download(*args, **kwargs):\n    raise RuntimeError('network model loading is disabled')",
    )

    patch_stage(root / "gen_video/gen_video_global.py", "global_model.safetensors")
    patch_stage(root / "gen_video/gen_video_local.py", "local_model.safetensors")

    # Upstream parses --cfg_scale as int, so any fractional guidance scale
    # (including the adapter's 5.0 default) exits before loading a model. The
    # pipelines take a float guidance scale.
    for stage in ("gen_video_global.py", "gen_video_local.py"):
        replace_once(
            root / "gen_video" / stage,
            '        "--cfg_scale",\n        type=int,',
            '        "--cfg_scale",\n        type=float,',
        )

    # Preserve the signed local-stage CFG argument instead of silently dropping it.
    local = root / "gen_video/gen_video_local.py"
    replace_once(
        local,
        "              enable_skip_layer=False, sigma_shift=5, num_inference_steps=48):",
        "              enable_skip_layer=False, sigma_shift=5, num_inference_steps=48, cfg_scale=5):",
    )
    replace_once(
        local,
        "    input_config['num_inference_steps'] = num_inference_steps",
        "    input_config['num_inference_steps'] = num_inference_steps\n    input_config['cfg_scale'] = cfg_scale",
    )
    replace_once(
        local,
        "                    num_inference_steps=args.num_inference_steps)",
        "                    num_inference_steps=args.num_inference_steps, cfg_scale=args.cfg_scale)",
    )

    for stage in (root / "gen_video/gen_video_global.py", local):
        value = stage.read_text(encoding="utf-8")
        forbidden = (
            'model_id="Wan-AI/Wan-Dancer-14B"',
            "use_usp=True",
            "dist.get_rank()",
        )
        present = [item for item in forbidden if item in value]
        if present:
            raise SystemExit(f"{stage}: unsafe upstream assumptions remain: {present}")


if __name__ == "__main__":
    main()
