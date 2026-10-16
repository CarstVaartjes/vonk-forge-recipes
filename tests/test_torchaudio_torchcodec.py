"""An adapter image that pins torchaudio >= 2.9 also pins TorchCodec.

Since torchaudio 2.9, ``torchaudio.load``, ``save`` and ``info`` go through
TorchCodec. Without it they raise ``ImportError`` at the call, which failed the
HunyuanVideo-Foley job after inference. Images pin torchaudio in lock and
requirements files and in Dockerfile ``pip``/``uv`` lines, so all of those are
scanned. An image whose code never loads or saves audio through torchaudio is
exempt; its reason records the evidence.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "adapters"
TORCHAUDIO_PIN = re.compile(r"\btorchaudio\s*==\s*(\d+)\.(\d+)")
TORCHCODEC_PIN = re.compile(r"\btorchcodec\s*==")
FIRST_TORCHCODEC_TORCHAUDIO = (2, 9)

_COMFYUI = (
    "ComfyUI 83071e1a never imports torchaudio (its audio nodes use PyAV), the "
    "image adds no custom nodes, and the job wrapper does no audio I/O."
)
_LTX2 = (
    "LTX-2 9ec55f9f ltx-core and ltx-pipelines decode and encode audio with PyAV "
    "(ltx_pipelines/utils/media_io) and use torchaudio only for MelSpectrogram "
    "and resample; torchaudio.load/save are only in ltx-trainer, which the "
    "image does not install."
)
EXEMPT = {
    "media/comfyui-core": _COMFYUI,
    "media/comfyui-qwen-image-2512-bf16": _COMFYUI,
    "media/comfyui-qwen-image-2512-fp8-lightning": _COMFYUI,
    "media/comfyui-qwen-image-edit-2511-fp8mixed": _COMFYUI,
    "media/comfyui-qwen-image-edit-2511-int8-convrot": _COMFYUI,
    "video/ltx2-pytorch": _LTX2,
    "video/ltx2-sync-native": _LTX2,
    "video/ltx23-sync-native-disk": _LTX2,
    "video/mova-pytorch": (
        "MOVA 1591b255 scripts/inference_single.py writes audio with the stdlib "
        "wave module and muxes it with the ffmpeg CLI (mova/utils/data.py); "
        "torchaudio.load is only under evaluation/, which the image never runs. "
        "descript-audiotools 0.7.2 loads through librosa and writes through "
        "soundfile; its torchaudio.info shim is reached only from "
        "AudioSignal.excerpt. The recipe takes text and an image, never audio."
    ),
    "llm/r0b0tlab-laguna-s-vllm-single": (
        "Serves Laguna-S 2.1, a text-only model (models/laguna-s-2-1-*.json: "
        "modalities text), so vLLM never decodes audio. Dockerfile.production "
        "is vendored byte-identical from upstream (vendored-upstream.json)."
    ),
    "minimax/a3refaat-m3-b12x-dual": (
        "Serves MiniMax-M3, a text-only model (models/minimax-m3-*.json: "
        "modalities text), so vLLM never decodes audio."
    ),
}


def _build_inputs(adapter: Path) -> list[Path]:
    return [
        path
        for path in adapter.rglob("*")
        if path.is_file()
        and (
            path.name.startswith("Dockerfile")
            or path.suffix in {".in", ".txt", ".lock", ".sh"}
        )
    ]


def _torchcodec_adapters() -> dict[str, bool]:
    """Adapters pinning torchaudio >= 2.9, mapped to whether they pin TorchCodec."""
    found: dict[str, bool] = {}
    for adapter in sorted(path for path in ADAPTERS.glob("*/*") if path.is_dir()):
        texts = [path.read_text(errors="replace") for path in _build_inputs(adapter)]
        if any(
            (int(major), int(minor)) >= FIRST_TORCHCODEC_TORCHAUDIO
            for text in texts
            for major, minor in TORCHAUDIO_PIN.findall(text)
        ):
            name = adapter.relative_to(ADAPTERS).as_posix()
            found[name] = any(TORCHCODEC_PIN.search(text) for text in texts)
    return found


def test_torchaudio_images_pin_torchcodec_or_record_why_not() -> None:
    adapters = _torchcodec_adapters()
    # The scan sees real pins of both packages, not an empty tree.
    assert any(adapters.values())
    missing = sorted(
        name for name, pinned in adapters.items() if not pinned and name not in EXEMPT
    )
    assert not missing, (
        "torchaudio >= 2.9 loads and saves through TorchCodec; pin torchcodec in "
        f"these adapter images or exempt them with evidence: {missing}"
    )


def test_every_exemption_is_still_needed() -> None:
    adapters = _torchcodec_adapters()
    stale = sorted(name for name in EXEMPT if adapters.get(name) is not False)
    assert not stale, f"these exemptions no longer apply; remove them: {stale}"
