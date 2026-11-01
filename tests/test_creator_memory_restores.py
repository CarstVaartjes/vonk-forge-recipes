"""Recipes follow the creator's memory settings (owner decision 2026-10-29).

One row per restored recipe. Our pre-emptive corrections are reverted; the
sweep re-proves each recipe, and a restored recipe that then fails on hardware
goes back to a lowered value with that failure cited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from catalog_documents import catalog_models
from documents import recipe as load_recipe
from documents import recipe_display
from vonk_forge_contracts import RecipeDefinition, document_sha256

from qualification.memory_need import (
    OS_FLOOR_BYTES,
    argument_value,
    load_inputs,
    memory_variants,
    per_rank_need,
    require_admissible_declaration,
    require_memory_need,
    vllm_utilization_envelope,
)

ROOT = Path(__file__).resolve().parents[1]
MIA = "https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/94ae731f4619f769be1f4dfe6f6607ce0a3e869d"
EUGR = "https://github.com/eugr/spark-vllm-docker/blob/4709188da2921c89b12ae1231a6e712c6ab536e9/recipes"
MIA_G26 = "https://github.com/MiaAI-Lab/Gemma-4-26B-A4B-DGX-Spark-18-concurrencies/blob/39e94192bd6f114a222d434cf2bf4dcf173248a5"
MIA_G31 = "https://github.com/MiaAI-Lab/Gemma-4-31B-IT-NVFP4-Recipe/blob/ac429cca70aedfa4afb446cefc5f5f1bfa1e8802"
MIA_MUSE = "https://github.com/MiaAI-Lab/Muse-Glimmer-30B-DGX-Spark-RTX-5090-6000-PRO/blob/1b71dc011010a3e2c373eafd8a41f12c7c7913fc"
MIA_Q36 = "https://github.com/MiaAI-Lab/Qwen3.6-27B-NVFP4-DFlash-DGX-Spark/blob/c9b071012e8e7a74cdb5277413332d0767ba7fff"
TONY_JASL = "https://github.com/tonyd2wild/DS4-Jasl-Sparkrun-GB10-Fix/blob/050f1ba36d376f6a629894be795ebb79bc74501d"
TONY_DS1M = "https://github.com/tonyd2wild/DeepSeek-V4-Flash-2x-Spark-1M/blob/7cc2c55c7b6428dcf49401dae63583c2e0141cdf"
MIA_DS1M = "https://github.com/MiaAI-Lab/DeepSeek-V4-Flash-Dual-DGX-Spark-1M-Context/blob/d84319952b41df812603830cd13ae921c39860f7"
MIA_LEAN = "https://github.com/MiaAI-Lab/Leanstral-1.5-119B-A6B_2x_DGX_Sparks_vLLM/blob/94d729896226db3eaf7f584e79b7c1f0028bab1d"
MIA_EXL3 = "https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/blob/6f7d1590ad49a2b8995188e45d7b9db31e677452"
R0B0_GLM = "https://github.com/r0b0tlab/glm53-flash-nvfp4-sm121/blob/222697317a4a165767ab8ef17b44cf969a5ece09"
SERO = "https://github.com/0xSero/deepseek-spark/blob/97fd6f02e26870416d3c6b450d4fa3217d9db124/runtime/configs"
TONY_QT200 = "https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s/blob/162d7a766e4bc9de1809ff724339538380c43b0c"
TONY_NVFP4KV = "https://github.com/tonyd2wild/GLM-5.2-NVFP4-KV-4x-DGX-Spark-300kctx-42tok-s/blob/34e81562984bda993e0c9ed01ed6900c17e4857b"
TONY_G53 = "https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/blob/f5440eff7f80ae1b7f30898871a50707ba17d115"
EUGR_GLM = "https://github.com/eugr/spark-vllm-docker/blob/303b4a29a6bd390428d363358964ad298dc20625/recipes"
TONY_MIMO = "https://github.com/tonyd2wild/MiMo-V2.5-TP3-NVFP4-KV-3xDGX-Spark/blob/8024c85b5e5fc5c9edd9db7b40f607683372cb0c/recipes"
R0B0_Q36 = "https://github.com/r0b0tlab/qwen36-35b-a3b-nvfp4-sm121-vllm/blob/106fb53372c260e719c97d5b59e48142792783c7"
DROWZEYS = "https://github.com/drowzeys/keys-Qwen3.8-27B-NVFP4-DFlash2-Ablit-Cybersecurity-Unlock-1M-Context-YARN-Single-DGXSpark/blob/fca3d18bcab52b65e4a428b883cbcbcb0a7a3443"
AEON = "adapters/qwen/drowzeys-aeon-dflash2-single"
GMU = r'GMU="\$\{GMU:-([0-9.]+)\}"'
SHELL_UTILIZATION = r"--gpu-memory-utilization\s+([0-9.]+)"
UTILIZATION = r"gpu_memory_utilization:\s*([0-9.]+)"
FORBIDDEN = (
    "GB10 memory correction",
    "GB10 fit:",
    "Draft-model correction",
    "draft-model allowance",
    "Absolute KV correction",
)


@dataclass(frozen=True)
class Creator:
    source: str
    arguments: dict[str, str]
    absent: tuple[str, ...] = ()
    context_tokens: int | None = None
    options: dict[str, dict[str, str]] = field(default_factory=dict)
    vendored: tuple[str, str] | None = None
    vendored_options: dict[str, tuple[str, str]] = field(default_factory=dict)
    # "option/choice" ids the creator's profile dropped.
    absent_choices: tuple[str, ...] = ()
    peak: int | None = None  # None: the pool-less vLLM envelope
    os_floor_reserve: bool = False


CREATOR = {
    "glm-5-3-flash-exl3-dflash2-vllm-mia-four": Creator(
        source=f"{MIA}/.env.tp4.example#L115",
        arguments={"gpu-memory-utilization": "0.75"},
        absent=("kv-cache-memory-bytes",),
        os_floor_reserve=True,
    ),
    "glm-5-3-flash-exl3-dflash2-vllm-mia-triple": Creator(
        source=f"{MIA}/.env.tp3.example#L172",
        arguments={"gpu-memory-utilization": "0.80"},
        absent=("kv-cache-memory-bytes",),
        os_floor_reserve=True,
    ),
    "glm-5-3-flash-nvfp4-vllm-four": Creator(
        source="https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark/blob/2ac4e8dd661b12e6a1d7e69df711707abd6b9cd4/launch-glm53-tp4-24g.sh#L86",
        arguments={
            "gpu-memory-utilization": "0.85",
            "kv-cache-memory": "25769803776",
        },
        # Multi-Spark startup budget: 0.85 x total + 8 GB, beside the pinned pool.
        peak=119063746765,
    ),
    "laguna-s-2-1-nvfp4-r0b0tlab-vllm-single": Creator(
        source="https://github.com/r0b0tlab/laguna-s-2.1-nvfp4-sm121-vllm/blob/4cf44cf36b88b564031ddb19c41d7cb9cfe0f52a/sparkrun/recipes/laguna-s-2.1-nvfp4-vllm-r0b0tlab.yaml#L27",
        arguments={"gpu-memory-utilization": "0.85"},
        absent=("kv-cache-memory-bytes",),
    ),
    "gemma-4-26b-a4b-it-nvfp4-eugr-vllm-single": Creator(
        source=f"{EUGR}/gemma4-26b-a4b-nvfp4.yaml#L28",
        arguments={"gpu-memory-utilization": "0.7"},
        vendored=(
            "adapters/eugr/gemma4-26b-a4b-nvfp4/recipes/gemma4-26b-a4b-nvfp4.yaml",
            UTILIZATION,
        ),
    ),
    "nemotron-3-5-lightning-30b-a3b-nvfp4-eugr-vllm-single": Creator(
        source=f"{EUGR}/nemotron-3.5-lightning.yaml#L18",
        arguments={"gpu-memory-utilization": "0.7"},
        vendored=(
            "adapters/eugr/nemotron-3.5-lightning/recipes/nemotron-3.5-lightning.yaml",
            UTILIZATION,
        ),
    ),
    "nemotron-3-5-lightning-30b-a3b-nvfp4-eugr-vllm-dual": Creator(
        source=f"{EUGR}/nemotron-3.5-lightning.yaml#L18",
        arguments={"gpu-memory-utilization": "0.7"},
        vendored=(
            "adapters/eugr/nemotron-3.5-lightning/recipes/nemotron-3.5-lightning.yaml",
            UTILIZATION,
        ),
    ),
    "qwen3-6-35b-a3b-fp8-dflash-eugr-vllm-single": Creator(
        source=f"{EUGR}/qwen3.6-35b-a3b-fp8-dflash.yaml#L26",
        arguments={"gpu-memory-utilization": "0.8"},
        vendored=(
            "adapters/eugr/qwen3-6-35b-a3b-fp8-dflash/recipes/qwen3.6-35b-a3b-fp8-dflash.yaml",
            UTILIZATION,
        ),
    ),
    "qwen3-6-35b-a3b-fp8-dflash-eugr-vllm-dual": Creator(
        source=f"{EUGR}/qwen3.6-35b-a3b-fp8-dflash.yaml#L26",
        arguments={"gpu-memory-utilization": "0.8"},
        vendored=(
            "adapters/eugr/qwen3-6-35b-a3b-fp8-dflash/recipes/qwen3.6-35b-a3b-fp8-dflash.yaml",
            UTILIZATION,
        ),
    ),
    "qwen3-8-27b-nvfp4-dflash2-eugr-vllm-dual": Creator(
        source=f"{EUGR}/qwen3.8-27b-nvfp4-dflash2.yaml#L18",
        arguments={"gpu-memory-utilization": "0.8"},
        vendored=(
            "adapters/eugr/qwen3-8-27b-nvfp4-dflash2/recipes/qwen3.8-27b-nvfp4-dflash2.yaml",
            UTILIZATION,
        ),
    ),
    "gemma-4-26b-a4b-vllm-mia-single": Creator(
        source=f"{MIA_G26}/start.sh#L99",
        arguments={"gpu-memory-utilization": "0.70"},
        vendored=(
            "adapters/google/gemma4-26b-a4b-nvfp4-mia-vllm-single/upstream-start.sh",
            SHELL_UTILIZATION,
        ),
    ),
    "gemma-4-31b-it-nvfp4-vllm-mia-single": Creator(
        source=f"{MIA_G31}/start.sh#L156",
        arguments={"gpu-memory-utilization": "0.70"},
        options={"gpu-memory/util-085": {"gpu-memory-utilization": "0.85"}},
        # The creator's 0.95 alternative can never be admitted on a Spark (README#L209).
        absent_choices=("gpu-memory/util-088",),
        vendored=(
            "adapters/google/gemma4-31b-nvfp4-mia-vllm-single/upstream-start.sh",
            SHELL_UTILIZATION,
        ),
    ),
    "muse-glimmer-30b-nvfp4-vllm-mia-single": Creator(
        source=f"{MIA_MUSE}/start.sh#L37",
        arguments={"gpu-memory-utilization": "0.42"},
    ),
    "qwen3-6-27b-nvfp4-dflash-vllm-mia-single": Creator(
        source=f"{MIA_Q36}/start.sh#L83",
        arguments={"gpu-memory-utilization": "0.84"},
        vendored=("adapters/qwen/mia-nvfp4-dflash-single/start.sh", SHELL_UTILIZATION),
    ),
    "nemotron-3-5-lightning-30b-a3b-sfxnz-vllm-single": Creator(
        source="https://github.com/sfxnz/Nemotron-3.5-Lightning-vLLM-DGX-Spark/blob/0214edfd3888cebee3bf0abff7663e34a419bf38/run.sh#L10",
        arguments={"gpu-memory-utilization": "0.8"},
        options={"gpu_memory/util-0-7": {"gpu-memory-utilization": "0.7"}},
        # The kit's 0.91 profile pushed free RAM toward 1 GiB (README#L101); never admissible.
        absent_choices=("gpu_memory/util-0-91",),
    ),
    "nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single": Creator(
        source="https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/blob/bee7596271d1495f6992ae224aefde4410e816b8/README.md#L114",
        arguments={"gpu-memory-utilization": "0.85"},
    ),
    "laguna-s-2-1-nvfp4-vllm-single": Creator(
        source="https://huggingface.co/poolside/Laguna-S-2.1-NVFP4/blob/07614121b31898586430f189d27a25a0be310843/README.md#L185",
        arguments={"gpu-memory-utilization": "0.85"},
    ),
    "qwen3-8-27b-aeon-abliterated-dflash2-drowzeys-vllm-single": Creator(
        source=f"{DROWZEYS}/serve_b5_on_4.sh#L16",
        arguments={"gpu-memory-utilization": "0.70"},
        context_tokens=1048576,
        options={"seat/speed-16k": {"gpu-memory-utilization": "0.75"}},
        vendored=(f"{AEON}/serve_b5_on_4.sh", GMU),
        vendored_options={"seat/speed-16k": (f"{AEON}/serve_b5_16k.sh", GMU)},
    ),
    "qwen3-8-27b-nvfp4-mtp-r0b0tlab-vllm-single": Creator(
        source="https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-vllm/blob/a96186f8973023365bddb34b6b34fa889d4c40ae/scripts/serve.sh#L103",
        arguments={"gpu-memory-utilization": "0.70"},
        # The creator's DFlash2 profile runs the base 0.70 (serve.sh#L103).
        options={"speculation/dflash2": {"gpu-memory-utilization": "0.70"}},
    ),
    "deepseek-v4-flash-jasl-tonyd2wild-vllm-dual": Creator(
        source=f"{TONY_JASL}/recipe/deepseek4-flash-fp8-mtp-vllm-sparkfix.yaml#L16",
        arguments={"gpu-memory-utilization": "0.825"},
        absent=("kv-cache-memory-bytes",),
    ),
    "deepseek-v4-flash-mtp-tonyd2wild-vllm-dual": Creator(
        source=f"{TONY_DS1M}/launch/ds4-1m-launch.sh#L30",
        arguments={"gpu-memory-utilization": "0.82"},
        absent=("kv-cache-memory-bytes",),
        context_tokens=1000000,
        # The creator's 500K rollback profile (launch/ds4-500k-launch.sh#L30).
        options={"memory/profile-500k": {"gpu-memory-utilization": "0.80"}},
    ),
    "deepseek-v4-flash-vllm-mia-dual": Creator(
        source=f"{MIA_DS1M}/docker-compose.yml#L68",
        arguments={"gpu-memory-utilization": "0.83"},
        absent=("kv-cache-memory-bytes",),
        context_tokens=1000000,
        # The creator's out-of-memory fallback (README#L232).
        options={"memory/util-075": {"gpu-memory-utilization": "0.75"}},
    ),
    "glm-5-3-flash-nvidia-nvfp4-dflash2-vllm-dual": Creator(
        source=f"{R0B0_GLM}/serve/boot_dflash2.sh#L14",
        arguments={"gpu-memory-utilization": "0.86"},
        os_floor_reserve=True,
    ),
    "leanstral-1-5-119b-a6b-vllm-mia-dual": Creator(
        source=f"{MIA_LEAN}/start.sh#L21",
        arguments={"gpu-memory-utilization": "0.85"},
        absent=("kv-cache-memory-bytes",),
    ),
    "deepseek-v4-1-flash-exl3-mia-dual": Creator(
        source=f"{MIA_EXL3}/.env.example#L146",
        arguments={
            "gpu-memory-utilization": "0.88",
            "kv-cache-memory-bytes": "2684354560",
        },
        # Beside a pinned pool the utilization is only vLLM's startup check (#L143);
        # the honest peak is the per-rank loading floor.
        peak=115984824099,
    ),
    "deepseek-v4-flash-reap-162b-0xsero-vllm-single": Creator(
        source=f"{SERO}/k144-nospec-200k.env#L9",
        arguments={"gpu-memory-utilization": "0.88", "kv-cache-memory-bytes": "14G"},
        peak=120370379081,
    ),
    "deepseek-v4-flash-reap-180b-0xsero-vllm-single": Creator(
        source=f"{SERO}/k160-mtp2-200k.env#L9",
        arguments={"gpu-memory-utilization": "0.88", "kv-cache-memory-bytes": "6G"},
        peak=120370379081,
    ),
    "glm-5-2-quanttrio-200k-mtp-tonyd2wild-vllm-four": Creator(
        source=f"{TONY_QT200}/launch.sh#L209",
        arguments={
            "gpu-memory-utilization": "0.91",
            "kv-cache-memory-bytes": "10950000000",
        },
        context_tokens=200000,
        peak=120320253504,
    ),
    "glm-5-2-quanttrio-nvfp4-kv-tonyd2wild-vllm-four": Creator(
        source=f"{TONY_NVFP4KV}/serve/speednight-nvfp4.sh#L170",
        arguments={
            "gpu-memory-utilization": "0.91",
            "kv-cache-memory-bytes": "10950000000",
        },
        context_tokens=316000,
        peak=120320253504,
    ),
    "glm-5-3-int4-int8mix-mtp-tonyd2wild-vllm-four": Creator(
        source=f"{TONY_G53}/launch/launch-glm53-tp4.sh#L132",
        arguments={
            "gpu-memory-utilization": "0.91",
            "kv-cache-memory-bytes": "10950000000",
        },
        context_tokens=200000,
        peak=120260467668,
    ),
    "glm-5-3-flash-nvfp4-spark-eugr-vllm-dual": Creator(
        source=f"{EUGR_GLM}/glm-5.3-flash.yaml#L30",
        arguments={"gpu-memory-utilization": "0.87"},
        # The yaml declares kv_cache_memory_bytes 8G (#L29) its command never passes.
        absent=("kv-cache-memory-bytes",),
    ),
    "mimo-v2-5-omni-nvfp4-tonyd2wild-vllm-triple": Creator(
        source=f"{TONY_MIMO}/mimo-v25-tp3-omni-1mctx-6seq-mtp2-nvfp4kv.yaml#L34",
        arguments={"gpu-memory-utilization": "0.89"},
        absent=("kv-cache-memory-bytes",),
    ),
    "qwen3-6-35b-a3b-nvfp4-r0b0tlab-vllm-single": Creator(
        source=f"{R0B0_Q36}/Dockerfile.v025#L23",
        arguments={"gpu-memory-utilization": "0.88"},
    ),
}


@pytest.mark.parametrize("slug", sorted(CREATOR))
def test_recipe_follows_the_creators_memory_settings(slug: str) -> None:
    row, recipe = CREATOR[slug], load_recipe(slug)
    for name, expected in row.arguments.items():
        if row.vendored is not None:
            path, pattern = row.vendored
            match = re.search(pattern, (ROOT / path).read_text(encoding="utf-8"))
            assert match, (path, pattern)
            expected = match.group(1)
        assert str(argument_value(recipe, name)) == expected, (name, row.source)
    for name in row.absent:
        assert argument_value(recipe, name) is None, name
    if row.context_tokens is not None:
        assert argument_value(recipe, "max-model-len") == row.context_tokens
    for choice, values in row.options.items():
        option, value = choice.split("/")
        chosen = recipe.with_option_choices({option: value})
        for name, expected in values.items():
            if choice in row.vendored_options:
                path, pattern = row.vendored_options[choice]
                match = re.search(pattern, (ROOT / path).read_text(encoding="utf-8"))
                assert match, (path, pattern)
                expected = match.group(1)
            assert str(argument_value(chosen, name)) == expected, (choice, name)
    for choice in row.absent_choices:
        option, value = choice.split("/")
        offered = next(item for item in recipe.options if item.name == option)
        assert value not in [item.value for item in offered.choices], choice
    if row.peak is None:
        expected_peak = max(
            envelope
            for variant in memory_variants(recipe)
            if (envelope := vllm_utilization_envelope(variant)) is not None
        )
    else:
        expected_peak = row.peak
        need = per_rank_need(recipe, catalog_models(), load_inputs())
        assert expected_peak >= need.peak_bytes
    for role in recipe.topology.roles:
        memory = role.resources.memory
        # The creator's envelope is the floor; a role's honest measured peak may be higher.
        assert memory.peak_bytes >= expected_peak, role.name
        if row.os_floor_reserve:
            assert memory.reserve_bytes >= OS_FLOOR_BYTES, role.name
    description = recipe_display(slug).description
    assert not [text for text in FORBIDDEN if text in description]
    assert "Memory follows the creator's settings" in description


def test_lowering_a_restored_peak_by_one_byte_is_refused() -> None:
    slug = "laguna-s-2-1-nvfp4-r0b0tlab-vllm-single"
    recipe = load_recipe(slug)
    models, inputs = catalog_models(), load_inputs()
    digest = document_sha256(recipe.model_dump(mode="json", by_alias=True))
    require_memory_need(recipe, models, inputs, document_digest=digest, touched=True)
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes -= 1
    with pytest.raises(ValueError, match="peak understates"):
        require_memory_need(
            recipe, models, inputs, document_digest=digest, touched=True
        )


def test_lowering_the_mia_gemma_26b_peak_by_one_byte_is_refused() -> None:
    recipe = load_recipe("gemma-4-26b-a4b-vllm-mia-single")
    models, inputs = catalog_models(), load_inputs()
    digest = document_sha256(recipe.model_dump(mode="json", by_alias=True))
    require_memory_need(recipe, models, inputs, document_digest=digest, touched=True)
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes -= 1
    with pytest.raises(ValueError, match="peak understates"):
        require_memory_need(
            recipe, models, inputs, document_digest=digest, touched=True
        )


def test_the_creators_gemma_31b_095_choice_can_never_be_admitted() -> None:
    document = load_recipe("gemma-4-31b-it-nvfp4-vllm-mia-single").model_dump(
        mode="json", by_alias=True
    )
    require_admissible_declaration(RecipeDefinition.model_validate(document))
    option = next(item for item in document["options"] if item["name"] == "gpu-memory")
    option["choices"].append(
        {
            "args": [
                {
                    "name": "gpu-memory-utilization",
                    "setting": None,
                    "value": "0.95",
                }
            ],
            "help": "The creator's 0.95 estimate (README#L209).",
            "label": "0.95",
            "value": "util-095",
        }
    )
    widened = RecipeDefinition.model_validate(document)
    peak = max(
        envelope
        for variant in memory_variants(widened)
        if (envelope := vllm_utilization_envelope(variant)) is not None
    )
    for role in widened.topology.roles:
        role.resources.memory.peak_bytes = peak
    with pytest.raises(ValueError, match="not admissible"):
        require_admissible_declaration(widened)


@pytest.mark.parametrize(
    "slug",
    [
        "deepseek-v4-flash-reap-162b-0xsero-vllm-single",
        "deepseek-v4-flash-reap-180b-0xsero-vllm-single",
    ],
)
def test_lowering_a_reap_single_peak_below_its_loading_floor_is_refused(
    slug: str,
) -> None:
    models, inputs = catalog_models(), load_inputs()
    recipe = load_recipe(slug)
    digest = document_sha256(recipe.model_dump(mode="json", by_alias=True))
    require_memory_need(recipe, models, inputs, document_digest=digest, touched=True)
    floor = per_rank_need(recipe, models, inputs).peak_bytes
    below = load_recipe(slug)
    for role in below.topology.roles:
        role.resources.memory.peak_bytes = floor - 1
    with pytest.raises(ValueError, match="peak understates"):
        require_memory_need(below, models, inputs, document_digest=digest, touched=True)


def test_lowering_the_r0b0tlab_qwen36_single_peak_by_one_byte_is_refused() -> None:
    recipe = load_recipe("qwen3-6-35b-a3b-nvfp4-r0b0tlab-vllm-single")
    models, inputs = catalog_models(), load_inputs()
    digest = document_sha256(recipe.model_dump(mode="json", by_alias=True))
    require_memory_need(recipe, models, inputs, document_digest=digest, touched=True)
    for role in recipe.topology.roles:
        role.resources.memory.peak_bytes -= 1
    with pytest.raises(ValueError, match="peak understates"):
        require_memory_need(
            recipe, models, inputs, document_digest=digest, touched=True
        )
