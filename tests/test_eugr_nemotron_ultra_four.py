"""eugr's 4x-spark-cluster Nemotron 3 Ultra launch, run as the recipe states it on four Sparks."""

import json
import re
import runpy
from pathlib import Path

import pytest
from vonk_forge_contracts.recipe import RecipeGenerationSettings

from qualification import documents

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/eugr/nemotron-3-ultra-nvfp4"
UPSTREAM = ADAPTER / "recipes/4x-spark-cluster/nemotron-3-ultra-nvfp4.yaml"
RECIPE = "nemotron-3-ultra-550b-a55b-nvfp4-eugr-vllm-four"
# Arguments the recipe words differently from upstream's argv, each with its reason.
REWORDED = {
    "--distributed-executor-backend": "native multi-node launch, upstream's no-Ray cluster mode",
    "-cc.mode": "folded into --compilation-config",
    "-cc.cudagraph_mode": "folded into --compilation-config",
    "--cudagraph-capture-sizes": "folded into --compilation-config",
    "--max-model-len": "bound to context_tokens",
    "--max-num-seqs": "bound to concurrency",
    "--max-num-batched-tokens": "bound to max_batch_tokens",
    "--tensor-parallel-size": "topology tensor 4",
    "--host": "the Controller owns the listen address",
    "--port": "the Controller owns the port",
}


def upstream_flags():
    command = UPSTREAM.read_text().split("command: |", 1)[1]
    return re.findall(
        r"^\s+(-{1,2}[A-Za-z_.-]+)(?: ([^\\\n]+))? \\?$", command, re.MULTILINE
    )


def recipe_arguments():
    recipe = documents.recipe(RECIPE)
    return recipe, {a.name: a for a in recipe.runtime.arguments}


def test_every_upstream_flag_is_in_the_recipe_or_reworded_for_a_stated_reason():
    recipe, arguments = recipe_arguments()
    seen = set()
    for flag, value in upstream_flags():
        seen.add(flag)
        if flag in REWORDED:
            continue
        name = flag.lstrip("-").replace("_", "-")
        assert name in arguments, flag
        if value and not value.startswith("'") and not value.startswith("{"):
            assert str(arguments[name].value) == value, flag
    assert set(REWORDED) <= seen
    settings = recipe.settings
    assert isinstance(settings, RecipeGenerationSettings)
    assert settings.concurrency is not None and settings.max_batch_tokens is not None
    assert settings.context_tokens.value == 262144
    assert settings.concurrency.value == 2
    assert settings.max_batch_tokens.value == 16384
    assert arguments["tensor-parallel-size"].value == 4
    assert (
        recipe.topology.node_count == 4 and recipe.topology.parallelism.backend == "mp"
    )


def test_upstream_applies_no_mod_and_so_does_the_recipe():
    assert re.search(r"^mods: \[\]$", UPSTREAM.read_text(), re.MULTILINE)
    assert not list((ROOT / "adapters/eugr").glob("*ultra*/mods"))


def test_the_three_compilation_arguments_are_one_compilation_config():
    _, arguments = recipe_arguments()
    config = arguments["compilation-config"].value
    assert config == {
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [5, 10, 15, 20],
    }
    assert (
        json.loads(json.dumps(arguments["speculative-config"].value))["method"] == "mtp"
    )


def run_wrapper(monkeypatch, arguments):
    monkeypatch.setattr("sys.argv", ["vllm", *arguments])
    monkeypatch.setenv("VONK_LOCAL_ADDR", "192.0.2.2")
    monkeypatch.setenv("VONK_MASTER_ADDR", "192.0.2.1")
    monkeypatch.setenv("VONK_MASTER_PORT", "29500")
    for name in (
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "TP_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
    ):
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("NCCL_IB_GID_INDEX", "3")
    started = []
    monkeypatch.setattr("os.execv", lambda path, argv: started.append(argv))
    monkeypatch.setattr("shutil.which", lambda name, path=None: "/usr/local/bin/vllm")
    runpy.run_path(str(ADAPTER / "vllm-wrapper.py"))
    return started


def launch(nodes, rank):
    return [
        "serve",
        "/models",
        "--distributed-executor-backend",
        "mp",
        "--nnodes",
        str(nodes),
        "--node-rank",
        str(rank),
        *(["--headless"] if rank else []),
    ]


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_wrapper_starts_four_node_ranks(monkeypatch, rank):
    started = run_wrapper(monkeypatch, launch(4, rank))
    assert started and started[0][started[0].index("--nnodes") + 1] == "4"
    assert list(started[0][-4:]) == [
        "--master-addr",
        "192.0.2.1",
        "--master-port",
        "29500",
    ]


@pytest.mark.parametrize("arguments", [launch(2, 1), launch(4, 4)])
def test_wrapper_refuses_other_placements(monkeypatch, arguments):
    with pytest.raises(SystemExit, match="four-Spark launch"):
        run_wrapper(monkeypatch, arguments)
