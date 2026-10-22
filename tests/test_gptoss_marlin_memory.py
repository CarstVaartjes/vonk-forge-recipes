"""Exercise the old repack's allocation failure without CUDA or host mutation."""

from __future__ import annotations

import ast
import hashlib
import runpy
import weakref
from math import prod
from pathlib import Path
from types import SimpleNamespace

import pytest
from catalog_documents import catalog_models
from vonk_forge_contracts import RecipeDefinition
from vonk_forge_contracts.memory_admission import evaluate_memory_admission
from vonk_forge_contracts.recipe import RecipeRuntimeArgument

from qualification.memory_need import (
    IDLE_USE_BYTES,
    TOTAL_BYTES,
    argument_value,
    calculate_need,
    load_inputs,
    require_memory_need,
)

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters/nvidia/gpt-oss-120b-nvidia-vllm-single"
FIXTURE = ROOT / "tests/fixtures/gptoss-marlin-mxfp4.py.txt"
PATCH = runpy.run_path(str(ADAPTER / "patch-marlin.py"))["patch_source"]


class Tensor:
    device = "cpu"

    def __init__(self, data, shape, tracker, dtype="int32"):
        self.data = data
        self.shape = shape
        self.tracker = tracker
        self.dtype = dtype

    @property
    def nbytes(self):
        return len(self.data)

    def __getitem__(self, index):
        stride = self.nbytes // self.shape[0]
        return Tensor(
            self.data[index * stride : (index + 1) * stride],
            self.shape[1:],
            self.tracker,
            self.dtype,
        )

    def __setitem__(self, index, value):
        stride = self.nbytes // self.shape[0]
        self.data[index * stride : (index + 1) * stride] = value.data

    def view(self, dtype):
        return Tensor(
            self.data, (*self.shape[:-1], self.shape[-1] // 4), self.tracker, dtype
        )

    @property
    def T(self):
        return Tensor(self.data, tuple(reversed(self.shape)), self.tracker, self.dtype)

    def contiguous(self):
        return self.tracker.allocate(self.data, self.shape)

    def unsqueeze(self, axis):
        assert axis == 0
        return Tensor(self.data, (1, *self.shape), self.tracker, self.dtype)


class Allocations:
    def __init__(self, limit=None):
        self.live = []
        self.peak = 0
        self.limit = limit

    def allocate(self, data, shape):
        tensor = Tensor(bytearray(data), shape, self)
        self.live.append(weakref.ref(tensor))
        used = sum(
            value.nbytes
            for reference in self.live
            if (value := reference()) is not None
        )
        self.peak = max(self.peak, used)
        if self.limit is not None and used > self.limit:
            raise MemoryError("simulated unified-memory exhaustion")
        return tensor

    def cat(self, items, axis):
        assert axis == 0
        return self.allocate(
            b"".join(item.data for item in items), (len(items), *items[0].shape[1:])
        )

    def repack_kernel(self, **kwargs):
        source = kwargs["b_q_weight"]
        return self.allocate(bytes(value ^ 123 for value in source.data), source.shape)


def repack(source, tracker, name):
    tree = ast.parse(source)
    target = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "prepare_moe_mxfp4_layer_for_marlin"
    )
    nested = next(
        item
        for item in target.body
        if isinstance(item, ast.FunctionDef) and item.name == "repack_weight"
    )
    functions: list[ast.stmt] = []
    functions.extend(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "_repack_marlin_experts"
    )
    functions.append(nested)
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {
        "torch": SimpleNamespace(
            int32="int32",
            cat=tracker.cat,
            empty=lambda shape, dtype, device: tracker.allocate(
                bytearray(prod(shape) * 4), shape
            ),
        ),
        "ops": SimpleNamespace(gptq_marlin_repack=tracker.repack_kernel),
        "e": 128,
        "n": 32,
        "k": 64,
        "perm": None,
        "is_a_8bit": False,
    }
    exec(compile(module, str(FIXTURE), "exec"), namespace)  # noqa: S102 - isolated upstream function with fake allocations.
    shape = (128, 64, 32) if name == "w13" else (128, 64, 16)
    weight = Tensor(
        bytes(index % 256 for index in range(prod(shape))), shape, tracker, "uint8"
    )
    return namespace["repack_weight"](weight, name)


@pytest.mark.parametrize("name", ["w13", "w2"])
def test_repack_is_identical_and_fits_where_the_old_allocation_fails(name):
    source = FIXTURE.read_text()
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == (
        "4de8d6b46e3846e8996274872bf44c09319a41e93bd14b39ead7f172c9f09d62"
    )
    old = Allocations()
    expected = repack(source, old, name)
    new = Allocations()
    patched = PATCH(source)
    actual = repack(patched, new, name)
    assert actual.shape == expected.shape
    assert actual.data == expected.data
    assert new.peak < old.peak * 0.6
    limit = new.peak
    with pytest.raises(MemoryError, match="unified-memory exhaustion"):
        repack(source, Allocations(limit), name)
    assert repack(patched, Allocations(limit), name).data == expected.data


def test_build_rejects_source_drift_and_installs_the_backport():
    source = FIXTURE.read_text().replace("tensor_list = []", "tensor_list = list()", 1)
    with pytest.raises(ValueError, match="source drift"):
        PATCH(source)
    dockerfile = (ADAPTER / "Dockerfile").read_text()
    assert "COPY patch-marlin.py /opt/vonk/patch-marlin.py" in dockerfile
    assert "RUN python3 /opt/vonk/patch-marlin.py" in dockerfile


def test_full_context_need_and_absolute_pool_replace_the_failed_fraction():
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes/gpt-oss-120b-nvidia-vllm-single.json").read_text()
    )
    inputs = load_inputs()
    need = calculate_need(recipe, catalog_models(), inputs.geometries)
    assert need.weights_bytes == 65_248_893_184
    assert need.kv_bytes == 9_663_676_416
    assert need.resident_expansion_bytes == 10_000_000_000
    assert need.total_bytes == 92_912_569_600 < need.limit_bytes
    assert argument_value(recipe, "max-model-len") == 131_072
    assert argument_value(recipe, "max-num-seqs") == 1
    assert argument_value(recipe, "enforce-eager") is True
    assert argument_value(recipe, "max-num-batched-tokens") == 2048
    assert argument_value(recipe, "gpu-memory-utilization") == "0.649858"
    assert argument_value(recipe, "kv-cache-memory-bytes") == need.kv_bytes
    assert recipe.topology.roles[0].resources.memory.peak_bytes == need.total_bytes
    require_memory_need(recipe, catalog_models(), inputs, document_digest="")
    # The actual shared admission contract charges host occupancy exactly once.
    admission = evaluate_memory_admission(
        pool_bytes=TOTAL_BYTES,
        occupied_bytes=IDLE_USE_BYTES,
        demand_bytes=need.total_bytes,
    )
    assert admission.admissible
    recipe.runtime.arguments = [
        item
        for item in recipe.runtime.arguments
        if item.name != "kv-cache-memory-bytes"
    ] + [RecipeRuntimeArgument(name="gpu-memory-utilization", value="0.85")]
    with pytest.raises(ValueError, match="absolute kv-cache-memory-bytes"):
        require_memory_need(recipe, catalog_models(), inputs, document_digest="")


def test_resident_expansion_rejects_graph_and_image_drift():
    recipe = RecipeDefinition.model_validate_json(
        (ROOT / "recipes/gpt-oss-120b-nvidia-vllm-single.json").read_text()
    )
    inputs = load_inputs()
    eager = next(
        item for item in recipe.runtime.arguments if item.name == "enforce-eager"
    )
    eager.value = False
    with pytest.raises(ValueError, match="requires disabled graphs"):
        calculate_need(recipe, catalog_models(), inputs.geometries)
    eager.value = True
    recipe.execution.build.base_image.digest = "0" * 64
    with pytest.raises(ValueError, match="does not match the pinned image"):
        calculate_need(recipe, catalog_models(), inputs.geometries)
