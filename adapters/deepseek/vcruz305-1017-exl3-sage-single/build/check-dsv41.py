import ast
import hashlib
import importlib.util
import os
from pathlib import Path

tilelang = Path(importlib.util.find_spec("tilelang").submodule_search_locations[0])
header = (tilelang / "src/tl_templates/cuda/instruction/mma_block_scale.h").read_text()
for marker in ("CUTLASS_ARCH_MMA_SM121A_ENABLED", "kind::mxf8f6f4", "scale_vec::2X"):
    assert marker in header, f"TileLang's block-scaled MMA lacks {marker}: SM12x patches missing"
assert "mxf8f6f4" in (tilelang / "cuda/op/gemm/gemm_mma_sm120.py").read_text()
print("TileLang SM12x block-scaled MMA present")

source = Path("/opt/spark3/candidate/vllm/vllm/tokenizers/deepseek_v41_encoding.py").read_text()
mappings = next(
    node.value for node in ast.parse(source).body
    if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "REASONING_EFFORT_MAPPINGS"
)
efforts = ast.literal_eval(mappings)
expected = {"minimal": 50, "low": 50, "medium": 75, "high": 75, "xhigh": 75, "max": 100}
assert efforts == expected, efforts
print("DeepSeek V4.1 reasoning efforts", efforts)

loader = Path("/opt/spark3/candidate/vllm/vllm/model_executor/model_loader")
assert "WeightSnapshot.for_load(self, vllm_config, model_config)" in (loader / "base_loader.py").read_text()
snapshot = ast.parse((loader / "weight_snapshot.py").read_text())
names = {node.name for node in snapshot.body if isinstance(node, ast.ClassDef)}
assert {"WeightSnapshot", "SnapshotStore", "LoadProbe"} <= names, names
print("Weight snapshot hook present (DSV41_WEIGHT_SNAPSHOT=1 enables it)")

core = Path("/opt/spark3/candidate/vllm/vllm/v1")
assert "self.kv_snapshot.restore()" in (core / "engine/core.py").read_text(), "KV snapshot restore hook (0008) missing"
assert "def save_kv_snapshot(" in (core / "engine/core.py").read_text(), "KV snapshot save utility (0008) missing"
kv = ast.parse((core / "core/kv_snapshot.py").read_text())
names = {node.name for node in kv.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
assert {"KVSnapshot", "SnapshotDir", "rank_save", "rank_restore", "check_table", "install_table"} <= names, names
router = Path("/opt/spark3/candidate/vllm/vllm/entrypoints/serve")
assert '"/v1/dsv41/kv_snapshot"' in (router / "kv_snapshot/api_router.py").read_text(), "KV snapshot route (0008) missing"
assert "attach_kv_snapshot_router(app)" in (router / "__init__.py").read_text(), "KV snapshot route (0008) not registered"
print("Persistent prefix KV cache hooks present (DSV41_KV_PERSIST=1 enables them, POST /v1/dsv41/kv_snapshot saves)")

vllm = Path("/opt/spark3/candidate/vllm/vllm")
for path, pin in (
    ("model_executor/layers/fused_moe/tilelang.py", "SCALE_CODES_FUSED_MOE_TILELANG_SHA256"),
    ("models/deepseek_v4_1/tilelang/gemm.py", "SCALE_CODES_TILELANG_GEMM_SHA256"),
    ("models/deepseek_v4_1/tilelang/moe.py", "SCALE_CODES_TILELANG_MOE_SHA256"),
):
    digest = hashlib.sha256((vllm / path).read_bytes()).hexdigest()
    assert digest == os.environ[pin], f"vllm/{path} is not the scale-codes overlay's file ({digest})"
print("MoE scale-codes overlay present")

spec = vllm / "v1/worker/gpu/spec_decode"
residual_mass = (spec / "rejection_sampler_utils.py").read_text()
assert "if logit_idx + 1 >= tl.num_programs(0):" in residual_mass, "vllm#60945 port (0003) missing"
assert "spec_step_seed(seed, i) if USE_BLOCK_VERIFICATION else seed" in residual_mass, "vllm#61161 port (0004) missing"
assert "def spec_step_seed(seed, step):" in (vllm / "v1/worker/gpu/sample/gumbel.py").read_text()
assert "spec_step_seed(tl.load(seeds_ptr + req" in (spec / "vocab_parallel.py").read_text()
print("Block verification ports present (vllm#60945 lookahead bound, vllm#61161 step seeds)")

ced = (vllm / "models/deepseek_v4_1/ced.py").read_text()
assert "def nonfinal_limits(" in ced, "CED non-final chunk skip (0005) missing"
state = (vllm / "models/deepseek_v4_1/nvidia/model_state.py").read_text()
assert 'os.environ.get("DSV41_CED_SKIP_NONFINAL", "1")' in state, "CED skip gate (0005) missing"
print("CED non-final prefill chunks keep one decoder row (DSV41_CED_SKIP_NONFINAL=0 keeps 128)")

model = (vllm / "models/deepseek_v4_1/nvidia/model.py").read_text()
assert "and not spec_config.use_dspark()" in model, "DSpark MTP buffer removal (0006) missing"
assert "prefill_sp_skips_mtp_buffer" not in model, "DSpark MTP buffer removal (0006) incomplete"
print("DSpark runs without the pre-collapse MTP buffer")

for path, pin in (
    ("v1/worker/gpu/spec_decode/adaptive_verification.py", "LINEAR_BUDGET_ADAPTIVE_VERIFICATION_SHA256"),
    ("config/speculative.py", "LINEAR_BUDGET_SPECULATIVE_SHA256"),
):
    digest = hashlib.sha256((vllm / path).read_bytes()).hexdigest()
    assert digest == os.environ[pin], f"vllm/{path} is not the linear-budget overlay's file ({digest})"
runner = (vllm / "v1/worker/gpu/model_runner.py").read_text()
assert "self.adaptive_verification.warm_tail_shapes(" in runner, "DSpark linear budget (0007) missing from the runner"
assert "adaptive_verification_row_cost=(" in runner, "DSpark linear budget (0007) missing from the runner"
assert runner.count("if pre_hc_hidden_states is not None:") == 2, "0006's None guard lost in the runner"
print("DSpark draft budget by a linear step cost (adaptive_verification_row_cost, default 0.02), nothing timed")
