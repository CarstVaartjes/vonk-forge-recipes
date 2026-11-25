"""Source pinning: every stock function rsfix replaces, copies or relies on is hashed
(sha256 of its exact source lines, decorators included) and compared with the text of
the pinned image (lmsysorg/sglang@sha256:efac853b...). Any difference refuses the patch.

Pure Python (ast + hashlib): runs before sglang, torch or triton is imported, and on a
machine without them (the CPU tests)."""

import ast
import hashlib
import os

REJECT = "kernels/ops/speculative/reject_sampling.py"
MARKOV = "kernels/ops/speculative/dspark/markov_sample.py"
BLOCKWISE = "kernels/ops/speculative/dspark/blockwise_rejection.py"
SELECTOR_KERNEL = "kernels/ops/speculative/dflash.py"
SELECTOR_MODEL = "srt/models/dflash.py"  # rewritten in part by the DFlash2 patch: pin functions only
SELECTOR_LILICORR = "kernels/ops/speculative/lilicorr.py"

MODULES = {
    "reject": ("sglang.kernels.ops.speculative.reject_sampling", REJECT),
    "markov": ("sglang.kernels.ops.speculative.dspark.markov_sample", MARKOV),
    "blockwise": ("sglang.kernels.ops.speculative.dspark.blockwise_rejection", BLOCKWISE),
    "selector_kernel": ("sglang.kernels.ops.speculative.dflash", SELECTOR_KERNEL),
    "selector_model": ("sglang.srt.models.dflash", SELECTOR_MODEL),
    "selector_lilicorr": ("sglang.kernels.ops.speculative.lilicorr", SELECTOR_LILICORR),
}

# (file, qualified name) -> sha256 of the stock source segment in the pinned image.
EXPECTED = {
    "reject": {
        "speculative_sampling_classic_kernel": "9de37d48646c13b5926744ce0197e60a0578cbc00ab8c85da6d705c8ee476142",
        "_split_chain_walk": "fe95c9b42e3cdc3fc2c7eba93d03a18ec76d78eb724d719c96f01518bdb8e9ee",
        "_split_residual_tile": "f4c7f9ea35265724db752eaca8740d27a98a22a7c89937a4bfa5ac2c0d22dc6a",
        "_split_sum_kernel": "1a2f3de4b045a66bf95408857002be8926e896d162e163c2d0dd7c944cbab663",
        "_split_cdf_kernel": "07b593c93ae5398c7658e6c20610229fa1e0dec71a8267020636987bd39862b6",
        "_split_scratch": "077fae2849bece406c5105eef748d9eb23b7bf791a99034d15d66bf8fbdde1d8",
        "_should_split_final_sampling": "854b0e4e49c359f3223b241a359c479f96b0a325882f664b7910bb22e2ef6c8e",
        "_chain_speculative_sampling_split": "3bb97501d5a7d109a179c00d2312c3515828b03d460b18a9bf8a75bbe4c59550",
        "chain_speculative_sampling_triton": "dfcb1d55ff0ae2d408e90d70323dab2afcd2c2ad250e0a2db5cd1b20021f4acc",
    },
    "markov": {
        "_round_to_step_dtype": "74f71ff436926fd3a6287f452fdd006341d2f007e4224c28f575789daa0a7e8d",
        "_markov_sample_partial_kernel": "1ff2c78319a9568cd96a9306e369d82a1086b4dd72d4bfc307603be3268f8523",
        "_select_block": "72a3df262afc1b3a90a2739750098e80e174256fd8483c17c51b306da89e6293",
        # not replaced, but the fixed partial kernel's results go through it: pinned so a changed finish pass refuses
        "_markov_sample_finish_kernel": "bde3a7fb212d3320d78eba93bde03a8735bc3a93d5a6da7478cdc56f68bbc489",
        "_philox": "c46afcee7ea04d047ca6f4477846b61dad16b0668f77d7896ffd3e5d8bfb559f",
        "_launch_partial": "c15a6962e31710fc7f651254ed8355261db3e9c741b38f7ff0db1ba46675a02d",
        "MarkovSampleStep.execute": "0e1ec0cabba6000806f58672ef4cd46e13c8b77dad23033307f396b30f3c2291",
        "MarkovSampleStep.triton": "ad9ed8ce8848357cb0ea817508cb2ef3e82e1dd148ee3fa52007cb4ed990a159",
        "MarkovSampleStep.torch": "03ac9ed2f04b5605450edf46bdddd65821787760eed6ba1d10d5ea54d208f55f",
    },
    "blockwise": {
        "_step_seed": "4d4ad2ace88056385ac59fbd34131a9a3a195d41b8de7efdc422e9ef3127abfb",
        "_murmur3_uniform32": "c2e564b30c191db91a82b28648d3c7f1687efd4c6c3dbaaad0fe3433094667e9",
        "_log1p_neg_stable": "8e451dbcf8c2d0aed9109b8044bd276e63379bf760a2e7a05b0ba6872a943c92",
        "_global_lse": "89e01fde32fedde91ca55fa996fed4dbea37e6c17083a45d6b077ee6f3d65544",
        "_stats_kernel": "3289820c7060dcc452a91150fe2cce3684b5fa6cae0662c06e7677faa7018092",
        "_rejection_kernel": "51d21d01889657f66b7a338093a6e63f0e5e6d99cf5d46f2a8499af9b9329239",
        "_resample_kernel": "b128331773a1aab0af906fb84ae08edc00b8e1a8c9d23673ccc30c92da87d57b",
        "_insert_kernel": "7d6a8d50ddd705c22971ef2f30002f8dd8514ef7ebfcec487d7f83c5a2f87634",
        "BlockwiseRejection.execute": "eb722f0999ffb7f2d060911bfe6e755a213ae2249d100510bded3e561ae01666",
        "blockwise_rejection_triton": "6d44cfb483facb4ad255f6761958f5951993e1aec4dbca2d5cf7f4a3501b2f9d",
        "blockwise_rejection_torch": "27bc311727a562c24edca9e00708007a3d86135ff8e3045806d9f8464aaf0031",
    },
    "selector_kernel": {
        "_selector_walk_kernel": "c86be883c10395cc05305a01eefcff7b3c05d1bc860a35cf28f0882344a222aa",
        "selector_walk_triton": "3bff00cb77d4445cc71ed958a84b10e495072969dc53e880498f503d8922cfaf",
    },
    "selector_model": {
        "CandidateSelector.sample_path": "9c33adcb303e1a9a2135e26d26c99fb3a7bb691a5911e51d6321b8beb0850f1d",
        "_follow_maps": "531a7ef36282d302a5ff00f13c379e4c838f9c68e39161c240d332c725e03524",
    },
    "selector_lilicorr": {
        "_selector_walk_torch": "12f5d1d9d7c2261ec2311974dc4caea9c1073e599fba3d66d205cbf51cbe9856",
        "lilicorr_sample_path": "72b13c555e5fb030bb5175df21fd5e988cdaf0bda37574aa2b23c92b4e0656c6",
    },
}


def _find(tree, qualname):
    node = tree
    for part in qualname.split("."):
        body = getattr(node, "body", [])
        hits = [
            n for n in body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part
        ]
        if len(hits) != 1:
            return None
        node = hits[0]
    return node


def segment(text, qualname):
    """Exact source lines of a top-level function / class method, decorators included,
    or None when the name is missing or defined more than once."""
    node = _find(ast.parse(text), qualname)
    if node is None:
        return None
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    lines = text.splitlines(keepends=True)
    return "".join(lines[start - 1: node.end_lineno])


def digest(text, qualname):
    seg = segment(text, qualname)
    return None if seg is None else hashlib.sha256(seg.encode("utf-8")).hexdigest()


def mismatches(part, text):
    """[(qualname, why)] for every pinned segment of target ``part`` that differs in ``text``."""
    bad = []
    for name, want in EXPECTED[part].items():
        got = digest(text, name)
        if got is None:
            bad.append((name, "missing or defined twice"))
        elif got != want:
            bad.append((name, f"sha256 {got[:12]} != pinned {want[:12]}"))
    return bad


def sglang_root():
    import importlib.util

    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("inkling_rsfix: the sglang package is not importable")
    return list(spec.submodule_search_locations)[0]


def check_part(part, root=None):
    """Raise RuntimeError unless the installed file of ``part`` matches the pinned text."""
    rel = MODULES[part][1]
    path = os.path.join(root or sglang_root(), rel)
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise RuntimeError(f"inkling_rsfix: patch target {path} is missing") from exc
    bad = mismatches(part, text)
    if bad:
        raise RuntimeError(f"inkling_rsfix: {path} differs from the pinned image: {bad}")
