"""The Mia 1104 GLM dual wrapper turns the Controller's placement into the kit's per-rank environment, or refuses."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

WRAPPER = (
    Path(__file__).resolve().parents[1]
    / "adapters/glm/mia-1104-tensorfold-dual/tensorfold-serve"
)
VENDORED_TEMPLATE = WRAPPER.parent / "chat_template.jinja"
# The two template paths the wrapper reads inside the container.
IMAGE_TEMPLATE_PATH = "/opt/vonk/templates/chat_template.jinja"
CHECKPOINT_TEMPLATE_PATH = "/models/target/chat_template.jinja"
EFFORT_BASE = (
    "{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined "
    "and reasoning_effort in ['low', 'high'] else 'max' -%}"
)
EFFORT_LINE = (
    "{%- set effective_reasoning_effort = (reasoning_effort if reasoning_effort in "
    "['low', 'high'] else 'max') if (reasoning_effort is defined and reasoning_effort is "
    "not none) else none -%}"
)
FABRIC = {
    "VONK_MASTER_ADDR": "192.0.2.10",
    "VONK_MASTER_PORT": "29500",
    "NCCL_IB_HCA": "=rocep1s0f0:1,roceP2p1s0f0:1",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_SOCKET_IFNAME": "enp1s0f0np0",
}
PLACEMENT = [
    "--tp",
    "2",
    "--rank",
    "0",
    "--master",
    "VONK_MASTER_ADDR",
    "--master-port",
    "VONK_MASTER_PORT",
]


def seen(stdout: str) -> tuple[list[str], dict[str, str]]:
    argv = [line[2:] for line in stdout.splitlines() if line.startswith("A:")]
    env = dict(
        line[2:].split("=", 1) for line in stdout.splitlines() if line.startswith("E:")
    )
    return argv, env


def wrapper_for(tmp_path: Path, checkpoint_template: str | None = None) -> Path:
    """The wrapper with its two template paths pointed under tmp_path (the container's /models and /opt are not here)."""
    vendored = tmp_path / "opt" / "chat_template.jinja"
    checkpoint = tmp_path / "ckpt" / "chat_template.jinja"
    vendored.parent.mkdir(exist_ok=True)
    checkpoint.parent.mkdir(exist_ok=True)
    vendored.write_bytes(VENDORED_TEMPLATE.read_bytes())
    checkpoint.write_text(
        checkpoint_template
        if checkpoint_template is not None
        else VENDORED_TEMPLATE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    text = WRAPPER.read_text(encoding="utf-8")
    for old, new in (
        (IMAGE_TEMPLATE_PATH, vendored),
        (CHECKPOINT_TEMPLATE_PATH, checkpoint),
    ):
        assert text.count(old) == 1
        text = text.replace(old, str(new))
    patched = tmp_path / "tensorfold-serve"
    patched.write_text(text, encoding="utf-8")
    return patched


def run(
    tmp_path: Path,
    args: list[str],
    checkpoint_template: str | None = None,
    **env: str,
):
    stub = tmp_path / "bin" / "tensorfold-native"
    stub.parent.mkdir(exist_ok=True)
    stub.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        "for a in sys.argv[1:]: print('A:' + a)\n"
        "for k, v in os.environ.items(): print('E:' + k + '=' + v)\n"
    )
    stub.chmod(0o755)
    base = {k: v for k, v in os.environ.items() if k not in FABRIC}
    base.update(
        PATH=f"{stub.parent}:{os.environ['PATH']}",
        HOME=str(tmp_path / "home"),
        TMPDIR=str(tmp_path / "tmp"),
    )
    return subprocess.run(
        ["bash", str(wrapper_for(tmp_path, checkpoint_template)), *args],
        env={**base, **env},
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("rank", ["0", "1"])
def test_both_ranks_get_the_kits_placement(tmp_path: Path, rank: str) -> None:
    placement = list(PLACEMENT)
    placement[3] = rank
    result = run(tmp_path, ["--port", "8888", "--vision", *placement], **FABRIC)
    assert result.returncode == 0, result.stderr
    argv, env = seen(result.stdout)
    assert argv == [
        "serve",
        "/models/target",
        "--port",
        "8888",
        "--vision",
        "--chat-template",
        f"{tmp_path}/tmp/vonk-template/chat_template.jinja",
    ]
    assert env["TF_GLM_RANK"] == rank
    assert env["TF_GLM_WORLD"] == "2"
    assert env["NCCL_COMM_ID"] == "192.0.2.10:29500"
    assert env["TF_ROCE_HCA"] == "rocep1s0f0,roceP2p1s0f0"
    assert env["NCCL_IB_HCA"] == FABRIC["NCCL_IB_HCA"]
    assert (tmp_path / "home").is_dir() and (tmp_path / "tmp").is_dir()


def served_template(tmp_path: Path, args: list[str]) -> str:
    result = run(tmp_path, ["--port", "8888", *args, *PLACEMENT], **FABRIC)
    assert result.returncode == 0, result.stderr
    argv = seen(result.stdout)[0]
    served = Path(argv[argv.index("--chat-template") + 1])
    assert served == tmp_path / "tmp/vonk-template/chat_template.jinja"
    return served.read_text(encoding="utf-8")


@pytest.mark.parametrize("args", [[], ["--vision"]])
def test_the_served_template_takes_the_effort_line_from_reasoning_effort(
    tmp_path: Path, args: list[str]
) -> None:
    """The kit's EFFORT_LINE=thinking: thinking off passes no reasoning_effort, so no `Reasoning Effort: Max` line."""
    original = VENDORED_TEMPLATE.read_text(encoding="utf-8")
    served = served_template(tmp_path, args)
    assert served == original.replace(EFFORT_BASE, EFFORT_LINE)
    assert EFFORT_BASE not in served and served.count(EFFORT_LINE) == 1
    # Thinking off passes no reasoning_effort: the line then sets none, so the template's `is not none` guard drops the
    # `Reasoning Effort` system line; thinking on keeps low/high and maps anything else to max.
    assert (
        "reasoning_effort is defined and reasoning_effort is not none) else none -%}"
        in served
    )


def test_the_vendored_template_stays_as_vendored(tmp_path: Path) -> None:
    before = VENDORED_TEMPLATE.read_bytes()
    served_template(tmp_path, ["--vision"])
    assert VENDORED_TEMPLATE.read_bytes() == before
    assert (tmp_path / "opt/chat_template.jinja").read_bytes() == before


def test_a_template_without_the_known_effort_line_does_not_start_the_engine(
    tmp_path: Path,
) -> None:
    result = run(
        tmp_path,
        list(PLACEMENT),
        checkpoint_template="{{ messages }}",
        **FABRIC,
    )
    assert result.returncode != 0 and result.stdout == ""
    assert "no effort line" in result.stderr


@pytest.mark.parametrize(
    "missing",
    [
        "VONK_MASTER_ADDR",
        "VONK_MASTER_PORT",
        "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX",
        "NCCL_SOCKET_IFNAME",
    ],
)
def test_a_missing_variable_does_not_start_the_engine(
    tmp_path: Path, missing: str
) -> None:
    result = run(
        tmp_path, list(PLACEMENT), **{k: v for k, v in FABRIC.items() if k != missing}
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert "tensorfold-serve:" in result.stderr


@pytest.mark.parametrize(
    "bad",
    [
        {"NCCL_IB_HCA": "=a:1,a:1"},
        {"NCCL_IB_HCA": "=a:1,b:1,c:1"},
        {"NCCL_IB_HCA": "=a:2"},
        {"NCCL_IB_GID_INDEX": "x"},
        {"VONK_MASTER_ADDR": "not-an-address"},
        {"VONK_MASTER_PORT": "80"},
    ],
)
def test_malformed_placement_does_not_start_the_engine(
    tmp_path: Path, bad: dict[str, str]
) -> None:
    result = run(tmp_path, list(PLACEMENT), **{**FABRIC, **bad})
    assert result.returncode != 0 and result.stdout == ""


def test_other_topologies_are_refused(tmp_path: Path) -> None:
    result = run(tmp_path, ["--tp", "3", *PLACEMENT[2:]], **FABRIC)
    assert result.returncode != 0 and result.stdout == ""
