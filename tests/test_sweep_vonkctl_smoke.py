"""The vonkctl wrapper, the smoke tests and the reviewed definitions they run."""

from __future__ import annotations

import json
import stat
import textwrap
from pathlib import Path

import pytest
from sweep_fakes import Gateway

from spark_sweep.catalog import Recipe
from spark_sweep.definitions import Definitions
from spark_sweep.smoke import (
    AssertionFailed,
    HttpConfig,
    check_assertions,
    run_case,
    smoke_readiness,
    smoke_service,
    stream_probe,
)
from spark_sweep.vonkctl import (
    OwnerProfileError,
    Reply,
    Vonkctl,
    VonkctlError,
    request_key,
)

ROOT = Path(__file__).resolve().parents[1]


def reply(code: int, document: object, argv: tuple[str, ...] = ()) -> Reply:
    return Reply(argv, code, document, "")


def runner_returning(code: int, out: str, err: str = ""):
    seen: list[list[str]] = []

    def run(argv, timeout):
        seen.append(list(argv))
        return code, out, err

    return run, seen


# -- vonkctl wrapper ------------------------------------------------------------------


def test_every_call_is_json_and_non_interactive_with_an_explicit_profile() -> None:
    run, seen = runner_returning(0, '{"ok": true}')
    assert Vonkctl("vonkctl", runner=run).call("profile", "progress", profile=10) == {
        "ok": True
    }
    assert seen[0] == [
        "vonkctl",
        "--json",
        "--no-input",
        "--profile",
        "10",
        "profile",
        "progress",
    ]


@pytest.mark.parametrize(
    "verb", ["add", "remove", "load", "cancel", "import", "configure"]
)
def test_profile_writes_refuse_owner_profiles_and_a_missing_profile(verb: str) -> None:
    run, seen = runner_returning(0, "{}")
    vk = Vonkctl("vonkctl", runner=run)
    for number in (1, 2, 3):
        with pytest.raises(OwnerProfileError):
            vk.run("profile", verb, profile=number)
    with pytest.raises(
        OwnerProfileError
    ):  # the default profile is the owner's profile 1
        vk.run("profile", verb)
    assert seen == []  # nothing reached vonkctl
    vk.run("profile", verb, profile=10)
    assert len(seen) == 1


def test_reading_an_owner_profile_is_allowed_and_a_deliberate_restore_can_write_it() -> (
    None
):
    run, seen = runner_returning(0, "{}")
    vk = Vonkctl("vonkctl", runner=run)
    vk.run("profile", "progress", profile=2)
    vk.run("profile", "export", profile=2)
    vk.run("profile", "load", "--yes", profile=2, allow_owner_write=True)
    assert len(seen) == 3
    with pytest.raises(OwnerProfileError):
        vk.run("run", "some-recipe")  # `vonkctl run` edits profile 1


def test_a_blocked_review_exits_2_but_is_still_a_document() -> None:
    run, _ = runner_returning(2, '{"allowed": false}')
    vk = Vonkctl("vonkctl", runner=run)
    assert vk.call("profile", "load", "--review", profile=10, tolerate=(2,)) == {
        "allowed": False
    }
    with pytest.raises(VonkctlError):
        vk.call("profile", "load", "--review", profile=10)


def test_error_documents_are_parsed_from_either_stream() -> None:
    error = json.dumps(
        {
            "error": "x",
            "error_type": "control_api",
            "code": "controller.unavailable",
            "detail": "dockerfile.heredoc_forbidden: no",
        }
    )
    for out, err in ((error, ""), ("", error)):
        run, _ = runner_returning(2, out, err)
        with pytest.raises(VonkctlError) as caught:
            Vonkctl("vonkctl", runner=run).call("recipe", "download", "x")
        assert caught.value.code == "controller.unavailable"
        assert "heredoc" in str(caught.value)


def test_progress_lines_before_the_document_are_tolerated() -> None:
    run, _ = runner_returning(0, 'queued\n{"state": "running"}\n')
    assert Vonkctl("vonkctl", runner=run).call("recipe", "progress", "op") == {
        "state": "running"
    }


def test_request_keys_are_deterministic_so_a_restart_reconnects() -> None:
    assert request_key("download", "a/b", "c1", 1) == request_key(
        "download", "a/b", "c1", 1
    )
    assert request_key("download", "a/b", "c1", 1) != request_key(
        "download", "a/b", "c1", 2
    )


def test_the_real_subprocess_runner_runs_an_executable(tmp_path: Path) -> None:
    script = tmp_path / "vonkctl"
    script.write_text(
        textwrap.dedent("""\
        #!/usr/bin/env python3
        import json, sys
        print(json.dumps({"argv": sys.argv[1:]}))
        sys.exit(0)
    """)
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    assert Vonkctl(str(script)).call("fleet")["argv"] == [
        "--json",
        "--no-input",
        "fleet",
    ]


# -- assertions ------------------------------------------------------------------------


def test_the_assertion_kinds_of_the_reviewed_cases() -> None:
    response = {
        "model": "m",
        "choices": [
            {
                "message": {
                    "content": " 391 ",
                    "tool_calls": [
                        {"function": {"name": "f", "arguments": '{"city": "A"}'}}
                    ],
                },
                "finish_reason": "stop",
            }
        ],
        "data": [{"id": "m"}, {"id": "n"}],
        "usage": {"completion_tokens": 3},
    }
    check_assertions(
        response,
        json.dumps(response),
        [
            {"kind": "path.equals", "path": "model", "value": "m"},
            {"kind": "path.count", "path": "choices", "value": 1},
            {
                "kind": "path.regex",
                "path": "choices.0.message.content",
                "value": r"^\s*391\s*$",
            },
            {"kind": "path.nonempty", "path": "data"},
            {"kind": "path.lte", "path": "usage.completion_tokens", "value": 3},
            {
                "kind": "path.json-equals",
                "path": "choices.0.message.tool_calls.0.function.arguments",
                "value": {"city": "A"},
            },
            {
                "kind": "array.path-count-equals",
                "path": "data",
                "item_path": "id",
                "value": "m",
                "count": 1,
            },
            {"kind": "raw.not-contains", "values": ["<think>"]},
            {"kind": "path.nonempty", "path": "choices"},
        ],
    )
    for bad in (
        {"kind": "path.equals", "path": "model", "value": "other"},
        {"kind": "path.regex", "path": "choices.0.message.content", "value": "^392$"},
        {"kind": "path.lte", "path": "usage.completion_tokens", "value": 2},
        {"kind": "raw.not-contains", "values": ["391"]},
        {"kind": "path.equals", "path": "no.such.path", "value": 1},
        {"kind": "mystery", "path": "model"},
    ):
        with pytest.raises(AssertionFailed):
            check_assertions(response, json.dumps(response), [bad])


# -- the reviewed definitions ----------------------------------------------------------------


@pytest.fixture(scope="module")
def definitions() -> Definitions:
    return Definitions.load()


def test_every_reviewed_service_case_renders_with_its_fixtures(
    definitions: Definitions,
) -> None:
    services = definitions.document["service_recipes"]
    assert services
    for key, service in services.items():
        cases = definitions.service_cases(key, service["alias"])
        assert [c["id"] for c in cases] == service["smoke_cases"], key
        assert json.dumps(cases).count("$ALIAS") == 0 and "$fixture" not in json.dumps(
            cases
        )


def test_a_vision_case_carries_its_image_as_a_data_uri(
    definitions: Definitions,
) -> None:
    key = next(
        k
        for k, s in definitions.document["service_recipes"].items()
        if "V_RED" in s["smoke_cases"]
    )
    case = next(c for c in definitions.service_cases(key, "x") if c["id"] == "V_RED")
    url = case["body"]["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,") and len(url) > 100


def test_the_authority_id_matches_the_derived_authority() -> None:
    from spark_sweep.cli import AUTHORITY_ID

    builder = (ROOT / "tools" / "build-qualification-authority").read_text()
    assert f'AUTHORITY_ID = "{AUTHORITY_ID}"' in builder


# -- smoke against a gateway ----------------------------------------------------------------------


def _recipe(adapter: str = "openai") -> Recipe:
    return Recipe(
        key="vonk-forge/x",
        title="x",
        node_count=1,
        usage=("chat",),
        engine="vllm",
        creator="",
        model_digests=(),
        model_selectors=(),
        memory_bytes=0,
        image_bytes=0,
        artifact_bytes=0,
        ports=(8000,),
        adapter=adapter,
        aliases=("x",),
        content_sha256="c",
        revision_id="r",
        local="cached",
        updated_at="",
        base_image="",
        checks=(
            {
                "kind": "openai.chat",
                "name": "bounded-chat",
                "request": {
                    "body": {
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 8,
                    }
                },
            },
        ),
    )


@pytest.fixture
def gateway():
    gateway = Gateway()
    root = gateway.start()
    yield gateway, root
    gateway.stop()


def test_the_stream_probe_measures_time_to_first_token_and_tokens_per_second(
    gateway,
) -> None:
    _, root = gateway
    probe = stream_probe(f"{root}/x/v1", "x", HttpConfig())
    assert (
        probe["tokens"] == 8
        and probe["ttft_ms"] >= 0
        and probe["tokens_per_second"] > 0
    )


def test_a_serving_recipe_runs_its_reviewed_cases_with_perf(
    gateway, definitions: Definitions
) -> None:
    _, root = gateway
    document = {
        **definitions.document,
        "service_recipes": {
            "vonk-forge/x": {"alias": "x", "smoke_cases": ["M0", "A391"]}
        },
    }
    result = smoke_service(
        _recipe(),
        Definitions(document, definitions.root),
        f"{root}/x/v1",
        "x",
        HttpConfig(),
    )
    assert (
        result.ok
        and result.kind == "service"
        and result.perf
        and [c["case_id"] for c in result.cases] == ["M0", "A391"]
    )


def test_without_reviewed_cases_the_recipes_own_declared_checks_run(
    gateway, definitions: Definitions
) -> None:
    _, root = gateway
    bare = Definitions(
        {**definitions.document, "service_recipes": {}}, definitions.root
    )
    result = smoke_service(_recipe(), bare, f"{root}/x/v1", "x", HttpConfig())
    assert result.ok and [c["case_id"] for c in result.cases] == [
        "models",
        "bounded-chat",
    ]


def test_a_gateway_error_is_a_smoke_failure_with_a_class(
    gateway, definitions: Definitions
) -> None:
    server, root = gateway
    server.broken.add("x")
    document = {
        **definitions.document,
        "service_recipes": {"vonk-forge/x": {"alias": "x", "smoke_cases": ["M0"]}},
    }
    result = smoke_service(
        _recipe(),
        Definitions(document, definitions.root),
        f"{root}/x/v1",
        "x",
        HttpConfig(),
    )
    assert not result.ok and result.failure is not None
    assert (result.failure.phase, result.failure.klass) == ("smoke", "smoke-request")


def test_a_wrong_answer_is_an_assertion_failure_naming_the_case(
    gateway, definitions: Definitions
) -> None:
    _, root = gateway
    case = {
        "id": "A323",
        "method": "POST",
        "path": "/chat/completions",
        "body": {"model": "x", "messages": []},
        "assertions": [
            {
                "kind": "path.regex",
                "path": "choices.0.message.content",
                "value": "^323$",
            }
        ],
    }
    with pytest.raises(AssertionFailed, match="path.regex"):
        run_case(f"{root}/x/v1", case, HttpConfig())


def test_the_client_key_is_sent_as_a_bearer_token(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    assert HttpConfig(key_file=key).headers()["Authorization"] == "Bearer secret-key"


def test_non_openai_recipes_are_checked_for_readiness_only() -> None:
    result = smoke_readiness("run-1")
    assert result.ok and result.kind == "readiness-only" and result.perf is None
