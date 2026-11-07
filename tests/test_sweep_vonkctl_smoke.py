"""The vonkctl wrapper, the smoke tests and the reviewed definitions they run."""

from __future__ import annotations

import ast
import dataclasses
import json
import stat
import textwrap
from pathlib import Path

import pytest
from sweep_fakes import Gateway
from vonk_forge_contracts.api import (
    ApiReasoningAlwaysOn,
    ApiReasoningNone,
    ApiReasoningSwitch,
    ApiRemainingContext,
    ApiTokenArgument,
    ApiTokenSetting,
    ApiTokenValue,
    ChatApi,
)
from vonk_forge_contracts.sweep import CliObservationStatus

from qualification.documents import recipe, recipe_api
from spark_sweep.catalog import Recipe
from spark_sweep.chat_probe import ChatFacts, chat_facts
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


def test_error_documents_are_parsed_from_stdout() -> None:
    error = json.dumps(
        {
            "error": "x",
            "error_type": "control_api",
            "code": "controller.unavailable",
            "detail": "dockerfile.heredoc_forbidden: no",
        }
    )
    for out, err in ((error, ""),):
        run, _ = runner_returning(2, out, err)
        with pytest.raises(VonkctlError) as caught:
            Vonkctl("vonkctl", runner=run).call("recipe", "download", "x")
        assert caught.value.code == "controller.unavailable"
        assert "heredoc" in str(caught.value)


@pytest.mark.parametrize(
    "out, err", [('queued\n{"state": "running"}\n', ""), ("", '{"state": "running"}')]
)
def test_only_a_single_stdout_json_document_is_a_receipt(out: str, err: str) -> None:
    run, _ = runner_returning(0, out, err)
    with pytest.raises(VonkctlError):
        Vonkctl("vonkctl", runner=run).call("recipe", "progress", "op")


@pytest.mark.parametrize("code", [0, 2, 124])
@pytest.mark.parametrize("out", ["", "null", "not JSON"])
def test_missing_json_is_transport_with_exit_and_stderr(code: int, out: str) -> None:
    from spark_sweep.verdict import Disposition, reply_disposition
    from spark_sweep.vonkctl import VonkctlTransportError

    run, _ = runner_returning(code, out, "injected connection failure")
    vk = Vonkctl(runner=run)
    result = vk.run("fleet")
    assert not result.ok
    assert reply_disposition(result) is Disposition.WAIT
    with pytest.raises(VonkctlTransportError) as caught:
        vk.call("fleet", tolerate=(2, 124))
    assert caught.value.reply is not None
    assert caught.value.reply.stdout == out
    assert f"exit {code}" in str(caught.value)
    assert "injected connection failure" in str(caught.value)


def test_real_cli_observation_envelope_is_not_a_controller_reply(
    tmp_path: Path,
) -> None:
    """Captured GET remove-review output from cli_outcome.Observation.document()."""
    from spark_sweep.verdict import Disposition, reply_disposition
    from spark_sweep.vonkctl import VonkctlTransportError

    document = json.loads(
        (ROOT / "tests/fixtures/sweep-removal/observation.json").read_text()
    )
    # Even a retained successful receipt is stale observation evidence, not a
    # fresh response that can authorize effects or settle an outstanding request.
    from spark_sweep.lifecycle import LifecycleState

    document["result"] = {"state": LifecycleState.SUCCEEDED}
    script = tmp_path / "vonkctl"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"print({json.dumps(document)!r})\n"
        "sys.exit(2)\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    vk = Vonkctl(str(script))
    result = vk.run("model", "remove", "a" * 64, "--fleet", "--review")
    assert not result.ok
    assert reply_disposition(result) is Disposition.WAIT
    with pytest.raises(VonkctlTransportError, match="control API connection was lost"):
        vk.call("model", "remove", "a" * 64, "--fleet", "--review", tolerate=(2,))
    assert CliObservationStatus.TIMED_OUT in result.error_text
    assert "exit 2" in result.error_text


def test_controller_error_keeps_its_cause_when_observation_is_attached() -> None:
    """cli._control_error attaches observation metadata to final refusals too."""
    from spark_sweep.vonkctl import VonkctlTransportError

    error = {
        "error_type": "control_api",
        "code": "controller.not_found",
        "detail": "Model not found",
        "observation": {"status": "ended"},
        "result": {},
    }
    run, _ = runner_returning(2, json.dumps(error))
    vk = Vonkctl(runner=run)
    assert vk.run("model", "show", "missing").is_not_found
    with pytest.raises(VonkctlError) as caught:
        vk.call("model", "show", "missing", tolerate=(2,))
    assert not isinstance(caught.value, VonkctlTransportError)
    assert caught.value.code == error["code"]
    assert error["detail"] in str(caught.value)


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


REASONING = "choices.0.message.reasoning_content"
NO_REASONING = {"kind": "path.empty", "path": REASONING}
SOME_REASONING = {"kind": "path.nonempty", "path": REASONING}


@pytest.mark.parametrize(
    "reasoning", [{}, {"reasoning_content": None}, {"reasoning_content": ""}]
)
def test_thinking_off_accepts_reasoning_absent_null_or_empty(
    reasoning: dict[str, object],
) -> None:
    response = {"choices": [{"message": {"content": "391", **reasoning}}]}
    check_assertions(response, json.dumps(response), [NO_REASONING])


def test_thinking_off_refuses_reasoning_and_a_missing_parent() -> None:
    response = {
        "choices": [{"message": {"content": "391", "reasoning_content": "hmm"}}]
    }
    with pytest.raises(AssertionFailed):
        check_assertions(response, json.dumps(response), [NO_REASONING])
    with pytest.raises(AssertionFailed, match="is missing"):
        check_assertions({"choices": []}, "{}", [NO_REASONING])


@pytest.mark.parametrize(
    "reasoning", [{}, {"reasoning_content": None}, {"reasoning_content": ""}]
)
def test_thinking_on_still_requires_reasoning(reasoning: dict[str, object]) -> None:
    response = {"choices": [{"message": {"content": "391", **reasoning}}]}
    with pytest.raises(AssertionFailed):
        check_assertions(response, json.dumps(response), [SOME_REASONING])
    present = {
        "choices": [{"message": {"content": "391", "reasoning_content": "17*23"}}]
    }
    check_assertions(present, json.dumps(present), [SOME_REASONING])


LEAK = {"kind": "raw.not-contains", "values": ["<|", "<think>", "</think>"]}


def _chat(message: dict[str, object], **choice: object) -> dict[str, object]:
    return {
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, **choice}],
    }


def test_a_stop_token_named_in_server_metadata_is_not_a_leak() -> None:
    """SGLang names the stop it matched; the answer itself is clean (phi-4, A323)."""
    response = _chat(
        {
            "role": "assistant",
            "content": "The product of 17 multiplied by 19 is 323.",
            "reasoning_content": None,
            "tool_calls": None,
        },
        finish_reason="stop",
        matched_stop="<|end|>",
        provider_specific_fields={"matched_stop": "<|end|>"},
    )
    check_assertions(response, json.dumps(response), [LEAK])


@pytest.mark.parametrize(
    "message",
    [
        {"content": "323<|im_end|>"},
        {"content": "<think>17 times 19</think>323"},
        {"content": "323", "reasoning_content": "<|channel>analysis"},
        {"content": [{"type": "text", "text": "323<|end|>"}]},
        {
            "content": None,
            "tool_calls": [{"function": {"name": "f", "arguments": "<|im_end|>"}}],
        },
    ],
)
def test_a_token_in_the_text_the_user_receives_still_fails(
    message: dict[str, object],
) -> None:
    response = _chat({"role": "assistant", **message})
    with pytest.raises(AssertionFailed, match="raw.not-contains"):
        check_assertions(response, json.dumps(response), [LEAK])


def test_completion_text_is_checked_too() -> None:
    response = {"choices": [{"text": "323</think>"}]}
    with pytest.raises(AssertionFailed):
        check_assertions(response, json.dumps(response), [LEAK])


def test_a_body_that_is_not_a_json_object_is_checked_raw() -> None:
    raw = "323<|im_end|> not json"
    with pytest.raises(AssertionFailed):
        check_assertions(None, raw, [LEAK])
    check_assertions(None, "323 and nothing else", [LEAK])


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


def test_a_timeout_is_a_failed_reply_for_reads_and_an_error_for_calls() -> None:
    from spark_sweep.vonkctl import VonkctlTimeout

    def run(argv, timeout):
        raise VonkctlTimeout(argv, None, "vonkctl timed out after 5s")

    vk = Vonkctl("vonkctl", runner=run)
    assert not vk.run("fleet").ok
    with pytest.raises(VonkctlError):
        vk.call("fleet")


# -- the declared bounded chat follows the recipe's own reasoning facts -------------------------

_BASE = "agents-a1-nvfp4-r0b0tlab-vllm-single"
_ARGUMENT = "deepseek-v4-1-flash-1017-mia-tensorfold-dual"  # --max-tokens 32768
_OPTION = (
    "deepseek-v4-1-flash-exl3-sfxnz-vllm-dual"  # option vision=text-only has a block
)


def _facts(
    slug: str = _BASE, reasoning=None, default_output=None, max_output=None
) -> ChatFacts:
    """The real recipe and facts of ``slug``, with the named facts replaced."""
    api = recipe_api(slug, ROOT).api
    assert isinstance(api, ChatApi)
    update = {
        key: value
        for key, value in (
            ("reasoning", reasoning),
            ("default_output_tokens", default_output),
            ("max_output_tokens", max_output),
        )
        if value is not None
    }
    return ChatFacts(api.model_copy(update=update), recipe(slug, ROOT))


_SWITCH = ApiReasoningSwitch(
    kind="switch",
    via="chat_template_kwargs",
    keys=["enable_thinking"],
    on=True,
    off=False,
    default="on",
    effort=None,
)
_ALWAYS_ON = ApiReasoningAlwaysOn(kind="always-on", effort=None)


def _bounded_chat_smoke(gateway, definitions, facts, max_tokens=64, thinking=200):
    server, root = gateway
    server.thinking["x"] = thinking
    recipe_row = _recipe()
    check = {
        **recipe_row.checks[0],
        "request": {
            "body": {
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": max_tokens,
            }
        },
    }
    recipe_row = dataclasses.replace(recipe_row, checks=(check,))
    bare = Definitions(
        {**definitions.document, "service_recipes": {}}, definitions.root
    )
    result = smoke_service(recipe_row, bare, f"{root}/x/v1", "x", HttpConfig(), facts)
    return result, server.bodies[-1]


def test_a_switch_route_is_asked_with_its_declared_off_value(
    gateway, definitions: Definitions
) -> None:
    result, sent = _bounded_chat_smoke(gateway, definitions, _facts(reasoning=_SWITCH))
    assert result.ok, result.failure
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["max_tokens"] == 64


def test_every_declared_key_at_a_top_level_placement_is_switched_off(
    gateway, definitions: Definitions
) -> None:
    switch = _SWITCH.model_copy(
        update={"via": "top_level", "keys": ["thinking", "think"]}
    )
    result, sent = _bounded_chat_smoke(gateway, definitions, _facts(reasoning=switch))
    assert result.ok, result.failure
    assert (sent["thinking"], sent["think"]) == (False, False)
    assert "chat_template_kwargs" not in sent


def test_an_always_on_route_with_a_numeric_default_gets_that_allowance_capped(
    gateway, definitions: Definitions
) -> None:
    facts = _facts(
        reasoning=_ALWAYS_ON,
        default_output=ApiTokenValue(value=1024),
        max_output=ApiTokenValue(value=512),
    )
    result, sent = _bounded_chat_smoke(gateway, definitions, facts, thinking=300)
    assert result.ok, result.failure
    assert sent["max_tokens"] == 512


def test_an_always_on_route_reads_its_default_from_the_runtime_argument(
    gateway, definitions: Definitions
) -> None:
    facts = _facts(
        _ARGUMENT,
        reasoning=_ALWAYS_ON,
        default_output=ApiTokenArgument(argument="max-tokens"),
        max_output=ApiTokenSetting(setting="context_tokens"),
    )
    result, sent = _bounded_chat_smoke(gateway, definitions, facts, thinking=5000)
    assert result.ok, result.failure
    assert sent["max_tokens"] == 32768


def test_an_always_on_route_defaulting_to_the_remaining_context_sends_no_cap(
    gateway, definitions: Definitions
) -> None:
    """The engine thinks far past any fixed budget; with no cap the reply still answers."""
    facts = _facts(
        reasoning=_ALWAYS_ON, default_output=ApiRemainingContext(remaining="context")
    )
    result, sent = _bounded_chat_smoke(gateway, definitions, facts, thinking=20000)
    assert result.ok, result.failure
    assert "max_tokens" not in sent


def test_a_route_without_reasoning_is_sent_as_declared(
    gateway, definitions: Definitions
) -> None:
    result, sent = _bounded_chat_smoke(
        gateway,
        definitions,
        _facts(reasoning=ApiReasoningNone(kind="none")),
        thinking=0,
    )
    assert result.ok, result.failure
    assert sent == {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
        "model": "x",
        "stream": False,
    }


def test_the_option_choice_of_the_run_selects_the_block(
    gateway, definitions: Definitions
) -> None:
    api = recipe_api(_OPTION, ROOT)
    always = api.option_choices[0].api.model_copy(update={"reasoning": _ALWAYS_ON})
    api = api.model_copy(
        update={
            "option_choices": [api.option_choices[0].model_copy(update={"api": always})]
        }
    )
    definition = recipe(_OPTION, ROOT)
    choice = {api.option_choices[0].option: api.option_choices[0].value}
    base = chat_facts(definition, api, {})
    chosen = chat_facts(definition, api, choice)
    assert base is not None and chosen is not None
    assert isinstance(base.api.reasoning, ApiReasoningSwitch)
    assert isinstance(chosen.api.reasoning, ApiReasoningAlwaysOn)


def test_without_facts_the_declared_check_runs_as_written_with_its_old_assertion(
    gateway, definitions: Definitions
) -> None:
    """A recipe with no published facts keeps main's assertion: it is not newly failed.

    The real fix for those recipes is publishing their api documents.
    """
    result, sent = _bounded_chat_smoke(gateway, definitions, None, thinking=200)
    assert result.ok, result.failure
    assert "chat_template_kwargs" not in sent and sent["max_tokens"] == 64


def test_with_facts_a_reply_that_is_all_thinking_fails_the_case() -> None:
    cut = {"choices": [{"message": {"content": " ", "reasoning_content": "hmm"}}]}
    with pytest.raises(AssertionFailed):
        check_assertions(cut, json.dumps(cut), [{"kind": "chat.nonempty"}])


def test_chat_nonempty_accepts_text_or_tool_calls_like_the_controller() -> None:
    text = {"choices": [{"message": {"content": "hi"}}]}
    tools = {"choices": [{"message": {"content": None, "tool_calls": [{"id": "1"}]}}]}
    for response in (text, tools):
        check_assertions(response, json.dumps(response), [{"kind": "chat.nonempty"}])


def test_every_recipe_declared_chat_case_goes_through_the_one_helper() -> None:
    """No second sender: a case built from a recipe's declared check uses ``bounded_chat_body``."""
    tree = ast.parse((ROOT / "spark_sweep/smoke.py").read_text())
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "body"
        and isinstance(node.value, ast.Name)
        and node.value.id == "request"
    ]
    assert reads, "the declared check body is read in smoke.py"
    # Every read of the declared body sits inside the arguments of the one helper.
    covered = {
        id(inner)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bounded_chat_body"
        for argument in node.args
        for inner in ast.walk(argument)
    }
    assert all(id(read) in covered for read in reads)
    for path in (ROOT / "spark_sweep").glob("*.py"):
        if path.name == "smoke.py":
            continue
        readers = [
            node
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Attribute)
            and node.attr == "checks"
            and isinstance(node.value, ast.Name)
            and node.value.id == "recipe"
        ]
        assert not readers, f"{path.name} reads recipe-declared checks outside smoke.py"
