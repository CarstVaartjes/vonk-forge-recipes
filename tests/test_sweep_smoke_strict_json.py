"""The sweep reads model replies as strictly as the platform: NaN and duplicate keys are refused."""

from __future__ import annotations

import io

import pytest

from spark_sweep import smoke


class _Reply(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


STRICT_REFUSED = [
    b'{"choices": [{"message": {"content": "hi"}}], "score": NaN}',
    b'{"choices": [{"message": {"content": "hi"}}], "score": Infinity}',
    b'{"choices": [{"message": {"content": "hi"}}], "choices": []}',
]


@pytest.mark.parametrize("reply", STRICT_REFUSED)
def test_run_case_refuses_a_non_strict_reply(monkeypatch, reply: bytes) -> None:
    monkeypatch.setattr(smoke, "_open", lambda *_: _Reply(reply))
    case = {"id": "c", "method": "GET", "path": "/x"}
    with pytest.raises(ValueError):
        smoke.run_case("http://x", case, smoke.HttpConfig())


@pytest.mark.parametrize("value", ['{"a": 1, "a": 1}'])
def test_json_equals_refuses_a_non_strict_value(value: str) -> None:
    assertion = {"kind": "path.json-equals", "path": "v", "value": {"a": 1}}
    with pytest.raises(smoke.AssertionFailed):
        smoke.check_assertions({"v": value}, "", [assertion])
