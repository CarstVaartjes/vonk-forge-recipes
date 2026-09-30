#!/usr/bin/env python3
"""Host-side tests for chat_template_kwargs.thinking / enable_thinking aliasing."""
import sys
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))

from encoding_compat import (  # noqa: E402
    PUBLISHER_REASONING_EFFORT,
    install_encoder,
    install_serving_chat,
    normalize_chat_template_kwargs,
)


def load_test_encoder(module_name):
    """Prefer the real SGLang module in-image; use the pinned source copy on hosts."""
    try:
        return importlib.import_module('sglang.srt.entrypoints.openai.encoding_dsv41')
    except ImportError:
        reference = (ROOT.parent / 'r0b0tlab-sglang-v41-four' /
                     'references' / 'encoding.py')
        assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
            '502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1')
        spec = importlib.util.spec_from_file_location(module_name, reference)
        encoder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(encoder)
        return encoder


def test_normalize_copies_enable_thinking():
    got = normalize_chat_template_kwargs({'enable_thinking': False})
    assert got == {'enable_thinking': False, 'thinking': False}
    got = normalize_chat_template_kwargs({'enable_thinking': True})
    assert got == {'enable_thinking': True, 'thinking': True}
    got = normalize_chat_template_kwargs({'enable_thinking': 0})
    assert got == {'enable_thinking': False, 'thinking': False}


def test_normalize_copies_thinking():
    got = normalize_chat_template_kwargs({'thinking': False})
    assert got == {'thinking': False, 'enable_thinking': False}
    got = normalize_chat_template_kwargs({'thinking': True, 'foo': 1})
    assert got == {'thinking': True, 'enable_thinking': True, 'foo': 1}


def test_thinking_wins_on_conflict():
    import logging
    logging.disable(logging.WARNING)
    try:
        got = normalize_chat_template_kwargs(
            {'thinking': False, 'enable_thinking': True})
    finally:
        logging.disable(logging.NOTSET)
    assert got['thinking'] is False and got['enable_thinking'] is False


def test_empty_and_none_pass_through():
    assert normalize_chat_template_kwargs(None) is None
    assert normalize_chat_template_kwargs({}) == {}


def test_install_serving_chat_rewrites_request():
    class FakeServing:
        class OpenAIServingChat:
            def _process_messages(self, request, is_multimodal):
                return request.chat_template_kwargs

    install_serving_chat(FakeServing)
    serving = FakeServing.OpenAIServingChat()
    request = SimpleNamespace(chat_template_kwargs={'enable_thinking': False})
    out = serving._process_messages(request, False)
    assert out == {'enable_thinking': False, 'thinking': False}
    assert request.chat_template_kwargs['thinking'] is False


def test_install_encoder_publisher_table():
    mappings = {'low': 25, 'high': 50, 'xhigh': 75, 'max': 100}
    module = SimpleNamespace(REASONING_EFFORT_MAPPINGS=mappings)
    install_encoder(module)
    assert module.REASONING_EFFORT_MAPPINGS == PUBLISHER_REASONING_EFFORT
    assert mappings is module.REASONING_EFFORT_MAPPINGS  # in-place; readers keep the dict
    assert 'xhigh' not in module.REASONING_EFFORT_MAPPINGS


def test_namespaced_openai_request_round_trips_through_encoder_and_parser():
    # This checked-in reference is byte-identical to the official dba1be0a
    # encoding/encoding.py used by the paired Model and is also a complete
    # implementation of the encoder/parser protocol, without model weights.
    encoder = load_test_encoder('deepseek_v41_reference')
    install_encoder(encoder)

    request = [
        {
            'role': 'system', 'content': 'Use available tools.',
            'tools': [{
                'type': 'function',
                'namespace': {'name': 'weather', 'description': 'Weather data.'},
                'function': {
                    'name': 'forecast',
                    'description': 'Get a forecast.',
                    'parameters': {
                        'type': 'object',
                        'properties': {'city': {'type': 'string'},
                                       'days': {'type': 'integer'}},
                        'required': ['city', 'days'],
                    },
                },
            }],
        },
        {'role': 'user', 'content': 'Forecast Paris for two days.'},
        {
            'role': 'assistant', 'content': '',
            'tool_calls': [{
                'type': 'function', 'id': 'call-1', 'namespace': 'weather',
                'function': {'name': 'forecast',
                             'arguments': {'city': 'Paris', 'days': 2}},
            }],
        },
    ]
    prompt = encoder.encode_messages(request, thinking_mode='chat')
    assert '"name": "weather::forecast"' in prompt
    assistant = prompt.rsplit('<｜Assistant｜>', 1)[1]
    completion = assistant.removeprefix('</think>')
    parsed = encoder.parse_message_from_completion_text(completion, 'chat')
    call = parsed['tool_calls'][0]
    assert call['namespace'] == 'weather'
    assert call['function']['name'] == 'forecast'
    assert json.loads(call['function']['arguments']) == {'city': 'Paris', 'days': 2}


def test_current_effort_values_match_encoder_and_reject_removed_xhigh():
    encoder = load_test_encoder('deepseek_v41_effort_reference')
    install_encoder(encoder)
    assert encoder.encode_messages(
        [{'role': 'user', 'content': 'Hi'}], 'thinking', reasoning_effort='low'
    ).startswith('<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 (')
    assert encoder.encode_messages(
        [{'role': 'user', 'content': 'Hi'}], 'thinking', reasoning_effort='high'
    ).startswith('<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (')
    try:
        encoder.encode_messages(
            [{'role': 'user', 'content': 'Hi'}], 'thinking', reasoning_effort='xhigh')
    except AssertionError as exc:
        assert "['low', 'high', 'max']" in str(exc)
    else:
        raise AssertionError('removed xhigh reasoning effort was accepted')


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f'[OK]   {fn.__name__}')
        except Exception as exc:
            failed += 1
            print(f'[FAIL] {fn.__name__}: {exc}')
    print(f'{len(tests) - failed}/{len(tests)} passed')
    sys.exit(1 if failed else 0)
