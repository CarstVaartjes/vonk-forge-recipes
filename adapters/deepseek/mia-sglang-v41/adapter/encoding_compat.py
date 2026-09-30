"""Publisher-aligned V4.1 prompt encoding: reasoning-effort table, thinking kwargs,
and a completion-token cap.

SGLang's OpenAI chat path only reads ``chat_template_kwargs.thinking``. vLLM and the
EXL3 recipe use ``enable_thinking``. Without an alias, ``enable_thinking: true`` is
ignored (SGLang's default thinking flag is off) and a client that only sends the
vLLM name cannot turn thinking on.

The pinned official encoder also owns the tool namespace wire format and the
named effort tiers. This hook ports its namespace-aware conversion helpers and
effort map into the SGLang image; the official ``xhigh`` tier was removed.

Chat completions that omit ``max_tokens`` fill the remaining context (1M on TP4).
That is how a racing-animation trial ran to 714k tokens. The 2x EXL3 recipe does
not install a server cap either — vLLM does the same fill — but every smoke and
README example there sends ``max_tokens``. This adapter fills *and* clamps at
``DSV41_MAX_NEW_TOKENS`` (default 32768; 0 restores the uncapped engine behaviour).
"""
import logging
import json
import os

logger = logging.getLogger(__name__)

# DeepSeek V4.1 encoding/encoding.py at dba1be0a. xhigh was removed upstream.
PUBLISHER_REASONING_EFFORT = {'low': 50, 'high': 75, 'max': 100}
DEFAULT_MAX_NEW_TOKENS = 32768


def to_json(value):
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps(value, ensure_ascii=True)


def split_tool_name(name, namespace=None):
    """Return a bare tool name and its namespace, accepting OpenAI's qualified form."""
    prefix, separator, bare_name = name.partition('::')
    if separator:
        assert namespace in (None, prefix), (
            f'Conflicting tool namespaces: {namespace} != {prefix}')
        namespace, name = prefix, bare_name
    assert '::' not in name, f'Tool name must not contain "::": {name}'
    assert namespace is None or '::' not in namespace, (
        f'Tool namespace must not contain "::": {namespace}')
    return namespace, name


def tool_name_for_encoding(tool):
    """Render DeepSeek's namespace::name wire form without losing namespace data."""
    namespace = tool.get('namespace')
    if isinstance(namespace, dict):
        namespace = namespace['name']
    namespace, name = split_tool_name(tool['name'], namespace)
    return name if namespace is None else f'{namespace}::{name}'


def tools_from_openai_format(tools):
    functions = []
    for tool in tools:
        function = dict(tool['function'])
        if tool.get('namespace') is not None:
            function['namespace'] = tool['namespace']
        function['name'] = tool_name_for_encoding(function)
        namespace = function.pop('namespace', None)
        if isinstance(namespace, dict) and namespace.get('description'):
            function['description'] = (
                namespace['description'] + '\n' + (function.get('description') or ''))
        functions.append(function)
    return functions


def tool_calls_from_openai_format(tool_calls):
    calls = []
    for tool_call in tool_calls:
        function = tool_call['function']
        namespace, name = split_tool_name(
            function['name'], tool_call.get('namespace') or function.get('namespace'))
        call = {'name': name, 'arguments': function['arguments']}
        if namespace is not None:
            call['namespace'] = namespace
        calls.append(call)
    return calls


def tool_calls_to_openai_format(tool_calls):
    calls = []
    for tool_call in tool_calls:
        call = {
            'type': 'function',
            'function': {
                'name': tool_call['name'],
                'arguments': tool_call['arguments'],
            },
        }
        if tool_call.get('namespace') is not None:
            call['namespace'] = tool_call['namespace']
        calls.append(call)
    return calls


def decode_dsml_to_arguments(tool_name, tool_args):
    """Decode DSML parameters, returning the namespace as a separate call field."""
    def decode_value(key, value, is_string):
        if is_string == 'true':
            value = to_json(value)
        return f'{to_json(key)}: {value}'

    arguments = '{' + ', '.join(
        decode_value(key, value, is_string)
        for key, (value, is_string) in tool_args.items()) + '}'
    namespace, name = split_tool_name(tool_name)
    call = {'name': name, 'arguments': arguments}
    if namespace is not None:
        call['namespace'] = namespace
    return call


def normalize_chat_template_kwargs(kwargs):
    """Copy ``enable_thinking`` onto ``thinking`` (and the reverse) in a new dict.

    ``thinking`` wins when both are present and disagree. ``None`` / empty input
    is returned unchanged so SGLang's ``SGLANG_DEFAULT_THINKING`` still applies.
    """
    if not kwargs:
        return kwargs
    out = dict(kwargs)
    has_thinking = 'thinking' in out
    has_enable = 'enable_thinking' in out
    if has_thinking and has_enable:
        thinking = bool(out['thinking'])
        enable = bool(out['enable_thinking'])
        if thinking != enable:
            logger.warning(
                'chat_template_kwargs thinking=%r disagrees with enable_thinking=%r; '
                'using thinking', out['thinking'], out['enable_thinking'])
        out['thinking'] = thinking
        out['enable_thinking'] = thinking
    elif has_enable:
        out['thinking'] = bool(out['enable_thinking'])
        out['enable_thinking'] = out['thinking']
    elif has_thinking:
        out['thinking'] = bool(out['thinking'])
        out['enable_thinking'] = out['thinking']
    return out


def configured_max_new_tokens():
    """Return the completion-token default/cap, or 0 to leave the engine uncapped."""
    raw = os.environ.get('DSV41_MAX_NEW_TOKENS', str(DEFAULT_MAX_NEW_TOKENS)).strip()
    try:
        return int(raw)
    except ValueError:
        logger.warning('DSV41_MAX_NEW_TOKENS=%r is not an int; using %s',
                       raw, DEFAULT_MAX_NEW_TOKENS)
        return DEFAULT_MAX_NEW_TOKENS


def apply_request_max_tokens(request):
    """Fill omitted max_tokens / max_completion_tokens and clamp to the cap.

    Mutates ``request`` in place and returns it. ``limit <= 0`` is a no-op.
    ``max_completion_tokens`` wins when both fields are set, matching SGLang.
    """
    limit = configured_max_new_tokens()
    if limit <= 0 or request is None:
        return request
    completion = getattr(request, 'max_completion_tokens', None)
    tokens = getattr(request, 'max_tokens', None)
    current = completion if completion is not None else tokens
    if current is None:
        request.max_tokens = limit
        return request
    if current > limit:
        if completion is not None:
            request.max_completion_tokens = limit
        else:
            request.max_tokens = limit
    return request


def install_encoder(module):
    """Align SGLang's V4.1 encoder with the pinned official dba1be0a encoder."""
    module.REASONING_EFFORT_MAPPINGS.clear()
    module.REASONING_EFFORT_MAPPINGS.update(PUBLISHER_REASONING_EFFORT)
    # These are the globals used by render_message, encode_messages and parsing.
    # Replacing the conversion points keeps the patch local to V4.1 tool wire data.
    module._split_tool_name = split_tool_name
    module._tool_name_for_encoding = tool_name_for_encoding
    module.tools_from_openai_format = tools_from_openai_format
    module.tool_calls_from_openai_format = tool_calls_from_openai_format
    module.tool_calls_to_openai_format = tool_calls_to_openai_format
    module.decode_dsml_to_arguments = decode_dsml_to_arguments


def install_serving_chat(module):
    """Rewrite thinking kwargs and apply the completion-token cap on every chat request."""
    cls = module.OpenAIServingChat
    original = cls._process_messages

    def _process_messages(self, request, is_multimodal):
        request.chat_template_kwargs = normalize_chat_template_kwargs(
            request.chat_template_kwargs)
        return original(self, request, is_multimodal)

    cls._process_messages = _process_messages

    original_convert = getattr(cls, '_convert_to_internal_request', None)
    if original_convert is not None:
        def _convert(self, request, *args, **kwargs):
            apply_request_max_tokens(request)
            return original_convert(self, request, *args, **kwargs)

        cls._convert_to_internal_request = _convert

    limit = configured_max_new_tokens()
    logger.warning(
        'DSV41 chat_template_kwargs: enable_thinking is accepted as an alias for thinking'
        + (f'; omitted max_tokens defaults and caps at {limit}' if limit > 0
           else '; DSV41_MAX_NEW_TOKENS=0 leaves completions uncapped'))
