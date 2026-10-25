"""Make a forced Inkling tool choice exactly one native call from the first token.

SGLang's ReasonerGrammarBackend applies a request's grammar only after the
reasoning parser's end token while the request requires reasoning. Inkling's
end token is <|end_message|>, which closes every block, and its reasoning is
always on, so for tool_choice "required" or a named function the model's first
call ran unconstrained and the grammar then forced a second call. Here such a
request skips reasoning, and its grammar is one invocation:
<|message_model|>name<|content_invoke_tool_json|>{"name":...,"args":<schema>}<|end_message|>.
The model emits <|message_model|> itself (the renderer adds no generation
prompt). Every other request keeps upstream's behaviour. The build fails when
the pinned upstream code this edits has changed.
"""

from importlib.util import find_spec
from pathlib import Path

MARK = "# Vonk: forced Inkling tool call"

REASONING_ANCHOR = '''    def _get_reasoning_from_request(self, request: ChatCompletionRequest) -> bool:
        """Determine whether reasoning mode should be enabled for this request.

        NOTE: This is predefined based on model's chat template
        """
        if not self.reasoning_parser:
            return False
'''
SKIP_REASONING = '''
        if self.reasoning_parser == "inkling" and (
            request.tool_choice == "required"
            or isinstance(request.tool_choice, ToolChoice)
        ):
            # Vonk: forced Inkling tool call: no reasoning phase, so the grammar
            # applies from the first token, not after <|end_message|>.
            return False
'''

DETECTOR_ANCHOR = "    def get_auto_tool_call_structural_tag(\n"
ONE_INVOCATION = '''    def get_structural_tag(
        self, tools=None, tool_choice="auto", thinking_mode=False,
        parallel_tool_calls=True,
    ):
        # Vonk: forced Inkling tool call: exactly one native invocation of the
        # required or named tool; auto keeps upstream's grammar.
        del thinking_mode, parallel_tool_calls
        if tool_choice == "required":
            chosen = list(tools or [])
        elif isinstance(tool_choice, str):
            return None
        else:
            name = tool_choice.function.name
            chosen = [tool for tool in tools or [] if tool.function.name == name]
        tags = [
            {
                "type": "tag",
                "begin": (
                    MESSAGE_MODEL + tool.function.name + CONTENT_INVOKE_TOOL_JSON
                    + '{"name":' + json.dumps(tool.function.name) + ',"args":'
                ),
                "content": {
                    "type": "json_schema",
                    "json_schema": tool.function.parameters or {"type": "object"},
                },
                "end": "}" + END_MESSAGE,
            }
            for tool in chosen
        ]
        if not tags:
            return None
        return StructuralTag.model_validate({
            "type": "structural_tag",
            "format": tags[0] if len(tags) == 1 else {"type": "or", "elements": tags},
        })

'''

EDITS = (
    (
        "srt/entrypoints/openai/serving_chat.py",
        REASONING_ANCHOR,
        REASONING_ANCHOR + SKIP_REASONING,
        (MARK,),
    ),
    (
        "srt/function_call/inkling_detector.py",
        DETECTOR_ANCHOR,
        ONE_INVOCATION + DETECTOR_ANCHOR,
        (MARK, "    def get_structural_tag("),
    ),
)


def main() -> None:
    spec = find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sglang is not installed in this image")
    root = Path(list(spec.submodule_search_locations)[0])
    patched = []
    for relative, anchor, replacement, absent in EDITS:
        path = root / relative
        source = path.read_text()
        if source.count(anchor) != 1 or any(text in source for text in absent):
            raise SystemExit(
                f"{path}: pinned SGLang code changed; review the forced Inkling "
                "tool-call patch"
            )
        source = source.replace(anchor, replacement, 1)
        compile(source, str(path), "exec")
        patched.append((path, source))
    for path, source in patched:
        path.write_text(source)


if __name__ == "__main__":
    main()
