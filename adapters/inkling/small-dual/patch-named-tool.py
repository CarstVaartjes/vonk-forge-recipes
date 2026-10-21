"""Constrain a named Inkling tool choice to one native invocation."""

from pathlib import Path
import sys

METHOD = r'''    def get_structural_tag(
        self, tools=None, tool_choice="auto", thinking_mode=False,
        parallel_tool_calls=True,
    ):
        # A named choice selects one invocation; auto/required retain the
        # upstream parallel-call policy. SGLang owns the reasoning prefix.
        if isinstance(tool_choice, str):
            return super().get_structural_tag(
                tools, tool_choice, thinking_mode, parallel_tool_calls
            )
        name = tool_choice.function.name
        tool = next(tool for tool in tools or [] if tool.function.name == name)
        return StructuralTag.model_validate({
            "type": "structural_tag",
            "format": {
                "type": "tag",
                "begin": (
                    f"{MESSAGE_MODEL}{name}{CONTENT_INVOKE_TOOL_JSON}"
                    + '{"name":' + json.dumps(name) + ',"args":'
                ),
                "content": {
                    "type": "json_schema",
                    # Keep the argument schema at its own root so local $ref
                    # and $defs keep their original resolution scope.
                    "json_schema": tool.function.parameters,
                },
                "end": "}" + END_MESSAGE,
            },
        })

'''

path = Path(sys.argv[1])
source = path.read_text()
anchor = "    def get_auto_tool_call_structural_tag(\n"
if source.count(anchor) != 1 or "    def get_structural_tag(" in source:
    raise SystemExit("pinned Inkling detector grammar changed; review the patch")
source = source.replace(anchor, METHOD + anchor, 1)
compile(source, str(path), "exec")
path.write_text(source)
