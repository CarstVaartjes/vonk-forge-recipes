"""Honor explicit Nemotron reasoning-off before the template-wide heuristic."""

import hashlib
from pathlib import Path

SOURCE_SHA256 = "ef08da01462536b690c86801ba96ca5b540ee73abd4d1fb1a69c64cc07561341"
ANCHOR = "        self.detector = detector_class(**kwargs)"
REPLACEMENT = """        # The template-wide force_reasoning heuristic also matches the ON
        # branch of hybrid templates when this request rendered the OFF branch.
        if (
            model_type.lower() == "nemotron_3"
            and chat_template_kwargs.get("enable_thinking") is False
        ):
            kwargs["force_reasoning"] = False

        self.detector = detector_class(**kwargs)"""


def patch_parser(path):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise SystemExit("Nemotron reasoning patch: unexpected pinned parser bytes")
    text = source.decode()
    if text.count(ANCHOR) != 1:
        raise SystemExit("Nemotron reasoning patch: expected one detector initializer")
    path.write_text(text.replace(ANCHOR, REPLACEMENT))


if __name__ == "__main__":
    patch_parser(Path("/sgl-workspace/sglang/python/sglang/srt/parser/reasoning_parser.py"))
