"""Register EXL3 in the pinned vLLM quantization registry (image build)."""

from pathlib import Path

import vllm

site = Path(vllm.__file__).resolve().parent
init = site / "model_executor/layers/quantization/__init__.py"
text = init.read_text()
changed = False
if '"exl3"' not in text.split("QuantizationMethods", 1)[-1][:800]:
    old = 'QuantizationMethods = Literal[\n    "awq",\n'
    new = 'QuantizationMethods = Literal[\n    "exl3",\n    "awq",\n'
    if text.count(old) == 1:
        text = text.replace(old, new)
        changed = True
    elif "QuantizationMethods = Literal[" in text and '"exl3"' not in text:
        text = text.replace(
            "QuantizationMethods = Literal[\n",
            'QuantizationMethods = Literal[\n    "exl3",\n',
            1,
        )
        changed = True
lazy = (
    "    # Update the `method_to_config` with customized quantization methods.\n"
    "    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)\n"
)
if "from .exl3 import Exl3Config" not in text and lazy in text:
    text = text.replace(
        lazy,
        "    from .exl3 import Exl3Config\n"
        '    method_to_config["exl3"] = Exl3Config\n' + lazy,
        1,
    )
    changed = True
if changed:
    init.write_text(text)
    print("registered exl3 in QUANTIZATION_METHODS")
else:
    print(
        "exl3 already present in quantization registry (overlay will replace the class)"
    )
