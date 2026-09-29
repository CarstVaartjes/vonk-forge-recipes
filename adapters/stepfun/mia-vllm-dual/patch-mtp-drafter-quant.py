"""Build-time form of upstream's MTP drafter patch (templates/launch-mtp.sh).

The Step 3.7 MTP tensors are BF16, so the vLLM step3p5_mtp.py drafter must not
inherit the target's NVFP4 quant_config. Idempotent; fails closed on anchor drift.
"""

from pathlib import Path

candidates: list[Path] = []
for base in (
    Path("/usr/local/lib/python3.12/dist-packages"),
    Path("/usr/local/lib/python3.12/site-packages"),
    Path("/opt/venv/lib/python3.12/site-packages"),
    Path("/opt/env/lib/python3.12/site-packages"),
):
    candidates += list(base.glob("vllm/model_executor/models/step3p5_mtp.py"))

if not candidates:
    raise SystemExit("Could not find step3p5_mtp.py")

path = candidates[0]
source = path.read_text()
if "local_mtp_unquant_hack" in source:
    print(f"[step37-mtp] Patch already present in {path}")
else:
    old = "        quant_config = vllm_config.quant_config\n"
    new = (
        "        # local_mtp_unquant_hack: grafted Step-3.7 MTP tensors are BF16,\n"
        "        # so do not create NVFP4-packed drafter params.\n"
        "        quant_config = None\n"
    )
    if old not in source:
        raise SystemExit(f"Could not find quant_config assignment in {path}")
    path.write_text(source.replace(old, new, 1))
    print(f"[step37-mtp] Patched {path}")
