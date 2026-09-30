"""Verify the vLLM CLI sentinel used to disable speculative decoding."""

import os

# This process checks CLI parsing, not inference. Explicit CPU mode lets the
# real vLLM parser construct defaults without requiring a builder GPU.
os.environ["VLLM_TARGET_DEVICE"] = "cpu"

from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def main() -> None:
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    parsed = parser.parse_args(
        ["--model", "/models/parser-only", "--speculative-config", "None"]
    )
    if parsed.speculative_config is not None:
        raise SystemExit(
            "vLLM did not parse --speculative-config None as disabled speculation"
        )


if __name__ == "__main__":
    main()
