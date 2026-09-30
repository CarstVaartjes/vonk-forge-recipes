"""Verify the vLLM CLI sentinel used to disable speculative decoding."""

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
