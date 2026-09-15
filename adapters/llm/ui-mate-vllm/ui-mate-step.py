#!/usr/bin/env python3
"""Run one official UI-Mate prediction without executing the returned actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.ui_mate_agent import UIMateAgent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="UI_Mate")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    arguments = parser.parse_args()

    if not arguments.image.is_file():
        parser.error(f"screenshot is not a file: {arguments.image}")

    agent = UIMateAgent(
        base_url=arguments.base_url,
        model=arguments.model,
        request_timeout=arguments.request_timeout,
    )
    response, actions = agent.predict(
        arguments.instruction,
        {"screenshot": arguments.image.read_bytes()},
    )
    print(
        json.dumps(
            {"response": response, "actions": actions},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0 if actions and actions != ["FAIL"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
