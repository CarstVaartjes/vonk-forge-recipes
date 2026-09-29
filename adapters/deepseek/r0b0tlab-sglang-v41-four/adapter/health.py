#!/usr/bin/env python3
"""Container healthcheck: GET /health, exit 0/1. Run with python3 -S —
imports only urllib to stay light and avoid engine imports."""
import sys
import urllib.request


def main() -> int:
    import os
    port = os.environ.get("PORT", "30000")
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=20
        ) as r:
            return 0 if r.status == 200 else 1
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
