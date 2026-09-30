#!/usr/bin/env bash
set -euo pipefail
PORT="${PORT:-8000}"
/usr/bin/python3 - <<PY
import json, sys, urllib.request
port = "${PORT}"
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
        data = json.load(r)
    ids = [m.get('id') for m in data.get('data', [])]
    assert ids, data
    print('healthy', ids)
except Exception as exc:
    print('unhealthy', repr(exc), file=sys.stderr)
    raise SystemExit(1)
PY
