#!/bin/bash
set -euo pipefail

python3 -m pip install --no-cache-dir \
  av==18.0.0 \
  scipy==1.18.0 \
  soundfile==0.14.0 \
  soxr==1.1.0

python3 - <<'PY'
import av
import scipy
import soundfile
import soxr

print("[inkling-audio-deps] audio imports passed")
PY
