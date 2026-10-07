#!/bin/sh
# Verify the Vonk packaging of the ggml-org llama.cpp image at build time.
set -eu
test -x /opt/vonk/bin/llama-server || { echo "Vonk llama-server wrapper is not executable" >&2; exit 1; }
test -x /app/llama-server || { echo "upstream llama-server is missing from the base image" >&2; exit 1; }
# Executability alone misses dynamic-loader failures in the pinned image.
# Match the managed runtime working directory rather than upstream /app.
(cd /tmp && timeout 20s /opt/vonk/bin/llama-server --help >/dev/null)
