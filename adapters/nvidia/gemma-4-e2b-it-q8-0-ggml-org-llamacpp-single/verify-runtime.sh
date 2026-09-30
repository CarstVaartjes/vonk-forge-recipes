#!/bin/sh
# Verify the Vonk packaging of the ggml-org llama.cpp image at build time.
set -eu
test -x /opt/vonk/bin/llama-server || { echo "Vonk llama-server wrapper is not executable" >&2; exit 1; }
test -x /app/llama-server || { echo "upstream llama-server is missing from the base image" >&2; exit 1; }
