#!/bin/bash
# The recipe's knobs (knobs.sh), then gr8's own entrypoint (gr8/docker/entrypoint.sh).
set -euo pipefail
. /opt/glm53f-2x/knobs.sh
exec /opt/gr8/docker/entrypoint.sh "$@"
