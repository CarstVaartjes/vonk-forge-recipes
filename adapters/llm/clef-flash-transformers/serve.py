# SPDX-License-Identifier: Apache-2.0
"""Run the creator's server (app.py, unchanged) behind the platform's serving flags.

The platform's ``pytorch-server`` harness launches ``pytorch-server --served-model-name N
--max-model-len L --host H --port P``; the wrapper passes those here. They become the environment
app.py reads, and the model's SystemOne answer is completed into the platform's
decision response (decision_response.py) where app.py calls ``jsm.systemone``.
"""

from __future__ import annotations

import argparse
import os

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--served-model-name", default="clef-flash")
parser.add_argument("--max-model-len", type=int, default=16384)
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
flags, _ = parser.parse_known_args()

os.environ["CLEF_SNAPSHOT"] = "/models"
os.environ["SERVED_MODEL_NAME"] = flags.served_model_name
os.environ["MAX_LENGTH"] = str(flags.max_model_len)

import app  # noqa: E402  (reads the environment above at import)
from decision_response import platform_response  # noqa: E402

native_systemone = app.jsm.systemone


def systemone(model, processor, request, max_length=16384):  # type: ignore[no-untyped-def]
    return platform_response(request, native_systemone(model, processor, request, max_length=max_length))


app.jsm.systemone = systemone

if __name__ == "__main__":
    import uvicorn

    # One worker: a second would load a second copy of the model.
    uvicorn.run(app.app, host=flags.host, port=flags.port, workers=1, log_level="info", timeout_graceful_shutdown=10)
