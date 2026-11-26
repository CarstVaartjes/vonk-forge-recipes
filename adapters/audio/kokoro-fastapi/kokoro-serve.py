#!/app/.venv/bin/python
"""Run the creator's Kokoro-FastAPI server behind the platform's serving flags.

The creator's image starts ``uvicorn api.src.main:app --host 0.0.0.0 --port 8880``
from ``/app`` (``docker/scripts/entrypoint.sh``) after ``download_model.py`` has put
``v1_0/kokoro-v1_0.pth`` and ``config.json`` under its model directory. Here the
platform's ``pytorch-server`` harness passes ``--served-model-name N --host H --port P``;
the weights are the pinned Model mounted read-only at ``/models/v1_0``, so nothing is
downloaded. The app's settings read the environment at import, so it is set first.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

APP_ROOT = "/app"
MODEL_ROOT = "/models"
KOKORO_MODEL = "kokoro-v1_0"


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--served-model-name", default="kokoro")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    flags, _ = parser.parse_known_args()

    os.environ["MODEL_DIR"] = MODEL_ROOT
    # Download links are written here; the image's /app is read-only at run time.
    os.environ["TEMP_FILE_DIR"] = os.path.join(tempfile.gettempdir(), "kokoro-fastapi")
    os.chdir(APP_ROOT)
    sys.path.insert(0, APP_ROOT)

    import uvicorn
    from api.src.main import app
    from api.src.routers import openai_compatible

    # The speech route accepts only the model names in its mapping; the served name is one.
    openai_compatible._openai_mappings["models"].setdefault(flags.served_model_name, KOKORO_MODEL)
    uvicorn.run(app, host=flags.host, port=flags.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
