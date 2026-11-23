"""The one owner of the repository root for the tooling code.

The code lives in ``src/recipe_tooling``; the data it reads (``qualification/``,
``recipes/``, ``models/``) stays at the repository root.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QUALIFICATION_DIR = REPO_ROOT / "qualification"
