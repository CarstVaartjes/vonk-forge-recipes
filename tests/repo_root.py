"""The one owner of the repository root and the test folders.

Tests import these names instead of deriving a root from ``__file__``, so moving a
test into a subfolder cannot change which tree it scans.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
FIXTURES = TESTS / "fixtures"
