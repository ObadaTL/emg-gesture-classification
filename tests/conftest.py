import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make `emg_classifier` (src-layout package) and `scripts/*.py` importable in tests
# without requiring an editable install first.
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
