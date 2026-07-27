"""Repository-relative paths shared by experiment entry points."""

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("BSPC_DATA_DIR", PACKAGE_ROOT / "data" / "eyesdata_processed_57"))
RESULTS_DIR = Path(os.environ.get("BSPC_RESULTS_DIR", PACKAGE_ROOT / "results"))

