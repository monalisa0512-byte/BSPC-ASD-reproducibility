"""Repository-relative paths shared by figure-generation entry points."""

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("BSPC_DATA_DIR", PACKAGE_ROOT / "data" / "eyesdata_processed_57"))
RAW_DIR = Path(os.environ.get("BSPC_RAW_DIR", PACKAGE_ROOT / "data" / "raw"))
RESULTS_DIR = Path(os.environ.get("BSPC_RESULTS_DIR", PACKAGE_ROOT / "results"))
OUTPUT_DIR = Path(os.environ.get("BSPC_FIGURE_OUTPUT_DIR", PACKAGE_ROOT / "generated_figures"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

