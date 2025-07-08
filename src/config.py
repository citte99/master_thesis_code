# config.py
from pathlib import Path
import os

# Define the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define directories
CATALOGS_DIR = PROJECT_ROOT / "catalogs"
TRAINED_CLASSIFIERS_DIR = PROJECT_ROOT / "trained_classifiers"
PSFS_DIR = PROJECT_ROOT / "psfs"
# Ensure directories exist
for directory in [CATALOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

