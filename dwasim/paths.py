"""Shared locations for data and generated artifacts.

Set DWASIM_WORKDIR to use a separate local research workspace. By default,
artifacts live beside the three entry scripts, never inside the package.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get('DWASIM_WORKDIR', Path(__file__).resolve().parents[1])).resolve()
RAW_ROOT = ROOT / 'data/raw'
PROCESSED_ROOT = ROOT / 'data/processed'
CACHE_ROOT = ROOT / 'cache'
RESULTS_ROOT = ROOT / 'results'
