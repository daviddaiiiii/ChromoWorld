#!/usr/bin/env python3
"""Run the maintained ChromoPix PNG-to-polygon inference implementation."""
from pathlib import Path
import runpy


if __name__ == "__main__":
    entry = Path(__file__).resolve().parents[2] / "Analyze/fig2-3/get-polygon_PNG2TXT.py"
    runpy.run_path(str(entry), run_name="__main__")
