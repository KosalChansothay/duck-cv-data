#!/usr/bin/env python3
"""
Objects365 Duck Dataset Extractor (Backward-compatible entry point)
Delegates to the modular `src.data.objects365` package.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_objects365 import main

if __name__ == "__main__":
    main()
