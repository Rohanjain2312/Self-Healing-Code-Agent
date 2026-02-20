"""
HuggingFace Spaces entry point.

HF Spaces expects app.py in the repository root.
This file simply delegates to demo/app.py.
"""

import sys
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from demo.app import build_app

demo = build_app()
demo.launch()
