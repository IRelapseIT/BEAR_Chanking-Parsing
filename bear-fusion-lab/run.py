"""Workspace launcher: reuse the existing beta core instead of copying it."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path[:0] = [str(root / "src"), str(root.parent / "bear-text-beta/src")]
from bear_fusion.cli import main

if __name__ == "__main__":
    main()
