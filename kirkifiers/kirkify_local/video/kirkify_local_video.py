#!/usr/bin/env python3
"""Local Kirkify video — skeleton (LoRA frame pipeline not implemented yet)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR / "giris.mp4"
DEFAULT_OUTPUT = SCRIPT_DIR / "cikis.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Kirkify video (stub)")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Local video Kirkify is not implemented yet.", file=sys.stderr)
    print(
        "Planned: extract frames → Qwen-Image + Charlie Kirk LoRA per frame → mux audio.",
        file=sys.stderr,
    )
    print(f"Expected input:  {args.source}", file=sys.stderr)
    print(f"Expected output: {args.out}", file=sys.stderr)
    print("For now use: make cloud-video", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
