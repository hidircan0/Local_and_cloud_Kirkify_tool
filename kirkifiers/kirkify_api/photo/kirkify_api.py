#!/usr/bin/env python3
"""Cloud face-swap kirkify via Replicate (photo)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import replicate

# Allow importing kirkifiers.common when run as a script from photo/
KIRKIFIERS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIRKIFIERS_DIR.parent))

from kirkifiers.common.replicate_faceswap import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_VERSION,
    load_token,
    resolve_version,
    swap_image_file,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = KIRKIFIERS_DIR / ".env"
DEFAULT_SOURCE = SCRIPT_DIR / "giris.jpg"
DEFAULT_TARGET = SCRIPT_DIR / "charlie_kirk.jpg"
DEFAULT_OUTPUT = SCRIPT_DIR / "cikis.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cloud Kirkify (Replicate face-swap)")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Scene / group photo")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Face to apply")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="owner/name on Replicate")
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help="Pinned version id (empty string = newest)",
    )
    parser.add_argument(
        "--all-faces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detect and swap every real face",
    )
    parser.add_argument("--max-faces", type=int, default=10, help="Safety cap")
    parser.add_argument("--pad", type=float, default=0.7, help="Mask/paste padding around face")
    parser.add_argument("--score", type=float, default=0.5, help="YuNet min confidence")
    parser.add_argument(
        "--min-face",
        type=int,
        default=80,
        help="Ignore faces smaller than this (px on short side)",
    )
    parser.add_argument("--env-file", type=Path, default=ENV_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    version = args.version or None

    try:
        token = load_token(args.env_file)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.source.is_file() or not args.target.is_file():
        print(
            f"Error: need both source and target images.\n"
            f"  giris: {args.source}\n"
            f"  face:  {args.target}",
            file=sys.stderr,
        )
        return 1

    client = replicate.Client(api_token=token)

    try:
        version_id = resolve_version(client, args.model, version)
        ref = f"{args.model}:{version_id}"
        print(f"Running {ref} (all_faces={args.all_faces}) ...")
        args.out.parent.mkdir(parents=True, exist_ok=True)

        result = swap_image_file(
            client,
            ref,
            args.source,
            args.target,
            pad=args.pad,
            score=args.score,
            min_face=args.min_face,
            max_faces=args.max_faces,
            all_faces=args.all_faces,
        )

        if not cv2.imwrite(str(args.out), result):
            raise RuntimeError(f"Failed to write {args.out}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("=" * 50)
    print(f"Done: {args.out}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
