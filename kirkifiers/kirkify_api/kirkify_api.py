#!/usr/bin/env python3
"""Cloud face-swap kirkify via Replicate."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import replicate
import requests
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR.parent / ".env"
DEFAULT_SOURCE = SCRIPT_DIR / "foto.jpg"
DEFAULT_TARGET = SCRIPT_DIR / "charlie_kirk.jpg"
DEFAULT_OUTPUT = SCRIPT_DIR / "api_sonuc.png"
# Pin a known working model; override with --version if needed.
DEFAULT_MODEL = "codeplugtech/face-swap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cloud Kirkify (Replicate face-swap)")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Input face photo")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Face to swap onto source")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="owner/name on Replicate")
    parser.add_argument(
        "--version",
        default=None,
        help="Optional pinned version id (default: newest listed version)",
    )
    parser.add_argument("--env-file", type=Path, default=ENV_PATH)
    return parser.parse_args()


def load_token(env_file: Path) -> str:
    load_dotenv(env_file)
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError(
            f"REPLICATE_API_TOKEN missing. Set it in {env_file} "
            "(see kirkifiers/.env.example)."
        )
    return token


def resolve_version(client: replicate.Client, model_slug: str, version: str | None) -> str:
    if version:
        return version
    model = client.models.get(model_slug)
    versions = model.versions.list()
    if not versions:
        raise RuntimeError(f"No versions found for model: {model_slug}")
    return versions[0].id


def main() -> int:
    args = parse_args()

    try:
        token = load_token(args.env_file)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.source.is_file() or not args.target.is_file():
        print(
            f"Error: need both source and target images.\n"
            f"  source: {args.source}\n"
            f"  target: {args.target}",
            file=sys.stderr,
        )
        return 1

    client = replicate.Client(api_token=token)

    try:
        version_id = resolve_version(client, args.model, args.version)
        print(f"Running {args.model}:{version_id} ...")

        with args.source.open("rb") as source_fh, args.target.open("rb") as target_fh:
            output = client.run(
                f"{args.model}:{version_id}",
                input={
                    "input_image": source_fh,
                    "swap_image": target_fh,
                },
            )

        image_url = output.url if hasattr(output, "url") else str(output)
        response = requests.get(image_url, timeout=120)
        response.raise_for_status()

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(response.content)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("=" * 50)
    print(f"Done: {args.out}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
