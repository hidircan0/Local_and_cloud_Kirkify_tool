#!/usr/bin/env python3
"""Local image-to-image kirkify using Qwen-Image + Charlie Kirk LoRA."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LORA = REPO_ROOT / "charliekirk-model" / "charlie_kirk_v2_qwen_image.safetensors"
DEFAULT_IMAGE = Path(__file__).resolve().parent / "foto.jpg"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "kirkify_sonuc.png"
BASE_MODEL = "Qwen/Qwen-Image"
DEFAULT_PROMPT = "Ch4rlie K!rk face, photorealistic, meme realism, detailed"
DEFAULT_NEGATIVE = "bad quality, blurry, deformed, distorted face"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    # float16 on CPU is often slow/unstable; keep fp32 there.
    if device == "cuda":
        return torch.float16
    return torch.float32


def assert_real_weight(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"LoRA not found: {path}")
    # Git LFS pointer files are tiny text stubs (~100-200 bytes).
    if path.stat().st_size < 1024:
        raise RuntimeError(
            f"LoRA looks like a Git LFS pointer, not real weights: {path}\n"
            "Run: git lfs pull"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Kirkify (Qwen-Image + LoRA)")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Source photo")
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA, help="LoRA .safetensors path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--size", type=int, default=512, help="Long-side resize target")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=os.getenv("DEVICE", "auto"),
    )
    return parser.parse_args()


def resize_long_side(image: Image.Image, long_side: int) -> Image.Image:
    w, h = image.size
    scale = long_side / max(w, h)
    if scale >= 1.0:
        return image
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)
    dtype = pick_dtype(device)

    try:
        assert_real_weight(args.lora)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.image.is_file():
        print(f"Error: source image not found: {args.image}", file=sys.stderr)
        return 1

    print(f"Loading pipeline on {device} ({dtype})...")
    try:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        pipe.to(device)
        print(f"Applying LoRA: {args.lora}")
        pipe.load_lora_weights(str(args.lora))
    except Exception as exc:
        print(f"Error loading model: {exc}", file=sys.stderr)
        return 1

    init_image = Image.open(args.image).convert("RGB")
    init_image = resize_long_side(init_image, args.size)

    print("Running image-to-image...")
    if device == "cpu":
        print("CPU inference can take a long time; leave it running in the background.")

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=init_image,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
    ).images[0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.out)
    print("=" * 40)
    print(f"Done: {args.out}")
    print("=" * 40)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
