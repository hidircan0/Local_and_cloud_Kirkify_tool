#!/usr/bin/env python3
"""Cloud face-swap kirkify via Replicate (multi-face: black-mask others → swap → paste)."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import replicate
import requests
from dotenv import load_dotenv
from replicate.exceptions import ModelError, ReplicateError

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR.parent / ".env"
DEFAULT_SOURCE = SCRIPT_DIR / "giris.jpg"
DEFAULT_TARGET = SCRIPT_DIR / "charlie_kirk.jpg"
DEFAULT_OUTPUT = SCRIPT_DIR / "cikis.png"
YUNET_PATH = SCRIPT_DIR.parent / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

# Known-good single-face swapper (we drive multi-face ourselves).
DEFAULT_MODEL = "codeplugtech/face-swap"
DEFAULT_VERSION = "278a81e7ebb22db98bcba54de985d22cc1abeead2754eb1f2af717247be69b34"


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


def ensure_yunet(path: Path = YUNET_PATH) -> Path:
    if path.is_file() and path.stat().st_size > 1000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face detector → {path}")
    response = requests.get(YUNET_URL, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def detect_faces(
    image_bgr: np.ndarray,
    *,
    score_threshold: float,
    min_face: int,
) -> list[tuple[int, int, int, int]]:
    """Return real face boxes, largest first. Drops tiny false positives."""
    ensure_yunet()
    h, w = image_bgr.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_PATH), "", (w, h), score_threshold, 0.3, 5000
    )
    detector.setInputSize((w, h))
    _retval, faces = detector.detect(image_bgr)
    if faces is None:
        return []

    boxes: list[tuple[int, int, int, int]] = []
    for face in faces:
        x, y, fw, fh = map(int, face[:4])
        x = max(0, x)
        y = max(0, y)
        fw = max(1, min(fw, w - x))
        fh = max(1, min(fh, h - y))
        if min(fw, fh) < min_face:
            continue
        if (fw * fh) < (w * h) * 0.002:
            continue
        boxes.append((x, y, fw, fh))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


def padded_region(
    box: tuple[int, int, int, int],
    *,
    image_shape: tuple[int, ...],
    pad: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    img_h, img_w = image_shape[:2]
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1.0 + pad)
    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(img_w, cx + side / 2))
    y1 = int(min(img_h, cy + side / 2))
    return x0, y0, x1, y1


def mask_other_faces(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    keep_idx: int,
    *,
    pad: float,
) -> np.ndarray:
    """Black out every face except keep_idx so the swapper locks onto one person."""
    out = image.copy()
    for i, box in enumerate(boxes):
        if i == keep_idx:
            continue
        x0, y0, x1, y1 = padded_region(box, image_shape=image.shape, pad=pad)
        out[y0:y1, x0:x1] = 0
    return out


def is_rate_limited(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "throttled" in text or "rate limit" in text


def run_with_retries(client: replicate.Client, ref: str, payload: dict, *, attempts: int = 8):
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return client.run(ref, input=payload)
        except (ReplicateError, ModelError) as exc:
            last = exc
            if is_rate_limited(exc):
                wait = 12 + attempt * 3
                print(f"Rate limited; retry in {wait}s ({attempt + 1}/{attempts})...")
                time.sleep(wait)
                continue
            raise
    assert last is not None
    raise last


def download_output(output) -> bytes:
    if output is None:
        raise RuntimeError(
            "Face-swap returned empty output (often: no usable face / strong profile)."
        )
    if isinstance(output, list):
        output = output[0]
    if hasattr(output, "read"):
        data = output.read()
        if not data:
            raise RuntimeError("Face-swap returned empty file output.")
        return data
    image_url = output.url if hasattr(output, "url") else str(output)
    if not image_url or image_url == "None":
        raise RuntimeError(
            "Face-swap returned empty output (often: no usable face / strong profile)."
        )
    response = requests.get(image_url, timeout=120)
    response.raise_for_status()
    return response.content


def decode_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode swapped image")
    return image


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(a, b)))


def swap_image(
    client: replicate.Client,
    ref: str,
    scene_bgr: np.ndarray,
    face_path: Path,
) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", scene_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("Failed to encode scene image")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(encoded.tobytes())
        tmp_path = Path(tmp.name)

    try:
        with tmp_path.open("rb") as scene_fh, face_path.open("rb") as face_fh:
            output = run_with_retries(
                client,
                ref,
                {"input_image": scene_fh, "swap_image": face_fh},
            )
        return decode_bgr(download_output(output))
    finally:
        tmp_path.unlink(missing_ok=True)


def swap_all_faces(
    client: replicate.Client,
    ref: str,
    scene_path: Path,
    face_path: Path,
    *,
    pad: float,
    score: float,
    min_face: int,
    max_faces: int,
) -> np.ndarray:
    scene = cv2.imread(str(scene_path))
    if scene is None:
        raise RuntimeError(f"Cannot read scene image: {scene_path}")

    boxes = detect_faces(scene, score_threshold=score, min_face=min_face)
    if not boxes:
        raise RuntimeError(
            "No usable faces detected. Try --min-face 40 or --score 0.35"
        )

    boxes = boxes[:max_faces]
    print(f"Detected {len(boxes)} usable face(s) (largest first).")
    for i, (x, y, w, h) in enumerate(boxes):
        print(f"  face {i + 1}: {w}x{h} @ ({x},{y})")

    result = scene.copy()
    swapped_count = 0

    for idx, box in enumerate(boxes):
        print(f"Swapping face {idx + 1}/{len(boxes)}...")
        masked = mask_other_faces(scene, boxes, idx, pad=pad)
        try:
            swapped_full = swap_image(client, ref, masked, face_path)
        except Exception as exc:
            print(f"  skipped face {idx + 1}: {exc}")
            continue

        if swapped_full.shape[:2] != scene.shape[:2]:
            swapped_full = cv2.resize(
                swapped_full,
                (scene.shape[1], scene.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        x0, y0, x1, y1 = padded_region(box, image_shape=scene.shape, pad=pad)
        before = result[y0:y1, x0:x1]
        after = swapped_full[y0:y1, x0:x1]
        diff = mean_abs_diff(before, after)
        if diff < 2.0:
            print(f"  skipped face {idx + 1}: swap looked unchanged (diff={diff:.2f})")
            continue

        # Classic paste (may leave black blocks where masks overlapped — preferred look).
        result[y0:y1, x0:x1] = after
        swapped_count += 1
        print(f"  ok face {idx + 1} (diff={diff:.2f})")
        time.sleep(1)

    if swapped_count == 0:
        raise RuntimeError("No faces could be swapped.")
    print(f"Swapped {swapped_count}/{len(boxes)} face(s).")
    if swapped_count < len(boxes):
        print("Note: some faces were skipped (tiny/profile/failed swap).")
    return result


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

        if args.all_faces:
            result = swap_all_faces(
                client,
                ref,
                args.source,
                args.target,
                pad=args.pad,
                score=args.score,
                min_face=args.min_face,
                max_faces=args.max_faces,
            )
        else:
            scene = cv2.imread(str(args.source))
            if scene is None:
                raise RuntimeError(f"Cannot read {args.source}")
            result = swap_image(client, ref, scene, args.target)

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
