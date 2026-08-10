"""Shared Replicate multi-face swap helpers for photo and video Kirkify."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import replicate
import requests
from dotenv import load_dotenv
from replicate.exceptions import ModelError, ReplicateError

KIRKIFIERS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_YUNET_PATH = KIRKIFIERS_DIR / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

DEFAULT_MODEL = "codeplugtech/face-swap"
DEFAULT_VERSION = "278a81e7ebb22db98bcba54de985d22cc1abeead2754eb1f2af717247be69b34"


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


def ensure_yunet(path: Path = DEFAULT_YUNET_PATH) -> Path:
    if path.is_file() and path.stat().st_size > 1000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face detector → {path}")
    response = requests.get(YUNET_URL, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def box_io_min(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    smaller = min(aw * ah, bw * bh)
    return inter / smaller if smaller > 0 else 0.0


def boxes_same_face(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    iou_thresh: float = 0.2,
    iomin_thresh: float = 0.4,
    center_factor: float = 0.75,
) -> bool:
    if box_iou(a, b) >= iou_thresh or box_io_min(a, b) >= iomin_thresh:
        return True
    acx, acy = a[0] + a[2] / 2, a[1] + a[3] / 2
    bcx, bcy = b[0] + b[2] / 2, b[1] + b[3] / 2
    dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    avg_side = (max(a[2], a[3]) + max(b[2], b[3])) / 2
    return dist < avg_side * center_factor


def nms_faces(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if len(boxes) <= 1:
        return boxes
    ordered = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for box in ordered:
        if any(boxes_same_face(box, prev) for prev in kept):
            continue
        kept.append(box)
    return kept


def detect_faces(
    image_bgr: np.ndarray,
    *,
    score_threshold: float,
    min_face: int,
    yunet_path: Path = DEFAULT_YUNET_PATH,
    quiet: bool = False,
) -> list[tuple[int, int, int, int]]:
    ensure_yunet(yunet_path)
    h, w = image_bgr.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(yunet_path), "", (w, h), score_threshold, 0.3, 5000
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

    before = len(boxes)
    boxes = nms_faces(boxes)
    if before > len(boxes) and not quiet:
        print(f"NMS: merged {before} → {len(boxes)} face box(es).")
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
    out = image.copy()
    for i, box in enumerate(boxes):
        if i == keep_idx:
            continue
        x0, y0, x1, y1 = padded_region(box, image_shape=image.shape, pad=pad)
        out[y0:y1, x0:x1] = 0

    kx0, ky0, kx1, ky1 = padded_region(
        boxes[keep_idx], image_shape=image.shape, pad=0.05
    )
    out[ky0:ky1, kx0:kx1] = image[ky0:ky1, kx0:kx1]
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


def swap_frame_bgr(
    client: replicate.Client,
    ref: str,
    scene: np.ndarray,
    face_path: Path,
    *,
    pad: float = 0.7,
    score: float = 0.5,
    min_face: int = 80,
    max_faces: int = 10,
    all_faces: bool = True,
    quiet: bool = False,
    pace_s: float = 1.0,
) -> np.ndarray:
    """Swap faces on an in-memory BGR frame. Returns swapped frame (or raises)."""
    if not all_faces:
        return swap_image(client, ref, scene, face_path)

    boxes = detect_faces(
        scene, score_threshold=score, min_face=min_face, quiet=quiet
    )
    if not boxes:
        raise RuntimeError(
            "No usable faces detected. Try --min-face 40 or --score 0.35"
        )

    boxes = boxes[:max_faces]
    if not quiet:
        print(f"Detected {len(boxes)} usable face(s) (largest first).")
        for i, (x, y, w, h) in enumerate(boxes):
            print(f"  face {i + 1}: {w}x{h} @ ({x},{y})")

    result = scene.copy()
    swapped_count = 0

    for idx, box in enumerate(boxes):
        if not quiet:
            print(f"Swapping face {idx + 1}/{len(boxes)}...")
        masked = mask_other_faces(scene, boxes, idx, pad=pad)
        try:
            swapped_full = swap_image(client, ref, masked, face_path)
        except Exception as exc:
            if not quiet:
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
            if not quiet:
                print(f"  skipped face {idx + 1}: swap looked unchanged (diff={diff:.2f})")
            continue

        result[y0:y1, x0:x1] = after
        swapped_count += 1
        if not quiet:
            print(f"  ok face {idx + 1} (diff={diff:.2f})")
        if pace_s > 0:
            time.sleep(pace_s)

    if swapped_count == 0:
        raise RuntimeError("No faces could be swapped.")
    if not quiet:
        print(f"Swapped {swapped_count}/{len(boxes)} face(s).")
        if swapped_count < len(boxes):
            print("Note: some faces were skipped (tiny/profile/failed swap).")
    return result


def swap_image_file(
    client: replicate.Client,
    ref: str,
    scene_path: Path,
    face_path: Path,
    *,
    pad: float = 0.7,
    score: float = 0.5,
    min_face: int = 80,
    max_faces: int = 10,
    all_faces: bool = True,
) -> np.ndarray:
    scene = cv2.imread(str(scene_path))
    if scene is None:
        raise RuntimeError(f"Cannot read scene image: {scene_path}")
    return swap_frame_bgr(
        client,
        ref,
        scene,
        face_path,
        pad=pad,
        score=score,
        min_face=min_face,
        max_faces=max_faces,
        all_faces=all_faces,
        quiet=False,
    )
