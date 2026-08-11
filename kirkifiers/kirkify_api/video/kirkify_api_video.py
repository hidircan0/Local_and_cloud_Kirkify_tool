#!/usr/bin/env python3
"""Cloud Kirkify for video — default native Replicate model; optional per-frame swap."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import replicate

KIRKIFIERS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KIRKIFIERS_DIR.parent))

from kirkifiers.common.replicate_faceswap import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_VERSION,
    download_output,
    load_token,
    resolve_version,
    run_with_retries,
    swap_frame_bgr,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PHOTO_DIR = SCRIPT_DIR.parent / "photo"
ENV_PATH = KIRKIFIERS_DIR / ".env"

DEFAULT_SOURCE = SCRIPT_DIR / "giris.mp4"
DEFAULT_TARGET = SCRIPT_DIR / "charlie_kirk.jpg"
DEFAULT_OUTPUT = SCRIPT_DIR / "cikis.mp4"

NATIVE_MODEL = "prunaai/p-video-replace"
NATIVE_VERSION = "fef98527c7ebe8affa2c4c662dde019990541491e8dba4bd8db6a16f12496abc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cloud Kirkify video")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Input video")
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="Face image (default: video/charlie_kirk.jpg or ../photo/charlie_kirk.jpg)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Output mp4")
    parser.add_argument(
        "--engine",
        choices=("native", "frames"),
        default="native",
        help="native = prunaai/p-video-replace (default); frames = per-frame multi-face swap",
    )
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame (frames engine)")
    parser.add_argument("--max-frames", type=int, default=0, help="Safety cap (0 = all)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Frame-swap model owner/name")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Frame-swap version id")
    parser.add_argument("--native-version", default=NATIVE_VERSION, help="Native video model version")
    parser.add_argument(
        "--all-faces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Multi-face per frame (frames engine)",
    )
    parser.add_argument("--max-faces", type=int, default=10)
    parser.add_argument("--pad", type=float, default=0.7)
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--min-face", type=int, default=80)
    parser.add_argument("--resolution", choices=("720p", "1080p"), default="720p")
    parser.add_argument("--turbo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--env-file", type=Path, default=ENV_PATH)
    return parser.parse_args()


def resolve_face_path(path: Path) -> Path:
    if path.is_file():
        return path
    fallback = PHOTO_DIR / "charlie_kirk.jpg"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"Face image not found: {path} (also tried {fallback})"
    )


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install with: sudo pacman -S ffmpeg")


def mux_audio(silent_video: Path, source_video: Path, out_path: Path) -> None:
    """Copy video from silent_video, audio from source_video if present."""
    ensure_ffmpeg()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(out_path),
    ]
    proc = subprocess_run(cmd)
    if proc.returncode != 0:
        # Fallback: no audio
        shutil.copyfile(silent_video, out_path)


def subprocess_run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def run_frames_engine(
    client: replicate.Client,
    *,
    source: Path,
    face: Path,
    out: Path,
    ref: str,
    stride: int,
    max_frames: int,
    all_faces: bool,
    pad: float,
    score: float,
    min_face: int,
    max_faces: int,
) -> None:
    ensure_ffmpeg()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video {width}x{height} @ {fps:.2f} fps, ~{total} frames, stride={stride}")

    with tempfile.TemporaryDirectory(prefix="kirkify_vid_") as tmp:
        tmp_dir = Path(tmp)
        silent = tmp_dir / "silent.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(silent), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter")

        idx = 0
        processed = 0
        last_swapped = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames > 0 and idx >= max_frames:
                break

            if idx % stride == 0:
                print(f"\n--- frame {idx} ---")
                try:
                    last_swapped = swap_frame_bgr(
                        client,
                        ref,
                        frame,
                        face,
                        pad=pad,
                        score=score,
                        min_face=min_face,
                        max_faces=max_faces,
                        all_faces=all_faces,
                        quiet=False,
                        pace_s=0.5,
                    )
                    processed += 1
                except Exception as exc:
                    print(f"  frame {idx} failed ({exc}); keeping original")
                    last_swapped = frame
            else:
                # Hold last swapped frame for skipped indices
                if last_swapped is None:
                    last_swapped = frame

            out_frame = last_swapped if last_swapped is not None else frame
            if out_frame.shape[1] != width or out_frame.shape[0] != height:
                out_frame = cv2.resize(out_frame, (width, height))
            writer.write(out_frame)
            idx += 1

        cap.release()
        writer.release()
        print(f"\nProcessed {processed} unique frame(s), wrote {idx} total.")

        out.parent.mkdir(parents=True, exist_ok=True)
        mux_audio(silent, source, out)


def run_native_engine(
    client: replicate.Client,
    *,
    source: Path,
    face: Path,
    out: Path,
    version: str,
    resolution: str,
    turbo: bool,
) -> None:
    ref = f"{NATIVE_MODEL}:{version}"
    print(f"Running native {ref} ...")
    with source.open("rb") as video_fh, face.open("rb") as face_fh:
        output = run_with_retries(
            client,
            ref,
            {
                "video": video_fh,
                "images": [face_fh],
                "resolution": resolution,
                "save_audio": True,
                "target_fps": "original",
                "turbo": turbo,
            },
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(download_output(output))
    print(f"Saved {out}")


def main() -> int:
    args = parse_args()

    try:
        token = load_token(args.env_file)
        face = resolve_face_path(args.target)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.source.is_file():
        print(f"Error: missing input video: {args.source}", file=sys.stderr)
        print("Place giris.mp4 in kirkifiers/kirkify_api/video/", file=sys.stderr)
        return 1

    client = replicate.Client(api_token=token)

    try:
        if args.engine == "frames":
            version = args.version or None
            version_id = resolve_version(client, args.model, version)
            ref = f"{args.model}:{version_id}"
            print(f"Engine=frames  model={ref}")
            run_frames_engine(
                client,
                source=args.source,
                face=face,
                out=args.out,
                ref=ref,
                stride=max(1, args.stride),
                max_frames=args.max_frames,
                all_faces=args.all_faces,
                pad=args.pad,
                score=args.score,
                min_face=args.min_face,
                max_faces=args.max_faces,
            )
        else:
            print(f"Engine=native  face={face}")
            run_native_engine(
                client,
                source=args.source,
                face=face,
                out=args.out,
                version=args.native_version,
                resolution=args.resolution,
                turbo=args.turbo,
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("=" * 50)
    print(f"Done: {args.out}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
