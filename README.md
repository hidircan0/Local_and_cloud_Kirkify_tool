# Kirkify

Toolkit that turns a photo (or video) into a Charlie Kirk look.

| Path | What it does | Command |
| --- | --- | --- |
| **Local photo** | Qwen-Image + Charlie Kirk LoRA img2img | `make local` |
| **Cloud photo** | Replicate multi-face swap | `make cloud` |
| **Cloud video** | Hybrid: per-frame swap or native video model | `make cloud-video` |
| **Local video** | Stub (not implemented yet) | `make local-video` |

LoRA weights: [huwhitememes/charliekirk_v1-2-qwen_image](https://huggingface.co/huwhitememes/charliekirk_v1-2-qwen_image). Trigger: `Ch4rlie K!rk`.

---

## Project layout

```text
kirkify/
├── charliekirk-model/          # LoRA .safetensors (Git LFS)
├── kirkifiers/
│   ├── .env / .env.example
│   ├── models/                 # YuNet face detector
│   ├── common/
│   │   └── replicate_faceswap.py
│   ├── kirkify_local/
│   │   ├── photo/              # make local
│   │   └── video/              # make local-video (stub)
│   └── kirkify_api/
│       ├── photo/              # make cloud
│       └── video/              # make cloud-video
├── Makefile
├── requirements-cloud.txt
├── requirements-local.txt
└── requirements.txt
```

```bash
git lfs install && git lfs pull
```

---

## Requirements

- Python 3.10+
- Cloud: [Replicate](https://replicate.com) API token
- Video: system `ffmpeg` (`sudo pacman -S ffmpeg`)
- Local photo: enough RAM/VRAM

```bash
cp kirkifiers/.env.example kirkifiers/.env
# set REPLICATE_API_TOKEN=
make setup-cloud   # or: make cloud / make cloud-video
```

---

## Photo usage

### Local (`make local`)

Put `giris.jpg` in `kirkifiers/kirkify_local/photo/`, then:

```bash
make local
DEVICE=cuda make local
ARGS='--strength 0.7 --size 768' make local
```

Output: `kirkifiers/kirkify_local/photo/cikis.png`

### Cloud (`make cloud`)

Under `kirkifiers/kirkify_api/photo/`: `giris.jpg` + `charlie_kirk.jpg` → `cikis.png`

```bash
make cloud
ARGS='--no-all-faces' make cloud
```

Pipeline: YuNet detect → NMS → black-mask others → `codeplugtech/face-swap` → paste.

---

## Video usage (cloud)

Under `kirkifiers/kirkify_api/video/`:

| File | Role |
| --- | --- |
| `giris.mp4` | Input video |
| `charlie_kirk.jpg` | Face (symlink to photo by default) |
| `cikis.mp4` | Result |

```bash
# Default: per-frame multi-face (same engine as photo)
make cloud-video

# Draft faster/cheaper: every 3rd frame, first 30 frames
ARGS='--stride 3 --max-frames 30' make cloud-video

# Native Replicate video model (faster wall-clock, different look)
ARGS='--engine native' make cloud-video
ARGS='--engine native --turbo --resolution 720p' make cloud-video
```

**Cost note:** `--engine frames` calls Replicate once per processed face per frame. Prefer `--stride` / `--max-frames` while testing. `--engine native` is one prediction for the whole clip (`prunaai/p-video-replace`).

**Local video** (`make local-video`) is a stub for now.

---

## LoRA model

| Field | Value |
| --- | --- |
| Base | `Qwen/Qwen-Image` |
| Files | `charlie_kirk_v1_qwen_image.safetensors`, `charlie_kirk_v2_qwen_image.safetensors` |
| Trigger | `Ch4rlie K!rk` |

---

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `giris.jpg` / `giris.mp4` missing | Put input in the matching `photo/` or `video/` folder |
| `ffmpeg` required | `sudo pacman -S ffmpeg` |
| empty face-swap / no faces | Overlapping detections (NMS should help); try `--no-all-faces` |
| LoRA looks like LFS pointer | `git lfs pull` (~226 MB each) |
| Rate limited | Wait / lower `--stride` load / top up Replicate credit |
