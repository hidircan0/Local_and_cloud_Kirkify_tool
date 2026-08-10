# Kirkify

Small experimental toolkit that turns a photo into a Charlie Kirk look.

Two paths:

| Path | What it does | Where it runs |
| --- | --- | --- |
| **Local LoRA** | Image-to-image with `Qwen/Qwen-Image` + Charlie Kirk LoRA | Your machine (CPU/GPU) |
| **Cloud face-swap** | Face swap via Replicate (all faces in group photos) | Cloud (API token required) |

LoRA weights come from [huwhitememes/charliekirk_v1-2-qwen_image](https://huggingface.co/huwhitememes/charliekirk_v1-2-qwen_image). Trigger word: `Ch4rlie K!rk`.

---

## Project layout

```text
kirkify/
├── charliekirk-model/
│   ├── charlie_kirk_v1_qwen_image.safetensors
│   └── charlie_kirk_v2_qwen_image.safetensors   # used by make local
├── kirkifiers/
│   ├── .env.example
│   ├── .env                    # REPLICATE_API_TOKEN (gitignored)
│   ├── kirkify_local/
│   │   └── kirkify_local.py    # giris.jpg → cikis.png
│   └── kirkify_api/
│       ├── kirkify_api.py      # giris.jpg + charlie_kirk.jpg → cikis.png
│       └── charlie_kirk.jpg
├── Makefile
├── requirements-cloud.txt      # lightweight (Replicate only)
├── requirements-local.txt      # torch + diffusers (+ cloud)
└── requirements.txt            # alias → local stack
```

Weights under `charliekirk-model/` are tracked with **Git LFS**:

```bash
git lfs install
git lfs pull
```

---

## Requirements

- Python 3.10+
- Local path: enough RAM/VRAM (`make local` auto-picks CUDA if available)
- Cloud path: [Replicate](https://replicate.com) API token

`make local` / `make cloud` install deps automatically when needed.

```bash
make setup-cloud   # optional; cloud-only (small)
make setup-local   # optional; full torch stack
# or just:
make cloud
make local
```

---

## Environment

```bash
cp kirkifiers/.env.example kirkifiers/.env
# edit REPLICATE_API_TOKEN=
```

---

## Usage

### Local LoRA (`make local`)

1. Put the source photo at `kirkifiers/kirkify_local/giris.jpg`.
2. Run:

```bash
make local
# GPU:
DEVICE=cuda make local
# extra flags:
ARGS='--strength 0.7 --size 768' make local
```

Output: `kirkifiers/kirkify_local/cikis.png`

Useful CLI flags (`kirkify_local.py`):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--image` | `giris.jpg` | Source photo |
| `--lora` | `charlie_kirk_v2_...` | LoRA weights |
| `--out` | `cikis.png` | Output path |
| `--strength` | `0.65` | Img2img strength |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |
| `--size` | `512` | Long-side resize |

### Cloud face-swap (`make cloud`)

Only three images matter:

| File | Role |
| --- | --- |
| `giris.jpg` | Your input photo |
| `charlie_kirk.jpg` | Face to apply |
| `cikis.png` | Result |

1. Set `REPLICATE_API_TOKEN` in `kirkifiers/.env`.
2. Put your photo at `kirkifiers/kirkify_api/giris.jpg`.
3. Run:

```bash
make cloud
```

Output: `kirkifiers/kirkify_api/cikis.png`

Defaults to a **detect → mask others → swap → paste** pipeline (YuNet + `codeplugtech/face-swap`) so group photos get every *swappable* face.

```bash
ARGS='--no-all-faces' make cloud           # only primary face (cheaper)
ARGS='--pad 0.9 --score 0.45' make cloud   # larger mask / more sensitive detect
```

Nearby faces can leave small black mask leftovers — that's expected with this pipeline and usually looks better than AI/original fill.

**Limitation:** strong profile views (side face + cigarette, etc.) are often skipped by InsightFace-based swappers even when detected. Prefer photos where faces look more toward the camera.

---

## LoRA model

| Field | Value |
| --- | --- |
| Base | `Qwen/Qwen-Image` |
| Files | `charlie_kirk_v1_qwen_image.safetensors`, `charlie_kirk_v2_qwen_image.safetensors` |
| Trigger | `Ch4rlie K!rk` |
| License | Apache-2.0 (see model card) |

Local script defaults to **v2**.

---

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `giris.jpg` missing | Put input photo in the script directory |
| `REPLICATE_API_TOKEN` error | Missing/empty `kirkifiers/.env` |
| LoRA looks like LFS pointer | Run `git lfs pull` (file should be ~226 MB) |
| OOM / very slow | Use `DEVICE=cuda` if you have a GPU; lower `--size` |

```bash
git lfs install && git lfs pull
ls -lh charliekirk-model/*.safetensors   # expect ~226M each
```
