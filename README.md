# Kirkify

Small experimental toolkit that turns a photo into a Charlie Kirk look.

Two paths:

| Path | What it does | Where it runs |
| --- | --- | --- |
| **Local LoRA** | Image-to-image with `Qwen/Qwen-Image` + Charlie Kirk LoRA | Your machine (CPU/GPU) |
| **Cloud face-swap** | Face swap via Replicate `codeplugtech/face-swap` | Cloud (API token required) |

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
│   │   └── kirkify_local.py
│   └── kirkify_api/
│       ├── kirkify_api.py
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

1. Put the source photo at `kirkifiers/kirkify_local/foto.jpg`.
2. Run:

```bash
make local
# GPU:
DEVICE=cuda make local
# extra flags:
ARGS='--strength 0.7 --size 768' make local
```

Output: `kirkifiers/kirkify_local/kirkify_sonuc.png`

Useful CLI flags (`kirkify_local.py`):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--image` | `foto.jpg` | Source photo |
| `--lora` | `charlie_kirk_v2_...` | LoRA weights |
| `--out` | `kirkify_sonuc.png` | Output path |
| `--strength` | `0.65` | Img2img strength |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |
| `--size` | `512` | Long-side resize |

### Cloud face-swap (`make cloud`)

1. Set `REPLICATE_API_TOKEN` in `kirkifiers/.env`.
2. Put the source face at `kirkifiers/kirkify_api/foto.jpg`.
3. Target face defaults to `charlie_kirk.jpg`.
4. Run:

```bash
make cloud
ARGS='--out result.png' make cloud
```

Output: `kirkifiers/kirkify_api/api_sonuc.png`

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
| `foto.jpg` missing | File not in the script directory |
| `REPLICATE_API_TOKEN` error | Missing/empty `kirkifiers/.env` |
| LoRA looks like LFS pointer | Run `git lfs pull` (file should be ~226 MB) |
| OOM / very slow | Use `DEVICE=cuda` if you have a GPU; lower `--size` |

```bash
git lfs install && git lfs pull
ls -lh charliekirk-model/*.safetensors   # expect ~226M each
```
