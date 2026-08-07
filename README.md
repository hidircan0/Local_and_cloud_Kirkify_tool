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
├── charliekirk-model/          # Hugging Face LoRA (v1 + v2 .safetensors)
├── kirkifiers/
│   ├── .env                    # REPLICATE_API_TOKEN (gitignored)
│   ├── kirkify_local/          # Local img2img pipeline
│   │   └── kirkify_local.py
│   └── kirkify_api/            # Replicate face-swap
│       ├── kirkify_api.py
│       └── charlie_kirk.jpg    # Swap target face
├── Makefile
├── model.py                    # Experimental transformers load stub
├── requirements.txt
└── venv/
```

LoRA `.safetensors` weights under `charliekirk-model/` are tracked with **Git LFS**. After clone:

```bash
git lfs install
git lfs pull
```

Root-level duplicate weight files are gitignored.

---

## Requirements

- Python 3.10+ (developed on 3.14)
- Local path: enough RAM/VRAM; the script defaults to **CPU** and resizes to `512×512`
- Cloud path: a [Replicate](https://replicate.com) account and API token

`make local` and `make cloud` run setup automatically when the venv is missing or `requirements.txt` changed. You can still run `make setup` alone if you want.

```bash
make setup   # optional; also implied by local/cloud
```

---

## Environment

`kirkifiers/.env`:

```env
REPLICATE_API_TOKEN=r8_...
```

`.env` is gitignored — do not commit the token.

---

## Usage

### Local LoRA (`make local`)

1. Put the source photo at `kirkifiers/kirkify_local/foto.jpg`.
2. Run:

```bash
make local
```

Output: `kirkifiers/kirkify_local/kirkify_sonuc.png`

Default prompt:

```text
Ch4rlie K!rk face, photorealistic, meme realism, detailed
```

Key parameters in `kirkify_local.py`:

- `strength=0.65` — how strongly the source photo is preserved
- `guidance_scale=7.5`
- `torch.float16` + `low_cpu_mem_usage=True`
- device: `cpu` (use `pipe.to("cuda")` for GPU)

> On CPU, Qwen-Image + LoRA can take a long time; leaving it in the background is normal.

### Cloud face-swap (`make cloud`)

1. Put `REPLICATE_API_TOKEN` in `kirkifiers/.env`.
2. Put the source face at `kirkifiers/kirkify_api/foto.jpg`.
3. Target face is already `charlie_kirk.jpg` (replace if you want).
4. Run:

```bash
make cloud
```

Output: `kirkifiers/kirkify_api/api_sonuc.png`

The script uses the latest `codeplugtech/face-swap` version, swaps `input_image` ↔ `swap_image`, and downloads the result.

---

## LoRA model

| Field | Value |
| --- | --- |
| Base | `Qwen/Qwen-Image` |
| Trainer | WaveSpeedAI LoRA Trainer |
| Steps | ~2000 |
| Rank | 16 |
| Trigger | `Ch4rlie K!rk` |
| License | Apache-2.0 (see model card) |

The repo ships both **v1** and **v2** weights; the local script uses v2.

---

## Notes

- `model.py` tries to load `~/kirkify` with `AutoModelForCausalLM`; it is not part of the img2img/face-swap flow.
- Cloud path needs network and Replicate quota; local path needs model download + disk/RAM.
- Use outputs for personal/meme purposes; commercial or deceptive use may conflict with the model card’s fair-use notes.

---

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `foto.jpg` missing | File not in the script’s directory |
| `REPLICATE_API_TOKEN` error | Missing/empty `kirkifiers/.env` |
| LoRA load error | Incomplete LFS checkout — run `git lfs pull` |
| OOM / very slow | CPU + heavy model; 512×512 and `float16` are already set |

```bash
git lfs install && git lfs pull
```
