ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON := $(ROOT)/venv/bin/python
PIP := $(ROOT)/venv/bin/pip
CLOUD_STAMP := $(ROOT)/venv/.setup-cloud
LOCAL_STAMP := $(ROOT)/venv/.setup-local

API_PHOTO := $(ROOT)/kirkifiers/kirkify_api/photo
API_VIDEO := $(ROOT)/kirkifiers/kirkify_api/video
LOCAL_PHOTO := $(ROOT)/kirkifiers/kirkify_local/photo
LOCAL_VIDEO := $(ROOT)/kirkifiers/kirkify_local/video

.PHONY: help setup setup-cloud setup-local local cloud local-video cloud-video

help:
	@echo "Targets:"
	@echo "  make setup         - full local stack (torch + cloud deps)"
	@echo "  make setup-cloud   - lightweight Replicate-only deps"
	@echo "  make setup-local   - local LoRA deps (includes cloud)"
	@echo "  make local         - photo LoRA img2img"
	@echo "  make cloud         - photo Replicate face-swap"
	@echo "  make cloud-video   - video Replicate (frames|native)"
	@echo "  make local-video   - video local stub (not implemented yet)"
	@echo ""
	@echo "Photo cloud: $(API_PHOTO)/giris.jpg + charlie_kirk.jpg → cikis.png"
	@echo "Video cloud: $(API_VIDEO)/giris.mp4 + charlie_kirk.jpg → cikis.mp4"
	@echo "  ARGS='--engine native' make cloud-video"
	@echo "  ARGS='--stride 3 --max-frames 30' make cloud-video"
	@echo "Optional: DEVICE=cuda make local"

setup: setup-local

setup-cloud: $(CLOUD_STAMP)

setup-local: $(LOCAL_STAMP)

$(CLOUD_STAMP): $(ROOT)/requirements-cloud.txt
	python -m venv $(ROOT)/venv
	$(PIP) install -r $(ROOT)/requirements-cloud.txt
	touch $(CLOUD_STAMP)

$(LOCAL_STAMP): $(ROOT)/requirements-local.txt $(ROOT)/requirements-cloud.txt
	python -m venv $(ROOT)/venv
	$(PIP) install -r $(ROOT)/requirements-local.txt
	touch $(CLOUD_STAMP)
	touch $(LOCAL_STAMP)

# Local Qwen-Image + Charlie Kirk LoRA (photo/)
local: setup-local
	@test -f $(LOCAL_PHOTO)/giris.jpg || \
		(echo "Missing $(LOCAL_PHOTO)/giris.jpg"; exit 1)
	@sz=$$(stat -c%s $(ROOT)/charliekirk-model/charlie_kirk_v2_qwen_image.safetensors 2>/dev/null || echo 0); \
		if [ "$$sz" -lt 1000000 ]; then \
			echo "Missing LoRA weights (or LFS pointer only, $$sz bytes). Run: git lfs pull"; exit 1; \
		fi
	cd $(LOCAL_PHOTO) && $(PYTHON) kirkify_local.py $(if $(DEVICE),--device $(DEVICE)) $(ARGS)

# Cloud face-swap via Replicate (photo/)
cloud: setup-cloud
	@test -f $(ROOT)/kirkifiers/.env || \
		(echo "Missing kirkifiers/.env — copy from .env.example"; exit 1)
	@test -f $(API_PHOTO)/giris.jpg || \
		(echo "Missing $(API_PHOTO)/giris.jpg"; exit 1)
	@test -f $(API_PHOTO)/charlie_kirk.jpg || \
		(echo "Missing $(API_PHOTO)/charlie_kirk.jpg"; exit 1)
	cd $(API_PHOTO) && $(PYTHON) kirkify_api.py $(ARGS)

# Cloud video (default engine=frames)
cloud-video: setup-cloud
	@test -f $(ROOT)/kirkifiers/.env || \
		(echo "Missing kirkifiers/.env — copy from .env.example"; exit 1)
	@test -f $(API_VIDEO)/giris.mp4 || \
		(echo "Missing $(API_VIDEO)/giris.mp4"; exit 1)
	@command -v ffmpeg >/dev/null || (echo "ffmpeg required: sudo pacman -S ffmpeg"; exit 1)
	cd $(API_VIDEO) && $(PYTHON) kirkify_api_video.py $(ARGS)

# Local video stub
local-video:
	cd $(LOCAL_VIDEO) && $(PYTHON) kirkify_local_video.py $(ARGS)
