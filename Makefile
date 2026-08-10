ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON := $(ROOT)/venv/bin/python
PIP := $(ROOT)/venv/bin/pip
CLOUD_STAMP := $(ROOT)/venv/.setup-cloud
LOCAL_STAMP := $(ROOT)/venv/.setup-local

.PHONY: help setup setup-cloud setup-local local cloud

help:
	@echo "Targets:"
	@echo "  make setup        - full local stack (torch + cloud deps)"
	@echo "  make setup-cloud  - lightweight Replicate-only deps"
	@echo "  make setup-local  - local LoRA deps (includes cloud)"
	@echo "  make local        - setup-local (if needed) + LoRA img2img"
	@echo "  make cloud        - setup-cloud (if needed) + Replicate face-swap"
	@echo ""
	@echo "Photos (cloud): giris.jpg + charlie_kirk.jpg → cikis.png"
	@echo "Photos (local): giris.jpg → cikis.png"
	@echo "Optional: DEVICE=cuda make local"
	@echo "Optional: ARGS='--strength 0.7' make local"

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

# Local Qwen-Image + Charlie Kirk LoRA
# Place source photo at: kirkifiers/kirkify_local/giris.jpg
local: setup-local
	@test -f $(ROOT)/kirkifiers/kirkify_local/giris.jpg || \
		(echo "Missing kirkifiers/kirkify_local/giris.jpg"; exit 1)
	@sz=$$(stat -c%s $(ROOT)/charliekirk-model/charlie_kirk_v2_qwen_image.safetensors 2>/dev/null || echo 0); \
		if [ "$$sz" -lt 1000000 ]; then \
			echo "Missing LoRA weights (or LFS pointer only, $$sz bytes). Run: git lfs pull"; exit 1; \
		fi
	cd $(ROOT)/kirkifiers/kirkify_local && $(PYTHON) kirkify_local.py $(if $(DEVICE),--device $(DEVICE)) $(ARGS)

# Cloud face-swap via Replicate
# Place: giris.jpg + charlie_kirk.jpg → cikis.png
# Requires REPLICATE_API_TOKEN in kirkifiers/.env
cloud: setup-cloud
	@test -f $(ROOT)/kirkifiers/.env || \
		(echo "Missing kirkifiers/.env — copy from .env.example"; exit 1)
	@test -f $(ROOT)/kirkifiers/kirkify_api/giris.jpg || \
		(echo "Missing kirkifiers/kirkify_api/giris.jpg"; exit 1)
	@test -f $(ROOT)/kirkifiers/kirkify_api/charlie_kirk.jpg || \
		(echo "Missing kirkifiers/kirkify_api/charlie_kirk.jpg"; exit 1)
	cd $(ROOT)/kirkifiers/kirkify_api && $(PYTHON) kirkify_api.py $(ARGS)
