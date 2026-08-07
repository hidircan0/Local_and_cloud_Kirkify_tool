ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON := $(ROOT)/venv/bin/python
PIP := $(ROOT)/venv/bin/pip
SETUP_STAMP := $(ROOT)/venv/.setup-complete

.PHONY: help setup local cloud

help:
	@echo "Targets:"
	@echo "  make setup  - create venv and install requirements"
	@echo "  make local  - setup (if needed) + local LoRA img2img"
	@echo "  make cloud  - setup (if needed) + Replicate face-swap"

setup: $(SETUP_STAMP)

$(SETUP_STAMP): $(ROOT)/requirements.txt
	python -m venv $(ROOT)/venv
	$(PIP) install -r $(ROOT)/requirements.txt
	touch $(SETUP_STAMP)

# Local Qwen-Image + Charlie Kirk LoRA (CPU by default)
# Place source photo at: kirkifiers/kirkify_local/foto.jpg
local: setup
	@test -f $(ROOT)/kirkifiers/kirkify_local/foto.jpg || \
		(echo "Missing kirkifiers/kirkify_local/foto.jpg"; exit 1)
	cd $(ROOT)/kirkifiers/kirkify_local && $(PYTHON) kirkify_local.py

# Cloud face-swap via Replicate
# Place source photo at: kirkifiers/kirkify_api/foto.jpg
# Requires REPLICATE_API_TOKEN in kirkifiers/.env
cloud: setup
	@test -f $(ROOT)/kirkifiers/.env || \
		(echo "Missing kirkifiers/.env (need REPLICATE_API_TOKEN)"; exit 1)
	@test -f $(ROOT)/kirkifiers/kirkify_api/foto.jpg || \
		(echo "Missing kirkifiers/kirkify_api/foto.jpg"; exit 1)
	@test -f $(ROOT)/kirkifiers/kirkify_api/charlie_kirk.jpg || \
		(echo "Missing kirkifiers/kirkify_api/charlie_kirk.jpg"; exit 1)
	cd $(ROOT)/kirkifiers/kirkify_api && $(PYTHON) kirkify_api.py
