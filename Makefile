PYTHON ?= 3.10
FFMPEG_VERSION ?= 7.1.1
FFMPEG_DIR ?= .ffmpeg
UNAME_S := $(shell uname -s)

install:
	uv sync --python $(PYTHON)

install-mlx:
	uv sync --python $(PYTHON) --extra mlx

install-verbatim:
	uv sync --python $(PYTHON) --extra verbatim

install-cpu: install
	uv pip install --python $(PYTHON) torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu --reinstall
	@echo "Note: if diarization fails with torchcodec/FFmpeg errors, install FFmpeg 4-7 and set FFMPEG_DLL_DIR (Windows) or add ffmpeg to PATH."

install-cuda: install
	uv pip install --python $(PYTHON) torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --reinstall
	@echo "Note: if diarization fails with torchcodec/FFmpeg errors, install FFmpeg 4-7 and set FFMPEG_DLL_DIR (Windows) or add ffmpeg to PATH."

run:
ifeq ($(UNAME_S),Darwin)
	$(MAKE) install-mlx
else
	$(MAKE) install-cpu
endif
	uv run transcribe-worker

# Auto-fix with Ruff (format + quick fixes)
.PHONY: fix
fix:
	ruff format .
	ruff check --fix --select I .
	ruff check --fix nfrx transcribe
