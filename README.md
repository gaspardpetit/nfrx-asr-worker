# nfrx-asr-worker

ASR transcribe worker for NFRX using Verbatim (transcription + diarization).

Quick start (uv):
1. `make install` (CPU) or `make install-cuda`
2. Update `config.yaml` (or set `TRANSCRIBE_CONFIG=/path/to/config.yaml`)
3. `make run` (or `uv run transcribe-worker`)

Notes:
- On macOS, the default backend is `mlx-vibevoice` with model `mlx-community/VibeVoice-ASR-4bit`.
- `mlx-audio` currently conflicts with the Verbatim diarization dependency graph, so install it separately on macOS with `uv pip install "mlx-audio>=0.4.1"` when you want the MLX backend.
- When `mlx-vibevoice` is enabled, external diarization is disabled because VibeVoice already returns speaker-aware segments.
- Override backend selection with `worker.backend`, `ASR_BACKEND`, or `--backend` using `verbatim` or `mlx-vibevoice`.
- `config.yaml` follows Verbatim’s CLI schema for transcription/formatting, plus `worker`/`auth` sections for NFRX.
- Diarization (pyannote) requires a Hugging Face token (`HUGGINGFACE_TOKEN`) and FFmpeg shared libs for torchcodec.
- If you see torchcodec/FFmpeg errors, install FFmpeg 4-7 and set `FFMPEG_DLL_DIR` (Windows) or add `ffmpeg` to PATH.
- Settings read from environment variables, then overridden by `.env` (if present). `config.yaml` can override both.
- Use `output.formats: [txt]` to return raw text, or include `json` for JSON output.
- Set `worker.strategy: "fast"` to use faster-whisper for single-speaker/no-diarization cases.
- Set `LOG_LEVEL=DEBUG` (or `worker.log_level` in config) for verbose tracing.
