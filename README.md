# nfrx-asr-worker

ASR transcribe worker for NFRX using Verbatim (transcription + diarization).

Quick start (uv):
1. `make install` (CPU) or `make install-cuda`
2. Update `config.yaml` (or set `TRANSCRIBE_CONFIG=/path/to/config.yaml`)
3. `make run` (or `uv run transcribe-worker`)

Notes:
- `config.yaml` follows Verbatim’s CLI schema for transcription/formatting, plus `worker`/`auth` sections for NFRX.
- Diarization (pyannote) requires a Hugging Face token (`HUGGINGFACE_TOKEN`) and FFmpeg shared libs for torchcodec.
- If you see torchcodec/FFmpeg errors, install FFmpeg 4-7 and set `FFMPEG_DLL_DIR` (Windows) or add `ffmpeg` to PATH.
- Settings read from environment variables, then overridden by `.env` (if present). `config.yaml` can override both.
- Use `output.formats: [txt]` to return raw text, or include `json` for JSON output.
- Set `worker.strategy: "fast"` to use faster-whisper for single-speaker/no-diarization cases.
- Set `LOG_LEVEL=DEBUG` (or `worker.log_level` in config) for verbose tracing.
