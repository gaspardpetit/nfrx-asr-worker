# nfrx-asr-worker

ASR transcribe worker for NFRX with optional backends for Verbatim and MLX VibeVoice, plus in-worker speaker embeddings for chunk reconciliation.

Quick start (uv):
1. Choose a backend install:
   - macOS / MLX: `uv sync --extra mlx`
   - Verbatim + diarization: `uv sync --extra verbatim`
2. Update `config.yaml` (or set `TRANSCRIBE_CONFIG=/path/to/config.yaml`)
3. Run `uv run transcribe-worker`

Notes:
- On macOS, the default backend is `mlx-vibevoice` with model `mlx-community/VibeVoice-ASR-8bit`.
- The `mlx` and `verbatim` extras are intentionally mutually exclusive because their dependency graphs conflict.
- When `mlx-vibevoice` is enabled, external diarization is disabled because VibeVoice already returns speaker-aware segments.
- Override backend selection with `worker.backend`, `ASR_BACKEND`, or `--backend` using `verbatim` or `mlx-vibevoice`.
- The worker runs `worker.concurrency` jobs concurrently in one process; default is `1` for `mlx-vibevoice` and `4` for other backends, configurable via `worker.concurrency`, `NFRX_CONCURRENCY`, or `--concurrency`.
- The MLX backend now defaults to chunked transcription for long audio: `20` minute chunks with `60` seconds of overlap, carrying the previous chunk transcript forward as prompt context.
- Speaker embeddings are part of the default stack and are built incrementally from resolved utterance audio ranges using SpeechBrain ECAPA so later chunks can be matched back to earlier speakers.
- Tune long-audio behavior with `transcription.mlx_max_tokens`, `transcription.mlx_chunk_seconds`, `transcription.mlx_chunk_overlap_seconds`, and `transcription.mlx_context_characters` (or the matching `MLX_*` environment variables / CLI flags).
- `config.yaml` follows Verbatim’s CLI schema for transcription/formatting, plus `worker`/`auth` sections for NFRX.
- Diarization (pyannote) requires a Hugging Face token (`HUGGINGFACE_TOKEN`) and FFmpeg shared libs for torchcodec.
- If you see torchcodec/FFmpeg errors, install FFmpeg 4-7 and set `FFMPEG_DLL_DIR` (Windows) or add `ffmpeg` to PATH.
- Settings read from environment variables, then overridden by `.env` (if present). `config.yaml` can override both.
- The default output is `jsonl`, with one JSON object per line containing `speaker`, `start`, `end`, and `text`. Use `output.formats: [txt]` for raw text or include `json` for a single JSON payload.
- With the MLX backend, status updates can now include `progress.current`, `progress.finish`, and `progress.percent`, plus the latest parsed utterance text.
- Overlap reconciliation uses both segment timing and normalized Levenshtein text similarity; see `tests/test_chunk_alignment.py` for the covered merge cases.
- Set `worker.strategy: "fast"` to use faster-whisper for single-speaker/no-diarization cases.
- Set `LOG_LEVEL=DEBUG` (or `worker.log_level` in config) for verbose tracing.
