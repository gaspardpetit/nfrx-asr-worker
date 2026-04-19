# nfrx-asr-worker

ASR transcribe worker for NFRX using Verbatim, plus in-worker speaker embeddings for chunk reconciliation.

Quick start (uv):
1. Install dependencies: `uv sync --extra verbatim`
2. Update `config.yaml` (or set `TRANSCRIBE_CONFIG=/path/to/config.yaml`)
3. Run `uv run transcribe-worker`

Notes:
- Override backend selection with `worker.backend`, `ASR_BACKEND`, or `--backend` using `verbatim`.
- The worker runs `worker.concurrency` jobs concurrently in one process; default is `4`, configurable via `worker.concurrency`, `NFRX_CONCURRENCY`, or `--concurrency`.
- Speaker embeddings are part of the default stack and are built incrementally from resolved utterance audio ranges using SpeechBrain ECAPA so later chunks can be matched back to earlier speakers.
- `config.yaml` follows Verbatim’s CLI schema for transcription/formatting, plus `worker`/`auth` sections for NFRX.
- Diarization (pyannote) requires a Hugging Face token (`HUGGINGFACE_TOKEN`) and FFmpeg shared libs for torchcodec.
- If you see torchcodec/FFmpeg errors, install FFmpeg 4-7 and set `FFMPEG_DLL_DIR` (Windows) or add `ffmpeg` to PATH.
- Settings read from environment variables, then overridden by `.env` (if present). `config.yaml` can override both.
- The default output is `jsonl`, with one JSON object per line containing `speaker`, `start`, `end`, and `text`. Use `output.formats: [txt]` for raw text or include `json` for a single JSON payload.
- Overlap reconciliation uses both segment timing and normalized Levenshtein text similarity; see `tests/test_chunk_alignment.py` for the covered merge cases.
- Set `worker.strategy: "fast"` to use faster-whisper for single-speaker/no-diarization cases.
- Set `LOG_LEVEL=DEBUG` (or `worker.log_level` in config) for verbose tracing.
