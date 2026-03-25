#!/usr/bin/env python3
"""ASR transcribe worker for NFRX using Verbatim (transcription + diarization)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, TypedDict

import httpx
from dotenv import load_dotenv

from transcribe.audio import AudioBuffer, AudioDecodeError, normalize_audio_buffer, slice_audio_buffer
from transcribe.speaker_embeddings import IncrementalSpeakerResolver, warmup_default_speaker_embedder

_VERBATIM_MODULES: Optional[SimpleNamespace] = None
_VERBATIM_IMPORT_ERROR: Optional[Exception] = None


def _load_verbatim_modules(required: bool = False) -> Optional[SimpleNamespace]:
    global _VERBATIM_MODULES, _VERBATIM_IMPORT_ERROR
    if _VERBATIM_MODULES is not None:
        return _VERBATIM_MODULES
    if _VERBATIM_IMPORT_ERROR is not None and not required:
        return None
    try:
        from verbatim_audio.sources.factory import create_audio_sources
        from verbatim_audio.sources.sourceconfig import SourceConfig
        from verbatim.config import Config as VerbatimConfig
        from verbatim.models import Models as VerbatimModels
        from verbatim.verbatim import Verbatim
        from verbatim.transcript.words import Utterance, Word
        from verbatim_files.format.docx import DocxTranscriptWriter
        from verbatim_files.format.txt import COLORSCHEME_NONE, TextIOTranscriptWriter
        from verbatim_files.format.writer import (
            LanguageStyle,
            ProbabilityStyle,
            SpeakerStyle,
            TimestampStyle,
            TranscriptWriterConfig,
        )
        from verbatim_cli import args as verbatim_args
        from verbatim_cli import configure as verbatim_configure
        from verbatim_cli.config_file import load_config_file, merge_args, select_profile
    except Exception as exc:
        _VERBATIM_IMPORT_ERROR = exc
        if required:
            raise RuntimeError(
                "The selected configuration requires the optional 'verbatim' dependencies. "
                "Install them with `uv sync --extra verbatim`."
            ) from exc
        return None
    _VERBATIM_MODULES = SimpleNamespace(
        create_audio_sources=create_audio_sources,
        SourceConfig=SourceConfig,
        VerbatimConfig=VerbatimConfig,
        VerbatimModels=VerbatimModels,
        Verbatim=Verbatim,
        Utterance=Utterance,
        Word=Word,
        DocxTranscriptWriter=DocxTranscriptWriter,
        COLORSCHEME_NONE=COLORSCHEME_NONE,
        TextIOTranscriptWriter=TextIOTranscriptWriter,
        LanguageStyle=LanguageStyle,
        ProbabilityStyle=ProbabilityStyle,
        SpeakerStyle=SpeakerStyle,
        TimestampStyle=TimestampStyle,
        TranscriptWriterConfig=TranscriptWriterConfig,
        verbatim_args=verbatim_args,
        verbatim_configure=verbatim_configure,
        load_config_file=load_config_file,
        merge_args=merge_args,
        select_profile=select_profile,
    )
    return _VERBATIM_MODULES


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _torch_mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    return bool(mps_backend.is_available())


class JobCanceled(Exception):
    pass


@dataclass
class AuthConfig:
    client_key: Optional[str] = None


class JobClaim(TypedDict):
    job_id: str
    type: str
    metadata: Dict[str, Any]


ProcessHandler = Callable[[JobClaim, bytes], Awaitable[Tuple[bytes, Optional[str]]]]
StatusHandler = Callable[[str, Optional[Dict[str, Any]]], Awaitable[None]]
StatusPayload = Dict[str, Any]

DEFAULT_BASE_URL = "https://nfrx.l3ia.ca/"
DEFAULT_LANGUAGES = ["en", "fr"]


@dataclass
class DiarizationConfig:
    enabled: bool = True
    strategy: str = "pyannote"
    speakers: Optional[int] = None
    huggingface_token: Optional[str] = None


@dataclass
class TranscriptionConfig:
    backend: str = "auto"  # auto|verbatim|mlx-vibevoice
    model: str = "large-v3"
    device: str = "auto"
    compute_type: str = "int8"
    language: Optional[list[str]] = None
    transcriber_backend: str = "auto"
    language_identifier_backend: str = "transcriber"
    mms_lid_model_size: str = "facebook/mms-lid-126"
    beam_size: Optional[int] = None
    best_of: Optional[int] = None
    patience: Optional[float] = None
    output_format: str = "json"  # json|text
    mlx_max_tokens: int = 32768
    mlx_repetition_penalty: float = 1.08
    mlx_repetition_context_size: int = 256
    mlx_chunk_seconds: float = 20.0 * 60.0
    mlx_chunk_overlap_seconds: float = 60.0
    mlx_context_characters: int = 8000


@dataclass
class WorkerConfig:
    base_url: str = DEFAULT_BASE_URL
    types: Optional[str] = "asr.transcribe"
    max_wait_seconds: int = 30
    concurrency: int = 4
    once: bool = False
    log_level: str = "INFO"
    backend: str = "auto"  # auto|verbatim|mlx-vibevoice
    strategy: str = "verbatim"  # verbatim|fast
    status_progress: bool = True
    status_interval_seconds: float = 2.0
    status_min_progress_seconds: float = 1.0
    status_include_utterance: bool = False
    status_watchdog_seconds: float = 20.0


class NfrxStatusPoster:
    def __init__(
        self,
        worker: "NfrxJobsWorker",
        min_interval_seconds: float,
        min_progress_seconds: float,
    ) -> None:
        self._worker = worker
        self._min_interval_seconds = min_interval_seconds
        self._min_progress_seconds = min_progress_seconds
        self._queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        self._task: Optional[asyncio.Task[None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._closed = False
        self._completed_jobs: set[str] = set()
        self._last_progress: dict[str, float] = {}
        self._last_progress_time: dict[str, float] = {}
        self._watchdogs: dict[str, asyncio.Task[None]] = {}

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._task is None:
            self._loop = loop
            self._task = loop.create_task(self._run())

    async def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)
        for task in self._watchdogs.values():
            task.cancel()
        if self._watchdogs:
            await asyncio.gather(*self._watchdogs.values(), return_exceptions=True)
        self._watchdogs.clear()
        if self._task is not None:
            await self._task

    def mark_done(self, job_id: str) -> None:
        self._completed_jobs.add(job_id)
        self._last_progress.pop(job_id, None)
        self._last_progress_time.pop(job_id, None)
        watchdog = self._watchdogs.pop(job_id, None)
        if watchdog is not None:
            watchdog.cancel()

    def enqueue(self, job_id: str, payload: StatusPayload, *, origin: str = "unknown") -> None:
        if self._closed or job_id in self._completed_jobs:
            return
        self._queue.put_nowait({"job_id": job_id, "payload": payload, "origin": origin})

    def start_watchdog(self, job_id: str, interval_seconds: float, payload: StatusPayload) -> None:
        if interval_seconds <= 0 or self._closed or job_id in self._completed_jobs:
            return
        if job_id in self._watchdogs:
            return
        loop = self._loop
        if loop is None:
            raise RuntimeError("Status poster not started")

        async def _watchdog() -> None:
            try:
                while True:
                    await asyncio.sleep(interval_seconds)
                    if self._closed or job_id in self._completed_jobs:
                        return
                    last_current = self._last_progress.get(job_id)
                    last_time = self._last_progress_time.get(job_id)
                    age = None if last_time is None else max(0.0, time.monotonic() - last_time)
                    logging.debug(
                        "status watchdog firing job=%s interval=%.1fs last_progress=%s last_progress_age=%s payload_keys=%s",
                        job_id,
                        interval_seconds,
                        "n/a" if last_current is None else f"{last_current:.2f}",
                        "n/a" if age is None else f"{age:.2f}s",
                        sorted(payload.keys()),
                    )
                    self.enqueue(job_id, payload, origin="watchdog")
            except asyncio.CancelledError:
                return

        self._watchdogs[job_id] = loop.create_task(_watchdog())

    def build_hook(
        self,
        job_id: str,
        include_utterance: bool,
    ) -> Callable[[Any], None]:
        loop = self._loop
        if loop is None:
            raise RuntimeError("Status poster not started")

        def hook(update: Any) -> None:
            payload = _status_update_to_payload(update, include_utterance)
            if payload is None:
                return
            loop.call_soon_threadsafe(partial(self.enqueue, job_id, payload, origin="model"))

        return hook

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            job_id = item.get("job_id")
            payload = item.get("payload") or {}
            origin = str(item.get("origin") or "unknown")
            if not job_id or job_id in self._completed_jobs:
                continue
            progress = payload.get("progress")
            if isinstance(progress, dict) and "current" in progress:
                try:
                    current = float(progress["current"])
                except (TypeError, ValueError):
                    current = None
                if current is not None:
                    now = time.monotonic()
                    last_current = self._last_progress.get(job_id)
                    last_time = self._last_progress_time.get(job_id, 0.0)
                    if (
                        last_current is not None
                        and (current - last_current) < self._min_progress_seconds
                        and (now - last_time) < self._min_interval_seconds
                    ):
                        logging.debug(
                            "status poster suppressed update job=%s origin=%s current=%.2f last=%.2f age=%.2fs min_progress=%.2f min_interval=%.2f",
                            job_id,
                            origin,
                            current,
                            last_current,
                            now - last_time,
                            self._min_progress_seconds,
                            self._min_interval_seconds,
                        )
                        continue
                    self._last_progress[job_id] = current
                    self._last_progress_time[job_id] = now
            progress_summary = None
            if isinstance(progress, dict) and "current" in progress:
                finish = progress.get("finish")
                progress_summary = f"{progress.get('current')}/{finish}" if finish is not None else str(progress.get("current"))
            logging.debug(
                "status poster sending job=%s origin=%s progress=%s has_utterance=%s payload_keys=%s",
                job_id,
                origin,
                progress_summary or "none",
                "utterance" in payload,
                sorted(payload.keys()),
            )
            try:
                await self._worker.update_status(job_id, "running", payload=payload)
            except JobCanceled:
                logging.info("job canceled while posting status: %s", job_id)
                self.mark_done(job_id)
            except httpx.HTTPError:
                logging.exception("status update failed for job %s", job_id)


class NfrxJobsWorker:
    def __init__(self, base_url: str, auth: AuthConfig) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth = auth
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=60.0)

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self._auth.client_key:
            headers["Authorization"] = f"Bearer {self._auth.client_key}"
        return headers

    def _raise_if_canceled(self, resp: httpx.Response, job_id: Optional[str]) -> None:
        if resp.status_code == 409:
            raise JobCanceled(job_id or "unknown")

    async def claim_job(
        self, types: Optional[list[str]] = None, max_wait_seconds: Optional[int] = None
    ) -> Optional[JobClaim]:
        payload: Dict[str, Any] = {}
        if types:
            payload["types"] = types
        if max_wait_seconds is not None:
            payload["max_wait_seconds"] = max_wait_seconds
        resp = await self._client.post("/api/jobs/claim", json=payload, headers=self._headers())
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        return resp.json()

    async def request_payload_channel(self, job_id: str) -> Dict[str, Any]:
        resp = await self._client.post(f"/api/jobs/{job_id}/payload", headers=self._headers())
        self._raise_if_canceled(resp, job_id)
        resp.raise_for_status()
        return resp.json()

    async def request_result_channel(self, job_id: str) -> Dict[str, Any]:
        resp = await self._client.post(f"/api/jobs/{job_id}/result", headers=self._headers())
        self._raise_if_canceled(resp, job_id)
        resp.raise_for_status()
        return resp.json()

    async def update_status(self, job_id: str, state: str, payload: Optional[Dict[str, Any]] = None) -> None:
        body = {"state": state}
        if payload:
            body.update(payload)
        resp = await self._client.post(
            f"/api/jobs/{job_id}/status", json=body, headers=self._headers()
        )
        self._raise_if_canceled(resp, job_id)
        resp.raise_for_status()

    async def read_payload(self, url: str, job_id: Optional[str] = None) -> bytes:
        resp = await self._client.get(url, headers=self._headers())
        self._raise_if_canceled(resp, job_id)
        resp.raise_for_status()
        return resp.content

    async def write_result(self, url: str, data: bytes, job_id: Optional[str] = None) -> None:
        resp = await self._client.post(url, content=data, headers=self._headers())
        self._raise_if_canceled(resp, job_id)
        resp.raise_for_status()

    async def run_once(
        self,
        handler: ProcessHandler,
        types: Optional[list[str]],
        max_wait_seconds: int,
        on_status: Optional[StatusHandler] = None,
    ) -> bool:
        job = await self.claim_job(types=types, max_wait_seconds=max_wait_seconds)
        if not job:
            logging.info("no job available")
            return False
        job_id = job.get("job_id") or job.get("id")
        metadata = job.get("metadata") or {}
        filename = metadata.get("filename") or metadata.get("name")
        content_type = metadata.get("content_type") or metadata.get("mime_type")
        logging.info("claimed job: %s", job_id)
        if filename or content_type:
            logging.info(
                "job metadata: filename=%s content_type=%s",
                filename or "-",
                content_type or "-",
            )
        try:
            await self.update_status(job_id, "claimed")
            if on_status:
                await on_status("claimed", None)

            payload_channel = await self.request_payload_channel(job_id)
            payload_url = payload_channel.get("reader_url") or payload_channel.get("url")
            if on_status:
                await on_status("awaiting_payload", payload_channel)
            payload_bytes = await self.read_payload(payload_url, job_id=job_id)
            logging.info("payload received: %s bytes", len(payload_bytes))

            await self.update_status(job_id, "running")
            if on_status:
                await on_status("running", None)
            result_bytes, error_message = await handler(job, payload_bytes)

            if error_message:
                logging.error("handler error: %s", error_message)
                await self.update_status(
                    job_id,
                    "failed",
                    payload={"error": {"code": "handler_error", "message": error_message}},
                )
                if on_status:
                    await on_status("failed", {"error": {"code": "handler_error", "message": error_message}})
                return True

            result_channel = await self.request_result_channel(job_id)
            result_url = result_channel.get("writer_url") or result_channel.get("url")
            if on_status:
                await on_status("awaiting_result", result_channel)
            await self.write_result(result_url, result_bytes, job_id=job_id)
            logging.info("result uploaded: %s bytes", len(result_bytes))

            await self.update_status(job_id, "completed")
            if on_status:
                await on_status("completed", None)
            return True
        except JobCanceled:
            logging.info("job canceled: %s", job_id)
            return True
        except httpx.HTTPError as exc:
            logging.exception("http error while handling job %s", job_id)
            try:
                await self.update_status(
                    job_id,
                    "failed",
                    payload={"error": {"code": "http_error", "message": str(exc)}},
                )
            except JobCanceled:
                logging.info("job canceled while reporting failure: %s", job_id)
            if on_status:
                await on_status("failed", {"error": {"code": "http_error", "message": str(exc)}})
            return True



def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        logging.info("config file not found: %s (using defaults/env)", path)
        return {}
    mods = _load_verbatim_modules(required=False)
    if mods is not None:
        data = mods.load_config_file(path)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "Config parsing requires either the optional 'verbatim' dependencies or PyYAML."
            ) from exc
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def _cfg_get(config: Dict[str, Any], *keys: str) -> Any:
    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _cfg_get_transcription(config: Dict[str, Any], key: str) -> Any:
    value = _cfg_get(config, "transcription", key)
    if value is not None:
        return value
    value = _cfg_get(config, "verbatim", key)
    if value is not None:
        return value
    if key == "model":
        return _cfg_get(config, "whisper_model_size") or _cfg_get(config, "model")
    return _cfg_get(config, key)


def _resolve(cli: Any, cfg: Any, env: Any, default: Any) -> Any:
    if cli is not None:
        return cli
    if cfg is not None:
        return cfg
    if env is not None:
        return env
    return default


def _normalize_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return str(value)


def _normalize_languages(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        return [value.strip()]
    if isinstance(value, list):
        cleaned = [str(v).strip() for v in value if str(v).strip()]
        return cleaned or None
    return [str(value)]

def _normalize_string_list(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        return [value.strip()]
    if isinstance(value, list):
        cleaned = [str(v).strip() for v in value if str(v).strip()]
        return cleaned or None
    return [str(value)]


def _coerce_enum(value: Any, enum_cls: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if not key:
            return None
        for member in enum_cls:
            if member.name.lower() == key:
                return member
    return None


def _output_format_from_formats(output_formats: list[str]) -> str:
    if "txt" in output_formats or "stdout" in output_formats or "stdout-nocolor" in output_formats:
        return "text"
    if "docx" in output_formats:
        return "docx"
    if "jsonl" in output_formats:
        return "jsonl"
    return "json"


def _status_update_to_payload(update: Any, include_utterance: bool) -> Optional[StatusPayload]:
    if update is None:
        return None
    if isinstance(update, dict):
        payload = dict(update)
        if not include_utterance:
            payload.pop("utterance", None)
        return payload if payload else None
    payload: StatusPayload = {}
    state = getattr(update, "state", None)
    if isinstance(state, str) and state.strip():
        payload["phase"] = state
    progress = getattr(update, "progress", None)
    if progress is not None:
        progress_payload: Dict[str, Any] = {}
        for key in ("current", "start", "finish", "units"):
            value = getattr(progress, key, None)
            if value is None:
                continue
            progress_payload[key] = float(value) if isinstance(value, (int, float)) else value
        if progress_payload:
            payload["progress"] = progress_payload
    if include_utterance:
        utterance = getattr(update, "utterance", None)
        if utterance is not None:
            utterance_payload: Dict[str, Any] = {
                "text": getattr(utterance, "text", None),
                "speaker": _public_speaker_label(getattr(utterance, "speaker", None)),
            }
            if hasattr(utterance, "get_start"):
                utterance_payload["start"] = float(utterance.get_start())
            if hasattr(utterance, "get_end"):
                utterance_payload["end"] = float(utterance.get_end())
            payload["utterance"] = utterance_payload
    if "progress" not in payload and "utterance" not in payload:
        return None
    return payload


def _apply_job_config(
    base_args: argparse.Namespace, job_cfg: Dict[str, Any]
) -> Tuple[argparse.Namespace, Optional[list[str]], Optional[list[str]], Optional[Any]]:
    if not job_cfg:
        return base_args, None, None, None

    mods = _load_verbatim_modules(required=False)

    args = argparse.Namespace(**vars(base_args))
    languages = _normalize_languages(job_cfg.get("languages"))
    if languages is not None:
        args.languages = languages

    diarize = _normalize_optional_str(job_cfg.get("diarize"))
    if diarize is not None:
        if diarize.strip().lower() in ("none", "off", "false", "0", "disable", "disabled"):
            args.diarize = None
            args.diarize_policy = None
            args.diarization = None
        else:
            args.diarize = diarize

    output_cfg = job_cfg.get("output") if isinstance(job_cfg.get("output"), dict) else {}
    output_files = _normalize_string_list(output_cfg.get("files"))
    if output_files is not None:
        for name in ("ass", "docx", "txt", "md", "jsonl", "json"):
            setattr(args, name, False)
        for item in output_files:
            key = item.strip().lower()
            if key in ("ass", "docx", "txt", "md", "jsonl", "json"):
                setattr(args, key, True)

    fmt_cfg = output_cfg.get("format") if isinstance(output_cfg.get("format"), dict) else {}
    if mods is not None:
        ts_style = _coerce_enum(fmt_cfg.get("timestamp"), mods.TimestampStyle)
        if ts_style is not None:
            args.format_timestamp = ts_style
        sp_style = _coerce_enum(fmt_cfg.get("speaker"), mods.SpeakerStyle)
        if sp_style is not None:
            args.format_speaker = sp_style
        prob_style = _coerce_enum(fmt_cfg.get("probability"), mods.ProbabilityStyle)
        if prob_style is not None:
            args.format_probability = prob_style
        lang_style = _coerce_enum(fmt_cfg.get("language"), mods.LanguageStyle)
        if lang_style is not None:
            args.format_language = lang_style

        write_config = mods.verbatim_configure.make_write_config(args, logging.getLogger().level)
        output_formats = (
            mods.verbatim_configure.build_output_formats(args, default_stdout=False)
            if output_files is not None
            else None
        )
    else:
        write_config = None
        output_formats = output_files
    return args, output_files, output_formats, write_config


def _resolve_device(device: str) -> str:
    if device == "auto":
        if platform.system() == "Darwin" and _torch_mps_available():
            return "mps"
        return "cuda" if _torch_cuda_available() else "cpu"
    return device


def _default_backend() -> str:
    return "verbatim"


def _normalize_backend(value: Any) -> str:
    backend = str(value or "").strip().lower()
    if backend in ("", "auto"):
        return _default_backend()
    if backend in ("verbatim", "mlx-vibevoice"):
        return backend
    raise ValueError(f"unsupported backend: {value}")


def _default_model_for_backend(backend: str) -> str:
    if backend == "mlx-vibevoice":
        return "mlx-community/VibeVoice-ASR-8bit"
    return "large-v3"


def _default_concurrency_for_backend(backend: str) -> int:
    if backend == "mlx-vibevoice":
        return 1
    return 4


def _resolve_bool(cli: Optional[bool], cfg: Any, env: Optional[str], default: bool) -> bool:
    if cli is not None:
        return bool(cli)
    if cfg is not None:
        return bool(cfg)
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "y")
    return default


def _resolve_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _redact_job_config(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            if key.lower() in ("huggingface_token", "token", "api_key", "apikey", "client_key"):
                redacted[key] = "***"
            else:
                redacted[key] = _redact_job_config(item)
        return redacted
    if isinstance(value, list):
        return [_redact_job_config(item) for item in value]
    return value


def _resolve_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _infer_suffix(filename: Optional[str], content_type: Optional[str]) -> str:
    if filename:
        _, ext = os.path.splitext(filename)
        if ext:
            return ext
    if content_type:
        ct = content_type.lower().strip()
        if ct in ("audio/wav", "audio/x-wav"):
            return ".wav"
        if ct in ("audio/mpeg", "audio/mp3"):
            return ".mp3"
        if ct in ("audio/mp4", "audio/m4a", "video/mp4"):
            return ".m4a"
        if ct in ("audio/ogg", "application/ogg"):
            return ".ogg"
        if ct in ("audio/flac",):
            return ".flac"
    return ".bin"

def _format_text_output(utterances: list[Any], write_config: Optional[Any]) -> str:
    mods = _load_verbatim_modules(required=False)
    if mods is None or write_config is None:
        segments: list[Dict[str, Any]] = []
        for utt in utterances:
            segments.append(
                {
                    "start": float(utt.get_start()) if hasattr(utt, "get_start") else 0.0,
                    "end": float(utt.get_end()) if hasattr(utt, "get_end") else 0.0,
                    "text": getattr(utt, "text", ""),
                    "speaker": getattr(utt, "speaker", None),
                }
            )
        return _format_text_from_segments(segments)
    writer = mods.TextIOTranscriptWriter(
        config=write_config,
        acknowledged_colours=mods.COLORSCHEME_NONE,
        unacknowledged_colours=mods.COLORSCHEME_NONE,
        unconfirmed_colors=mods.COLORSCHEME_NONE,
    )
    chunks: list[bytes] = [writer.format_start()]
    for utt in utterances:
        chunks.append(writer.format_utterance(utt))
    chunks.append(writer.format_end())
    return b"".join(chunks).decode("utf-8").strip()


def _format_docx_output(utterances: list[Any], write_config: Optional[Any]) -> bytes:
    mods = _load_verbatim_modules(required=True)
    if write_config is None:
        raise RuntimeError("docx output requires the optional 'verbatim' dependencies.")
    writer = mods.DocxTranscriptWriter(config=write_config)
    for utt in utterances:
        writer.format_utterance(utt)
    return writer.flush()


def _format_text_from_segments(segments: list[Dict[str, Any]]) -> str:
    lines: list[str] = []
    current_minute: Optional[int] = None
    for seg in segments:
        start = float(seg.get("start", 0.0))
        minute = int(start // 60)
        if current_minute is None or minute != current_minute:
            current_minute = minute
            lines.append(f"[{minute:02d}:00]")
        speaker = _public_segment_speaker(seg) or "SPEAKER"
        text = (seg.get("text") or "").strip()
        if text:
            lines.append(f"[{speaker}] {text}")
    return "\n".join(lines).strip()


def _public_speaker_label(speaker: Any) -> Optional[str]:
    if speaker is None:
        return None
    label = str(speaker).strip()
    if not label:
        return None
    upper = label.upper()
    if upper == "UNKNOWN":
        return None
    match = re.fullmatch(r"SPEAKER_(\d+)", upper)
    if match:
        return str(int(match.group(1)))
    if re.fullmatch(r"\d+", label):
        return str(int(label))
    return None


def _public_segment_speaker(segment: Dict[str, Any]) -> Optional[str]:
    speaker = segment.get("speaker")
    if "_chunk_index" in segment or "_local_speaker" in segment:
        label = str(speaker or "").strip()
        if not re.fullmatch(r"SPEAKER_\d+", label, re.IGNORECASE):
            return None
    return _public_speaker_label(speaker)


def _format_jsonl_output(segments: list[Dict[str, Any]]) -> bytes:
    lines: list[str] = []
    for seg in segments:
        payload = {
            "speaker": _public_segment_speaker(seg),
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", seg.get("start", 0.0))),
            "text": (seg.get("text") or "").strip(),
        }
        lines.append(json.dumps(payload, ensure_ascii=False))
    text = "\n".join(lines)
    if text:
        text += "\n"
    return text.encode("utf-8")


def _public_segments(segments: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    public_segments: list[Dict[str, Any]] = []
    for seg in segments:
        updated = dict(seg)
        updated["speaker"] = _public_segment_speaker(seg)
        public_segments.append(updated)
    return public_segments


def _segments_to_plain_text(segments: list[Dict[str, Any]]) -> str:
    return " ".join((seg.get("text") or "").strip() for seg in segments if (seg.get("text") or "").strip()).strip()


def _debug_text_preview(value: str, limit: int = 240) -> str:
    if not value:
        return ""
    normalized = value.replace("\n", "\\n").replace("\r", "\\r")
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def _normalize_alignment_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (char_a != char_b)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _levenshtein_similarity(a: str, b: str) -> float:
    norm_a = _normalize_alignment_text(a)
    norm_b = _normalize_alignment_text(b)
    if not norm_a and not norm_b:
        return 1.0
    if not norm_a or not norm_b:
        return 0.0
    distance = _levenshtein_distance(norm_a, norm_b)
    return max(0.0, 1.0 - (distance / max(len(norm_a), len(norm_b))))


def _segment_overlap_ratio(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    left_start = float(left.get("start", 0.0))
    left_end = float(left.get("end", left_start))
    right_start = float(right.get("start", 0.0))
    right_end = float(right.get("end", right_start))
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    if overlap <= 0:
        return 0.0
    left_duration = max(0.001, left_end - left_start)
    right_duration = max(0.001, right_end - right_start)
    return overlap / max(left_duration, right_duration)


def _resolved_alignment_speaker(segment: Dict[str, Any]) -> Optional[str]:
    speaker = segment.get("speaker")
    if speaker is None:
        return None
    label = str(speaker).strip()
    if not label:
        return None
    if re.fullmatch(r"SPEAKER_\d+", label, re.IGNORECASE):
        return label.upper()
    return None


def _segment_alignment_score(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    text_score = _levenshtein_similarity(str(left.get("text") or ""), str(right.get("text") or ""))
    time_score = _segment_overlap_ratio(left, right)
    if text_score < 0.45 and time_score < 0.45:
        return 0.0
    left_speaker = _resolved_alignment_speaker(left)
    right_speaker = _resolved_alignment_speaker(right)
    if left_speaker is not None and right_speaker is not None and left_speaker != right_speaker:
        return 0.0
    score = text_score * 0.7 + time_score * 0.3
    if left_speaker is not None and right_speaker is not None and left_speaker == right_speaker:
        score += 0.15
    return min(score, 1.0)


def _is_more_complete_text(candidate: str, baseline: str) -> bool:
    candidate_norm = _normalize_alignment_text(candidate)
    baseline_norm = _normalize_alignment_text(baseline)
    if not candidate_norm or not baseline_norm:
        return False
    if len(candidate_norm) <= len(baseline_norm):
        return False
    prefix = candidate_norm[: len(baseline_norm)]
    return _levenshtein_similarity(prefix, baseline_norm) >= 0.85


def _align_overlap_segments(
    previous_overlap: list[Dict[str, Any]],
    current_overlap: list[Dict[str, Any]],
) -> list[tuple[int, int, float]]:
    if not previous_overlap or not current_overlap:
        return []

    prev_len = len(previous_overlap)
    curr_len = len(current_overlap)
    scores = [[0.0] * (curr_len + 1) for _ in range(prev_len + 1)]
    trace: list[list[tuple[str, float]]] = [[("", 0.0)] * (curr_len + 1) for _ in range(prev_len + 1)]

    for i in range(1, prev_len + 1):
        for j in range(1, curr_len + 1):
            best_score = scores[i - 1][j]
            best_trace = ("up", 0.0)

            if scores[i][j - 1] > best_score:
                best_score = scores[i][j - 1]
                best_trace = ("left", 0.0)

            match_score = _segment_alignment_score(previous_overlap[i - 1], current_overlap[j - 1])
            if match_score > 0:
                diagonal_score = scores[i - 1][j - 1] + match_score
                if diagonal_score >= best_score:
                    best_score = diagonal_score
                    best_trace = ("diag", match_score)

            scores[i][j] = best_score
            trace[i][j] = best_trace

    matches: list[tuple[int, int, float]] = []
    i = prev_len
    j = curr_len
    while i > 0 and j > 0:
        direction, match_score = trace[i][j]
        if direction == "diag" and match_score > 0:
            matches.append((i - 1, j - 1, match_score))
            i -= 1
            j -= 1
        elif direction == "left":
            j -= 1
        else:
            i -= 1
    matches.reverse()
    return matches


def _edge_confidence(segment: Dict[str, Any], overlap_start: float, overlap_end: float, prefer_previous: bool) -> float:
    midpoint = (float(segment.get("start", 0.0)) + float(segment.get("end", segment.get("start", 0.0)))) / 2.0
    if prefer_previous:
        return max(0.0, overlap_end - midpoint)
    return max(0.0, midpoint - overlap_start)


def _debug_segment_summary(segment: Dict[str, Any]) -> str:
    speaker = segment.get("speaker")
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    text = " ".join(str(segment.get("text") or "").strip().split())
    if len(text) > 96:
        text = text[:93] + "..."
    return f"{start:.2f}-{end:.2f} speaker={speaker!r} text={text!r}"


def _debug_segments_summary(name: str, segments: list[Dict[str, Any]]) -> str:
    if not segments:
        return f"{name}=<empty>"
    previews = ", ".join(_debug_segment_summary(seg) for seg in segments[:5])
    suffix = "" if len(segments) <= 5 else f", ... (+{len(segments) - 5} more)"
    return f"{name}[{len(segments)}]=[{previews}{suffix}]"


def _merge_chunk_segments(
    previous_segments: list[Dict[str, Any]],
    current_segments: list[Dict[str, Any]],
    overlap_start: float,
    overlap_end: float,
) -> list[Dict[str, Any]]:
    if not previous_segments:
        return list(current_segments)
    if not current_segments:
        return list(previous_segments)

    previous_overlap_indices = [idx for idx, seg in enumerate(previous_segments) if float(seg.get("end", seg.get("start", 0.0))) > overlap_start]
    current_overlap_indices = [idx for idx, seg in enumerate(current_segments) if float(seg.get("start", 0.0)) < overlap_end]

    previous_overlap = [previous_segments[idx] for idx in previous_overlap_indices]
    current_overlap = [current_segments[idx] for idx in current_overlap_indices]
    matches = _align_overlap_segments(previous_overlap, current_overlap)
    logging.debug(
        "merge_chunk_segments start=%.2fs end=%.2fs %s %s matches=%s",
        overlap_start,
        overlap_end,
        _debug_segments_summary("previous_overlap", previous_overlap),
        _debug_segments_summary("current_overlap", current_overlap),
        len(matches),
    )

    drop_previous: set[int] = set()
    drop_current: set[int] = set()
    for prev_local_idx, curr_local_idx, score in matches:
        prev_idx = previous_overlap_indices[prev_local_idx]
        curr_idx = current_overlap_indices[curr_local_idx]
        prev_seg = previous_segments[prev_idx]
        curr_seg = current_segments[curr_idx]

        prev_conf = _edge_confidence(prev_seg, overlap_start, overlap_end, prefer_previous=True)
        curr_conf = _edge_confidence(curr_seg, overlap_start, overlap_end, prefer_previous=False)
        prev_text = str(prev_seg.get("text") or "")
        curr_text = str(curr_seg.get("text") or "")
        decision = "keep-previous"
        if _is_more_complete_text(curr_text, prev_text) and score >= 0.55:
            drop_previous.add(prev_idx)
            decision = "prefer-current-more-complete"
        elif _is_more_complete_text(prev_text, curr_text) and score >= 0.55:
            drop_current.add(curr_idx)
            decision = "prefer-previous-more-complete"
        elif curr_conf > prev_conf and score >= 0.55:
            drop_previous.add(prev_idx)
            decision = "prefer-current-edge-confidence"
        else:
            drop_current.add(curr_idx)
            decision = "prefer-previous-edge-confidence"
        logging.debug(
            "merge_chunk_segments match score=%.4f decision=%s prev_conf=%.4f curr_conf=%.4f previous=(%s) current=(%s)",
            score,
            decision,
            prev_conf,
            curr_conf,
            _debug_segment_summary(prev_seg),
            _debug_segment_summary(curr_seg),
        )

    for curr_idx in current_overlap_indices:
        if curr_idx in drop_current:
            continue
        curr_seg = current_segments[curr_idx]
        for prev_idx in previous_overlap_indices:
            if prev_idx in drop_previous:
                continue
            prev_seg = previous_segments[prev_idx]
            score = _segment_alignment_score(prev_seg, curr_seg)
            if score >= 0.82:
                drop_current.add(curr_idx)
                logging.debug(
                    "merge_chunk_segments dropping current as duplicate score=%.4f current=(%s) previous=(%s)",
                    score,
                    _debug_segment_summary(curr_seg),
                    _debug_segment_summary(prev_seg),
                )
                break

    merged = [seg for idx, seg in enumerate(previous_segments) if idx not in drop_previous]
    merged.extend(seg for idx, seg in enumerate(current_segments) if idx not in drop_current)
    merged.sort(key=lambda seg: (float(seg.get("start", 0.0)), float(seg.get("end", seg.get("start", 0.0)))))
    logging.debug(
        "merge_chunk_segments result previous_dropped=%s current_dropped=%s merged_segments=%s",
        sorted(drop_previous),
        sorted(drop_current),
        len(merged),
    )
    return merged


def _segment_midpoint(segment: Dict[str, Any]) -> float:
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    return (start + end) / 2.0


def _resolve_overlap_window(
    previous_overlap: list[Dict[str, Any]],
    current_overlap: list[Dict[str, Any]],
    overlap_start: float,
    overlap_end: float,
) -> list[Dict[str, Any]]:
    if not previous_overlap and not current_overlap:
        return []
    if not previous_overlap:
        return list(current_overlap)
    if not current_overlap:
        return list(previous_overlap)

    midpoint = overlap_start + ((overlap_end - overlap_start) / 2.0)
    previous_indices = list(range(len(previous_overlap)))
    current_indices = list(range(len(current_overlap)))
    matches = _align_overlap_segments(previous_overlap, current_overlap)
    logging.debug(
        "resolve_overlap_window start=%.2fs end=%.2fs midpoint=%.2fs %s %s matches=%s",
        overlap_start,
        overlap_end,
        midpoint,
        _debug_segments_summary("previous", previous_overlap),
        _debug_segments_summary("current", current_overlap),
        len(matches),
    )

    consumed_previous: set[int] = set()
    consumed_current: set[int] = set()
    resolved: list[Dict[str, Any]] = []

    for prev_idx, curr_idx, score in matches:
        consumed_previous.add(prev_idx)
        consumed_current.add(curr_idx)
        previous_seg = previous_overlap[prev_idx]
        current_seg = current_overlap[curr_idx]
        previous_text = str(previous_seg.get("text") or "")
        current_text = str(current_seg.get("text") or "")
        if _normalize_alignment_text(previous_text) == _normalize_alignment_text(current_text):
            chosen = previous_seg if _segment_midpoint(previous_seg) <= midpoint else current_seg
            choice_reason = "same-text-midpoint"
        elif _segment_midpoint(previous_seg) < midpoint:
            chosen = previous_seg
            choice_reason = "first-half-prefer-previous"
        else:
            chosen = current_seg
            choice_reason = "second-half-prefer-current"
        logging.debug(
            "resolve_overlap_window match score=%.4f reason=%s previous=(%s) current=(%s) chosen=(%s)",
            score,
            choice_reason,
            _debug_segment_summary(previous_seg),
            _debug_segment_summary(current_seg),
            _debug_segment_summary(chosen),
        )
        resolved.append(chosen)

    # Preserve unmatched overlap speech from both sides; for online updates it is
    # better to dedupe later than to silently drop a real utterance.
    for prev_idx in previous_indices:
        if prev_idx in consumed_previous:
            continue
        logging.debug(
            "resolve_overlap_window preserving unmatched previous=(%s)",
            _debug_segment_summary(previous_overlap[prev_idx]),
        )
        resolved.append(previous_overlap[prev_idx])

    for curr_idx in current_indices:
        if curr_idx in consumed_current:
            continue
        logging.debug(
            "resolve_overlap_window preserving unmatched current=(%s)",
            _debug_segment_summary(current_overlap[curr_idx]),
        )
        resolved.append(current_overlap[curr_idx])

    resolved.sort(key=lambda seg: (float(seg.get("start", 0.0)), float(seg.get("end", seg.get("start", 0.0)))))

    deduped: list[Dict[str, Any]] = []
    for seg in resolved:
        if deduped and _segment_alignment_score(deduped[-1], seg) >= 0.9:
            logging.debug(
                "resolve_overlap_window deduping segment=(%s) against previous=(%s)",
                _debug_segment_summary(seg),
                _debug_segment_summary(deduped[-1]),
            )
            continue
        deduped.append(seg)
    logging.debug("resolve_overlap_window result %s", _debug_segments_summary("deduped", deduped))
    return deduped


def _status_payload_from_segment(segment: Dict[str, Any], finish: Optional[float]) -> StatusPayload:
    end = float(segment.get("end", segment.get("start", 0.0)))
    progress: Dict[str, Any] = {"current": round(end, 2), "units": "seconds"}
    if finish is not None:
        progress["finish"] = round(finish, 2)
        if finish > 0:
            progress["percent"] = round(max(0.0, min(100.0, (end / finish) * 100.0)), 1)
    payload: StatusPayload = {
        "phase": "transcribing",
        "progress": progress,
        "utterance": {
            "text": str(segment.get("text") or "").strip(),
            "speaker": _public_segment_speaker(segment),
            "start": round(float(segment.get("start", 0.0)), 2),
            "end": round(end, 2),
        },
    }
    return payload


class ChunkResolvedEmitter:
    def __init__(
        self,
        *,
        overlap_seconds: float,
        finish: Optional[float],
        status_hook: Optional[Callable[[Any], None]],
        segment_transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        segment_observer: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._overlap_seconds = max(0.0, overlap_seconds)
        self._finish = finish
        self._status_hook = status_hook
        self._segment_transformer = segment_transformer
        self._segment_observer = segment_observer
        self._pending_tail: list[Dict[str, Any]] = []
        self._emitted: list[Dict[str, Any]] = []

    @staticmethod
    def _segment_time_bounds(segments: list[Dict[str, Any]]) -> tuple[float, float]:
        if not segments:
            return 0.0, 0.0
        start = float(segments[0].get("start", 0.0))
        end = float(segments[-1].get("end", segments[-1].get("start", 0.0)))
        return start, end

    def _emit(self, segments: list[Dict[str, Any]], *, reason: str) -> None:
        if not segments:
            logging.info("resolved emitter skipped empty emission reason=%s", reason)
            return
        start, end = self._segment_time_bounds(segments)
        logging.info(
            "resolved emitter emitting reason=%s segments=%s start=%.2fs end=%.2fs",
            reason,
            len(segments),
            start,
            end,
        )
        emitted_segments: list[Dict[str, Any]] = []
        for segment in segments:
            resolved = dict(segment)
            if self._segment_transformer is not None:
                resolved = self._segment_transformer(resolved)
            logging.debug("resolved emitter utterance reason=%s %s", reason, _debug_segment_summary(resolved))
            emitted_segments.append(resolved)
            if self._segment_observer is not None:
                self._segment_observer(dict(resolved))
        self._emitted.extend(emitted_segments)
        if self._status_hook is None:
            return
        for segment in emitted_segments:
            self._status_hook(_status_payload_from_segment(segment, self._finish))

    def process_chunk(
        self,
        segments: list[Dict[str, Any]],
        *,
        chunk_start: float,
        chunk_end: float,
        is_last: bool,
    ) -> None:
        logging.info(
            "resolved emitter processing chunk start=%.2fs end=%.2fs is_last=%s overlap=%.2fs pending_tail=%s incoming_segments=%s",
            chunk_start,
            chunk_end,
            is_last,
            self._overlap_seconds,
            len(self._pending_tail),
            len(segments),
        )
        if self._overlap_seconds <= 0:
            self._emit(segments, reason="no-overlap")
            return

        overlap_end = min(chunk_end, chunk_start + self._overlap_seconds)
        current_head = [seg for seg in segments if float(seg.get("start", 0.0)) < overlap_end]
        current_body = [seg for seg in segments if float(seg.get("start", 0.0)) >= overlap_end]
        had_pending_tail = bool(self._pending_tail)

        first_non_overlap: list[Dict[str, Any]] = []
        if had_pending_tail:
            resolved_overlap = _resolve_overlap_window(self._pending_tail, current_head, chunk_start, overlap_end)
            logging.info(
                "resolved emitter resolved overlap window start=%.2fs end=%.2fs previous_tail=%s current_head=%s resolved=%s",
                chunk_start,
                overlap_end,
                len(self._pending_tail),
                len(current_head),
                len(resolved_overlap),
            )
            self._emit(resolved_overlap, reason="resolved-overlap")
        else:
            first_non_overlap = [seg for seg in segments if float(seg.get("end", seg.get("start", 0.0))) <= max(chunk_end - self._overlap_seconds, 0.0)]
            self._emit(first_non_overlap, reason="initial-non-overlap")

        if is_last:
            if not self._pending_tail:
                trailing = [seg for seg in segments if seg not in first_non_overlap]
                self._emit(trailing, reason="last-trailing")
            else:
                self._emit(current_body, reason="last-body")
            self._pending_tail = []
            return

        tail_start = max(chunk_start, chunk_end - self._overlap_seconds)
        current_non_overlap = [
            seg for seg in segments if float(seg.get("start", 0.0)) >= overlap_end and float(seg.get("end", seg.get("start", 0.0))) <= tail_start
        ]
        if had_pending_tail:
            self._emit(current_non_overlap, reason="mid-body")
        self._pending_tail = [seg for seg in segments if float(seg.get("end", seg.get("start", 0.0))) > tail_start]
        pending_start, pending_end = self._segment_time_bounds(self._pending_tail)
        logging.info(
            "resolved emitter buffered tail segments=%s start=%.2fs end=%.2fs",
            len(self._pending_tail),
            pending_start,
            pending_end,
        )

    def finalize(self) -> None:
        if self._pending_tail:
            logging.info("resolved emitter finalizing pending tail segments=%s", len(self._pending_tail))
            self._emit(self._pending_tail, reason="finalize-tail")
            self._pending_tail = []


class VerbatimService:
    def __init__(
        self,
        config: TranscriptionConfig,
        diarization: DiarizationConfig,
        source_config: Any,
        write_config: Optional[Any],
    ) -> None:
        mods = _load_verbatim_modules(required=True)
        self._base_config = config
        self._diarization = diarization
        self._source_config = source_config
        self._write_config = write_config
        self._mods = mods
        self._models = mods.VerbatimModels(
            device=config.device,
            whisper_model_size=config.model,
            stream=False,
            transcriber=None,
            transcriber_backend=config.transcriber_backend,
        )

    def _build_config(
        self, language: Optional[list[str]], input_label: str
    ) -> Tuple[Any, str, str]:
        cfg = self._mods.VerbatimConfig(
            device=self._base_config.device,
            whisper_model_size=self._base_config.model,
            stream=False,
            working_dir=None,
        )
        cfg.transcriber_backend = self._base_config.transcriber_backend
        cfg.language_identifier_backend = self._base_config.language_identifier_backend
        cfg.mms_lid_model_size = self._base_config.mms_lid_model_size
        if language:
            cfg.lang = language
        if self._base_config.beam_size is not None:
            cfg.whisper_beam_size = self._base_config.beam_size
        if self._base_config.best_of is not None:
            cfg.whisper_best_of = self._base_config.best_of
        if self._base_config.patience is not None:
            cfg.whisper_patience = self._base_config.patience

        if self._diarization.huggingface_token:
            os.environ["HUGGINGFACE_TOKEN"] = self._diarization.huggingface_token

        basename = os.path.splitext(os.path.basename(input_label))[0] or "verbatim"
        output_prefix = os.path.join(cfg.output_dir, basename)
        working_prefix = basename if cfg.working_dir is None else os.path.join(cfg.working_dir, basename)
        return cfg, output_prefix, working_prefix

    def transcribe_file(
        self,
        path: str,
        language: Optional[list[str]],
        status_hook: Optional[Callable[[Any], None]] = None,
    ) -> Tuple[str, list[Dict[str, Any]], list[Any]]:
        cfg, output_prefix, working_prefix = self._build_config(language, path)
        try:
            with open(path, "rb") as fh:
                cfg.cache.set_bytes(path, fh.read())
        except OSError as exc:
            raise RuntimeError(f"Failed to read input file '{path}': {exc}") from exc

        transcriber = self._mods.Verbatim(cfg, models=self._models, status_hook=status_hook)
        utterances: list[Any] = []
        for audio_source in self._mods.create_audio_sources(
            source_config=self._source_config,
            device=cfg.device,
            cache=cfg.cache,
            input_source=path,
            start_time="00:00.000",
            stop_time="",
            working_prefix_no_ext=working_prefix,
            output_prefix_no_ext=output_prefix,
            stream=False,
        ):
            with audio_source.open() as audio_stream:
                for utterance, _unack, _unconfirmed in transcriber.transcribe(
                    audio_stream=audio_stream, working_prefix_no_ext=working_prefix
                ):
                    utterances.append(utterance)

        segments: list[Dict[str, Any]] = []
        text_parts: list[str] = []
        for utt in utterances:
            segments.append(
                {
                    "start": float(utt.get_start()),
                    "end": float(utt.get_end()),
                    "text": utt.text,
                    "speaker": utt.speaker,
                }
            )
            text_parts.append(utt.text)
            return "".join(text_parts).strip(), segments, utterances

    def transcribe_bytes(
        self,
        payload: bytes,
        input_label: str,
        language: Optional[list[str]],
        source_config: Optional[Any] = None,
        status_hook: Optional[Callable[[Any], None]] = None,
    ) -> Tuple[str, list[Dict[str, Any]], list[Any]]:
        cfg, output_prefix, working_prefix = self._build_config(language, input_label)
        cfg.cache.set_bytes(input_label, payload)
        use_source_config = source_config or self._source_config

        transcriber = self._mods.Verbatim(cfg, models=self._models, status_hook=status_hook)
        utterances: list[Any] = []
        for audio_source in self._mods.create_audio_sources(
            source_config=use_source_config,
            device=cfg.device,
            cache=cfg.cache,
            input_source=input_label,
            start_time="00:00.000",
            stop_time="",
            working_prefix_no_ext=working_prefix,
            output_prefix_no_ext=output_prefix,
            stream=False,
        ):
            with audio_source.open() as audio_stream:
                for utterance, _unack, _unconfirmed in transcriber.transcribe(
                    audio_stream=audio_stream, working_prefix_no_ext=working_prefix
                ):
                    utterances.append(utterance)

        segments: list[Dict[str, Any]] = []
        text_parts: list[str] = []
        for utt in utterances:
            segments.append(
                {
                    "start": float(utt.get_start()),
                    "end": float(utt.get_end()),
                    "text": utt.text,
                    "speaker": utt.speaker,
                }
            )
            text_parts.append(utt.text)
        return "".join(text_parts).strip(), segments, utterances


class DiarizationService:
    def __init__(self, config: DiarizationConfig, device: str) -> None:
        self._config = config
        self._device = device
        self._pipeline: Optional[Any] = None

    def _get_pipeline(self) -> Optional[Any]:
        if not self._config.enabled:
            return None
        if self._config.strategy != "pyannote":
            logging.warning("Unsupported diarization strategy '%s'; diarization disabled.", self._config.strategy)
            return None
        token = (self._config.huggingface_token or os.environ.get("HUGGINGFACE_TOKEN") or "").strip()
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN is required for pyannote diarization.")
        if self._pipeline is None:
            try:
                from pyannote.audio import Pipeline
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "Diarization requires the optional 'verbatim' dependencies. Install them with `uv sync --extra verbatim`."
                ) from exc
            self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
            self._pipeline.to(torch.device(self._device))
        return self._pipeline

    def diarize(self, path: str) -> list[Dict[str, Any]]:
        pipeline = self._get_pipeline()
        if pipeline is None:
            return []
        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError(
                "Diarization requires the optional 'verbatim' dependencies. Install them with `uv sync --extra verbatim`."
            ) from exc
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=self._config.speakers)
        if hasattr(diarization, "speaker_diarization"):
            diarization = diarization.speaker_diarization
        turns: list[Dict[str, Any]] = []
        for turn, _track, speaker in diarization.itertracks(yield_label=True):  # type: ignore[assignment]
            turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
        return turns


class FastWhisperService:
    def __init__(self, config: TranscriptionConfig) -> None:
        logging.info("loading fast whisper model: %s (device=%s)", config.model, config.device)
        self._config = config
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Fast whisper requires the base worker dependencies. Install them with `uv sync`."
            ) from exc
        self._model = WhisperModel(config.model, device=config.device, compute_type="int8")

    def transcribe_file(
        self,
        path: str,
        language: Optional[list[str]],
        diarization_turns: list[Dict[str, Any]],
    ) -> Tuple[str, list[Dict[str, Any]], Optional[list[Any]]]:
        lang = None
        if language:
            # faster-whisper accepts a single language code; use first.
            lang = language[0]
        segments, info = self._model.transcribe(
            path,
            language=lang,
            beam_size=self._config.beam_size or 5,
            best_of=self._config.best_of or 5,
            patience=self._config.patience or 1.0,
        )
        logging.info("transcribe complete: language=%s duration=%.2fs", info.language, info.duration)
        text_parts: list[str] = []
        segment_list: list[Dict[str, Any]] = []
        for seg in segments:
            text_parts.append(seg.text)
            segment_list.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text})

        if diarization_turns:
            for seg in segment_list:
                best_speaker = None
                best_overlap = 0.0
                for turn in diarization_turns:
                    overlap = max(0.0, min(seg["end"], turn["end"]) - max(seg["start"], turn["start"]))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = turn["speaker"]
                seg["speaker"] = best_speaker

        utterances = _segments_to_utterances(segment_list, language)
        return "".join(text_parts).strip(), segment_list, utterances


class MlxVibeVoiceService:
    def __init__(self, config: TranscriptionConfig) -> None:
        self._config = config
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        if platform.system() != "Darwin":
            raise RuntimeError("MLX backend requires macOS.")
        try:
            from mlx_audio.stt.utils import load as load_model
        except ImportError:
            try:
                from mlx_audio.stt.utils import load_model  # type: ignore[attr-defined]
            except ImportError as exc:
                raise RuntimeError(
                    "MLX backend requires mlx-audio. Install it on macOS before selecting the MLX backend."
                ) from exc
        logging.info("loading MLX VibeVoice model: %s", self._config.model)
        self._model = load_model(self._config.model)
        return self._model

    def warmup(self) -> None:
        self._get_model()

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _extract_stream_status(cls, chunk: Any, stream_finish: Optional[float] = None) -> Optional[StatusPayload]:
        if chunk is None:
            return None

        text: Optional[str] = None
        current: Optional[float] = None
        finish: Optional[float] = None

        if isinstance(chunk, str):
            text = chunk.strip() or None
        elif isinstance(chunk, dict):
            text = str(chunk.get("text") or chunk.get("Content") or "").strip() or None
            current = cls._coerce_float(
                chunk.get("end_time")
                or chunk.get("End")
                or chunk.get("current_time")
                or chunk.get("timestamp")
                or chunk.get("time")
            )
            finish = cls._coerce_float(chunk.get("duration") or chunk.get("finish"))
        else:
            text = str(getattr(chunk, "text", "") or "").strip() or None
            current = cls._coerce_float(
                getattr(chunk, "end_time", None)
                or getattr(chunk, "current_time", None)
                or getattr(chunk, "timestamp", None)
                or getattr(chunk, "time", None)
            )
            finish = cls._coerce_float(getattr(chunk, "duration", None) or getattr(chunk, "finish", None))

        if finish is None:
            finish = stream_finish

        payload: StatusPayload = {"phase": "transcribing"}
        if current is not None:
            progress_payload: Dict[str, Any] = {
                "current": round(current, 2),
                "units": "seconds",
            }
            if finish is not None:
                progress_payload["finish"] = round(finish, 2)
                if finish > 0:
                    progress_payload["percent"] = round(max(0.0, min(100.0, (current / finish) * 100.0)), 1)
            payload["progress"] = progress_payload
        if text:
            payload["utterance"] = {"text": text}
        return payload if "progress" in payload or "utterance" in payload else None

    @staticmethod
    def _normalize_segment(seg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(seg, dict):
            return None
        start = seg.get("start", seg.get("start_time", seg.get("Start", 0.0)))
        end = seg.get("end", seg.get("end_time", seg.get("End", start)))
        speaker = seg.get("speaker", seg.get("speaker_id", seg.get("Speaker")))
        text = str(seg.get("text", seg.get("Content", ""))).strip()
        normalized: Dict[str, Any] = {
            "start": float(start),
            "end": float(end),
            "text": text,
        }
        if speaker is not None:
            normalized["speaker"] = str(speaker)
        return normalized

    def _build_context(self, previous_text: str) -> Optional[str]:
        context = (previous_text or "").strip()
        if not context:
            return None
        if self._config.mlx_context_characters > 0:
            context = context[-self._config.mlx_context_characters :]
        return (
            "Use this prior transcript for continuity of names, terminology, and topic. "
            "Preserve consistent wording when it fits the audio.\n\n"
            f"{context}"
        )

    @staticmethod
    def _iter_chunk_windows(duration_seconds: float, chunk_seconds: float, overlap_seconds: float) -> list[tuple[float, float]]:
        if duration_seconds <= 0:
            return []
        if chunk_seconds <= 0 or duration_seconds <= chunk_seconds:
            return [(0.0, duration_seconds)]
        stride = chunk_seconds - max(0.0, min(overlap_seconds, chunk_seconds - 1.0))
        if stride <= 0:
            stride = chunk_seconds
        windows: list[tuple[float, float]] = []
        start = 0.0
        while start < duration_seconds:
            end = min(duration_seconds, start + chunk_seconds)
            windows.append((start, end))
            if end >= duration_seconds:
                break
            start += stride
        return windows

    @staticmethod
    def _iter_json_objects(buffer: str) -> tuple[list[Dict[str, Any]], str]:
        objects: list[Dict[str, Any]] = []
        in_string = False
        escape = False
        depth = 0
        start_idx: Optional[int] = None
        consumed_upto = 0

        for idx, char in enumerate(buffer):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue

            if char == "{":
                if depth == 0:
                    start_idx = idx
                depth += 1
                continue

            if char == "}":
                if depth == 0:
                    continue
                depth -= 1
                if depth == 0 and start_idx is not None:
                    candidate = buffer[start_idx : idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        objects.append(parsed)
                        consumed_upto = idx + 1
                    start_idx = None

        remainder = buffer[consumed_upto:] if consumed_upto else buffer
        return objects, remainder

    def _stream_transcribe(
        self,
        model: Any,
        audio: AudioBuffer,
        context: Optional[str],
        status_hook: Optional[Callable[[Any], None]],
        progress_offset: float = 0.0,
        progress_finish: Optional[float] = None,
        emit_utterances: bool = True,
    ) -> list[Dict[str, Any]]:
        segments: list[Dict[str, Any]] = []
        stream_transcribe = getattr(model, "stream_transcribe", None)
        if not callable(stream_transcribe):
            return segments
        if audio.samples.size == 0:
            raise AudioDecodeError("cannot stream-transcribe an empty audio buffer")

        try:
            buffer = ""
            estimated_finish = progress_finish if progress_finish is not None else audio.duration_seconds
            stream_items = 0
            status_events = 0
            first_activity_logged = False
            raw_text_chunks = 0
            raw_text_characters = 0
            parsed_object_count = 0
            nonprogress_payloads = 0
            progress_payloads = 0
            last_stream_yield_at = time.monotonic()
            last_progress_at: Optional[float] = None
            last_progress_value: Optional[float] = None
            last_text_diagnostic_at = last_stream_yield_at
            last_payload_diagnostic_at = last_stream_yield_at
            last_parsed_object_at = last_stream_yield_at
            last_parse_stall_log_at = 0.0
            logging.info(
                "MLX stream start offset=%.2fs duration=%.3fs finish=%.3fs context_chars=%s max_tokens=%s repetition_penalty=%.3f repetition_context_size=%s emit_utterances=%s",
                progress_offset,
                audio.duration_seconds,
                estimated_finish,
                len(context or ""),
                self._config.mlx_max_tokens,
                self._config.mlx_repetition_penalty,
                self._config.mlx_repetition_context_size,
                emit_utterances,
            )
            for chunk in stream_transcribe(
                audio=audio.samples,
                sampling_rate=audio.sample_rate,
                context=context,
                max_tokens=self._config.mlx_max_tokens,
                repetition_penalty=self._config.mlx_repetition_penalty,
                repetition_context_size=self._config.mlx_repetition_context_size,
            ):
                stream_items += 1
                now = time.monotonic()
                idle_since_last_yield = max(0.0, now - last_stream_yield_at)
                last_stream_yield_at = now
                if isinstance(chunk, str):
                    raw_text_chunks += 1
                    raw_text_characters += len(chunk)
                    logging.debug(
                        "MLX stream raw chunk offset=%.2fs item=%s chars=%s text='%s'",
                        progress_offset,
                        stream_items,
                        len(chunk),
                        _debug_text_preview(chunk, limit=400),
                    )
                    buffer += chunk
                    objects, buffer = self._iter_json_objects(buffer)
                    if objects:
                        parsed_object_count += len(objects)
                        last_parsed_object_at = now
                    if raw_text_chunks == 1 or idle_since_last_yield >= 5.0 or (now - last_text_diagnostic_at) >= 10.0:
                        logging.debug(
                            "MLX stream raw yield offset=%.2fs item=%s idle=%.2fs chars=%s total_text_chars=%s buffer_chars=%s parsed_objects=%s segments=%s",
                            progress_offset,
                            stream_items,
                            idle_since_last_yield,
                            len(chunk),
                            raw_text_characters,
                            len(buffer),
                            parsed_object_count,
                            len(segments),
                        )
                        last_text_diagnostic_at = now
                    parse_stall_age = max(0.0, now - last_parsed_object_at)
                    if not objects and len(buffer) >= 1024 and parse_stall_age >= 5.0 and (now - last_parse_stall_log_at) >= 10.0:
                        logging.debug(
                            "MLX stream parse stall offset=%.2fs item=%s buffer_chars=%s parsed_objects=%s stall_age=%.2fs buffer_head='%s' buffer_tail='%s'",
                            progress_offset,
                            stream_items,
                            len(buffer),
                            parsed_object_count,
                            parse_stall_age,
                            _debug_text_preview(buffer[:240]),
                            _debug_text_preview(buffer[-240:]),
                        )
                        last_parse_stall_log_at = now
                    if objects:
                        if not first_activity_logged:
                            logging.info(
                                "MLX stream produced first parsed JSON object offset=%.2fs items=%s",
                                progress_offset,
                                stream_items,
                            )
                            first_activity_logged = True
                        for obj in objects:
                            normalized = self._normalize_segment(obj)
                            if normalized is not None:
                                normalized["start"] = round(normalized["start"] + progress_offset, 2)
                                normalized["end"] = round(normalized["end"] + progress_offset, 2)
                                segments.append(normalized)
                            payload = self._extract_stream_status(obj, estimated_finish)
                            if payload:
                                progress = payload.get("progress")
                                if isinstance(progress, dict) and "current" in progress:
                                    progress_payloads += 1
                                else:
                                    nonprogress_payloads += 1
                            if payload and progress_offset:
                                progress = payload.get("progress")
                                if isinstance(progress, dict) and "current" in progress:
                                    progress["current"] = round(float(progress["current"]) + progress_offset, 2)
                                    if estimated_finish is not None:
                                        progress["finish"] = round(estimated_finish, 2)
                                        if estimated_finish > 0:
                                            progress["percent"] = round(
                                                max(0.0, min(100.0, (float(progress["current"]) / estimated_finish) * 100.0)),
                                                1,
                                            )
                            if payload:
                                progress = payload.get("progress")
                                if isinstance(progress, dict) and "current" in progress:
                                    try:
                                        current_progress = float(progress["current"])
                                    except (TypeError, ValueError):
                                        current_progress = None
                                    if current_progress is not None:
                                        delta = None if last_progress_value is None else current_progress - last_progress_value
                                        since_last_progress = None if last_progress_at is None else max(0.0, now - last_progress_at)
                                        logging.debug(
                                            "MLX stream progress offset=%.2fs item=%s current=%.2f delta=%s since_last=%s segments=%s buffer_chars=%s",
                                            progress_offset,
                                            stream_items,
                                            current_progress,
                                            "n/a" if delta is None else f"{delta:.2f}",
                                            "n/a" if since_last_progress is None else f"{since_last_progress:.2f}s",
                                            len(segments),
                                            len(buffer),
                                        )
                                        last_progress_value = current_progress
                                        last_progress_at = now
                                elif (now - last_payload_diagnostic_at) >= 10.0:
                                    logging.debug(
                                        "MLX stream non-progress payload offset=%.2fs item=%s keys=%s nonprogress_payloads=%s segments=%s buffer_chars=%s",
                                        progress_offset,
                                        stream_items,
                                        sorted(payload.keys()),
                                        nonprogress_payloads,
                                        len(segments),
                                        len(buffer),
                                    )
                                    last_payload_diagnostic_at = now
                            if payload and not emit_utterances:
                                payload.pop("utterance", None)
                            if payload and status_hook is not None:
                                status_hook(payload)
                                status_events += 1
                    continue
                payload = self._extract_stream_status(chunk, estimated_finish)
                if payload:
                    progress = payload.get("progress")
                    if isinstance(progress, dict) and "current" in progress:
                        progress_payloads += 1
                    else:
                        nonprogress_payloads += 1
                if payload and not first_activity_logged:
                    logging.info(
                        "MLX stream produced first status payload offset=%.2fs items=%s keys=%s",
                        progress_offset,
                        stream_items,
                        sorted(payload.keys()),
                    )
                    first_activity_logged = True
                if payload and progress_offset:
                    progress = payload.get("progress")
                    if isinstance(progress, dict) and "current" in progress:
                        progress["current"] = round(float(progress["current"]) + progress_offset, 2)
                        if estimated_finish is not None:
                            progress["finish"] = round(estimated_finish, 2)
                            if estimated_finish > 0:
                                progress["percent"] = round(
                                max(0.0, min(100.0, (float(progress["current"]) / estimated_finish) * 100.0)),
                                1,
                            )
                if payload:
                    progress = payload.get("progress")
                    if isinstance(progress, dict) and "current" in progress:
                        try:
                            current_progress = float(progress["current"])
                        except (TypeError, ValueError):
                            current_progress = None
                        if current_progress is not None:
                            delta = None if last_progress_value is None else current_progress - last_progress_value
                            since_last_progress = None if last_progress_at is None else max(0.0, now - last_progress_at)
                            logging.debug(
                                "MLX stream progress offset=%.2fs item=%s current=%.2f delta=%s since_last=%s segments=%s raw_text_chunks=%s trailing_buffer_chars=%s",
                                progress_offset,
                                stream_items,
                                current_progress,
                                "n/a" if delta is None else f"{delta:.2f}",
                                "n/a" if since_last_progress is None else f"{since_last_progress:.2f}s",
                                len(segments),
                                raw_text_chunks,
                                len(buffer),
                            )
                            last_progress_value = current_progress
                            last_progress_at = now
                    elif stream_items == 1 or idle_since_last_yield >= 5.0 or (now - last_payload_diagnostic_at) >= 10.0:
                        logging.debug(
                            "MLX stream payload without progress offset=%.2fs item=%s idle=%.2fs keys=%s nonprogress_payloads=%s segments=%s",
                            progress_offset,
                            stream_items,
                            idle_since_last_yield,
                            sorted(payload.keys()),
                            nonprogress_payloads,
                            len(segments),
                        )
                        last_payload_diagnostic_at = now
                if payload and not emit_utterances:
                    payload.pop("utterance", None)
                if payload and status_hook is not None:
                    status_hook(payload)
                    status_events += 1
        except AudioDecodeError:
            raise
        except Exception:
            logging.exception(
                "MLX VibeVoice streaming status failed; continuing with final generate() result "
                "(samples=%s sample_rate=%s duration=%.3fs offset=%.3fs)",
                audio.samples.size,
                audio.sample_rate,
                audio.duration_seconds,
                progress_offset,
            )
            return []
        last_end = 0.0 if not segments else float(segments[-1].get("end", segments[-1].get("start", 0.0)))
        logging.info(
            "MLX stream finished offset=%.2fs segments=%s status_events=%s stream_items=%s raw_text_chunks=%s parsed_objects=%s progress_payloads=%s nonprogress_payloads=%s trailing_buffer_chars=%s last_end=%.2fs",
            progress_offset,
            len(segments),
            status_events,
            stream_items,
            raw_text_chunks,
            parsed_object_count,
            progress_payloads,
            nonprogress_payloads,
            len(buffer),
            last_end,
        )
        return segments

    def _transcribe_single_file(
        self,
        model: Any,
        audio: AudioBuffer,
        language: Optional[list[str]],
        context: Optional[str],
        status_hook: Optional[Callable[[Any], None]],
        progress_offset: float = 0.0,
        progress_finish: Optional[float] = None,
        emit_utterances: bool = True,
    ) -> tuple[str, list[Dict[str, Any]]]:
        if language:
            logging.info("MLX VibeVoice ignores explicit language selection; using model auto-detection.")
        if audio.samples.size == 0:
            raise AudioDecodeError("cannot transcribe an empty audio buffer")
        segments = self._stream_transcribe(
            model,
            audio,
            context,
            status_hook,
            progress_offset=progress_offset,
            progress_finish=progress_finish,
            emit_utterances=emit_utterances,
        )
        if segments:
            logging.info(
                "MLX transcription completed from streaming offset=%.2fs segments=%s text_chars=%s",
                progress_offset,
                len(segments),
                len(_segments_to_plain_text(segments)),
            )
            return _segments_to_plain_text(segments), segments

        logging.warning(
            "MLX VibeVoice streaming yielded no segments; falling back to generate() "
            "(samples=%s sample_rate=%s duration=%.3fs offset=%.3fs context_chars=%s)",
            audio.samples.size,
            audio.sample_rate,
            audio.duration_seconds,
            progress_offset,
            len(context or ""),
        )
        logging.info(
            "MLX generate fallback start offset=%.2fs duration=%.3fs context_chars=%s max_tokens=%s repetition_penalty=%.3f repetition_context_size=%s",
            progress_offset,
            audio.duration_seconds,
            len(context or ""),
            self._config.mlx_max_tokens,
            self._config.mlx_repetition_penalty,
            self._config.mlx_repetition_context_size,
        )
        result = model.generate(
            audio=audio.samples,
            sampling_rate=audio.sample_rate,
            context=context,
            max_tokens=self._config.mlx_max_tokens,
            temperature=0.0,
            repetition_penalty=self._config.mlx_repetition_penalty,
            repetition_context_size=self._config.mlx_repetition_context_size,
        )
        raw_segments = getattr(result, "segments", None) or []
        if not raw_segments:
            raw_text = getattr(result, "text", None)
            if isinstance(raw_text, str):
                try:
                    parsed_text = json.loads(raw_text)
                except json.JSONDecodeError:
                    parsed_text = None
                if isinstance(parsed_text, list):
                    raw_segments = [seg for seg in parsed_text if isinstance(seg, dict)]
        normalized_segments: list[Dict[str, Any]] = []
        for seg in raw_segments:
            normalized = self._normalize_segment(seg)
            if normalized is None:
                continue
            normalized["start"] = round(normalized["start"] + progress_offset, 2)
            normalized["end"] = round(normalized["end"] + progress_offset, 2)
            normalized_segments.append(normalized)
        text = _segments_to_plain_text(normalized_segments)
        if not text:
            text = (getattr(result, "text", None) or "").strip()
        logging.info(
            "MLX generate fallback finished offset=%.2fs raw_segments=%s normalized_segments=%s text_chars=%s",
            progress_offset,
            len(raw_segments),
            len(normalized_segments),
            len(text),
        )
        return text, normalized_segments

    def transcribe_file(
        self,
        path: str,
        language: Optional[list[str]],
        status_hook: Optional[Callable[[Any], None]] = None,
    ) -> Tuple[str, list[Dict[str, Any]], Optional[list[Any]]]:
        with open(path, "rb") as handle:
            payload = handle.read()
        return self.transcribe_bytes(payload, os.path.basename(path), language, status_hook)

    def transcribe_bytes(
        self,
        payload: bytes,
        input_label: str,
        language: Optional[list[str]],
        status_hook: Optional[Callable[[Any], None]] = None,
    ) -> Tuple[str, list[Dict[str, Any]], Optional[list[Any]]]:
        model = self._get_model()
        try:
            audio = normalize_audio_buffer(payload, target_sample_rate=24000)
        except AudioDecodeError as exc:
            raise AudioDecodeError(
                f"failed to decode '{input_label}' ({len(payload)} bytes) into MLX audio: {exc}"
            ) from exc
        logging.info(
            "decoded MLX audio input=%s bytes=%s sample_rate=%s samples=%s duration=%.3fs",
            input_label,
            len(payload),
            audio.sample_rate,
            audio.samples.size,
            audio.duration_seconds,
        )
        total_duration = audio.duration_seconds
        chunk_seconds = max(0.0, self._config.mlx_chunk_seconds)
        overlap_seconds = max(0.0, min(self._config.mlx_chunk_overlap_seconds, max(0.0, chunk_seconds - 1.0)))
        embeddings_enabled = chunk_seconds > 0 and total_duration > chunk_seconds
        speaker_resolver = IncrementalSpeakerResolver(audio=audio, enabled=embeddings_enabled)
        try:
            if chunk_seconds <= 0 or total_duration <= chunk_seconds:
                logging.info(
                    "running MLX single-window transcription duration=%.3fs embeddings_enabled=%s",
                    total_duration,
                    embeddings_enabled,
                )
                text, segments = self._transcribe_single_file(
                    model,
                    audio,
                    language,
                    context=None,
                    status_hook=status_hook,
                    progress_finish=total_duration,
                    emit_utterances=False,
                )
                annotated_segments = [
                    speaker_resolver.annotate_segment(segment, chunk_index=1)
                    for segment in segments
                ]
                resolved_emitter = ChunkResolvedEmitter(
                    overlap_seconds=0.0,
                    finish=total_duration,
                    status_hook=status_hook,
                    segment_transformer=speaker_resolver.label_for_emission,
                    segment_observer=speaker_resolver.observe_emitted_segment,
                )
                resolved_emitter.process_chunk(
                    annotated_segments,
                    chunk_start=0.0,
                    chunk_end=total_duration,
                    is_last=True,
                )
                logging.info("waiting for speaker resolution on single-window transcription segments=%s", len(annotated_segments))
                annotated_segments = speaker_resolver.relabel_segments(annotated_segments)
                utterances = _segments_to_utterances(annotated_segments, language)
                logging.info(
                    "single-window transcription complete segments=%s utterances=%s text_chars=%s",
                    len(annotated_segments),
                    0 if utterances is None else len(utterances),
                    len(_segments_to_plain_text(annotated_segments)),
                )
                return _segments_to_plain_text(annotated_segments), annotated_segments, utterances

            windows = self._iter_chunk_windows(total_duration, chunk_seconds, overlap_seconds)
            logging.info(
                "chunking MLX transcription into %s windows (chunk=%.1fs overlap=%.1fs max_tokens=%s)",
                len(windows),
                chunk_seconds,
                overlap_seconds,
                self._config.mlx_max_tokens,
            )

            merged_segments: list[Dict[str, Any]] = []
            previous_chunk_text = ""
            resolved_emitter = ChunkResolvedEmitter(
                overlap_seconds=overlap_seconds,
                finish=total_duration,
                status_hook=status_hook,
                segment_transformer=speaker_resolver.label_for_emission,
                segment_observer=speaker_resolver.observe_emitted_segment,
            )
            for chunk_index, (chunk_start, chunk_end) in enumerate(windows, start=1):
                chunk_audio = slice_audio_buffer(audio, chunk_start, chunk_end)
                context = self._build_context(previous_chunk_text)
                logging.info(
                    "transcribing MLX chunk %s/%s start=%.2fs end=%.2fs context_chars=%s samples=%s duration=%.3fs",
                    chunk_index,
                    len(windows),
                    chunk_start,
                    chunk_end,
                    len(context or ""),
                    chunk_audio.samples.size,
                    chunk_audio.duration_seconds,
                )
                chunk_text, chunk_segments = self._transcribe_single_file(
                    model,
                    chunk_audio,
                    language,
                    context=context,
                    status_hook=status_hook,
                    progress_offset=chunk_start,
                    progress_finish=total_duration,
                    emit_utterances=False,
                )
                logging.info(
                    "completed MLX chunk %s/%s start=%.2fs end=%.2fs segments=%s text_chars=%s",
                    chunk_index,
                    len(windows),
                    chunk_start,
                    chunk_end,
                    len(chunk_segments),
                    len(chunk_text),
                )
                annotated_chunk_segments: list[Dict[str, Any]] = []
                for segment in chunk_segments:
                    annotated = speaker_resolver.annotate_segment(segment, chunk_index=chunk_index)
                    annotated["speaker"] = speaker_resolver.label_for_emission(annotated).get("speaker")
                    annotated_chunk_segments.append(annotated)
                chunk_segments = annotated_chunk_segments

                if not merged_segments:
                    merged_segments = chunk_segments
                else:
                    overlap_start = chunk_start
                    overlap_end = min(chunk_end, chunk_start + overlap_seconds)
                    logging.info(
                        "merging MLX chunk %s/%s into transcript overlap_start=%.2fs overlap_end=%.2fs prior_segments=%s current_segments=%s",
                        chunk_index,
                        len(windows),
                        overlap_start,
                        overlap_end,
                        len(merged_segments),
                        len(chunk_segments),
                    )
                    merged_segments = _merge_chunk_segments(merged_segments, chunk_segments, overlap_start, overlap_end)
                resolved_emitter.process_chunk(
                    chunk_segments,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    is_last=chunk_index == len(windows),
                )
                logging.info(
                    "resolved emitter processed chunk %s/%s merged_segments=%s emitted_segments=%s",
                    chunk_index,
                    len(windows),
                    len(merged_segments),
                    len(resolved_emitter._emitted),
                )
                previous_chunk_text = chunk_text

            logging.info("finalizing resolved emitter after %s chunks", len(windows))
            resolved_emitter.finalize()
            logging.info("waiting for final speaker resolution merged_segments=%s", len(merged_segments))
            merged_segments = speaker_resolver.relabel_segments(merged_segments)
            text = _segments_to_plain_text(merged_segments)
            utterances = _segments_to_utterances(merged_segments, language)
            logging.info(
                "chunked MLX transcription complete chunks=%s segments=%s utterances=%s text_chars=%s",
                len(windows),
                len(merged_segments),
                0 if utterances is None else len(utterances),
                len(text),
            )
            return text, merged_segments, utterances
        finally:
            speaker_resolver.close()


def _segments_to_utterances(segments: list[Dict[str, Any]], languages: Optional[list[str]]) -> Optional[list[Any]]:
    mods = _load_verbatim_modules(required=False)
    if mods is None:
        return None
    lang = languages[0] if languages else "en"
    utterances: list[Any] = []
    for idx, seg in enumerate(segments):
        start_sec = float(seg.get("start", 0.0))
        end_sec = float(seg.get("end", start_sec))
        start_ts = int(start_sec * 16000)
        end_ts = int(end_sec * 16000)
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        tokens = text.split()
        if not tokens:
            continue
        duration = max(1, end_ts - start_ts)
        step = max(1, duration // len(tokens))
        words: list[Any] = []
        for w_idx, token in enumerate(tokens):
            w_start = start_ts + w_idx * step
            w_end = start_ts + (w_idx + 1) * step
            if w_idx == len(tokens) - 1:
                w_end = end_ts
            word_text = token if w_idx == 0 else f" {token}"
            words.append(mods.Word(start_ts=w_start, end_ts=w_end, word=word_text, probability=1.0, lang=lang))
        utterances.append(
            mods.Utterance.from_words(
                utterance_id=str(idx),
                words=words,
                speaker=_public_segment_speaker(seg),
            )
        )
    return utterances


async def transcribe_handler(
    verbatim_service: Optional[VerbatimService],
    mlx_service: MlxVibeVoiceService,
    job: JobClaim,
    payload: bytes,
    language: Optional[list[str]],
    backend: str,
    strategy: str,
    diarization_service: DiarizationService,
    diarize_enabled: bool,
    base_verbatim_args: argparse.Namespace,
    base_output_formats: list[str],
    status_poster: Optional[NfrxStatusPoster],
    status_include_utterance: bool,
    status_watchdog_seconds: float,
) -> Tuple[bytes, Optional[str]]:
    job_id = job.get("job_id") or job.get("id") or ""
    status_hook: Optional[Callable[[Any], None]] = None
    if status_poster and job_id:
        status_hook = status_poster.build_hook(job_id, status_include_utterance)
        status_poster.start_watchdog(job_id, status_watchdog_seconds, {"message": "working"})
    try:
        metadata = job.get("metadata") or {}
        filename = metadata.get("filename") or metadata.get("name")
        content_type = metadata.get("content_type") or metadata.get("mime_type")
        if strategy == "fast":
            return b"", "fast strategy requires file-backed input; temp files are disabled"
        if backend == "mlx-vibevoice" and diarize_enabled:
            logging.info("MLX backend selected; external diarization is disabled.")

        job_cfg = metadata.get("config")
        if job_cfg is not None and not isinstance(job_cfg, dict):
            logging.warning("job config ignored; expected mapping but got %s", type(job_cfg).__name__)
            job_cfg = None
        if job_cfg:
            logging.info("job config: %s", _redact_job_config(job_cfg))

        job_args, output_files, output_formats, job_write_config = _apply_job_config(
            base_verbatim_args, job_cfg or {}
        )
        if output_formats is None:
            output_formats = base_output_formats
        job_language = getattr(job_args, "languages", None) or language

        output_format = _output_format_from_formats(output_formats)
        write_config = job_write_config or (verbatim_service._write_config if verbatim_service else None)
        source_config = verbatim_service._source_config if verbatim_service else None
        mods = _load_verbatim_modules(required=False)
        if job_cfg and mods is not None and verbatim_service is not None:
            speakers_resolved = mods.verbatim_configure.resolve_speakers(job_args)
            source_config = mods.verbatim_configure.make_source_config(job_args, speakers_resolved)

        suffix = _infer_suffix(filename, content_type)
        base = os.path.splitext(os.path.basename(filename or "payload"))[0] or "payload"
        input_label = f"{base}{suffix}"

        logging.info("starting transcription (input=%s)", input_label)
        if backend == "mlx-vibevoice":
            text, segments, utterances = await asyncio.to_thread(
                mlx_service.transcribe_bytes,
                payload,
                input_label,
                job_language,
                status_hook,
            )
        else:
            if verbatim_service is None:
                return b"", "verbatim backend selected but optional verbatim dependencies are not installed"
            text, segments, utterances = await asyncio.to_thread(
                verbatim_service.transcribe_bytes,
                payload,
                input_label,
                job_language,
                source_config,
                status_hook,
            )
        if output_format == "text":
            if utterances is not None:
                formatted = _format_text_output(utterances, write_config)
            else:
                formatted = _format_text_from_segments(segments)
            logging.info("transcription produced %s chars (text output)", len(formatted))
            logging.info("returning text payload (%s bytes)", len(formatted.encode("utf-8")))
            return formatted.encode("utf-8"), None
        if output_format == "docx":
            if utterances is None:
                return b"", "docx output requires utterances"
            formatted_docx = _format_docx_output(utterances, write_config)
            logging.info("transcription produced %s bytes (docx output)", len(formatted_docx))
            logging.info("returning docx payload (%s bytes)", len(formatted_docx))
            return formatted_docx, None
        if output_format == "jsonl":
            formatted_jsonl = _format_jsonl_output(segments)
            logging.info("transcription produced %s lines (jsonl output)", len(segments))
            logging.info("returning jsonl payload (%s bytes)", len(formatted_jsonl))
            return formatted_jsonl, None

        result = {"text": text, "segments": _public_segments(segments)}
        logging.info("transcription produced %s chars, %s segments (json output)", len(text), len(segments))
        logging.info("returning json payload (%s bytes)", len(json.dumps(result).encode("utf-8")))
        return json.dumps(result).encode("utf-8"), None
    except Exception as exc:
        logging.exception("transcription failure")
        return b"", str(exc)
    finally:
        if status_poster and job_id:
            status_poster.mark_done(job_id)


async def run(args: argparse.Namespace) -> int:
    config_data = _load_config(args.config)
    logging.info("Loaded config file: %s", args.config)

    requested_backend = _normalize_backend(
        _resolve(args.backend, _cfg_get(config_data, "worker", "backend"), os.environ.get("ASR_BACKEND"), "auto")
    )
    requested_diarization_strategy = _resolve(
        args.diarization_strategy,
        _cfg_get(config_data, "diarization", "strategy"),
        os.environ.get("DIARIZATION_STRATEGY"),
        None,
    )
    requested_diarization_speakers = _resolve(
        args.diarization_speakers,
        _cfg_get(config_data, "diarization", "speakers"),
        os.environ.get("DIARIZATION_SPEAKERS"),
        None,
    )
    verbatim_mods = _load_verbatim_modules(required=False)

    if verbatim_mods is not None:
        verbatim_parser = verbatim_mods.verbatim_args.build_parser(include_input=False)
        verbatim_defaults = verbatim_parser.parse_args([])
        profile_overrides = verbatim_mods.select_profile(config_data, filename=None) if config_data else {}
        verbatim_user = argparse.Namespace(**vars(verbatim_defaults))
        if args.language:
            verbatim_user.languages = [args.language]
        elif getattr(verbatim_user, "languages", None) in (None, []):
            verbatim_user.languages = list(DEFAULT_LANGUAGES)
        if args.beam_size is not None:
            verbatim_user.nb_beams = args.beam_size
        if requested_diarization_strategy:
            verbatim_user.diarize = requested_diarization_strategy
        if requested_diarization_speakers is not None:
            verbatim_user.speakers = requested_diarization_speakers
        verbatim_cfg_args = verbatim_mods.merge_args(verbatim_defaults, profile_overrides, verbatim_user)
    else:
        verbatim_cfg_args = argparse.Namespace(
            languages=[args.language] if args.language else list(DEFAULT_LANGUAGES),
            nb_beams=args.beam_size,
            diarize=requested_diarization_strategy,
            speakers=requested_diarization_speakers,
            cpu=False,
        )

    worker_cfg = WorkerConfig(
        base_url=_resolve(args.base_url, _cfg_get(config_data, "worker", "base_url"), os.environ.get("NFRX_BASE_URL"), DEFAULT_BASE_URL),
        types=_resolve(args.types, _cfg_get(config_data, "worker", "types"), os.environ.get("NFRX_JOB_TYPES"), "asr.transcribe"),
        max_wait_seconds=_resolve_int(
            _resolve(args.max_wait_seconds, _cfg_get(config_data, "worker", "max_wait_seconds"), os.environ.get("NFRX_MAX_WAIT_SECONDS"), 30),
            30,
        ),
        concurrency=max(
            1,
            _resolve_int(
                _resolve(
                    args.concurrency,
                    _cfg_get(config_data, "worker", "concurrency"),
                    os.environ.get("NFRX_CONCURRENCY"),
                    _default_concurrency_for_backend(requested_backend),
                ),
                _default_concurrency_for_backend(requested_backend),
            )
            or _default_concurrency_for_backend(requested_backend),
        ),
        once=_resolve_bool(args.once, _cfg_get(config_data, "worker", "once"), os.environ.get("NFRX_ONCE"), False),
        log_level=_resolve(args.log_level, _cfg_get(config_data, "worker", "log_level"), os.environ.get("LOG_LEVEL"), "INFO"),
        backend=requested_backend,
        strategy=_resolve(args.strategy, _cfg_get(config_data, "worker", "strategy"), os.environ.get("ASR_STRATEGY"), "verbatim"),
        status_progress=_resolve_bool(
            args.status_progress,
            _cfg_get(config_data, "worker", "status_progress"),
            os.environ.get("NFRX_STATUS_PROGRESS"),
            True,
        ),
        status_interval_seconds=_resolve_float(
            _resolve(
                args.status_interval_seconds,
                _cfg_get(config_data, "worker", "status_interval_seconds"),
                os.environ.get("NFRX_STATUS_INTERVAL_SECONDS"),
                2.0,
            ),
            2.0,
        ),
        status_min_progress_seconds=_resolve_float(
            _resolve(
                args.status_min_progress_seconds,
                _cfg_get(config_data, "worker", "status_min_progress_seconds"),
                os.environ.get("NFRX_STATUS_MIN_PROGRESS_SECONDS"),
                1.0,
            ),
            1.0,
        ),
        status_include_utterance=_resolve_bool(
            args.status_include_utterance,
            _cfg_get(config_data, "worker", "status_include_utterance"),
            os.environ.get("NFRX_STATUS_INCLUDE_UTTERANCE"),
            False,
        ),
        status_watchdog_seconds=_resolve_float(
            _resolve(
                args.status_watchdog_seconds,
                _cfg_get(config_data, "worker", "status_watchdog_seconds"),
                os.environ.get("NFRX_STATUS_WATCHDOG_SECONDS"),
                20.0,
            ),
            20.0,
        ),
    )

    logging.getLogger().setLevel(getattr(logging, str(worker_cfg.log_level).upper(), logging.INFO))

    auth = AuthConfig(
        client_key=_resolve(args.client_key, _cfg_get(config_data, "auth", "client_key"), os.environ.get("NFRX_CLIENT_KEY"), None)
    )

    speakers_resolved = (
        verbatim_mods.verbatim_configure.resolve_speakers(verbatim_cfg_args)
        if verbatim_mods is not None
        else getattr(verbatim_cfg_args, "speakers", None)
    )
    diarization_cfg = DiarizationConfig(
        enabled=_resolve_bool(args.diarization, _cfg_get(config_data, "diarization", "enabled"), os.environ.get("DIARIZATION_ENABLED"), True),
        strategy=_resolve(
            args.diarization_strategy,
            _cfg_get(config_data, "diarization", "strategy"),
            os.environ.get("DIARIZATION_STRATEGY"),
            getattr(verbatim_cfg_args, "diarize", None) or "pyannote",
        ),
        speakers=_resolve_int(
            _resolve(
                args.diarization_speakers,
                _cfg_get(config_data, "diarization", "speakers"),
                os.environ.get("DIARIZATION_SPEAKERS"),
                speakers_resolved,
            )
        ),
        huggingface_token=_resolve(
            args.huggingface_token,
            _cfg_get(config_data, "diarization", "huggingface_token"),
            os.environ.get("HUGGINGFACE_TOKEN"),
            None,
        ),
    )

    if verbatim_mods is not None:
        output_formats = verbatim_mods.verbatim_configure.build_output_formats(verbatim_cfg_args, default_stdout=False)
        if not output_formats:
            output_formats = ["jsonl"]
    else:
        configured_output = _cfg_get_transcription(config_data, "output_format") or _cfg_get_transcription(config_data, "output")
        if str(configured_output or "").strip().lower() == "text":
            output_formats = ["txt"]
        elif str(configured_output or "").strip().lower() == "json":
            output_formats = ["json"]
        else:
            output_formats = ["jsonl"]
    output_format = _output_format_from_formats(output_formats)
    logging.info("Output formats: %s (selected output_format=%s)", output_formats, output_format)

    resolved_device = _resolve(
        args.device,
        _cfg_get_transcription(config_data, "device"),
        os.environ.get("WHISPER_DEVICE"),
        "cpu" if getattr(verbatim_cfg_args, "cpu", False) else "auto",
    )
    resolved_backend = requested_backend
    transcription_cfg = TranscriptionConfig(
        backend=resolved_backend,
        model=_resolve(
            args.model,
            _cfg_get_transcription(config_data, "model"),
            os.environ.get("WHISPER_MODEL"),
            _default_model_for_backend(resolved_backend),
        ),
        device=_resolve_device(resolved_device),
        language=_normalize_languages(
            _resolve(
                args.language,
                _cfg_get_transcription(config_data, "language") or getattr(verbatim_cfg_args, "languages", None),
                os.environ.get("WHISPER_LANGUAGE"),
                list(DEFAULT_LANGUAGES),
            )
        ),
        transcriber_backend=_resolve(
            None,
            _cfg_get_transcription(config_data, "transcriber_backend") or getattr(verbatim_cfg_args, "transcriber_backend", None),
            os.environ.get("VERBATIM_TRANSCRIBER_BACKEND"),
            "auto",
        ),
        language_identifier_backend=_resolve(
            None,
            _cfg_get_transcription(config_data, "language_identifier_backend") or getattr(verbatim_cfg_args, "language_identifier_backend", None),
            os.environ.get("VERBATIM_LANGUAGE_IDENTIFIER_BACKEND"),
            "transcriber",
        ),
        mms_lid_model_size=_resolve(
            None,
            _cfg_get_transcription(config_data, "mms_lid_model_size") or getattr(verbatim_cfg_args, "mms_lid_model_size", None),
            os.environ.get("VERBATIM_MMS_LID_MODEL_SIZE"),
            "facebook/mms-lid-126",
        ),
        beam_size=_resolve_int(
            _resolve(args.beam_size, _cfg_get_transcription(config_data, "beam_size"), os.environ.get("WHISPER_BEAM_SIZE"), getattr(verbatim_cfg_args, "nb_beams", None))
        ),
        best_of=_resolve_int(
            _resolve(args.best_of, _cfg_get_transcription(config_data, "best_of"), os.environ.get("WHISPER_BEST_OF"), None)
        ),
        patience=_resolve_float(
            _resolve(args.patience, _cfg_get_transcription(config_data, "patience"), os.environ.get("WHISPER_PATIENCE"), None)
        ),
        output_format=_resolve(
            None,
            _cfg_get_transcription(config_data, "output_format") or _cfg_get_transcription(config_data, "output"),
            os.environ.get("TRANSCRIPTION_OUTPUT_FORMAT"),
            "jsonl",
        ),
        mlx_max_tokens=max(
            1024,
            _resolve_int(
                _resolve(
                    getattr(args, "mlx_max_tokens", None),
                    _cfg_get_transcription(config_data, "mlx_max_tokens"),
                    os.environ.get("MLX_MAX_TOKENS"),
                    32768,
                ),
                32768,
            )
            or 32768,
        ),
        mlx_repetition_penalty=max(
            1.0,
            _resolve_float(
                _resolve(
                    getattr(args, "mlx_repetition_penalty", None),
                    _cfg_get_transcription(config_data, "mlx_repetition_penalty"),
                    os.environ.get("MLX_REPETITION_PENALTY"),
                    1.08,
                ),
                1.08,
            )
            or 1.08,
        ),
        mlx_repetition_context_size=max(
            1,
            _resolve_int(
                _resolve(
                    getattr(args, "mlx_repetition_context_size", None),
                    _cfg_get_transcription(config_data, "mlx_repetition_context_size"),
                    os.environ.get("MLX_REPETITION_CONTEXT_SIZE"),
                    256,
                ),
                256,
            )
            or 256,
        ),
        mlx_chunk_seconds=max(
            0.0,
            _resolve_float(
                _resolve(
                    getattr(args, "mlx_chunk_seconds", None),
                    _cfg_get_transcription(config_data, "mlx_chunk_seconds"),
                    os.environ.get("MLX_CHUNK_SECONDS"),
                    20.0 * 60.0,
                ),
                20.0 * 60.0,
            )
            or 0.0,
        ),
        mlx_chunk_overlap_seconds=max(
            0.0,
            _resolve_float(
                _resolve(
                    getattr(args, "mlx_chunk_overlap_seconds", None),
                    _cfg_get_transcription(config_data, "mlx_chunk_overlap_seconds"),
                    os.environ.get("MLX_CHUNK_OVERLAP_SECONDS"),
                    60.0,
                ),
                60.0,
            )
            or 0.0,
        ),
        mlx_context_characters=max(
            0,
            _resolve_int(
                _resolve(
                    getattr(args, "mlx_context_characters", None),
                    _cfg_get_transcription(config_data, "mlx_context_characters"),
                    os.environ.get("MLX_CONTEXT_CHARACTERS"),
                    8000,
                ),
                8000,
            )
            or 0,
        ),
    )

    # If config explicitly sets output formats (e.g., [txt]), honor that over env overrides.
    if output_formats and "json" not in output_formats:
        transcription_cfg.output_format = "text"

    write_config = (
        verbatim_mods.verbatim_configure.make_write_config(verbatim_cfg_args, logging.getLogger().level)
        if verbatim_mods is not None
        else None
    )
    source_config = (
        verbatim_mods.verbatim_configure.make_source_config(verbatim_cfg_args, speakers_resolved)
        if verbatim_mods is not None
        else None
    )

    worker = NfrxJobsWorker(worker_cfg.base_url, auth)
    status_poster: Optional[NfrxStatusPoster] = None
    if worker_cfg.status_progress:
        status_poster = NfrxStatusPoster(
            worker,
            min_interval_seconds=worker_cfg.status_interval_seconds,
            min_progress_seconds=worker_cfg.status_min_progress_seconds,
        )
        status_poster.start(asyncio.get_running_loop())
    verbatim_service = (
        VerbatimService(transcription_cfg, diarization_cfg, source_config, write_config)
        if verbatim_mods is not None
        else None
    )
    mlx_service = MlxVibeVoiceService(transcription_cfg)
    diarization_service = DiarizationService(diarization_cfg, transcription_cfg.device)
    if worker_cfg.backend == "verbatim" and verbatim_service is None:
        raise RuntimeError(
            "The 'verbatim' backend is selected but optional verbatim dependencies are not installed. "
            "Install them with `uv sync --extra verbatim`."
        )
    try:
        types = worker_cfg.types.split(",") if worker_cfg.types else None
        effective_concurrency = 1 if worker_cfg.once else worker_cfg.concurrency
        logging.info(
            "worker starting: backend=%s model=%s concurrency=%s",
            transcription_cfg.backend,
            transcription_cfg.model,
            effective_concurrency,
        )
        warmup_tasks: list[asyncio.Task[Any]] = []
        if transcription_cfg.backend == "mlx-vibevoice":
            async def _warm_mlx() -> None:
                try:
                    logging.info("warming MLX VibeVoice model")
                    await asyncio.to_thread(mlx_service.warmup)
                    logging.info("MLX VibeVoice warmup complete")
                except Exception:
                    logging.exception("MLX VibeVoice warmup failed; continuing without startup warmup")

            warmup_tasks.append(asyncio.create_task(_warm_mlx()))

            async def _warm_speaker_embeddings() -> None:
                try:
                    logging.info("warming speaker embedding model")
                    await asyncio.to_thread(warmup_default_speaker_embedder, target_sample_rate=16000)
                    logging.info("speaker embedding warmup complete")
                except Exception:
                    logging.exception("speaker embedding warmup failed; continuing without startup warmup")

            warmup_tasks.append(asyncio.create_task(_warm_speaker_embeddings()))

        async def _handle_job(job: JobClaim, payload: bytes) -> Tuple[bytes, Optional[str]]:
            backend = _normalize_backend(job.get("metadata", {}).get("backend", worker_cfg.backend))
            return await transcribe_handler(
                verbatim_service,
                mlx_service,
                job,
                payload,
                transcription_cfg.language,
                backend,
                job.get("metadata", {}).get("strategy", worker_cfg.strategy),
                diarization_service,
                diarization_cfg.enabled
                and diarization_cfg.speakers not in (1, "1")
                and backend != "mlx-vibevoice",
                verbatim_cfg_args,
                output_formats,
                status_poster,
                worker_cfg.status_include_utterance,
                worker_cfg.status_watchdog_seconds,
            )

        async def _worker_loop(slot: int) -> None:
            while True:
                handled = await worker.run_once(
                    handler=_handle_job,
                    types=types,
                    max_wait_seconds=worker_cfg.max_wait_seconds,
                )
                if handled and worker_cfg.once:
                    logging.info("worker slot %s handled one job; exiting due to --once", slot)
                    return

        await asyncio.gather(*(_worker_loop(slot) for slot in range(effective_concurrency)))
    finally:
        for task in locals().get("warmup_tasks", []):
            if not task.done():
                task.cancel()
        if status_poster:
            await status_poster.stop()
        await worker.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NFRX ASR transcribe worker")
    parser.add_argument("--config", default=os.environ.get("TRANSCRIBE_CONFIG", "config.yaml"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--client-key", default=None)
    parser.add_argument("--types", default=None, help="comma-separated job types")
    parser.add_argument("--max-wait-seconds", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None, help="number of concurrent job workers")
    parser.add_argument("--once", action="store_true", default=None, help="exit after handling one job")
    parser.add_argument("--model", default=None, help="ASR model id, size, or local path")
    parser.add_argument("--device", default=None, help="inference device: auto|cpu|cuda|mps")
    parser.add_argument("--compute-type", default=None, help="unused (kept for compatibility)")
    parser.add_argument("--backend", default=None, help="backend: auto|verbatim|mlx-vibevoice")
    parser.add_argument("--strategy", default=None, help="transcription strategy: verbatim|fast")
    parser.add_argument("--language", default=None)
    parser.add_argument("--beam-size", default=None, type=int)
    parser.add_argument("--best-of", default=None, type=int)
    parser.add_argument("--patience", default=None, type=float)
    parser.add_argument("--mlx-max-tokens", type=int, default=None)
    parser.add_argument("--mlx-chunk-seconds", type=float, default=None)
    parser.add_argument("--mlx-chunk-overlap-seconds", type=float, default=None)
    parser.add_argument("--mlx-context-characters", type=int, default=None)
    parser.add_argument("--diarization", action="store_true", default=None)
    parser.add_argument("--no-diarization", dest="diarization", action="store_false")
    parser.add_argument("--diarization-strategy", default=None)
    parser.add_argument("--diarization-speakers", type=int, default=None)
    parser.add_argument("--huggingface-token", default=None)
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--status-progress", action="store_true", default=None)
    parser.add_argument("--no-status-progress", dest="status_progress", action="store_false")
    parser.add_argument("--status-interval-seconds", type=float, default=None)
    parser.add_argument("--status-min-progress-seconds", type=float, default=None)
    parser.add_argument("--status-include-utterance", action="store_true", default=None)
    parser.add_argument("--status-watchdog-seconds", type=float, default=None)
    return parser


def main() -> int:
    load_dotenv(override=True)
    parser = build_parser()
    args = parser.parse_args()
    if isinstance(args.language, str) and not args.language.strip():
        args.language = None
    logging.basicConfig(
        level=getattr(logging, str(args.log_level or os.environ.get("LOG_LEVEL", "INFO")).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return asyncio.run(run(args))
    except httpx.HTTPError as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
