#!/usr/bin/env python3
"""ASR transcribe worker for NFRX using Verbatim (transcription + diarization)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, TypedDict

import httpx
from dotenv import load_dotenv
import torch
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
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torchaudio


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


@dataclass
class DiarizationConfig:
    enabled: bool = True
    strategy: str = "pyannote"
    speakers: Optional[int] = None
    huggingface_token: Optional[str] = None


@dataclass
class TranscriptionConfig:
    model: str = "large-v3"
    device: str = "auto"
    compute_type: str = "int8"
    language: Optional[list[str]] = None
    beam_size: Optional[int] = None
    best_of: Optional[int] = None
    patience: Optional[float] = None
    output_format: str = "json"  # json|text


@dataclass
class WorkerConfig:
    base_url: str = DEFAULT_BASE_URL
    types: Optional[str] = "asr.transcribe"
    max_wait_seconds: int = 30
    once: bool = False
    log_level: str = "INFO"
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

    def enqueue(self, job_id: str, payload: StatusPayload) -> None:
        if self._closed or job_id in self._completed_jobs:
            return
        self._queue.put_nowait({"job_id": job_id, "payload": payload})

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
                    self.enqueue(job_id, payload)
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
            loop.call_soon_threadsafe(self.enqueue, job_id, payload)

        return hook

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            job_id = item.get("job_id")
            payload = item.get("payload") or {}
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
                        continue
                    self._last_progress[job_id] = current
                    self._last_progress_time[job_id] = now
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
    data = load_config_file(path)
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
    return "json"


def _status_update_to_payload(update: Any, include_utterance: bool) -> Optional[StatusPayload]:
    if update is None:
        return None
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
                "speaker": getattr(utterance, "speaker", None),
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
) -> Tuple[argparse.Namespace, Optional[list[str]], Optional[list[str]], Optional[TranscriptWriterConfig]]:
    if not job_cfg:
        return base_args, None, None, None

    args = argparse.Namespace(**vars(base_args))
    languages = _normalize_languages(job_cfg.get("languages"))
    if languages is not None:
        args.languages = languages

    diarize = _normalize_optional_str(job_cfg.get("diarize"))
    if diarize is not None:
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
    ts_style = _coerce_enum(fmt_cfg.get("timestamp"), TimestampStyle)
    if ts_style is not None:
        args.format_timestamp = ts_style
    sp_style = _coerce_enum(fmt_cfg.get("speaker"), SpeakerStyle)
    if sp_style is not None:
        args.format_speaker = sp_style
    prob_style = _coerce_enum(fmt_cfg.get("probability"), ProbabilityStyle)
    if prob_style is not None:
        args.format_probability = prob_style
    lang_style = _coerce_enum(fmt_cfg.get("language"), LanguageStyle)
    if lang_style is not None:
        args.format_language = lang_style

    write_config = verbatim_configure.make_write_config(args, logging.getLogger().level)
    output_formats = (
        verbatim_configure.build_output_formats(args, default_stdout=False)
        if output_files is not None
        else None
    )
    return args, output_files, output_formats, write_config


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


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

def _format_text_output(utterances: list[Any], write_config: TranscriptWriterConfig) -> str:
    writer = TextIOTranscriptWriter(
        config=write_config,
        acknowledged_colours=COLORSCHEME_NONE,
        unacknowledged_colours=COLORSCHEME_NONE,
        unconfirmed_colors=COLORSCHEME_NONE,
    )
    chunks: list[bytes] = [writer.format_start()]
    for utt in utterances:
        chunks.append(writer.format_utterance(utt))
    chunks.append(writer.format_end())
    return b"".join(chunks).decode("utf-8").strip()


def _format_docx_output(utterances: list[Any], write_config: TranscriptWriterConfig) -> bytes:
    writer = DocxTranscriptWriter(config=write_config)
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
        speaker = seg.get("speaker") or "SPEAKER"
        text = (seg.get("text") or "").strip()
        if text:
            lines.append(f"[{speaker}] {text}")
    return "\n".join(lines).strip()


class VerbatimService:
    def __init__(
        self,
        config: TranscriptionConfig,
        diarization: DiarizationConfig,
        source_config: SourceConfig,
        write_config: TranscriptWriterConfig,
    ) -> None:
        self._base_config = config
        self._diarization = diarization
        self._source_config = source_config
        self._write_config = write_config
        self._models = VerbatimModels(
            device=config.device,
            whisper_model_size=config.model,
            stream=False,
            transcriber=None,
        )

    def _build_config(
        self, language: Optional[list[str]], input_label: str
    ) -> Tuple[VerbatimConfig, str, str]:
        cfg = VerbatimConfig(
            device=self._base_config.device,
            whisper_model_size=self._base_config.model,
            stream=False,
            working_dir=None,
        )
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

        transcriber = Verbatim(cfg, models=self._models, status_hook=status_hook)
        utterances: list[Any] = []
        for audio_source in create_audio_sources(
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
        source_config: Optional[SourceConfig] = None,
        status_hook: Optional[Callable[[Any], None]] = None,
    ) -> Tuple[str, list[Dict[str, Any]], list[Any]]:
        cfg, output_prefix, working_prefix = self._build_config(language, input_label)
        cfg.cache.set_bytes(input_label, payload)
        use_source_config = source_config or self._source_config

        transcriber = Verbatim(cfg, models=self._models, status_hook=status_hook)
        utterances: list[Any] = []
        for audio_source in create_audio_sources(
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
        self._pipeline: Optional[Pipeline] = None

    def _get_pipeline(self) -> Optional[Pipeline]:
        if not self._config.enabled:
            return None
        if self._config.strategy != "pyannote":
            logging.warning("Unsupported diarization strategy '%s'; diarization disabled.", self._config.strategy)
            return None
        token = (self._config.huggingface_token or os.environ.get("HUGGINGFACE_TOKEN") or "").strip()
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN is required for pyannote diarization.")
        if self._pipeline is None:
            self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
            self._pipeline.to(torch.device(self._device))
        return self._pipeline

    def diarize(self, path: str) -> list[Dict[str, Any]]:
        pipeline = self._get_pipeline()
        if pipeline is None:
            return []
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
        self._model = WhisperModel(config.model, device=config.device, compute_type="int8")

    def transcribe_file(
        self,
        path: str,
        language: Optional[list[str]],
        diarization_turns: list[Dict[str, Any]],
    ) -> Tuple[str, list[Dict[str, Any]], list[Utterance]]:
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


def _segments_to_utterances(segments: list[Dict[str, Any]], languages: Optional[list[str]]) -> list[Utterance]:
    lang = languages[0] if languages else "en"
    utterances: list[Utterance] = []
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
        words: list[Word] = []
        for w_idx, token in enumerate(tokens):
            w_start = start_ts + w_idx * step
            w_end = start_ts + (w_idx + 1) * step
            if w_idx == len(tokens) - 1:
                w_end = end_ts
            word_text = token if w_idx == 0 else f" {token}"
            words.append(Word(start_ts=w_start, end_ts=w_end, word=word_text, probability=1.0, lang=lang))
        utterances.append(Utterance.from_words(utterance_id=str(idx), words=words, speaker=seg.get("speaker")))
    return utterances


async def transcribe_handler(
    verbatim_service: VerbatimService,
    job: JobClaim,
    payload: bytes,
    language: Optional[list[str]],
    strategy: str,
    fast_service: FastWhisperService,
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

        job_cfg = metadata.get("config")
        if job_cfg is not None and not isinstance(job_cfg, dict):
            logging.warning("job config ignored; expected mapping but got %s", type(job_cfg).__name__)
            job_cfg = None

        job_args, output_files, output_formats, job_write_config = _apply_job_config(
            base_verbatim_args, job_cfg or {}
        )
        if output_formats is None:
            output_formats = base_output_formats

        output_format = _output_format_from_formats(output_formats)
        write_config = job_write_config or verbatim_service._write_config
        source_config = verbatim_service._source_config
        if job_cfg:
            speakers_resolved = verbatim_configure.resolve_speakers(job_args)
            source_config = verbatim_configure.make_source_config(job_args, speakers_resolved)

        suffix = _infer_suffix(filename, content_type)
        base = os.path.splitext(os.path.basename(filename or "payload"))[0] or "payload"
        input_label = f"{base}{suffix}"

        logging.info("starting transcription (input=%s)", input_label)
        text, segments, utterances = await asyncio.to_thread(
            verbatim_service.transcribe_bytes,
            payload,
            input_label,
            language,
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

        result = {"text": text, "segments": segments}
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

    verbatim_parser = verbatim_args.build_parser(include_input=False)
    verbatim_defaults = verbatim_parser.parse_args([])
    profile_overrides = select_profile(config_data, filename=None) if config_data else {}
    verbatim_user = argparse.Namespace(**vars(verbatim_defaults))
    if args.language:
        verbatim_user.languages = [args.language]
    if args.beam_size is not None:
        verbatim_user.nb_beams = args.beam_size
    if args.diarization_strategy:
        verbatim_user.diarize = args.diarization_strategy
    if args.diarization_speakers is not None:
        verbatim_user.speakers = args.diarization_speakers
    verbatim_cfg_args = merge_args(verbatim_defaults, profile_overrides, verbatim_user)

    worker_cfg = WorkerConfig(
        base_url=_resolve(args.base_url, _cfg_get(config_data, "worker", "base_url"), os.environ.get("NFRX_BASE_URL"), DEFAULT_BASE_URL),
        types=_resolve(args.types, _cfg_get(config_data, "worker", "types"), os.environ.get("NFRX_JOB_TYPES"), "asr.transcribe"),
        max_wait_seconds=_resolve_int(
            _resolve(args.max_wait_seconds, _cfg_get(config_data, "worker", "max_wait_seconds"), os.environ.get("NFRX_MAX_WAIT_SECONDS"), 30),
            30,
        ),
        once=_resolve_bool(args.once, _cfg_get(config_data, "worker", "once"), os.environ.get("NFRX_ONCE"), False),
        log_level=_resolve(args.log_level, _cfg_get(config_data, "worker", "log_level"), os.environ.get("LOG_LEVEL"), "INFO"),
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

    speakers_resolved = verbatim_configure.resolve_speakers(verbatim_cfg_args)
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

    output_formats = verbatim_configure.build_output_formats(verbatim_cfg_args)
    output_format = "json"
    if "txt" in output_formats or "stdout" in output_formats or "stdout-nocolor" in output_formats:
        output_format = "text"
    logging.info("Output formats: %s (selected output_format=%s)", output_formats, output_format)

    resolved_device = _resolve(
        args.device,
        _cfg_get_transcription(config_data, "device"),
        os.environ.get("WHISPER_DEVICE"),
        "cpu" if getattr(verbatim_cfg_args, "cpu", False) else "auto",
    )
    transcription_cfg = TranscriptionConfig(
        model=_resolve(args.model, _cfg_get_transcription(config_data, "model"), os.environ.get("WHISPER_MODEL"), "large-v3"),
        device=_resolve_device(resolved_device),
        language=_normalize_languages(
            _resolve(
                args.language,
                _cfg_get_transcription(config_data, "language") or getattr(verbatim_cfg_args, "languages", None),
                os.environ.get("WHISPER_LANGUAGE"),
                None,
            )
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
            output_format,
        ),
    )

    # If config explicitly sets output formats (e.g., [txt]), honor that over env overrides.
    if output_formats and "json" not in output_formats:
        transcription_cfg.output_format = "text"

    write_config = verbatim_configure.make_write_config(verbatim_cfg_args, logging.getLogger().level)
    source_config = verbatim_configure.make_source_config(verbatim_cfg_args, speakers_resolved)

    worker = NfrxJobsWorker(worker_cfg.base_url, auth)
    status_poster: Optional[NfrxStatusPoster] = None
    if worker_cfg.status_progress:
        status_poster = NfrxStatusPoster(
            worker,
            min_interval_seconds=worker_cfg.status_interval_seconds,
            min_progress_seconds=worker_cfg.status_min_progress_seconds,
        )
        status_poster.start(asyncio.get_running_loop())
    verbatim_service = VerbatimService(transcription_cfg, diarization_cfg, source_config, write_config)
    fast_service = FastWhisperService(transcription_cfg)
    diarization_service = DiarizationService(diarization_cfg, transcription_cfg.device)
    try:
        types = worker_cfg.types.split(",") if worker_cfg.types else None
        while True:
            handled = await worker.run_once(
                handler=lambda job, payload: transcribe_handler(
                    verbatim_service,
                    job,
                    payload,
                    transcription_cfg.language,
                    job.get("metadata", {}).get("strategy", worker_cfg.strategy),
                    fast_service,
                    diarization_service,
                    diarization_cfg.enabled and diarization_cfg.speakers not in (1, "1"),
                    verbatim_cfg_args,
                    output_formats,
                    status_poster,
                    worker_cfg.status_include_utterance,
                    worker_cfg.status_watchdog_seconds,
                ),
                types=types,
                max_wait_seconds=worker_cfg.max_wait_seconds,
            )
            if handled and worker_cfg.once:
                return 0
    finally:
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
    parser.add_argument("--once", action="store_true", default=None, help="exit after handling one job")
    parser.add_argument("--model", default=None, help="whisper model size or path")
    parser.add_argument("--device", default=None, help="whisper device: auto|cpu|cuda|mps")
    parser.add_argument("--compute-type", default=None, help="unused (kept for compatibility)")
    parser.add_argument("--strategy", default=None, help="transcription strategy: verbatim|fast")
    parser.add_argument("--language", default=None)
    parser.add_argument("--beam-size", default=None, type=int)
    parser.add_argument("--best-of", default=None, type=int)
    parser.add_argument("--patience", default=None, type=float)
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
