from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np


class AudioDecodeError(RuntimeError):
    pass


@dataclass
class AudioBuffer:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(self.samples.shape[-1]) / float(self.sample_rate)


def _validate_audio_array(audio_array: np.ndarray, *, source: str) -> np.ndarray:
    if audio_array.size == 0:
        raise AudioDecodeError(f"{source} produced an empty audio buffer")
    if audio_array.ndim == 1:
        mono = audio_array
    else:
        if audio_array.shape[1] == 0:
            raise AudioDecodeError(f"{source} produced audio with zero channels")
        mono = audio_array.mean(axis=1)
    if mono.size == 0:
        raise AudioDecodeError(f"{source} produced an empty mono audio buffer")
    if not np.isfinite(mono).any():
        raise AudioDecodeError(f"{source} produced only non-finite samples")
    return np.ascontiguousarray(mono, dtype=np.float32)


def _decode_audio_bytes_pyav(payload: bytes, *, sample_rate: int) -> np.ndarray:
    import av
    from scipy.signal import resample_poly

    if not payload:
        raise AudioDecodeError("input payload is empty")
    try:
        container = av.open(io.BytesIO(payload))
    except Exception as exc:
        raise AudioDecodeError(f"PyAV could not open the audio payload: {exc}") from exc

    try:
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            raise AudioDecodeError("PyAV found no audio stream in payload")
        stream = audio_streams[0]
        source_rate = int(stream.codec_context.sample_rate or 0)
        if source_rate <= 0:
            raise AudioDecodeError(f"PyAV reported invalid sample rate: {source_rate!r}")

        collected: list[np.ndarray] = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            frame_array = np.asarray(arr, dtype=np.float32)
            if frame_array.ndim == 1:
                frame_array = frame_array[np.newaxis, :]
            elif frame_array.ndim != 2:
                raise AudioDecodeError(f"Unexpected decoded frame shape from PyAV: {frame_array.shape}")
            if frame_array.size == 0 or frame_array.shape[1] == 0:
                continue
            if frame_array.shape[0] > 1:
                frame_array = frame_array.mean(axis=0, keepdims=True)

            mono_frame = np.squeeze(frame_array, axis=0)
            if source_rate != sample_rate:
                mono_frame = resample_poly(mono_frame, sample_rate, source_rate).astype(np.float32, copy=False)
            collected.append(np.asarray(mono_frame, dtype=np.float32))

        if not collected:
            raise AudioDecodeError("PyAV decode produced no audio frames")

        audio = np.concatenate(collected, axis=0)
        return _validate_audio_array(audio, source="PyAV decode")
    except AudioDecodeError:
        raise
    except Exception as exc:
        raise AudioDecodeError(f"PyAV decode failed: {exc}") from exc
    finally:
        try:
            container.close()
        except Exception:
            pass


def decode_audio_bytes(payload: bytes) -> AudioBuffer:
    raise AudioDecodeError("decode_audio_bytes() now requires a target sample rate; use normalize_audio_buffer() instead")


def normalize_audio_buffer(
    payload: bytes,
    *,
    target_sample_rate: int,
) -> AudioBuffer:
    if target_sample_rate <= 0:
        raise AudioDecodeError(f"invalid target sample rate: {target_sample_rate}")
    samples = _decode_audio_bytes_pyav(payload, sample_rate=target_sample_rate)
    return AudioBuffer(samples=samples, sample_rate=target_sample_rate)


def slice_audio_buffer(audio: AudioBuffer, start_seconds: float, end_seconds: float) -> AudioBuffer:
    start_idx = max(0, int(start_seconds * audio.sample_rate))
    end_idx = min(audio.samples.shape[-1], int(end_seconds * audio.sample_rate))
    if end_idx < start_idx:
        end_idx = start_idx
    sliced = np.ascontiguousarray(audio.samples[start_idx:end_idx], dtype=np.float32)
    if sliced.size == 0:
        raise AudioDecodeError(
            f"audio slice is empty for start={start_seconds:.3f}s end={end_seconds:.3f}s sample_rate={audio.sample_rate}"
        )
    return AudioBuffer(samples=sliced, sample_rate=audio.sample_rate)


def audio_buffer_from_any(audio: Any, sample_rate: int | None = None) -> AudioBuffer:
    if isinstance(audio, AudioBuffer):
        return audio
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.squeeze()
    if samples.ndim != 1:
        raise ValueError("Expected mono 1-D audio samples")
    if sample_rate is None:
        raise ValueError("sample_rate is required when constructing AudioBuffer from raw samples")
    return AudioBuffer(samples=_validate_audio_array(samples, source="raw audio"), sample_rate=int(sample_rate))
