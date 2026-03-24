from __future__ import annotations

import concurrent.futures
import inspect
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np

from transcribe.audio import AudioBuffer, AudioDecodeError, slice_audio_buffer


class SpeakerEmbeddingError(RuntimeError):
    pass


class SpeakerEmbeddingExtractor(Protocol):
    def embed(self, samples: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        ...


@dataclass
class SpeakerEmbeddingConfig:
    sample_rate: int = 16000
    min_utterance_seconds: float = 1.5
    seed_seconds: float = 6.0
    similarity_threshold: float = 0.72
    merge_seed_seconds: float = 8.0
    merge_similarity_threshold: float = 0.9
    merge_margin: float = 0.08
    max_workers: int = 2


def _normalize_local_speaker_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    label = str(value).strip()
    if not label:
        return None
    normalized = label.lower()
    if normalized in {"unknown", "unk", "noise", "[noise]", "silence", "[silence]", "null", "none"}:
        return None
    return label


@dataclass
class SpeakerProfile:
    speaker_id: str
    centroid: Optional[np.ndarray] = None
    total_seconds: float = 0.0
    embedding_count: int = 0
    exemplars: list[np.ndarray] = field(default_factory=list)

    def is_ready(self, seed_seconds: float) -> bool:
        return self.centroid is not None and self.total_seconds >= seed_seconds

    def update(self, embedding: np.ndarray, duration_seconds: float) -> None:
        duration = max(0.0, float(duration_seconds))
        vector = _normalize_embedding(embedding)
        if self.centroid is None or self.total_seconds <= 0:
            self.centroid = vector
            self.total_seconds = duration
            self.embedding_count = 1
            self.exemplars = [vector]
            return

        weight_existing = self.total_seconds
        weight_new = max(duration, 0.25)
        merged = ((self.centroid * weight_existing) + (vector * weight_new)) / (weight_existing + weight_new)
        self.centroid = _normalize_embedding(merged)
        self.total_seconds += duration
        self.embedding_count += 1
        self.exemplars.append(vector)
        if len(self.exemplars) > 8:
            self.exemplars = self.exemplars[-8:]


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise SpeakerEmbeddingError("speaker embedding vector is empty or non-finite")
    return vector / norm


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vec = _normalize_embedding(left)
    right_vec = _normalize_embedding(right)
    return float(np.dot(left_vec, right_vec))


class SpeechBrainSpeakerEmbedder:
    def __init__(self, *, target_sample_rate: int = 16000) -> None:
        self._target_sample_rate = target_sample_rate
        self._classifier: Any = None
        self._lock = threading.Lock()
        self._init_error: Optional[Exception] = None

    @staticmethod
    def _ensure_hf_hub_compat() -> None:
        try:
            import huggingface_hub
        except ImportError as exc:
            raise SpeakerEmbeddingError("speaker embeddings require huggingface_hub") from exc

        hf_hub_download = getattr(huggingface_hub, "hf_hub_download", None)
        if not callable(hf_hub_download):
            raise SpeakerEmbeddingError("huggingface_hub.hf_hub_download is unavailable")
        try:
            signature = inspect.signature(hf_hub_download)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and "use_auth_token" in signature.parameters:
            return
        if getattr(hf_hub_download, "_nfrx_compat_wrapped", False):
            return

        def compat_hf_hub_download(*args: Any, **kwargs: Any) -> Any:
            if "use_auth_token" in kwargs and "token" not in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            else:
                kwargs.pop("use_auth_token", None)
            try:
                return hf_hub_download(*args, **kwargs)
            except Exception as exc:
                filename = kwargs.get("filename")
                if filename == "custom.py" and exc.__class__.__name__ == "RemoteEntryNotFoundError":
                    raise ValueError("File not found on HF hub") from exc
                raise

        setattr(compat_hf_hub_download, "_nfrx_compat_wrapped", True)
        huggingface_hub.hf_hub_download = compat_hf_hub_download

    def _get_classifier(self) -> Any:
        with self._lock:
            if self._classifier is not None:
                return self._classifier
            if self._init_error is not None:
                raise SpeakerEmbeddingError(f"speaker embedder initialization previously failed: {self._init_error}") from self._init_error
            try:
                self._ensure_hf_hub_compat()
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError as exc:
                self._init_error = exc
                raise SpeakerEmbeddingError(
                    "speaker embeddings require speechbrain. Install project dependencies before running the worker."
                ) from exc
            try:
                self._classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
            except Exception as exc:
                self._init_error = exc
                raise SpeakerEmbeddingError(f"failed to initialize SpeechBrain speaker embedder: {exc}") from exc
            return self._classifier

    def initialize(self) -> None:
        self._get_classifier()

    def embed(self, samples: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return None
        if sample_rate != self._target_sample_rate:
            try:
                from scipy.signal import resample_poly
            except ImportError as exc:
                raise SpeakerEmbeddingError("speaker embeddings require scipy for resampling") from exc
            audio = resample_poly(audio, self._target_sample_rate, sample_rate).astype(np.float32, copy=False)
        if audio.size == 0:
            return None
        try:
            import torch
        except ImportError as exc:
            raise SpeakerEmbeddingError("speaker embeddings require torch") from exc

        classifier = self._get_classifier()
        waveform = torch.from_numpy(audio).unsqueeze(0)
        wav_lens = torch.tensor([1.0], dtype=torch.float32)
        try:
            embedding = classifier.encode_batch(waveform, wav_lens=wav_lens)
        except TypeError:
            embedding = classifier.encode_batch(waveform, wav_lens)
        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().numpy()
        return _normalize_embedding(np.asarray(embedding, dtype=np.float32))


_DEFAULT_SPEAKER_EMBEDDER: Optional[SpeechBrainSpeakerEmbedder] = None
_DEFAULT_SPEAKER_EMBEDDER_LOCK = threading.Lock()


def get_default_speaker_embedder(*, target_sample_rate: int = 16000) -> SpeechBrainSpeakerEmbedder:
    global _DEFAULT_SPEAKER_EMBEDDER
    with _DEFAULT_SPEAKER_EMBEDDER_LOCK:
        if _DEFAULT_SPEAKER_EMBEDDER is None:
            _DEFAULT_SPEAKER_EMBEDDER = SpeechBrainSpeakerEmbedder(target_sample_rate=target_sample_rate)
        return _DEFAULT_SPEAKER_EMBEDDER


def warmup_default_speaker_embedder(*, target_sample_rate: int = 16000) -> None:
    get_default_speaker_embedder(target_sample_rate=target_sample_rate).initialize()


class IncrementalSpeakerResolver:
    def __init__(
        self,
        *,
        audio: AudioBuffer,
        config: Optional[SpeakerEmbeddingConfig] = None,
        extractor: Optional[SpeakerEmbeddingExtractor] = None,
        enabled: bool = True,
    ) -> None:
        self._audio = audio
        self._config = config or SpeakerEmbeddingConfig()
        self._extractor = extractor or get_default_speaker_embedder(target_sample_rate=self._config.sample_rate)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, self._config.max_workers),
            thread_name_prefix="speaker-embed",
        )
        self._lock = threading.Lock()
        self._profiles: dict[str, SpeakerProfile] = {}
        self._chunk_assignments: dict[tuple[int, Optional[str]], str] = {}
        self._pending: list[concurrent.futures.Future[None]] = []
        self._observed_segments: set[tuple[int, Optional[str], float, float]] = set()
        self._speaker_counter = 0
        self._canonical_ids: dict[str, str] = {}
        self._embedding_enabled = bool(enabled)
        self._embedder_initialized = False

    def _initialize_embedder(self) -> None:
        if not self._embedding_enabled or self._embedder_initialized:
            return
        initialize = getattr(self._extractor, "initialize", None)
        if not callable(initialize):
            self._embedder_initialized = True
            return
        try:
            initialize()
            self._embedder_initialized = True
        except Exception as exc:
            self._embedding_enabled = False
            logging.exception("speaker embedding initialization failed; continuing without speaker embeddings")
            logging.info("speaker embeddings disabled: %s", exc)

    def close(self) -> None:
        self.wait()
        self._executor.shutdown(wait=True)

    def wait(self) -> None:
        pending = list(self._pending)
        for future in pending:
            try:
                future.result()
            except Exception:
                self._embedding_enabled = False
                logging.exception("speaker embedding update failed; continuing without speaker embeddings")
        self._pending.clear()

    def annotate_segment(self, segment: dict[str, Any], *, chunk_index: int) -> dict[str, Any]:
        annotated = dict(segment)
        local_speaker = _normalize_local_speaker_label(annotated.get("speaker"))
        annotated["_chunk_index"] = int(chunk_index)
        annotated["_local_speaker"] = local_speaker
        return annotated

    def label_for_emission(self, segment: dict[str, Any]) -> dict[str, Any]:
        labeled = dict(segment)
        chunk_index = int(labeled.get("_chunk_index", 1))
        local_speaker = _normalize_local_speaker_label(labeled.get("_local_speaker"))
        if local_speaker is None:
            labeled["speaker"] = None
            return labeled
        global_speaker = self._current_assignment(chunk_index, local_speaker)
        # Chunk 1 local speaker labels are already file-scoped enough to expose as
        # stable public IDs, even when embeddings are disabled for performance.
        if global_speaker is None and chunk_index == 1:
            with self._lock:
                global_speaker = self._assign_new_global_locked(chunk_index, local_speaker)
        if global_speaker is None:
            global_speaker = f"chunk{chunk_index}:{local_speaker}"
        labeled["speaker"] = global_speaker
        return labeled

    def observe_emitted_segment(self, segment: dict[str, Any]) -> None:
        chunk_index = int(segment.get("_chunk_index", 1))
        local_speaker = _normalize_local_speaker_label(segment.get("_local_speaker"))
        if local_speaker is None:
            return
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        duration = max(0.0, end - start)
        text = str(segment.get("text") or "").strip()
        if duration < self._config.min_utterance_seconds:
            return
        if not text or text.lower() in {"[silence]", "[noise]"}:
            return
        if not self._embedding_enabled:
            return
        self._initialize_embedder()
        if not self._embedding_enabled:
            return
        dedupe_key = (chunk_index, local_speaker, round(start, 2), round(end, 2))
        with self._lock:
            if dedupe_key in self._observed_segments:
                return
            self._observed_segments.add(dedupe_key)
        try:
            audio_slice = slice_audio_buffer(self._audio, start, end)
        except AudioDecodeError:
            logging.debug("Skipping speaker embedding for empty audio slice %.2f-%.2f", start, end)
            return
        future = self._executor.submit(self._embed_and_update, chunk_index, local_speaker, audio_slice, duration)
        self._pending.append(future)

    def relabel_segments(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.wait()
        relabeled: list[dict[str, Any]] = []
        for segment in segments:
            updated = dict(segment)
            chunk_index = int(updated.get("_chunk_index", 1))
            local_speaker = _normalize_local_speaker_label(updated.get("_local_speaker"))
            if local_speaker is None:
                updated["speaker"] = None
                relabeled.append(updated)
                continue
            global_speaker = self._current_assignment(chunk_index, local_speaker)
            if global_speaker is None:
                updated["speaker"] = f"chunk{chunk_index}:{local_speaker}"
            else:
                updated["speaker"] = global_speaker
            relabeled.append(updated)
        return relabeled

    def profiles_snapshot(self) -> dict[str, SpeakerProfile]:
        with self._lock:
            return {speaker_id: SpeakerProfile(**{
                "speaker_id": profile.speaker_id,
                "centroid": None if profile.centroid is None else np.array(profile.centroid, copy=True),
                "total_seconds": profile.total_seconds,
                "embedding_count": profile.embedding_count,
                "exemplars": [np.array(item, copy=True) for item in profile.exemplars],
            }) for speaker_id, profile in self._profiles.items()}

    def _embed_and_update(self, chunk_index: int, local_speaker: str, audio_slice: AudioBuffer, duration: float) -> None:
        embedding = self._extractor.embed(audio_slice.samples, audio_slice.sample_rate)
        if embedding is None:
            return
        with self._lock:
            global_speaker = self._resolve_canonical_locked(self._chunk_assignments.get((chunk_index, local_speaker)))
            if global_speaker is None:
                global_speaker, best_match, best_score = self._match_or_create_global_locked(embedding)
                self._chunk_assignments[(chunk_index, local_speaker)] = global_speaker
                if best_match is None:
                    logging.info(
                        "speaker embedding chunk=%s local=%s created new global speaker=%s (no prior ready match)",
                        chunk_index,
                        local_speaker,
                        global_speaker,
                    )
                else:
                    logging.info(
                        "speaker embedding chunk=%s local=%s assigned global=%s best_match=%s similarity=%.4f threshold=%.4f",
                        chunk_index,
                        local_speaker,
                        global_speaker,
                        best_match,
                        best_score,
                        self._config.similarity_threshold,
                    )
            profile = self._profiles[self._resolve_canonical_locked(global_speaker)]
            prior_centroid = None if profile.centroid is None else np.array(profile.centroid, copy=True)
            prior_total_seconds = profile.total_seconds
            profile.update(np.asarray(embedding, dtype=np.float32), duration)
            global_speaker = self._maybe_merge_global_locked(profile.speaker_id)
            profile = self._profiles[global_speaker]
            alignment = None if prior_centroid is None else _cosine_similarity(prior_centroid, embedding)
            competitor_scores: list[tuple[str, float]] = []
            for speaker_id, other_profile in self._profiles.items():
                if speaker_id == global_speaker or other_profile.centroid is None:
                    continue
                competitor_scores.append((speaker_id, _cosine_similarity(other_profile.centroid, profile.centroid)))
            competitor_scores.sort(key=lambda item: item[1], reverse=True)
            top_competitors = ", ".join(
                f"{speaker_id}={score:.4f}" for speaker_id, score in competitor_scores[:3]
            ) or "none"
            logging.info(
                "speaker embedding update chunk=%s local=%s global=%s duration=%.2fs prior_seconds=%.2fs "
                "alignment_to_prior=%s centroid_neighbors=%s",
                chunk_index,
                local_speaker,
                global_speaker,
                duration,
                prior_total_seconds,
                "n/a" if alignment is None else f"{alignment:.4f}",
                top_competitors,
            )

    def _current_assignment(self, chunk_index: int, local_speaker: str) -> Optional[str]:
        with self._lock:
            return self._resolve_canonical_locked(self._chunk_assignments.get((chunk_index, local_speaker)))

    def _match_or_create_global_locked(self, embedding: np.ndarray) -> tuple[str, Optional[str], float]:
        best_speaker: Optional[str] = None
        best_score = -1.0
        for speaker_id, profile in self._profiles.items():
            if not profile.is_ready(self._config.seed_seconds):
                continue
            if profile.centroid is None:
                continue
            score = _cosine_similarity(profile.centroid, embedding)
            if score > best_score:
                best_score = score
                best_speaker = speaker_id
        if best_speaker is not None and best_score >= self._config.similarity_threshold:
            return best_speaker, best_speaker, best_score
        return self._assign_new_global_locked(None, None), best_speaker, best_score

    def _resolve_canonical_locked(self, speaker_id: Optional[str]) -> Optional[str]:
        if speaker_id is None:
            return None
        resolved = speaker_id
        while resolved in self._canonical_ids:
            next_resolved = self._canonical_ids[resolved]
            if next_resolved == resolved:
                break
            resolved = next_resolved
        return resolved

    def _canonical_sort_key(self, speaker_id: str) -> tuple[int, str]:
        suffix = speaker_id.split("_")[-1]
        try:
            return (int(suffix), speaker_id)
        except ValueError:
            return (10**9, speaker_id)

    def _merge_profiles_locked(self, winner_id: str, loser_id: str) -> str:
        winner_id = self._resolve_canonical_locked(winner_id) or winner_id
        loser_id = self._resolve_canonical_locked(loser_id) or loser_id
        if winner_id == loser_id:
            return winner_id
        winner = self._profiles[winner_id]
        loser = self._profiles[loser_id]
        if loser.centroid is not None:
            winner.update(loser.centroid, max(loser.total_seconds, 0.25))
            winner.embedding_count += max(0, loser.embedding_count - 1)
            winner.exemplars.extend(loser.exemplars)
            if len(winner.exemplars) > 8:
                winner.exemplars = winner.exemplars[-8:]
        self._canonical_ids[loser_id] = winner_id
        for assignment_key, assigned in list(self._chunk_assignments.items()):
            if assigned == loser_id or self._resolve_canonical_locked(assigned) == loser_id:
                self._chunk_assignments[assignment_key] = winner_id
        self._profiles.pop(loser_id, None)
        return winner_id

    def _maybe_merge_global_locked(self, speaker_id: str) -> str:
        canonical_id = self._resolve_canonical_locked(speaker_id) or speaker_id
        profile = self._profiles.get(canonical_id)
        if profile is None or not profile.is_ready(self._config.merge_seed_seconds) or profile.centroid is None:
            return canonical_id

        best_other: Optional[str] = None
        best_score = -1.0
        second_best = -1.0
        for other_id, other_profile in self._profiles.items():
            if other_id == canonical_id or other_profile.centroid is None:
                continue
            if not other_profile.is_ready(self._config.merge_seed_seconds):
                continue
            score = _cosine_similarity(profile.centroid, other_profile.centroid)
            if score > best_score:
                second_best = best_score
                best_score = score
                best_other = other_id
            elif score > second_best:
                second_best = score

        if best_other is None:
            return canonical_id
        margin = best_score - max(second_best, -1.0)
        if best_score < self._config.merge_similarity_threshold or margin < self._config.merge_margin:
            return canonical_id

        winner_id, loser_id = sorted([canonical_id, best_other], key=self._canonical_sort_key)
        merged = self._merge_profiles_locked(winner_id, loser_id)
        logging.info(
            "speaker embedding merge winner=%s loser=%s similarity=%.4f margin=%.4f",
            merged,
            loser_id,
            best_score,
            margin,
        )
        return merged

    def _assign_new_global_locked(self, chunk_index: Optional[int], local_speaker: Optional[str]) -> str:
        speaker_id = f"SPEAKER_{self._speaker_counter:02d}"
        self._speaker_counter += 1
        self._profiles[speaker_id] = SpeakerProfile(speaker_id=speaker_id)
        if chunk_index is not None and local_speaker is not None:
            self._chunk_assignments[(chunk_index, local_speaker)] = speaker_id
        return speaker_id
