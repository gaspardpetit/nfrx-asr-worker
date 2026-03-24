import unittest

import numpy as np

from transcribe.audio import AudioBuffer
from transcribe.speaker_embeddings import (
    IncrementalSpeakerResolver,
    SpeakerEmbeddingConfig,
    _normalize_local_speaker_label,
)


class FakeExtractor:
    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        mean_value = float(np.mean(samples))
        if mean_value >= 0:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)


class FailingInitExtractor:
    def initialize(self) -> None:
        raise RuntimeError("boom")

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        raise AssertionError("embed should not be called after init failure")


class RecordingExtractor:
    def __init__(self) -> None:
        self.initialized = 0

    def initialize(self) -> None:
        self.initialized += 1

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class SpeakerEmbeddingTests(unittest.TestCase):
    def test_chunk_one_assigns_stable_global_speakers_immediately(self) -> None:
        audio = AudioBuffer(samples=np.ones(24000 * 8, dtype=np.float32), sample_rate=24000)
        resolver = IncrementalSpeakerResolver(
            audio=audio,
            config=SpeakerEmbeddingConfig(min_utterance_seconds=0.5, seed_seconds=1.0, max_workers=1),
            extractor=FakeExtractor(),
        )
        try:
            first = resolver.label_for_emission(
                resolver.annotate_segment({"speaker": "0", "start": 0.0, "end": 2.0, "text": "Alpha"}, chunk_index=1)
            )
            second = resolver.label_for_emission(
                resolver.annotate_segment({"speaker": "1", "start": 2.0, "end": 4.0, "text": "Beta"}, chunk_index=1)
            )
            again = resolver.label_for_emission(
                resolver.annotate_segment({"speaker": "0", "start": 4.0, "end": 5.0, "text": "Alpha again"}, chunk_index=1)
            )
            self.assertEqual(first["speaker"], "SPEAKER_00")
            self.assertEqual(second["speaker"], "SPEAKER_01")
            self.assertEqual(again["speaker"], "SPEAKER_00")
        finally:
            resolver.close()

    def test_chunk_two_can_match_existing_global_speaker_after_embedding_update(self) -> None:
        samples = np.concatenate(
            [
                np.ones(24000 * 3, dtype=np.float32),
                np.ones(24000 * 3, dtype=np.float32),
                -np.ones(24000 * 3, dtype=np.float32),
            ]
        )
        audio = AudioBuffer(samples=samples, sample_rate=24000)
        resolver = IncrementalSpeakerResolver(
            audio=audio,
            config=SpeakerEmbeddingConfig(
                min_utterance_seconds=0.5,
                seed_seconds=1.0,
                similarity_threshold=0.7,
                max_workers=1,
            ),
            extractor=FakeExtractor(),
        )
        try:
            chunk_one_segment = resolver.label_for_emission(
                resolver.annotate_segment({"speaker": "0", "start": 0.0, "end": 3.0, "text": "Seed speaker"}, chunk_index=1)
            )
            resolver.observe_emitted_segment(chunk_one_segment)
            resolver.wait()

            chunk_two_segment = resolver.annotate_segment(
                {"speaker": "7", "start": 3.0, "end": 6.0, "text": "Same person later"},
                chunk_index=2,
            )
            provisional = resolver.label_for_emission(chunk_two_segment)
            self.assertEqual(provisional["speaker"], "chunk2:7")

            resolver.observe_emitted_segment(provisional)
            relabeled = resolver.relabel_segments([chunk_two_segment])
            self.assertEqual(relabeled[0]["speaker"], "SPEAKER_00")
        finally:
            resolver.close()

    def test_chunk_two_creates_new_global_speaker_when_embedding_differs(self) -> None:
        samples = np.concatenate(
            [
                np.ones(24000 * 3, dtype=np.float32),
                -np.ones(24000 * 3, dtype=np.float32),
            ]
        )
        audio = AudioBuffer(samples=samples, sample_rate=24000)
        resolver = IncrementalSpeakerResolver(
            audio=audio,
            config=SpeakerEmbeddingConfig(
                min_utterance_seconds=0.5,
                seed_seconds=1.0,
                similarity_threshold=0.9,
                max_workers=1,
            ),
            extractor=FakeExtractor(),
        )
        try:
            seed = resolver.label_for_emission(
                resolver.annotate_segment({"speaker": "0", "start": 0.0, "end": 3.0, "text": "Seed"}, chunk_index=1)
            )
            resolver.observe_emitted_segment(seed)
            resolver.wait()

            new_speaker = resolver.annotate_segment(
                {"speaker": "9", "start": 3.0, "end": 6.0, "text": "Different voice"},
                chunk_index=2,
            )
            resolver.observe_emitted_segment(resolver.label_for_emission(new_speaker))
            relabeled = resolver.relabel_segments([new_speaker])
            self.assertEqual(relabeled[0]["speaker"], "SPEAKER_01")
        finally:
            resolver.close()

    def test_relabel_segments_preserves_internal_unresolved_label(self) -> None:
        audio = AudioBuffer(samples=np.ones(24000 * 4, dtype=np.float32), sample_rate=24000)
        resolver = IncrementalSpeakerResolver(
            audio=audio,
            config=SpeakerEmbeddingConfig(min_utterance_seconds=10.0, seed_seconds=10.0, max_workers=1),
            extractor=FakeExtractor(),
        )
        try:
            unresolved = resolver.annotate_segment(
                {"speaker": "5", "start": 0.0, "end": 1.0, "text": "Too short to embed"},
                chunk_index=2,
            )
            relabeled = resolver.relabel_segments([unresolved])
            self.assertEqual(relabeled[0]["speaker"], "chunk2:5")
        finally:
            resolver.close()

    def test_null_speaker_labels_are_preserved_and_not_embedded(self) -> None:
        audio = AudioBuffer(samples=np.ones(24000 * 4, dtype=np.float32), sample_rate=24000)
        resolver = IncrementalSpeakerResolver(
            audio=audio,
            config=SpeakerEmbeddingConfig(min_utterance_seconds=0.5, seed_seconds=1.0, max_workers=1),
            extractor=FakeExtractor(),
        )
        try:
            annotated = resolver.annotate_segment(
                {"speaker": None, "start": 0.0, "end": 2.0, "text": "[Noise]"},
                chunk_index=1,
            )
            emitted = resolver.label_for_emission(annotated)
            self.assertIsNone(emitted["speaker"])
            resolver.observe_emitted_segment(emitted)
            relabeled = resolver.relabel_segments([annotated])
            self.assertIsNone(relabeled[0]["speaker"])
            self.assertEqual(resolver.profiles_snapshot(), {})
        finally:
            resolver.close()

    def test_embedding_init_failure_disables_embeddings_without_raising(self) -> None:
        audio = AudioBuffer(samples=np.ones(24000 * 4, dtype=np.float32), sample_rate=24000)
        resolver = IncrementalSpeakerResolver(
            audio=audio,
            config=SpeakerEmbeddingConfig(min_utterance_seconds=0.5, seed_seconds=1.0, max_workers=1),
            extractor=FailingInitExtractor(),
        )
        try:
            first = resolver.annotate_segment({"speaker": "0", "start": 0.0, "end": 2.0, "text": "Alpha"}, chunk_index=1)
            emitted = resolver.label_for_emission(first)
            self.assertEqual(emitted["speaker"], "SPEAKER_00")
            resolver.observe_emitted_segment(emitted)
            relabeled = resolver.relabel_segments([first])
            self.assertEqual(relabeled[0]["speaker"], "SPEAKER_00")
        finally:
            resolver.close()

    def test_unknown_like_labels_normalize_to_none(self) -> None:
        self.assertIsNone(_normalize_local_speaker_label("UNKNOWN"))
        self.assertIsNone(_normalize_local_speaker_label("unknown"))
        self.assertIsNone(_normalize_local_speaker_label(" Noise "))
        self.assertIsNone(_normalize_local_speaker_label("[silence]"))
        self.assertEqual(_normalize_local_speaker_label("0"), "0")

    def test_disabled_embeddings_skip_embedder_initialization(self) -> None:
        audio = AudioBuffer(samples=np.ones(24000 * 4, dtype=np.float32), sample_rate=24000)
        extractor = RecordingExtractor()
        resolver = IncrementalSpeakerResolver(
            audio=audio,
            config=SpeakerEmbeddingConfig(min_utterance_seconds=0.5, seed_seconds=1.0, max_workers=1),
            extractor=extractor,
            enabled=False,
        )
        try:
            segment = resolver.annotate_segment({"speaker": "0", "start": 0.0, "end": 2.0, "text": "Alpha"}, chunk_index=1)
            emitted = resolver.label_for_emission(segment)
            self.assertEqual(emitted["speaker"], "SPEAKER_00")
            resolver.observe_emitted_segment(emitted)
            relabeled = resolver.relabel_segments([segment])
            self.assertEqual(relabeled[0]["speaker"], "SPEAKER_00")
            self.assertEqual(extractor.initialized, 0)
        finally:
            resolver.close()

    def test_chunk_one_oversegmentation_can_merge_to_single_global_speaker(self) -> None:
        samples = np.concatenate(
            [
                np.ones(24000 * 3, dtype=np.float32),
                np.ones(24000 * 3, dtype=np.float32),
            ]
        )
        audio = AudioBuffer(samples=samples, sample_rate=24000)
        resolver = IncrementalSpeakerResolver(
            audio=audio,
            config=SpeakerEmbeddingConfig(
                min_utterance_seconds=0.5,
                seed_seconds=1.0,
                merge_seed_seconds=1.0,
                merge_similarity_threshold=0.95,
                merge_margin=0.1,
                max_workers=1,
            ),
            extractor=FakeExtractor(),
        )
        try:
            first = resolver.annotate_segment({"speaker": "0", "start": 0.0, "end": 3.0, "text": "Alpha"}, chunk_index=1)
            second = resolver.annotate_segment({"speaker": "1", "start": 3.0, "end": 6.0, "text": "Alpha split"}, chunk_index=1)
            self.assertEqual(resolver.label_for_emission(first)["speaker"], "SPEAKER_00")
            self.assertEqual(resolver.label_for_emission(second)["speaker"], "SPEAKER_01")
            resolver.observe_emitted_segment(resolver.label_for_emission(first))
            resolver.observe_emitted_segment(resolver.label_for_emission(second))
            relabeled = resolver.relabel_segments([first, second])
            self.assertEqual(relabeled[0]["speaker"], "SPEAKER_00")
            self.assertEqual(relabeled[1]["speaker"], "SPEAKER_00")
            self.assertEqual(list(resolver.profiles_snapshot().keys()), ["SPEAKER_00"])
        finally:
            resolver.close()

    def test_true_different_speakers_do_not_merge(self) -> None:
        samples = np.concatenate(
            [
                np.ones(24000 * 3, dtype=np.float32),
                -np.ones(24000 * 3, dtype=np.float32),
            ]
        )
        audio = AudioBuffer(samples=samples, sample_rate=24000)
        resolver = IncrementalSpeakerResolver(
            audio=audio,
            config=SpeakerEmbeddingConfig(
                min_utterance_seconds=0.5,
                seed_seconds=1.0,
                merge_seed_seconds=1.0,
                merge_similarity_threshold=0.8,
                merge_margin=0.05,
                max_workers=1,
            ),
            extractor=FakeExtractor(),
        )
        try:
            first = resolver.annotate_segment({"speaker": "0", "start": 0.0, "end": 3.0, "text": "Alpha"}, chunk_index=1)
            second = resolver.annotate_segment({"speaker": "1", "start": 3.0, "end": 6.0, "text": "Beta"}, chunk_index=1)
            resolver.observe_emitted_segment(resolver.label_for_emission(first))
            resolver.observe_emitted_segment(resolver.label_for_emission(second))
            relabeled = resolver.relabel_segments([first, second])
            self.assertEqual(relabeled[0]["speaker"], "SPEAKER_00")
            self.assertEqual(relabeled[1]["speaker"], "SPEAKER_01")
            self.assertEqual(sorted(resolver.profiles_snapshot().keys()), ["SPEAKER_00", "SPEAKER_01"])
        finally:
            resolver.close()

    def test_canonical_assignment_stays_stable_after_merge(self) -> None:
        samples = np.concatenate(
            [
                np.ones(24000 * 3, dtype=np.float32),
                np.ones(24000 * 3, dtype=np.float32),
                np.ones(24000 * 3, dtype=np.float32),
            ]
        )
        audio = AudioBuffer(samples=samples, sample_rate=24000)
        resolver = IncrementalSpeakerResolver(
            audio=audio,
            config=SpeakerEmbeddingConfig(
                min_utterance_seconds=0.5,
                seed_seconds=1.0,
                similarity_threshold=0.7,
                merge_seed_seconds=1.0,
                merge_similarity_threshold=0.95,
                merge_margin=0.1,
                max_workers=1,
            ),
            extractor=FakeExtractor(),
        )
        try:
            first = resolver.annotate_segment({"speaker": "0", "start": 0.0, "end": 3.0, "text": "Alpha"}, chunk_index=1)
            second = resolver.annotate_segment({"speaker": "1", "start": 3.0, "end": 6.0, "text": "Alpha split"}, chunk_index=1)
            resolver.observe_emitted_segment(resolver.label_for_emission(first))
            resolver.observe_emitted_segment(resolver.label_for_emission(second))
            resolver.wait()

            later = resolver.annotate_segment({"speaker": "7", "start": 6.0, "end": 9.0, "text": "Alpha later"}, chunk_index=2)
            resolver.observe_emitted_segment(resolver.label_for_emission(later))
            relabeled = resolver.relabel_segments([later])
            self.assertEqual(relabeled[0]["speaker"], "SPEAKER_00")
        finally:
            resolver.close()


if __name__ == "__main__":
    unittest.main()
