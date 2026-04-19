import unittest
from transcribe.worker import (
    ChunkResolvedEmitter,
    _align_overlap_segments,
    _format_jsonl_output,
    _public_segment_speaker,
    _levenshtein_similarity,
    _merge_chunk_segments,
    _public_speaker_label,
    _resolve_overlap_window,
)


class ChunkAlignmentTests(unittest.TestCase):
    def test_levenshtein_similarity_handles_minor_text_differences(self) -> None:
        score = _levenshtein_similarity(
            "Thank you for your attention. We wish you a pleasant flight.",
            "thank you for your attention we wish you a very pleasant flight",
        )
        self.assertGreater(score, 0.75)

    def test_align_overlap_segments_matches_monotonic_pairs(self) -> None:
        previous = [
            {"start": 1140.0, "end": 1145.0, "text": "Alice introduces the quarterly numbers."},
            {"start": 1145.0, "end": 1152.0, "text": "Bob asks about revenue guidance."},
        ]
        current = [
            {"start": 1140.3, "end": 1145.4, "text": "Alice introduces quarterly numbers."},
            {"start": 1145.2, "end": 1152.3, "text": "Bob asks about the revenue guidance."},
        ]
        matches = _align_overlap_segments(previous, current)
        self.assertEqual([(prev_idx, curr_idx) for prev_idx, curr_idx, _score in matches], [(0, 0), (1, 1)])

    def test_align_overlap_segments_does_not_cross_match_same_text_when_resolved_speakers_differ(self) -> None:
        previous = [
            {"start": 1140.0, "end": 1142.0, "text": "Yes.", "speaker": "SPEAKER_00"},
            {"start": 1142.0, "end": 1144.0, "text": "Yes.", "speaker": "SPEAKER_01"},
        ]
        current = [
            {"start": 1140.1, "end": 1142.1, "text": "Yes.", "speaker": "SPEAKER_01"},
            {"start": 1142.1, "end": 1144.1, "text": "Yes.", "speaker": "SPEAKER_00"},
        ]
        matches = _align_overlap_segments(previous, current)
        self.assertFalse(any(prev_idx == 0 and curr_idx == 0 for prev_idx, curr_idx, _score in matches))
        self.assertFalse(any(prev_idx == 1 and curr_idx == 1 for prev_idx, curr_idx, _score in matches))
        for prev_idx, curr_idx, _score in matches:
            self.assertEqual(previous[prev_idx]["speaker"], current[curr_idx]["speaker"])

    def test_merge_chunk_segments_drops_duplicate_current_overlap(self) -> None:
        previous = [
            {"start": 1130.0, "end": 1140.0, "text": "Lead in before the overlap.", "speaker": "0"},
            {"start": 1140.0, "end": 1148.0, "text": "Alice introduces the quarterly numbers.", "speaker": "0"},
            {"start": 1148.0, "end": 1160.0, "text": "Bob asks about revenue guidance.", "speaker": "1"},
        ]
        current = [
            {"start": 1140.2, "end": 1148.1, "text": "Alice introduces quarterly numbers.", "speaker": "0"},
            {"start": 1148.1, "end": 1160.2, "text": "Bob asks about the revenue guidance.", "speaker": "1"},
            {"start": 1160.2, "end": 1168.0, "text": "Alice answers with a cautious outlook.", "speaker": "0"},
        ]
        merged = _merge_chunk_segments(previous, current, overlap_start=1140.0, overlap_end=1200.0)
        self.assertEqual(len(merged), 4)
        self.assertEqual(merged[-1]["text"], "Alice answers with a cautious outlook.")
        self.assertEqual(
            [segment["text"] for segment in merged[:3]],
            [
                "Lead in before the overlap.",
                "Alice introduces the quarterly numbers.",
                "Bob asks about revenue guidance.",
            ],
        )

    def test_merge_chunk_segments_can_prefer_better_current_boundary_segment(self) -> None:
        previous = [
            {"start": 1140.0, "end": 1145.0, "text": "Alice introduces the quar", "speaker": "0"},
            {"start": 1145.0, "end": 1152.0, "text": "Bob asks about revenue guidance.", "speaker": "1"},
        ]
        current = [
            {"start": 1140.0, "end": 1145.2, "text": "Alice introduces the quarterly numbers.", "speaker": "0"},
            {"start": 1145.2, "end": 1152.1, "text": "Bob asks about revenue guidance.", "speaker": "1"},
            {"start": 1152.1, "end": 1160.0, "text": "Alice clarifies the operating margin.", "speaker": "0"},
        ]
        merged = _merge_chunk_segments(previous, current, overlap_start=1140.0, overlap_end=1200.0)
        self.assertEqual(
            [segment["text"] for segment in merged],
            [
                "Alice introduces the quarterly numbers.",
                "Bob asks about revenue guidance.",
                "Alice clarifies the operating margin.",
            ],
        )

    def test_resolve_overlap_window_prefers_previous_in_first_half_and_current_in_second_half(self) -> None:
        previous_overlap = [
            {"start": 1140.0, "end": 1146.0, "text": "Alice introduces the quarterly num", "speaker": "0"},
            {"start": 1170.0, "end": 1178.0, "text": "Bob asks about guidance in broad terms.", "speaker": "1"},
        ]
        current_overlap = [
            {"start": 1140.1, "end": 1146.1, "text": "Alice introduces the quarterly numbers.", "speaker": "0"},
            {"start": 1170.1, "end": 1178.2, "text": "Bob asks about detailed revenue guidance.", "speaker": "1"},
        ]
        resolved = _resolve_overlap_window(previous_overlap, current_overlap, 1140.0, 1200.0)
        self.assertEqual(
            [segment["text"] for segment in resolved],
            [
                "Alice introduces the quarterly num",
                "Bob asks about detailed revenue guidance.",
            ],
        )

    def test_resolve_overlap_window_handles_empty_previous_overlap(self) -> None:
        current_overlap = [
            {"start": 1145.0, "end": 1150.0, "text": "Only current has speech here.", "speaker": "0"},
        ]
        resolved = _resolve_overlap_window([], current_overlap, 1140.0, 1200.0)
        self.assertEqual(resolved, current_overlap)

    def test_resolve_overlap_window_keeps_unmatched_segments_from_both_sides(self) -> None:
        previous_overlap = [
            {"start": 1172.0, "end": 1178.0, "text": "Board approves the budget.", "speaker": "0"},
        ]
        current_overlap = [
            {"start": 1142.0, "end": 1144.0, "text": "Please state your name.", "speaker": "0"},
        ]
        resolved = _resolve_overlap_window(previous_overlap, current_overlap, 1140.0, 1200.0)
        self.assertEqual(
            [segment["text"] for segment in resolved],
            ["Please state your name.", "Board approves the budget."],
        )

    def test_merge_chunk_segments_preserves_speaker_identity_for_repeated_short_phrases(self) -> None:
        previous = [
            {"start": 1140.0, "end": 1142.0, "text": "Okay.", "speaker": "SPEAKER_00"},
            {"start": 1142.0, "end": 1144.0, "text": "Okay.", "speaker": "SPEAKER_01"},
        ]
        current = [
            {"start": 1140.1, "end": 1142.1, "text": "Okay.", "speaker": "SPEAKER_00"},
            {"start": 1142.1, "end": 1144.1, "text": "Okay.", "speaker": "SPEAKER_01"},
            {"start": 1144.1, "end": 1146.0, "text": "Moving on.", "speaker": "SPEAKER_00"},
        ]
        merged = _merge_chunk_segments(previous, current, overlap_start=1140.0, overlap_end=1145.0)
        self.assertEqual(
            [(segment["text"], segment["speaker"]) for segment in merged],
            [
                ("Okay.", "SPEAKER_00"),
                ("Okay.", "SPEAKER_01"),
                ("Moving on.", "SPEAKER_00"),
            ],
        )

    def test_unresolved_speakers_can_still_align_by_text_and_time(self) -> None:
        previous = [
            {"start": 1140.0, "end": 1145.0, "text": "Only current has speech here.", "speaker": "chunk1:0"},
        ]
        current = [
            {"start": 1140.1, "end": 1145.1, "text": "Only current has speech here.", "speaker": "chunk2:7"},
        ]
        matches = _align_overlap_segments(previous, current)
        self.assertEqual([(prev_idx, curr_idx) for prev_idx, curr_idx, _score in matches], [(0, 0)])

    def test_chunk_resolved_emitter_delays_overlap_until_next_chunk(self) -> None:
        emitted = []

        def hook(payload):
            emitted.append(payload["utterance"]["text"])

        emitter = ChunkResolvedEmitter(overlap_seconds=60.0, finish=240.0, status_hook=hook)
        chunk_one = [
            {"start": 0.0, "end": 20.0, "text": "Opening section.", "speaker": "0"},
            {"start": 70.0, "end": 110.0, "text": "Tail overlap from previous chunk.", "speaker": "0"},
        ]
        emitter.process_chunk(chunk_one, chunk_start=0.0, chunk_end=120.0, is_last=False)
        self.assertEqual(emitted, ["Opening section."])

        chunk_two = [
            {"start": 60.0, "end": 90.0, "text": "Tail overlap from previous chunk.", "speaker": "0"},
            {"start": 130.0, "end": 150.0, "text": "Chunk two body.", "speaker": "1"},
        ]
        emitter.process_chunk(chunk_two, chunk_start=60.0, chunk_end=180.0, is_last=True)
        self.assertEqual(
            emitted,
            [
                "Opening section.",
                "Tail overlap from previous chunk.",
                "Chunk two body.",
            ],
        )

    def test_chunk_resolved_emitter_with_zero_overlap_applies_transformer_before_status(self) -> None:
        emitted = []
        observed = []

        def hook(payload):
            emitted.append(payload["utterance"]["speaker"])

        def transform(segment):
            updated = dict(segment)
            updated["speaker"] = "SPEAKER_03"
            return updated

        def observe(segment):
            observed.append(segment["speaker"])

        emitter = ChunkResolvedEmitter(
            overlap_seconds=0.0,
            finish=10.0,
            status_hook=hook,
            segment_transformer=transform,
            segment_observer=observe,
        )
        emitter.process_chunk(
            [{"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "chunk1:0", "_chunk_index": 1, "_local_speaker": "0"}],
            chunk_start=0.0,
            chunk_end=1.0,
            is_last=True,
        )
        self.assertEqual(emitted, ["3"])
        self.assertEqual(observed, ["SPEAKER_03"])

    def test_chunk_resolved_emitter_first_chunk_does_not_double_emit_non_overlap_body(self) -> None:
        emitted = []

        def hook(payload):
            emitted.append(payload["utterance"]["text"])

        emitter = ChunkResolvedEmitter(overlap_seconds=60.0, finish=1515.0, status_hook=hook)
        chunk_one = [
            {"start": 0.0, "end": 20.0, "text": "Opening section.", "speaker": "0"},
            {"start": 62.03, "end": 80.0, "text": "Body one.", "speaker": "0"},
            {"start": 200.0, "end": 240.0, "text": "Body two.", "speaker": "1"},
            {"start": 1128.78, "end": 1200.0, "text": "Tail overlap.", "speaker": "0"},
        ]

        emitter.process_chunk(chunk_one, chunk_start=0.0, chunk_end=1200.0, is_last=False)

        self.assertEqual(
            emitted,
            [
                "Opening section.",
                "Body one.",
                "Body two.",
            ],
        )

    def test_chunk_resolved_emitter_last_chunk_does_not_double_emit_after_transform(self) -> None:
        emitted = []

        def hook(payload):
            emitted.append(payload["utterance"]["text"])

        def transform(segment):
            updated = dict(segment)
            updated["speaker"] = "SPEAKER_03"
            return updated

        emitter = ChunkResolvedEmitter(
            overlap_seconds=60.0,
            finish=180.0,
            status_hook=hook,
            segment_transformer=transform,
        )
        emitter.process_chunk(
            [{"start": 0.0, "end": 10.0, "text": "A", "speaker": "chunk1:0", "_chunk_index": 1, "_local_speaker": "0"}],
            chunk_start=0.0,
            chunk_end=120.0,
            is_last=False,
        )
        emitter.process_chunk(
            [
                {"start": 60.0, "end": 90.0, "text": "B", "speaker": "chunk2:0", "_chunk_index": 2, "_local_speaker": "0"},
                {"start": 130.0, "end": 150.0, "text": "C", "speaker": "chunk2:0", "_chunk_index": 2, "_local_speaker": "0"},
            ],
            chunk_start=60.0,
            chunk_end=180.0,
            is_last=True,
        )
        self.assertEqual(emitted, ["A", "B", "C"])

    def test_public_speaker_label_maps_internal_labels_to_public_shape(self) -> None:
        self.assertEqual(_public_speaker_label("SPEAKER_00"), "0")
        self.assertEqual(_public_speaker_label("1"), "1")
        self.assertIsNone(_public_speaker_label("UNKNOWN"))
        self.assertIsNone(_public_speaker_label("chunk2:7"))

    def test_jsonl_output_uses_public_speaker_labels(self) -> None:
        payload = _format_jsonl_output(
            [
                {"speaker": "SPEAKER_02", "start": 0.0, "end": 1.0, "text": "Hello"},
                {"speaker": "chunk2:7", "_chunk_index": 2, "_local_speaker": "7", "start": 1.0, "end": 2.0, "text": "World"},
            ]
        ).decode("utf-8")
        self.assertIn('"speaker": "2"', payload)
        self.assertIn('"speaker": null', payload)

    def test_public_segment_speaker_hides_unresolved_chunk_local_labels(self) -> None:
        self.assertIsNone(_public_segment_speaker({"speaker": "chunk2:0", "_chunk_index": 2, "_local_speaker": "0"}))
        self.assertEqual(_public_segment_speaker({"speaker": "SPEAKER_03", "_chunk_index": 2, "_local_speaker": "0"}), "3")

if __name__ == "__main__":
    unittest.main()
