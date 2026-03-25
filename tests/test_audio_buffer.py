import unittest
from types import SimpleNamespace

import numpy as np

from transcribe.audio import (
    AudioBuffer,
    AudioDecodeError,
    _mono_from_pyav_frame_array,
    audio_buffer_from_any,
    normalize_audio_buffer,
    slice_audio_buffer,
)


class AudioBufferTests(unittest.TestCase):
    def test_audio_buffer_duration_seconds(self) -> None:
        audio = AudioBuffer(samples=np.zeros(24000, dtype=np.float32), sample_rate=24000)
        self.assertEqual(audio.duration_seconds, 1.0)

    def test_slice_audio_buffer_uses_sample_boundaries(self) -> None:
        base = AudioBuffer(samples=np.arange(20, dtype=np.float32), sample_rate=10)
        sliced = slice_audio_buffer(base, 0.5, 1.4)
        np.testing.assert_array_equal(sliced.samples, np.arange(5, 14, dtype=np.float32))
        self.assertEqual(sliced.sample_rate, 10)

    def test_audio_buffer_from_any_requires_sample_rate(self) -> None:
        with self.assertRaises(ValueError):
            audio_buffer_from_any(np.arange(10, dtype=np.float32))

    def test_audio_buffer_from_any_rejects_empty_audio(self) -> None:
        with self.assertRaises(AudioDecodeError):
            audio_buffer_from_any(np.array([], dtype=np.float32), sample_rate=24000)

    def test_slice_audio_buffer_rejects_empty_slice(self) -> None:
        base = AudioBuffer(samples=np.arange(20, dtype=np.float32), sample_rate=10)
        with self.assertRaises(AudioDecodeError):
            slice_audio_buffer(base, 3.0, 3.0)

    def test_normalize_audio_buffer_rejects_empty_payload(self) -> None:
        with self.assertRaises(AudioDecodeError):
            normalize_audio_buffer(b"", target_sample_rate=24000)

    def test_mono_from_pyav_frame_array_handles_planar_stereo(self) -> None:
        frame = np.array(
            [
                [1.0, 3.0, 5.0],
                [3.0, 5.0, 7.0],
            ],
            dtype=np.float32,
        )
        sample_format = SimpleNamespace(is_planar=True)
        mono = _mono_from_pyav_frame_array(frame, channels=2, sample_format=sample_format)
        np.testing.assert_array_equal(mono, np.array([2.0, 4.0, 6.0], dtype=np.float32))

    def test_mono_from_pyav_frame_array_handles_packed_stereo(self) -> None:
        frame = np.array(
            [
                [1.0, 3.0],
                [3.0, 5.0],
                [5.0, 7.0],
            ],
            dtype=np.float32,
        )
        sample_format = SimpleNamespace(is_planar=False)
        mono = _mono_from_pyav_frame_array(frame, channels=2, sample_format=sample_format)
        np.testing.assert_array_equal(mono, np.array([2.0, 4.0, 6.0], dtype=np.float32))

    def test_mono_from_pyav_frame_array_handles_single_plane_packed_stereo(self) -> None:
        frame = np.array([[1.0, 3.0, 3.0, 5.0, 5.0, 7.0]], dtype=np.float32)
        sample_format = SimpleNamespace(is_planar=False)
        mono = _mono_from_pyav_frame_array(frame, channels=2, sample_format=sample_format)
        np.testing.assert_array_equal(mono, np.array([2.0, 4.0, 6.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
