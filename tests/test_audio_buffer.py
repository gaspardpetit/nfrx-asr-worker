import unittest

import numpy as np

from transcribe.audio import AudioBuffer, AudioDecodeError, audio_buffer_from_any, normalize_audio_buffer, slice_audio_buffer


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


if __name__ == "__main__":
    unittest.main()
