"""Tests for audio cleanup pipeline (dead air trimming, hold music detection, speech stitching)."""

import pytest
import numpy as np
import soundfile as sf
from services.audio.cleanup import (
    _merge_close_segments,
    _stitch_segments,
    _remove_hold_music_from_segments,
    _is_hold_music,
    _basic_energy_vad,
    cleanup_audio,
)


class TestMergeCloseSegments:
    def test_no_merge_large_gaps(self):
        timestamps = [
            {"start": 0.0, "end": 1.0},
            {"start": 5.0, "end": 6.0},
        ]
        result = _merge_close_segments(timestamps, 2.0)
        assert len(result) == 2

    def test_merge_small_gaps(self):
        timestamps = [
            {"start": 0.0, "end": 1.0},
            {"start": 1.5, "end": 2.5},
        ]
        result = _merge_close_segments(timestamps, 2.0)
        assert len(result) == 1
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 2.5

    def test_merge_chain(self):
        timestamps = [
            {"start": 0.0, "end": 1.0},
            {"start": 1.5, "end": 2.0},
            {"start": 2.3, "end": 3.0},
        ]
        result = _merge_close_segments(timestamps, 2.0)
        assert len(result) == 1
        assert result[0]["end"] == 3.0

    def test_empty_input(self):
        assert _merge_close_segments([], 2.0) == []

    def test_single_segment(self):
        result = _merge_close_segments([{"start": 1.0, "end": 2.0}], 2.0)
        assert len(result) == 1


class TestStitchSegments:
    def test_single_segment(self):
        audio = np.ones(16000, dtype=np.float32)
        segments = [{"start": 0.0, "end": 1.0}]
        result = _stitch_segments(audio, 16000, segments, 0.2)
        assert len(result) == 16000  # No padding for single segment

    def test_two_segments_with_padding(self):
        audio = np.ones(48000, dtype=np.float32)  # 3 seconds
        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 2.0, "end": 3.0},
        ]
        result = _stitch_segments(audio, 16000, segments, 0.2)
        # 1s + 0.2s padding + 1s = 2.2s = 35200 samples
        assert len(result) == 35200

    def test_empty_segments(self):
        audio = np.ones(16000, dtype=np.float32)
        result = _stitch_segments(audio, 16000, [], 0.2)
        assert len(result) == 0

    def test_preserves_audio_content(self):
        audio = np.arange(32000, dtype=np.float32)
        segments = [{"start": 0.0, "end": 0.5}]
        result = _stitch_segments(audio, 16000, segments, 0.2)
        np.testing.assert_array_equal(result, audio[:8000])


class TestRemoveHoldMusic:
    def test_no_hold_music(self):
        segments = [{"start": 0.0, "end": 5.0}]
        result = _remove_hold_music_from_segments(segments, [])
        assert len(result) == 1
        assert result[0]["start"] == 0.0

    def test_trim_overlap(self):
        segments = [{"start": 0.0, "end": 10.0}]
        hold_music = [(3.0, 6.0)]
        result = _remove_hold_music_from_segments(segments, hold_music)
        assert len(result) == 2
        assert result[0]["end"] == 3.0
        assert result[1]["start"] == 6.0

    def test_no_overlap(self):
        segments = [{"start": 0.0, "end": 2.0}]
        hold_music = [(5.0, 8.0)]
        result = _remove_hold_music_from_segments(segments, hold_music)
        assert len(result) == 1


class TestBasicEnergyVad:
    def test_silence_returns_empty(self):
        silence = np.zeros(16000, dtype=np.float32)
        result = _basic_energy_vad(silence, 16000)
        assert result == []

    def test_speech_detected(self):
        # Create audio with speech-like energy in middle
        audio = np.zeros(48000, dtype=np.float32)
        audio[16000:32000] = np.random.randn(16000).astype(np.float32) * 0.5
        result = _basic_energy_vad(audio, 16000)
        assert len(result) >= 1


class TestIsHoldMusic:
    def test_silence_is_not_music(self):
        silence = np.zeros(32000, dtype=np.float32)
        assert _is_hold_music(silence, 16000) is False

    def test_returns_bool(self):
        tone = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 48000)).astype(np.float32)
        result = _is_hold_music(tone, 16000)
        assert isinstance(result, bool)


class TestCleanupAudioIntegration:
    def test_fallback_on_read_failure(self):
        result = cleanup_audio("/nonexistent/file.wav", "/tmp/out.wav")
        assert result["cleanup_applied"] is False
        assert result["output_path"] == "/nonexistent/file.wav"

    def test_no_cleanup_needed(self):
        """Audio that is mostly speech should not be cleaned."""
        import tempfile, os
        audio = np.random.randn(80000).astype(np.float32) * 0.3
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, 16000, subtype="PCM_16")
            tmp_path = f.name
        try:
            result = cleanup_audio(
                tmp_path,
                tmp_path.replace(".wav", "_clean.wav"),
                speech_timestamps=[{"start": 0.0, "end": 5.0}],
            )
            assert result["cleanup_applied"] is False
        finally:
            os.unlink(tmp_path)
            if os.path.exists(tmp_path.replace(".wav", "_clean.wav")):
                os.unlink(tmp_path.replace(".wav", "_clean.wav"))

    def test_dead_air_removed(self):
        """Audio with large silence gap should have dead air removed."""
        import tempfile, os
        sr = 16000
        # 2s speech + 5s silence + 2s speech = 9s total
        speech1 = np.random.randn(2 * sr).astype(np.float32) * 0.3
        silence = np.zeros(5 * sr, dtype=np.float32)
        speech2 = np.random.randn(2 * sr).astype(np.float32) * 0.3
        audio = np.concatenate([speech1, silence, speech2])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr, subtype="PCM_16")
            tmp_path = f.name

        out_path = tmp_path.replace(".wav", "_clean.wav")
        try:
            result = cleanup_audio(
                tmp_path,
                out_path,
                speech_timestamps=[
                    {"start": 0.0, "end": 2.0},
                    {"start": 7.0, "end": 9.0},
                ],
            )
            assert result["cleanup_applied"] is True
            assert result["cleaned_duration"] < result["original_duration"]
            assert result["segments_stitched"] == 2
        finally:
            os.unlink(tmp_path)
            if os.path.exists(out_path):
                os.unlink(out_path)
