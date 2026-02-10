"""Stage 2.5: Audio Cleanup — trim dead air, detect hold music, stitch speech.

Runs on CPU only. Uses speech_timestamps from quality.py to avoid recomputing VAD.
Input: 16kHz mono WAV from normalizer.
Output: cleaned WAV with dead air and hold music removed.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.warning("librosa not installed — hold music detection disabled")

SAMPLE_RATE = 16000
DEAD_AIR_THRESHOLD_SEC = 2.0
PADDING_SEC = 0.200  # 200ms padding between stitched segments
HOLD_MUSIC_MIN_DURATION_SEC = 3.0


def cleanup_audio(
    wav_path: str,
    output_path: str,
    speech_timestamps: list[dict] | None = None,
) -> dict:
    """Clean audio by removing dead air, hold music, and stitching speech segments.

    Args:
        wav_path: Path to normalized 16kHz mono WAV file
        output_path: Path for cleaned audio output
        speech_timestamps: Pre-computed speech segments from quality.py.
            Each dict has 'start' and 'end' (float, seconds).
            If None, a simple energy-based VAD is computed as fallback.

    Returns:
        dict with cleanup metadata (output_path, durations, flags)
    """
    try:
        audio, sr = sf.read(wav_path)
    except Exception as e:
        logger.error(f"Could not read audio for cleanup: {e}")
        return _fallback_result(wav_path, output_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    original_duration = len(audio) / sr

    # If no speech timestamps provided, compute basic VAD as fallback
    if speech_timestamps is None or len(speech_timestamps) == 0:
        logger.warning("No speech_timestamps provided — computing basic VAD for cleanup")
        speech_timestamps = _basic_energy_vad(audio, sr)

    if not speech_timestamps:
        logger.warning("No speech detected in audio — skipping cleanup")
        return _fallback_result(wav_path, output_path, original_duration=original_duration)

    # Step 1: Detect hold music in gaps between speech segments
    hold_music_ranges = _detect_hold_music(audio, sr, speech_timestamps)
    hold_music_duration = sum(end - start for start, end in hold_music_ranges)

    # Step 2: Merge speech segments separated by small gaps (< threshold)
    merged_segments = _merge_close_segments(speech_timestamps, DEAD_AIR_THRESHOLD_SEC)

    # Step 3: Remove hold music segments from consideration
    clean_segments = _remove_hold_music_from_segments(merged_segments, hold_music_ranges)

    # Step 4: Stitch speech segments with padding
    cleaned_audio = _stitch_segments(audio, sr, clean_segments, PADDING_SEC)

    if len(cleaned_audio) == 0:
        logger.warning("Cleanup produced empty audio — falling back to original")
        return _fallback_result(wav_path, output_path, original_duration=original_duration)

    cleaned_duration = len(cleaned_audio) / sr
    dead_air_removed = original_duration - cleaned_duration - hold_music_duration

    # Only write if we actually removed something meaningful (>1s)
    if original_duration - cleaned_duration < 1.0:
        logger.info("Cleanup: less than 1s to remove — using original audio")
        return _fallback_result(wav_path, output_path, original_duration=original_duration)

    # Write cleaned audio
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, cleaned_audio, sr, subtype="PCM_16")
    except Exception as e:
        logger.error(f"Failed to write cleaned audio: {e}")
        return _fallback_result(wav_path, output_path, original_duration=original_duration)

    result = {
        "output_path": output_path,
        "original_duration": round(original_duration, 2),
        "cleaned_duration": round(cleaned_duration, 2),
        "dead_air_removed_sec": round(max(0, dead_air_removed), 2),
        "hold_music_detected": len(hold_music_ranges) > 0,
        "hold_music_removed_sec": round(hold_music_duration, 2),
        "segments_stitched": len(clean_segments),
        "cleanup_applied": True,
    }

    logger.info(
        f"Audio cleanup: {original_duration:.1f}s → {cleaned_duration:.1f}s "
        f"(removed {original_duration - cleaned_duration:.1f}s: "
        f"dead_air={max(0, dead_air_removed):.1f}s, hold_music={hold_music_duration:.1f}s, "
        f"{len(clean_segments)} segments stitched)"
    )
    return result


def _fallback_result(
    wav_path: str,
    output_path: str,
    original_duration: float = 0.0,
) -> dict:
    """Return metadata indicating cleanup was not applied."""
    return {
        "output_path": wav_path,
        "original_duration": round(original_duration, 2),
        "cleaned_duration": round(original_duration, 2),
        "dead_air_removed_sec": 0.0,
        "hold_music_detected": False,
        "hold_music_removed_sec": 0.0,
        "segments_stitched": 0,
        "cleanup_applied": False,
    }


def _basic_energy_vad(audio: np.ndarray, sr: int) -> list[dict]:
    """Simple energy-based VAD as fallback when speech_timestamps not provided."""
    frame_length = int(0.025 * sr)  # 25ms
    hop_length = int(0.010 * sr)    # 10ms

    energies = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        energies.append(np.sum(frame ** 2) / frame_length)
    energies = np.array(energies)

    if len(energies) == 0:
        return []

    threshold = np.median(energies) * 1.5
    is_speech = energies > threshold

    timestamps = []
    in_speech = False
    start = 0.0
    for i, s in enumerate(is_speech):
        if s and not in_speech:
            start = i * hop_length / sr
            in_speech = True
        elif not s and in_speech:
            end = i * hop_length / sr
            timestamps.append({"start": start, "end": end})
            in_speech = False
    if in_speech:
        timestamps.append({"start": start, "end": len(audio) / sr})

    return timestamps


def _detect_hold_music(
    audio: np.ndarray,
    sr: int,
    speech_timestamps: list[dict],
) -> list[tuple[float, float]]:
    """Detect hold music in gaps between speech segments.

    Uses librosa spectral features: high centroid, low ZCR variance,
    consistent RMS energy — 2 of 3 must agree.
    """
    if not HAS_LIBROSA:
        return []

    # Find gaps between speech segments longer than threshold
    sorted_timestamps = sorted(speech_timestamps, key=lambda x: x["start"])
    gaps = []

    # Check beginning before first speech
    if sorted_timestamps and sorted_timestamps[0]["start"] >= HOLD_MUSIC_MIN_DURATION_SEC:
        gaps.append((0.0, sorted_timestamps[0]["start"]))

    # Check gaps between segments
    for i in range(len(sorted_timestamps) - 1):
        gap_start = sorted_timestamps[i]["end"]
        gap_end = sorted_timestamps[i + 1]["start"]
        if gap_end - gap_start >= HOLD_MUSIC_MIN_DURATION_SEC:
            gaps.append((gap_start, gap_end))

    # Check end after last speech
    if sorted_timestamps:
        total_dur = len(audio) / sr
        if total_dur - sorted_timestamps[-1]["end"] >= HOLD_MUSIC_MIN_DURATION_SEC:
            gaps.append((sorted_timestamps[-1]["end"], total_dur))

    hold_music_ranges = []
    for gap_start, gap_end in gaps:
        start_sample = int(gap_start * sr)
        end_sample = min(int(gap_end * sr), len(audio))
        chunk = audio[start_sample:end_sample]

        if len(chunk) < sr:  # Need at least 1 second
            continue

        if _is_hold_music(chunk, sr):
            hold_music_ranges.append((gap_start, gap_end))

    return hold_music_ranges


def _is_hold_music(audio_chunk: np.ndarray, sr: int) -> bool:
    """Determine if an audio chunk is hold music using spectral features.

    Three heuristics (any two must pass):
    1. Spectral centroid consistently high (>2000 Hz) — music is brighter
    2. Zero-crossing rate has low variance — periodic signals (music)
    3. RMS energy is consistent — music has steady volume
    """
    try:
        centroid = librosa.feature.spectral_centroid(y=audio_chunk, sr=sr)[0]
        mean_centroid = np.mean(centroid)

        zcr = librosa.feature.zero_crossing_rate(audio_chunk)[0]
        zcr_variance = np.var(zcr)

        rms = librosa.feature.rms(y=audio_chunk)[0]
        rms_mean = np.mean(rms)
        rms_cv = np.std(rms) / max(rms_mean, 1e-10)

        votes = 0
        if mean_centroid > 2000:
            votes += 1
        if zcr_variance < 0.005:
            votes += 1
        if rms_mean > 0.01 and rms_cv < 0.5:
            votes += 1

        return votes >= 2

    except Exception as e:
        logger.debug(f"Hold music detection failed for chunk: {e}")
        return False


def _merge_close_segments(
    speech_timestamps: list[dict],
    gap_threshold: float,
) -> list[dict]:
    """Merge speech segments separated by less than gap_threshold seconds."""
    if not speech_timestamps:
        return []

    sorted_ts = sorted(speech_timestamps, key=lambda x: x["start"])
    merged = [{"start": sorted_ts[0]["start"], "end": sorted_ts[0]["end"]}]

    for seg in sorted_ts[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        if gap < gap_threshold:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append({"start": seg["start"], "end": seg["end"]})

    return merged


def _remove_hold_music_from_segments(
    segments: list[dict],
    hold_music_ranges: list[tuple[float, float]],
) -> list[dict]:
    """Remove hold music time ranges from segment list, trimming overlaps."""
    if not hold_music_ranges:
        return segments

    clean = []
    for seg in segments:
        remaining = [(seg["start"], seg["end"])]
        for hm_start, hm_end in hold_music_ranges:
            next_remaining = []
            for r_start, r_end in remaining:
                if r_end <= hm_start or r_start >= hm_end:
                    next_remaining.append((r_start, r_end))
                else:
                    if r_start < hm_start:
                        next_remaining.append((r_start, hm_start))
                    if r_end > hm_end:
                        next_remaining.append((hm_end, r_end))
            remaining = next_remaining

        for r_start, r_end in remaining:
            if r_end - r_start > 0.1:
                clean.append({"start": r_start, "end": r_end})

    return clean


def _stitch_segments(
    audio: np.ndarray,
    sr: int,
    segments: list[dict],
    padding_sec: float,
) -> np.ndarray:
    """Concatenate speech segments with silence padding between them."""
    if not segments:
        return np.array([], dtype=audio.dtype)

    padding_samples = int(padding_sec * sr)
    padding = np.zeros(padding_samples, dtype=audio.dtype)

    chunks = []
    for i, seg in enumerate(segments):
        start_sample = max(0, int(seg["start"] * sr))
        end_sample = min(len(audio), int(seg["end"] * sr))
        chunk = audio[start_sample:end_sample]

        if len(chunk) > 0:
            chunks.append(chunk)
            if i < len(segments) - 1:
                chunks.append(padding)

    if not chunks:
        return np.array([], dtype=audio.dtype)

    return np.concatenate(chunks)
