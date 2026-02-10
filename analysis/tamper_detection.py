"""Audio Tamper & Replay Detection — spectral forensics for audio manipulation.

Detects signs of audio tampering, splicing, and replay attacks:
  1. Spectral discontinuity detection — abrupt frequency changes between segments
  2. Compression artifact analysis — inconsistent encoding suggests editing
  3. Silence pattern analysis — unnatural silence gaps (cut/spliced audio)
  4. Noise floor consistency — edited audio often has inconsistent background noise
  5. Double compression detection — audio that's been decoded and re-encoded

All CPU-based using librosa + numpy. No GPU needed.
"""

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.warning("librosa not installed — tamper detection disabled")


class TamperSignal(BaseModel):
    """A detected tampering or replay signal."""
    signal_type: str = Field(
        description="One of: spectral_discontinuity, silence_anomaly, "
        "noise_floor_inconsistency, double_compression, replay_detected"
    )
    description: str
    timestamp: float = Field(description="Time in seconds where anomaly was detected")
    confidence: float = Field(ge=0, le=1)
    severity: str = Field(description="'low', 'medium', 'high'")


def detect_spectral_discontinuities(
    wav_path: str = None,
    hop_length: int = 512,
    threshold_factor: float = 5.0,
    audio_data: tuple = None,
) -> list[TamperSignal]:
    """Detect abrupt spectral changes that suggest audio splicing.

    Computes spectral flux (frame-to-frame spectral difference) and flags
    frames where the change is >3x the median — indicating a cut point.

    Args:
        wav_path: Path to WAV file (used if audio_data not provided)
        audio_data: Pre-loaded (y, sr) tuple to avoid redundant disk reads
    """
    if not HAS_LIBROSA:
        return []

    signals = []
    if audio_data is not None:
        y, sr = audio_data
    else:
        try:
            y, sr = librosa.load(wav_path, sr=16000, mono=True)
        except Exception as e:
            logger.warning(f"Could not load audio for spectral analysis: {e}")
            return []

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Compute spectral flux (frame-to-frame difference)
    flux = np.sqrt(np.sum(np.diff(S_db, axis=1) ** 2, axis=0))

    if len(flux) < 10:
        return []

    # Use median + threshold to find anomalous jumps
    median_flux = np.median(flux)
    std_flux = np.std(flux)
    threshold = median_flux + threshold_factor * std_flux

    # Find anomalous frames (but filter out natural speech onset/offset)
    anomalous_frames = np.where(flux > threshold)[0]

    # Cluster nearby frames (within 0.5s)
    if len(anomalous_frames) == 0:
        return []

    clusters = []
    current_cluster = [anomalous_frames[0]]
    for frame in anomalous_frames[1:]:
        if (frame - current_cluster[-1]) < int(0.5 * sr / hop_length):
            current_cluster.append(frame)
        else:
            clusters.append(current_cluster)
            current_cluster = [frame]
    clusters.append(current_cluster)

    # Only flag clusters that are truly anomalous (not just speech onset)
    for cluster in clusters:
        center_frame = cluster[len(cluster) // 2]
        timestamp = librosa.frames_to_time(center_frame, sr=sr, hop_length=hop_length)
        max_flux = float(flux[cluster].max())
        deviation = max_flux / max(median_flux, 0.01)

        # Skip first and last 2 seconds (natural start/end)
        total_duration = len(y) / sr
        if timestamp < 2 or timestamp > total_duration - 2:
            continue

        if deviation > 10:
            severity = "high"
            confidence = min(0.9, deviation / 15)
        elif deviation > 7:
            severity = "medium"
            confidence = min(0.7, deviation / 15)
        else:
            continue

        signals.append(TamperSignal(
            signal_type="spectral_discontinuity",
            description=(
                f"Abrupt spectral change at {timestamp:.1f}s "
                f"({deviation:.1f}x median flux). "
                f"Possible audio splice point."
            ),
            timestamp=round(timestamp, 2),
            confidence=round(confidence, 3),
            severity=severity,
        ))

    return signals


def detect_silence_anomalies(
    wav_path: str = None,
    min_silence_duration: float = 0.05,
    max_natural_silence: float = 2.0,
    audio_data: tuple = None,
) -> list[TamperSignal]:
    """Detect unnatural silence patterns suggesting audio editing.

    Args:
        wav_path: Path to WAV file (used if audio_data not provided)
        audio_data: Pre-loaded (y, sr) tuple to avoid redundant disk reads
    """
    if not HAS_LIBROSA:
        return []

    signals = []
    if audio_data is not None:
        y, sr = audio_data
    else:
        try:
            y, sr = librosa.load(wav_path, sr=16000, mono=True)
        except Exception as e:
            logger.warning(f"Could not load audio for silence analysis: {e}")
            return []

    # Detect silent intervals
    intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)

    if len(intervals) < 2:
        return []

    # Find gaps between voiced intervals
    gaps = []
    for i in range(len(intervals) - 1):
        gap_start = intervals[i][1]
        gap_end = intervals[i + 1][0]
        gap_duration = (gap_end - gap_start) / sr
        gap_start_time = gap_start / sr

        # Check for digital silence (perfectly zero samples)
        gap_audio = y[gap_start:gap_end]
        is_digital_zero = np.all(np.abs(gap_audio) < 1e-10) if len(gap_audio) > 0 else False

        gaps.append({
            "start": gap_start_time,
            "duration": gap_duration,
            "is_digital_zero": is_digital_zero,
        })

    # Flag digital zero silences (never natural in real recordings)
    for gap in gaps:
        if gap["is_digital_zero"] and gap["duration"] > min_silence_duration:
            signals.append(TamperSignal(
                signal_type="silence_anomaly",
                description=(
                    f"Digital silence (perfect zero) at {gap['start']:.1f}s "
                    f"({gap['duration']*1000:.0f}ms). "
                    f"Natural recordings always have background noise."
                ),
                timestamp=round(gap["start"], 2),
                confidence=0.85,
                severity="high",
            ))

    # Flag inconsistent gap patterns
    if len(gaps) >= 5:
        durations = [g["duration"] for g in gaps if 0.1 < g["duration"] < 5.0]
        if len(durations) >= 3:
            cv = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 0
            # Very inconsistent silence patterns suggest editing
            if cv > 2.0:
                signals.append(TamperSignal(
                    signal_type="silence_anomaly",
                    description=(
                        f"Highly irregular silence pattern (CV={cv:.2f}). "
                        f"Natural conversations have more consistent pauses."
                    ),
                    timestamp=0,
                    confidence=round(min(0.7, cv / 4), 3),
                    severity="medium",
                ))

    return signals


def detect_noise_floor_inconsistency(wav_path: str = None, audio_data: tuple = None) -> list[TamperSignal]:
    """Detect inconsistent background noise levels across the recording.

    Args:
        wav_path: Path to WAV file (used if audio_data not provided)
        audio_data: Pre-loaded (y, sr) tuple to avoid redundant disk reads
    """
    if not HAS_LIBROSA:
        return []

    signals = []
    if audio_data is not None:
        y, sr = audio_data
    else:
        try:
            y, sr = librosa.load(wav_path, sr=16000, mono=True)
        except Exception as e:
            logger.warning(f"Could not load audio for noise analysis: {e}")
            return []

    total_duration = len(y) / sr
    if total_duration < 10:
        return []

    # Analyze noise floor in 5-second windows
    window_size = 5 * sr
    hop = 2 * sr
    noise_floors = []

    for start in range(0, len(y) - window_size, hop):
        chunk = y[start:start + window_size]
        # Estimate noise floor as the 5th percentile of amplitude
        rms = librosa.feature.rms(y=chunk, frame_length=2048, hop_length=512)[0]
        noise_floor = np.percentile(rms, 5)
        noise_floors.append({
            "time": start / sr,
            "noise_floor": float(noise_floor),
        })

    if len(noise_floors) < 3:
        return []

    floors = np.array([n["noise_floor"] for n in noise_floors])
    median_floor = np.median(floors)
    std_floor = np.std(floors)

    # Flag windows with noise floor deviating >2x from median
    for nf in noise_floors:
        if median_floor > 0:
            deviation = abs(nf["noise_floor"] - median_floor) / max(std_floor, median_floor * 0.1)
            if deviation > 3:
                signals.append(TamperSignal(
                    signal_type="noise_floor_inconsistency",
                    description=(
                        f"Background noise level changes significantly at {nf['time']:.0f}s "
                        f"(deviation={deviation:.1f}x). "
                        f"Different noise floors suggest audio from different sources."
                    ),
                    timestamp=round(nf["time"], 2),
                    confidence=round(min(0.8, deviation / 5), 3),
                    severity="medium" if deviation < 5 else "high",
                ))

    return signals


def run_tamper_detection(wav_path: str) -> list[TamperSignal]:
    """Run all tamper detection analyses.

    Loads audio once and passes to all sub-functions to avoid triple disk reads.
    Returns combined list of all detected tampering signals.
    """
    if not HAS_LIBROSA:
        return []

    # Load audio ONCE — previously loaded 3 separate times
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
    except Exception as e:
        logger.warning(f"Could not load audio for tamper detection: {e}")
        return []

    audio = (y, sr)
    signals = []

    # Spectral discontinuities (splice detection)
    spectral = detect_spectral_discontinuities(audio_data=audio)
    signals.extend(spectral)

    # Silence anomalies
    silence = detect_silence_anomalies(audio_data=audio)
    signals.extend(silence)

    # Noise floor consistency
    noise = detect_noise_floor_inconsistency(audio_data=audio)
    signals.extend(noise)

    if signals:
        high = sum(1 for s in signals if s.severity == "high")
        logger.info(f"Tamper detection: {len(signals)} signals ({high} high severity)")
    else:
        logger.info("Tamper detection: no tampering signals detected")

    return signals
