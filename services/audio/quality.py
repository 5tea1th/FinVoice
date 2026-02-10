"""Stage 2: Audio Quality Assessment — SNR, clipping, speech ratio, spectral quality."""

import numpy as np
import soundfile as sf
import opensmile
from loguru import logger


def assess_audio_quality(wav_path: str) -> dict:
    """Comprehensive audio quality assessment producing a 0-100 score.

    Four signals weighted:
      - SNR (40%): Speech power vs noise power via VAD
      - Clipping (20%): Samples hitting ±1.0 amplitude ceiling
      - Speech-to-silence ratio (20%): % of audio containing speech
      - Spectral quality (20%): HNR and spectral features via openSMILE

    Returns:
        dict with quality_score (0-100), quality_flag, snr_db, and component scores
    """
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert to mono if needed

    # 1. SNR estimation via simple energy-based VAD
    snr_score, snr_db, speech_timestamps = _compute_snr(audio, sr)

    # 2. Clipping detection
    clipping_score = _compute_clipping_score(audio)

    # 3. Speech-to-silence ratio
    speech_ratio_score, speech_pct = _compute_speech_ratio(speech_timestamps, len(audio), sr)

    # 4. Spectral quality via openSMILE
    spectral_score = _compute_spectral_score(wav_path)

    # Weighted combination
    quality_score = int(
        snr_score * 0.4
        + clipping_score * 0.2
        + speech_ratio_score * 0.2
        + spectral_score * 0.2
    )
    quality_score = max(0, min(100, quality_score))

    if quality_score >= 70:
        flag = "TRUSTWORTHY"
    elif quality_score >= 40:
        flag = "DEGRADED"
    else:
        flag = "UNRELIABLE"

    result = {
        "quality_score": quality_score,
        "quality_flag": flag,
        "snr_db": round(snr_db, 1),
        "clipping_detected": clipping_score < 80,
        "speech_percentage": round(speech_pct, 1),
        "speech_timestamps": speech_timestamps,
        "component_scores": {
            "snr": round(snr_score, 1),
            "clipping": round(clipping_score, 1),
            "speech_ratio": round(speech_ratio_score, 1),
            "spectral": round(spectral_score, 1),
        },
    }

    logger.info(f"Audio quality: {quality_score}/100 ({flag}) — SNR: {snr_db:.1f}dB")
    return result


def _compute_snr(audio: np.ndarray, sr: int) -> tuple[float, float, list]:
    """Estimate SNR using energy-based voice activity detection."""
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop

    # Compute frame energies
    energies = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        energies.append(np.sum(frame ** 2) / frame_length)
    energies = np.array(energies)

    if len(energies) == 0:
        return 0.0, 0.0, []

    # Simple threshold: frames above median energy are "speech"
    threshold = np.median(energies) * 1.5
    is_speech = energies > threshold

    speech_energy = np.mean(energies[is_speech]) if np.any(is_speech) else 1e-10
    noise_energy = np.mean(energies[~is_speech]) if np.any(~is_speech) else 1e-10

    snr_db = 10 * np.log10(speech_energy / max(noise_energy, 1e-10))
    snr_db = max(0, min(40, snr_db))  # Clip to reasonable range

    # Score: 0dB → 0, 10dB → 50, 20dB+ → 100
    snr_score = min(100, snr_db * 5)

    # Build speech timestamps for other functions
    speech_timestamps = []
    in_speech = False
    start = 0
    for i, s in enumerate(is_speech):
        if s and not in_speech:
            start = i * hop_length / sr
            in_speech = True
        elif not s and in_speech:
            end = i * hop_length / sr
            speech_timestamps.append({"start": start, "end": end})
            in_speech = False
    if in_speech:
        speech_timestamps.append({"start": start, "end": len(audio) / sr})

    return snr_score, snr_db, speech_timestamps


def _compute_clipping_score(audio: np.ndarray) -> float:
    """Check for clipping (samples at ±1.0). Returns 0-100 score."""
    clipped = np.sum(np.abs(audio) >= 0.99)
    clip_pct = clipped / max(len(audio), 1) * 100

    # 0% clipped → 100, 2%+ clipped → 0
    if clip_pct < 0.1:
        return 100.0
    elif clip_pct < 0.5:
        return 80.0
    elif clip_pct < 2.0:
        return 50.0
    else:
        return 20.0


def _compute_speech_ratio(
    speech_timestamps: list, total_samples: int, sr: int
) -> tuple[float, float]:
    """Compute speech-to-total ratio. Returns (score, percentage)."""
    total_duration = total_samples / sr
    if total_duration == 0:
        return 0.0, 0.0

    speech_duration = sum(seg["end"] - seg["start"] for seg in speech_timestamps)
    speech_pct = (speech_duration / total_duration) * 100

    # Normal calls: 60-80% speech. Below 30% is suspicious.
    if speech_pct >= 50:
        score = 100.0
    elif speech_pct >= 30:
        score = 60.0
    elif speech_pct >= 15:
        score = 30.0
    else:
        score = 10.0

    return score, speech_pct


def _compute_spectral_score(wav_path: str) -> float:
    """Extract spectral quality features via openSMILE eGeMAPS."""
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        features = smile.process_file(wav_path)

        # Use HNR (Harmonics-to-Noise Ratio) as primary spectral quality indicator
        # Higher HNR = cleaner voice signal
        hnr_col = [c for c in features.columns if "HNR" in c.upper() or "hnr" in c.lower()]
        if hnr_col:
            hnr_mean = features[hnr_col[0]].values[0]
            # HNR: 0-5 dB = poor, 5-15 dB = fair, 15+ dB = good
            spectral_score = min(100, max(0, hnr_mean * 5))
        else:
            spectral_score = 50.0  # Default if HNR not found

        return spectral_score
    except Exception as e:
        logger.warning(f"openSMILE spectral analysis failed: {e}")
        return 50.0  # Neutral default on failure
