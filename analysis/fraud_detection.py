"""Fraud Detection — voice stress analysis, coached speech detection, behavioral patterns.

Runs on CPU (Parselmouth + numpy). No GPU needed.
On multi-machine setups, emotion2vec runs on Bravo — this module handles Alpha's CPU-only signals.
"""

import numpy as np
from loguru import logger
from config.schemas import FraudSignal

try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False
    logger.warning("Parselmouth not installed — voice stress analysis disabled")


def analyze_voice_stress(wav_path: str, segments: list) -> list[FraudSignal]:
    """Analyze vocal stress indicators per speaker using Parselmouth.

    Measures jitter, shimmer, pitch variability, and HNR.
    Establishes per-speaker baseline from first 30s, then flags deviations.
    """
    if not HAS_PARSELMOUTH:
        return []

    signals = []
    try:
        sound = parselmouth.Sound(wav_path)
    except Exception as e:
        logger.warning(f"Could not load audio for voice stress: {e}")
        return []

    # Group segments by speaker
    speakers = {}
    for i, seg in enumerate(segments):
        speaker = seg.get("speaker", "unknown")
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append((i, seg))

    for speaker, segs in speakers.items():
        if len(segs) < 3:
            continue

        # Extract vocal features per segment
        features = []
        for seg_idx, seg in segs:
            start = seg.get("start", 0)
            end = seg.get("end", start + 1)
            if end - start < 0.5:
                continue

            try:
                chunk = sound.extract_part(start, min(end, sound.xmax), preserve_times=False)
                feats = _extract_vocal_features(chunk)
                if feats:
                    feats["seg_idx"] = seg_idx
                    feats["start"] = start
                    features.append(feats)
            except Exception:
                continue

        if len(features) < 3:
            continue

        # Baseline: first 30s of this speaker's segments
        baseline_feats = [f for f in features if f["start"] < 30]
        if len(baseline_feats) < 2:
            baseline_feats = features[:3]

        baseline = {
            "pitch_mean": np.mean([f["pitch_mean"] for f in baseline_feats if f["pitch_mean"] > 0]) or 150,
            "pitch_std": np.mean([f["pitch_std"] for f in baseline_feats]) or 20,
            "jitter": np.mean([f["jitter"] for f in baseline_feats]) or 0.01,
            "shimmer": np.mean([f["shimmer"] for f in baseline_feats]) or 0.03,
            "hnr": np.mean([f["hnr"] for f in baseline_feats]) or 15,
        }

        # Check each segment against baseline
        for feat in features:
            stress_score = _compute_stress_deviation(feat, baseline)
            if stress_score > 1.5:
                signals.append(FraudSignal(
                    signal_type="voice_stress_anomaly",
                    description=(
                        f"Speaker '{speaker}' shows elevated stress at {feat['start']:.0f}s "
                        f"(deviation={stress_score:.1f}x baseline). "
                        f"Pitch: {feat['pitch_mean']:.0f}Hz (baseline: {baseline['pitch_mean']:.0f}Hz), "
                        f"Jitter: {feat['jitter']:.4f} (baseline: {baseline['jitter']:.4f})"
                    ),
                    segment_id=feat["seg_idx"],
                    confidence=min(0.9, stress_score / 3),
                ))

    return signals


def detect_coached_speech(segments: list) -> list[FraudSignal]:
    """Detect unnaturally consistent speech rate suggesting rehearsed/read responses.

    Natural speech has variable WPM (words per minute). Rehearsed or read speech
    maintains unnaturally consistent pacing.
    """
    signals = []

    speakers = {}
    for i, seg in enumerate(segments):
        speaker = seg.get("speaker", "unknown")
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append((i, seg))

    for speaker, segs in speakers.items():
        wpm_values = []
        for seg_idx, seg in segs:
            text = seg.get("text", "")
            start = seg.get("start", 0)
            end = seg.get("end", start)
            duration = end - start
            if duration < 2:
                continue
            word_count = len(text.split())
            wpm = (word_count / duration) * 60
            if 50 < wpm < 300:  # filter outliers
                wpm_values.append((seg_idx, wpm))

        if len(wpm_values) < 5:
            continue

        wpms = [w[1] for w in wpm_values]
        mean_wpm = np.mean(wpms)
        std_wpm = np.std(wpms)
        cv = std_wpm / mean_wpm if mean_wpm > 0 else 1

        # Natural speech CV is typically 0.15-0.35
        # Coached/read speech CV < 0.10
        if cv < 0.10 and mean_wpm > 100:
            signals.append(FraudSignal(
                signal_type="coached_speech",
                description=(
                    f"Speaker '{speaker}' has unusually consistent speech rate: "
                    f"WPM={mean_wpm:.0f} (std={std_wpm:.1f}, CV={cv:.3f}). "
                    f"Natural speech typically has CV > 0.15. "
                    f"This may indicate rehearsed or read responses."
                ),
                segment_id=wpm_values[0][0],
                confidence=min(0.85, (0.15 - cv) / 0.15),
            ))

    return signals


def detect_speaker_anomalies(segments: list, expected_speakers: int = 2) -> list[FraudSignal]:
    """Detect unexpected number of speakers (third-party on the line)."""
    signals = []
    unique_speakers = set()
    for seg in segments:
        speaker = seg.get("speaker")
        if speaker:
            unique_speakers.add(speaker)

    if len(unique_speakers) > expected_speakers:
        signals.append(FraudSignal(
            signal_type="third_party_detected",
            description=(
                f"Expected {expected_speakers} speakers but detected {len(unique_speakers)}: "
                f"{', '.join(sorted(unique_speakers))}. "
                f"Possible third-party coaching or unauthorized participant."
            ),
            segment_id=0,
            confidence=0.7,
        ))

    return signals


def detect_content_scam_indicators(segments: list) -> list[FraudSignal]:
    """Detect content-based scam patterns in transcript text.

    Phrases are loaded from data/vocab/fraud_indicators.json, which is built
    from real HuggingFace scam conversation datasets:
    - BothBosu/Scammer-Conversation (1K dialogues, 10 scam types)
    - BothBosu/multi-agent-scam-conversation (1.6K dialogues)
    - shakeleoatmeal/phone-scam-detection-synthetic (1.8K dialogues)

    Falls back to minimal hardcoded set if vocab file hasn't been generated.
    """
    from analysis.vocab_loader import get_fraud_indicators
    indicators = get_fraud_indicators()

    signals = []
    full_text = " ".join(s.get("text", "").lower() for s in segments)

    # ── Remote access tool requests (highest risk) ──
    remote_phrases = indicators.get("remote_access_tools", [])
    if not remote_phrases:
        # Minimal fallback if vocab not built
        remote_phrases = ["anydesk", "teamviewer", "ammyy", "ultraviewer",
                          "rustdesk", "remote desktop", "share your screen"]
    for phrase in remote_phrases:
        if phrase in full_text:
            seg_id = next((i for i, s in enumerate(segments) if phrase in s.get("text", "").lower()), 0)
            signals.append(FraudSignal(
                signal_type="remote_access_request",
                description=f"Remote access tool request detected: '{phrase}'. Common in tech support scams.",
                segment_id=seg_id,
                confidence=0.85,
            ))
            break

    # ── Urgency / pressure tactics ──
    urgency_phrases = indicators.get("urgency_pressure", [])
    if not urgency_phrases:
        urgency_phrases = ["right now", "immediately", "before it expires",
                           "last chance", "act fast", "limited time"]
    urgency_hits = [p for p in urgency_phrases if p in full_text]
    if len(urgency_hits) >= 2:
        signals.append(FraudSignal(
            signal_type="urgency_pressure",
            description=f"Multiple urgency/pressure phrases ({len(urgency_hits)}): {', '.join(urgency_hits[:5])}.",
            segment_id=0,
            confidence=min(0.5 + len(urgency_hits) * 0.1, 0.9),
        ))

    # ── Credential / access requests ──
    credential_phrases = indicators.get("credential_requests", [])
    if not credential_phrases:
        credential_phrases = ["give me your password", "tell me the otp",
                              "share your pin", "verification code"]
    import re
    for phrase in credential_phrases:
        if re.search(re.escape(phrase), full_text):
            signals.append(FraudSignal(
                signal_type="credential_request",
                description=f"Credential/access request detected: '{phrase}'.",
                segment_id=0,
                confidence=0.75,
            ))
            break

    # ── Authority impersonation ──
    authority_phrases = indicators.get("authority_impersonation", [])
    if not authority_phrases:
        authority_phrases = ["internal revenue", "social security administration",
                             "law enforcement", "department of treasury"]
    for phrase in authority_phrases:
        if phrase in full_text:
            signals.append(FraudSignal(
                signal_type="authority_impersonation",
                description=f"Authority impersonation detected: '{phrase}'. Scammers impersonate government agencies.",
                segment_id=0,
                confidence=0.80,
            ))
            break

    # ── Vague organizational identity ──
    vague_phrases = indicators.get("vague_identity", [])
    if not vague_phrases:
        vague_phrases = ["online company", "finance department",
                         "technical department", "security department"]
    for phrase in vague_phrases:
        if phrase in full_text:
            signals.append(FraudSignal(
                signal_type="vague_identity",
                description=f"Vague organizational identity: '{phrase}'. Legitimate callers identify specifically.",
                segment_id=0,
                confidence=0.6,
            ))
            break

    # ── Financial threats ──
    threat_phrases = indicators.get("financial_threats", [])
    if not threat_phrases:
        threat_phrases = ["arrest warrant", "legal action", "you will be arrested",
                          "suspend your account", "freeze your account"]
    threat_hits = [p for p in threat_phrases if p in full_text]
    if threat_hits:
        signals.append(FraudSignal(
            signal_type="financial_threat",
            description=f"Financial threats detected: {', '.join(threat_hits[:3])}.",
            segment_id=0,
            confidence=min(0.6 + len(threat_hits) * 0.1, 0.9),
        ))

    # ── Social engineering / trust manipulation ──
    social_phrases = indicators.get("social_engineering", [])
    if not social_phrases:
        social_phrases = ["for your protection", "your account has been compromised",
                          "unauthorized transaction", "security check"]
    social_hits = [p for p in social_phrases if p in full_text]
    if len(social_hits) >= 2:
        signals.append(FraudSignal(
            signal_type="social_engineering",
            description=f"Social engineering patterns ({len(social_hits)}): {', '.join(social_hits[:3])}.",
            segment_id=0,
            confidence=min(0.5 + len(social_hits) * 0.1, 0.85),
        ))

    # ── Agent dominates conversation (>80% talk time = monologue/script) ──
    agent_dur = sum(s.get("end", 0) - s.get("start", 0) for s in segments
                    if s.get("speaker", "").lower() in ("agent", "speaker_00"))
    total_dur = sum(s.get("end", 0) - s.get("start", 0) for s in segments)
    if total_dur > 60 and agent_dur / max(total_dur, 1) > 0.80:
        signals.append(FraudSignal(
            signal_type="agent_monologue",
            description=f"Agent spoke {agent_dur / total_dur * 100:.0f}% of call — highly scripted behavior.",
            segment_id=0,
            confidence=0.5,
        ))

    return signals


def run_fraud_detection(wav_path: str, segments: list) -> list[FraudSignal]:
    """Run all fraud detection analyses and return combined signals."""
    signals = []

    # Voice stress analysis (Parselmouth)
    stress_signals = analyze_voice_stress(wav_path, segments)
    signals.extend(stress_signals)

    # Coached speech detection (WPM variance)
    coached_signals = detect_coached_speech(segments)
    signals.extend(coached_signals)

    # Speaker count anomalies
    speaker_signals = detect_speaker_anomalies(segments)
    signals.extend(speaker_signals)

    # Content-based scam indicators (social engineering patterns)
    content_signals = detect_content_scam_indicators(segments)
    signals.extend(content_signals)

    if signals:
        logger.info(f"Fraud detection: {len(signals)} signals found")
    else:
        logger.info("Fraud detection: no signals detected")

    return signals


def _extract_vocal_features(sound_chunk) -> dict | None:
    """Extract vocal stress features from a Parselmouth Sound object."""
    try:
        pitch = call(sound_chunk, "To Pitch", 0.0, 75, 600)
        pitch_values = pitch.selected_array["frequency"]
        pitch_values = pitch_values[pitch_values > 0]

        if len(pitch_values) < 3:
            return None

        point_process = call(sound_chunk, "To PointProcess (periodic, cc)", 75, 600)
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call(
            [sound_chunk, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )

        harmonicity = call(sound_chunk, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)

        return {
            "pitch_mean": float(np.mean(pitch_values)),
            "pitch_std": float(np.std(pitch_values)),
            "jitter": float(jitter) if not np.isnan(jitter) else 0.01,
            "shimmer": float(shimmer) if not np.isnan(shimmer) else 0.03,
            "hnr": float(hnr) if not np.isnan(hnr) else 15.0,
        }
    except Exception:
        return None


def refine_fraud_with_emotions(
    fraud_signals: list[FraudSignal],
    segment_emotions: list,
    segments: list,
) -> list[FraudSignal]:
    """Cross-reference fraud signals with emotion2vec results to adjust confidence.

    Logic:
    - voice_stress + angry customer → confidence × 0.6 (legitimate anger explains pitch rise)
    - voice_stress + neutral/fearful → confidence × 1.3 (stress without expected emotion = deception)
    - coached_speech + flat emotions → confidence × 1.25 (monotone = rehearsed)
    - coached_speech + varied emotions → confidence × 0.5 (natural variation = not coached)
    - third_party_detected → unchanged (emotion irrelevant)
    - Also generates new "emotional_incongruence" signals
    """
    if not segment_emotions or not fraud_signals:
        # Also check for incongruence even without fraud signals
        if segment_emotions and segments:
            emotion_by_segment = _build_emotion_lookup(segment_emotions)
            new_signals = _detect_emotional_incongruence(segment_emotions, segments, emotion_by_segment)
            return list(fraud_signals) + new_signals
        return list(fraud_signals)

    emotion_by_segment = _build_emotion_lookup(segment_emotions)
    refined = []

    for signal in fraud_signals:
        # Deep copy by reconstructing
        new_confidence = signal.confidence

        if signal.signal_type == "voice_stress_anomaly":
            emotion_data = emotion_by_segment.get(signal.segment_id)
            if emotion_data:
                dominant = emotion_data.get("emotion", "")
                if dominant in ("angry", "anger"):
                    # Legitimate anger explains vocal stress — reduce confidence
                    new_confidence = signal.confidence * 0.6
                elif dominant in ("neutral", "fearful", "fear", "sad"):
                    # Stress without expected emotion = more suspicious
                    new_confidence = signal.confidence * 1.3

        elif signal.signal_type == "coached_speech":
            # Check emotion diversity across all customer segments
            diversity = _compute_emotion_diversity(segment_emotions, segments)
            if diversity < 0.25:
                # Flat emotions = rehearsed
                new_confidence = signal.confidence * 1.25
            elif diversity > 0.5:
                # Varied emotions = natural, not coached
                new_confidence = signal.confidence * 0.5

        # third_party_detected and others pass through unchanged

        new_confidence = max(0.0, min(1.0, new_confidence))
        refined.append(FraudSignal(
            signal_type=signal.signal_type,
            description=signal.description,
            segment_id=signal.segment_id,
            confidence=round(new_confidence, 4),
        ))

    # Add emotional incongruence signals
    incongruence_signals = _detect_emotional_incongruence(segment_emotions, segments, emotion_by_segment)
    refined.extend(incongruence_signals)

    if refined != list(fraud_signals):
        logger.info(f"Fraud refinement: {len(fraud_signals)} signals → {len(refined)} after emotion convergence")

    return refined


def _build_emotion_lookup(segment_emotions: list) -> dict:
    """Build segment_id → emotion data lookup from emotion2vec results."""
    lookup = {}
    for emo in segment_emotions:
        if hasattr(emo, "model_dump"):
            emo_dict = emo.model_dump()
        elif isinstance(emo, dict):
            emo_dict = emo
        else:
            continue
        seg_id = emo_dict.get("segment_id")
        if seg_id is not None:
            lookup[seg_id] = emo_dict
    return lookup


def _compute_emotion_diversity(segment_emotions: list, segments: list) -> float:
    """Compute emotion diversity for customer segments (0 = all same, 1 = all different).

    Uses normalized entropy of emotion distribution.
    """
    customer_emotions = []
    for emo in segment_emotions:
        if hasattr(emo, "model_dump"):
            emo_dict = emo.model_dump()
        elif isinstance(emo, dict):
            emo_dict = emo
        else:
            continue
        seg_id = emo_dict.get("segment_id")
        if seg_id is not None and seg_id < len(segments):
            speaker = segments[seg_id].get("speaker", "")
            if speaker.lower() in ("customer", "speaker_01"):
                customer_emotions.append(emo_dict.get("emotion", "neutral"))

    if len(customer_emotions) < 2:
        return 0.5  # Not enough data — neutral

    # Count unique emotions
    from collections import Counter
    counts = Counter(customer_emotions)
    total = len(customer_emotions)
    n_unique = len(counts)

    if n_unique <= 1:
        return 0.0

    # Normalized entropy
    import math
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_entropy = math.log2(n_unique)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _detect_emotional_incongruence(
    segment_emotions: list,
    segments: list,
    emotion_by_segment: dict,
) -> list[FraudSignal]:
    """Detect emotional incongruence — fear/surprise during routine verification questions.

    If agent asks a verification question (DOB, account number, PAN, Aadhaar)
    and customer responds with fear (score > 0.5) or surprise (score > 0.6),
    this is suspicious (legitimate callers don't fear their own identity questions).
    """
    VERIFICATION_KEYWORDS = (
        "date of birth", "dob", "account number", "pan number", "pan card",
        "aadhaar", "aadhar", "verify", "verification", "confirm your",
        "mother's maiden", "maiden name", "security question",
    )

    signals = []

    for i, seg in enumerate(segments):
        speaker = seg.get("speaker", "").lower()
        text = seg.get("text", "").lower()

        # Check if this is an agent verification question
        if speaker not in ("agent", "speaker_00"):
            continue
        if not any(kw in text for kw in VERIFICATION_KEYWORDS):
            continue

        # Look at the customer's next response (i+1 or i+2)
        for response_idx in range(i + 1, min(i + 3, len(segments))):
            response_seg = segments[response_idx]
            response_speaker = response_seg.get("speaker", "").lower()
            if response_speaker not in ("customer", "speaker_01"):
                continue

            emo_data = emotion_by_segment.get(response_idx)
            if not emo_data:
                break

            dominant = emo_data.get("emotion", "")
            score = emo_data.get("score", 0)
            all_scores = emo_data.get("all_scores", {})

            # Check for fear or surprise
            fear_score = all_scores.get("fearful", all_scores.get("fear", 0))
            surprise_score = all_scores.get("surprised", all_scores.get("surprise", 0))

            is_fearful = dominant in ("fearful", "fear") and score > 0.5
            is_surprised = dominant in ("surprised", "surprise") and score > 0.6
            has_high_fear = fear_score > 0.5
            has_high_surprise = surprise_score > 0.6

            if is_fearful or is_surprised or has_high_fear or has_high_surprise:
                signals.append(FraudSignal(
                    signal_type="emotional_incongruence",
                    description=(
                        f"Customer shows {dominant} (score={score:.2f}) when asked "
                        f"verification question at segment {i}: '{seg.get('text', '')[:60]}'. "
                        f"Legitimate callers typically show neutral emotion during identity verification."
                    ),
                    segment_id=response_idx,
                    confidence=min(0.85, max(fear_score, surprise_score)),
                ))
            break  # Only check first customer response

    return signals


def _compute_stress_deviation(features: dict, baseline: dict) -> float:
    """Compute how many standard deviations a segment deviates from baseline.

    Higher values indicate more vocal stress. Returns a multiplier where:
    - 1.0 = normal
    - 1.5+ = elevated stress
    - 2.0+ = significant stress
    """
    deviations = []

    # Pitch increase under stress
    if baseline["pitch_mean"] > 0 and features["pitch_mean"] > 0:
        pitch_dev = (features["pitch_mean"] - baseline["pitch_mean"]) / max(baseline["pitch_std"], 10)
        deviations.append(max(0, pitch_dev))

    # Jitter increases under stress
    if baseline["jitter"] > 0:
        jitter_dev = features["jitter"] / baseline["jitter"]
        deviations.append(max(0, jitter_dev - 1))

    # Shimmer increases under stress
    if baseline["shimmer"] > 0:
        shimmer_dev = features["shimmer"] / baseline["shimmer"]
        deviations.append(max(0, shimmer_dev - 1))

    # HNR decreases under stress
    if baseline["hnr"] > 0:
        hnr_dev = (baseline["hnr"] - features["hnr"]) / max(baseline["hnr"], 1)
        deviations.append(max(0, hnr_dev))

    return float(np.mean(deviations)) + 1.0 if deviations else 1.0
