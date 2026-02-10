"""Tests for fraud + emotion2vec convergence (refine_fraud_with_emotions)."""

import pytest
from config.schemas import FraudSignal
from analysis.fraud_detection import refine_fraud_with_emotions


def _make_signal(signal_type: str, confidence: float, segment_id: int = 0) -> FraudSignal:
    return FraudSignal(
        signal_type=signal_type,
        description=f"Test {signal_type}",
        segment_id=segment_id,
        confidence=confidence,
    )


def _make_emotion(segment_id: int, emotion: str, score: float = 0.7, all_scores: dict = None) -> dict:
    return {
        "segment_id": segment_id,
        "emotion": emotion,
        "score": score,
        "all_scores": all_scores or {emotion: score},
    }


def _make_segment(speaker: str, text: str = "some text", start: float = 0.0, end: float = 1.0) -> dict:
    return {"speaker": speaker, "text": text, "start": start, "end": end}


class TestEmptyInputs:
    def test_empty_emotions_returns_originals(self):
        signals = [_make_signal("voice_stress_anomaly", 0.7)]
        result = refine_fraud_with_emotions(signals, [], [])
        assert len(result) == 1
        assert result[0].confidence == 0.7

    def test_empty_signals_returns_empty(self):
        result = refine_fraud_with_emotions([], [], [])
        assert result == []

    def test_both_empty(self):
        result = refine_fraud_with_emotions([], [], [])
        assert result == []

    def test_originals_not_mutated(self):
        signals = [_make_signal("voice_stress_anomaly", 0.7)]
        emotions = [_make_emotion(0, "angry")]
        segments = [_make_segment("customer")]
        original_confidence = signals[0].confidence

        refine_fraud_with_emotions(signals, emotions, segments)
        assert signals[0].confidence == original_confidence


class TestVoiceStressRefinement:
    def test_angry_reduces_confidence(self):
        signals = [_make_signal("voice_stress_anomaly", 0.8, segment_id=0)]
        emotions = [_make_emotion(0, "angry")]
        segments = [_make_segment("customer")]
        result = refine_fraud_with_emotions(signals, emotions, segments)
        assert result[0].confidence < 0.8
        assert result[0].confidence == pytest.approx(0.8 * 0.6, abs=0.01)

    def test_anger_reduces_confidence(self):
        """emotion2vec may report 'anger' instead of 'angry'."""
        signals = [_make_signal("voice_stress_anomaly", 0.8, segment_id=0)]
        emotions = [_make_emotion(0, "anger")]
        segments = [_make_segment("customer")]
        result = refine_fraud_with_emotions(signals, emotions, segments)
        assert result[0].confidence < 0.8

    def test_neutral_increases_confidence(self):
        signals = [_make_signal("voice_stress_anomaly", 0.5, segment_id=0)]
        emotions = [_make_emotion(0, "neutral")]
        segments = [_make_segment("customer")]
        result = refine_fraud_with_emotions(signals, emotions, segments)
        assert result[0].confidence > 0.5
        assert result[0].confidence == pytest.approx(0.5 * 1.3, abs=0.01)

    def test_fearful_increases_confidence(self):
        signals = [_make_signal("voice_stress_anomaly", 0.5, segment_id=0)]
        emotions = [_make_emotion(0, "fearful")]
        segments = [_make_segment("customer")]
        result = refine_fraud_with_emotions(signals, emotions, segments)
        assert result[0].confidence > 0.5

    def test_sad_increases_confidence(self):
        signals = [_make_signal("voice_stress_anomaly", 0.5, segment_id=0)]
        emotions = [_make_emotion(0, "sad")]
        segments = [_make_segment("customer")]
        result = refine_fraud_with_emotions(signals, emotions, segments)
        assert result[0].confidence > 0.5

    def test_happy_unchanged(self):
        signals = [_make_signal("voice_stress_anomaly", 0.5, segment_id=0)]
        emotions = [_make_emotion(0, "happy")]
        segments = [_make_segment("customer")]
        result = refine_fraud_with_emotions(signals, emotions, segments)
        assert result[0].confidence == 0.5

    def test_no_matching_emotion_unchanged(self):
        signals = [_make_signal("voice_stress_anomaly", 0.5, segment_id=0)]
        emotions = [_make_emotion(5, "angry")]  # Different segment
        segments = [_make_segment("customer")]
        result = refine_fraud_with_emotions(signals, emotions, segments)
        assert result[0].confidence == 0.5


class TestCoachedSpeechRefinement:
    def test_flat_emotions_increases_confidence(self):
        """All neutral = rehearsed, should increase coached_speech confidence."""
        signals = [_make_signal("coached_speech", 0.6)]
        # All customer segments are neutral = flat diversity < 0.25
        emotions = [
            _make_emotion(i, "neutral") for i in range(5)
        ]
        segments = [_make_segment("customer") for _ in range(5)]
        result = refine_fraud_with_emotions(signals, emotions, segments)
        assert result[0].confidence > 0.6

    def test_varied_emotions_reduces_confidence(self):
        """Natural emotion variety = not coached."""
        signals = [_make_signal("coached_speech", 0.6)]
        emotions = [
            _make_emotion(0, "neutral"),
            _make_emotion(1, "angry"),
            _make_emotion(2, "happy"),
            _make_emotion(3, "sad"),
            _make_emotion(4, "fearful"),
        ]
        segments = [_make_segment("customer") for _ in range(5)]
        result = refine_fraud_with_emotions(signals, emotions, segments)
        assert result[0].confidence < 0.6


class TestThirdPartyUnchanged:
    def test_third_party_passthrough(self):
        signals = [_make_signal("third_party_detected", 0.7)]
        emotions = [_make_emotion(0, "angry")]
        segments = [_make_segment("customer")]
        result = refine_fraud_with_emotions(signals, emotions, segments)
        assert result[0].confidence == 0.7
        assert result[0].signal_type == "third_party_detected"


class TestConfidenceBounds:
    def test_confidence_capped_at_1(self):
        signals = [_make_signal("voice_stress_anomaly", 0.9, segment_id=0)]
        emotions = [_make_emotion(0, "neutral")]
        segments = [_make_segment("customer")]
        result = refine_fraud_with_emotions(signals, emotions, segments)
        assert result[0].confidence <= 1.0

    def test_confidence_floored_at_0(self):
        signals = [_make_signal("voice_stress_anomaly", 0.01, segment_id=0)]
        emotions = [_make_emotion(0, "angry")]
        segments = [_make_segment("customer")]
        result = refine_fraud_with_emotions(signals, emotions, segments)
        assert result[0].confidence >= 0.0


class TestEmotionalIncongruence:
    def test_fear_during_verification(self):
        """Fear when asked verification question = suspicious."""
        segments = [
            _make_segment("agent", "Can you please confirm your date of birth?"),
            _make_segment("customer", "Uh... it's... January 15th, 1990"),
        ]
        emotions = [
            _make_emotion(0, "neutral", 0.8),
            _make_emotion(1, "fearful", 0.7, {"fearful": 0.7, "neutral": 0.2}),
        ]
        result = refine_fraud_with_emotions([], emotions, segments)
        incongruence = [s for s in result if s.signal_type == "emotional_incongruence"]
        assert len(incongruence) == 1
        assert incongruence[0].segment_id == 1

    def test_surprise_during_verification(self):
        segments = [
            _make_segment("agent", "Please verify your account number"),
            _make_segment("customer", "Oh, um, let me think..."),
        ]
        emotions = [
            _make_emotion(0, "neutral", 0.8),
            _make_emotion(1, "surprised", 0.7, {"surprised": 0.7, "neutral": 0.2}),
        ]
        result = refine_fraud_with_emotions([], emotions, segments)
        incongruence = [s for s in result if s.signal_type == "emotional_incongruence"]
        assert len(incongruence) == 1

    def test_neutral_during_verification_no_signal(self):
        """Neutral during verification = expected, no signal."""
        segments = [
            _make_segment("agent", "Can you confirm your date of birth?"),
            _make_segment("customer", "January 15th, 1990"),
        ]
        emotions = [
            _make_emotion(0, "neutral", 0.8),
            _make_emotion(1, "neutral", 0.9, {"neutral": 0.9, "fearful": 0.05}),
        ]
        result = refine_fraud_with_emotions([], emotions, segments)
        incongruence = [s for s in result if s.signal_type == "emotional_incongruence"]
        assert len(incongruence) == 0

    def test_fear_during_non_verification_no_signal(self):
        """Fear during non-verification = not flagged by incongruence detector."""
        segments = [
            _make_segment("agent", "How are you doing today?"),
            _make_segment("customer", "I'm worried about my payment"),
        ]
        emotions = [
            _make_emotion(0, "neutral", 0.8),
            _make_emotion(1, "fearful", 0.7, {"fearful": 0.7}),
        ]
        result = refine_fraud_with_emotions([], emotions, segments)
        incongruence = [s for s in result if s.signal_type == "emotional_incongruence"]
        assert len(incongruence) == 0

    def test_combined_with_existing_signals(self):
        """Incongruence signals should be appended to refined fraud signals."""
        signals = [_make_signal("voice_stress_anomaly", 0.7, segment_id=1)]
        segments = [
            _make_segment("agent", "Confirm your PAN number please"),
            _make_segment("customer", "Uh... let me check..."),
        ]
        emotions = [
            _make_emotion(0, "neutral", 0.8),
            _make_emotion(1, "fearful", 0.7, {"fearful": 0.7, "neutral": 0.2}),
        ]
        result = refine_fraud_with_emotions(signals, emotions, segments)
        types = [s.signal_type for s in result]
        assert "voice_stress_anomaly" in types
        assert "emotional_incongruence" in types
