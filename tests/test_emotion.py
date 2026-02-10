"""Tests for emotion2vec analyzer and emotion summary."""

import pytest
from services.emotion.emotion2vec_analyzer import (
    SegmentEmotion,
    get_emotion_summary,
    EMOTION_LABELS,
)


# ── Emotion Summary (no model needed) ──

class TestEmotionSummary:
    def test_empty_emotions(self):
        summary = get_emotion_summary([])
        assert summary["dominant_emotion"] == "neutral"
        assert summary["emotion_distribution"] == {}
        assert summary["escalation_moments"] == []

    def test_single_emotion(self):
        emotions = [SegmentEmotion(
            segment_id=0, speaker="customer", emotion="happy",
            emotion_score=0.9, all_scores={"happy": 0.9, "neutral": 0.1},
        )]
        summary = get_emotion_summary(emotions)
        assert summary["dominant_emotion"] == "happy"
        assert summary["emotion_distribution"]["happy"] == 1.0

    def test_mixed_emotions(self):
        emotions = [
            SegmentEmotion(segment_id=0, speaker="customer", emotion="neutral",
                           emotion_score=0.8, all_scores={"neutral": 0.8}),
            SegmentEmotion(segment_id=1, speaker="customer", emotion="angry",
                           emotion_score=0.7, all_scores={"angry": 0.7}),
            SegmentEmotion(segment_id=2, speaker="customer", emotion="neutral",
                           emotion_score=0.85, all_scores={"neutral": 0.85}),
        ]
        summary = get_emotion_summary(emotions)
        assert summary["dominant_emotion"] == "neutral"
        assert "neutral" in summary["emotion_distribution"]
        assert "angry" in summary["emotion_distribution"]

    def test_escalation_detected(self):
        emotions = [
            SegmentEmotion(segment_id=0, speaker="customer", emotion="neutral",
                           emotion_score=0.9, all_scores={"neutral": 0.9}),
            SegmentEmotion(segment_id=5, speaker="customer", emotion="angry",
                           emotion_score=0.85, all_scores={"angry": 0.85}),
        ]
        summary = get_emotion_summary(emotions)
        assert len(summary["escalation_moments"]) == 1
        assert summary["escalation_moments"][0]["emotion"] == "angry"
        assert summary["escalation_moments"][0]["segment_id"] == 5

    def test_fearful_escalation(self):
        emotions = [
            SegmentEmotion(segment_id=3, speaker="customer", emotion="fearful",
                           emotion_score=0.75, all_scores={"fearful": 0.75}),
        ]
        summary = get_emotion_summary(emotions)
        assert len(summary["escalation_moments"]) == 1

    def test_low_score_no_escalation(self):
        """Angry/fearful with low confidence should NOT trigger escalation."""
        emotions = [
            SegmentEmotion(segment_id=0, speaker="customer", emotion="angry",
                           emotion_score=0.4, all_scores={"angry": 0.4}),
        ]
        summary = get_emotion_summary(emotions)
        assert len(summary["escalation_moments"]) == 0


class TestEmotionLabels:
    def test_all_8_labels_exist(self):
        assert len(EMOTION_LABELS) == 8
        assert "angry" in EMOTION_LABELS
        assert "happy" in EMOTION_LABELS
        assert "neutral" in EMOTION_LABELS
        assert "sad" in EMOTION_LABELS


class TestSegmentEmotionModel:
    def test_valid_creation(self):
        e = SegmentEmotion(
            segment_id=0, speaker="customer", emotion="happy",
            emotion_score=0.95, all_scores={"happy": 0.95, "neutral": 0.05},
        )
        assert e.emotion == "happy"
        assert e.emotion_score == 0.95

    def test_score_bounds(self):
        with pytest.raises(Exception):
            SegmentEmotion(
                segment_id=0, speaker="customer", emotion="happy",
                emotion_score=1.5, all_scores={},
            )
