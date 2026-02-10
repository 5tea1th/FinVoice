"""Tests for dual-model sentiment analysis (FinBERT EN + XLM-R multilingual)."""

import pytest
from analysis.sentiment import (
    analyze_sentiment,
    analyze_sentiment_batch,
    compute_sentiment_trajectories,
    classify_for_llm_routing,
    get_dominant_emotion,
    _normalize_label,
    _is_agent_speaker,
    _is_customer_speaker,
    HAS_FINBERT,
)

pytestmark = pytest.mark.skipif(not HAS_FINBERT, reason="transformers not installed")


# ── Label Normalization ──

def test_normalize_label_lowercase():
    assert _normalize_label("Positive") == "positive"
    assert _normalize_label("NEGATIVE") == "negative"
    assert _normalize_label("neutral") == "neutral"


def test_normalize_label_strips_whitespace():
    assert _normalize_label("  positive  ") == "positive"


# ── Speaker Detection ──

def test_agent_speaker_labels():
    assert _is_agent_speaker("agent") is True
    assert _is_agent_speaker("SPEAKER_00") is True
    assert _is_agent_speaker("customer") is False


def test_customer_speaker_labels():
    assert _is_customer_speaker("customer") is True
    assert _is_customer_speaker("SPEAKER_01") is True
    assert _is_customer_speaker("agent") is False


# ── Sentiment Analysis ──

def test_analyze_sentiment_returns_dict():
    result = analyze_sentiment("I am very happy with this service")
    assert "label" in result
    assert "score" in result
    assert result["label"] in ("positive", "negative", "neutral")
    assert 0.0 <= result["score"] <= 1.0


def test_analyze_sentiment_positive():
    result = analyze_sentiment("This is excellent news for investors, profits are up 50%")
    assert result["label"] == "positive"
    assert result["score"] > 0.5


def test_analyze_sentiment_negative():
    result = analyze_sentiment("This is terrible, the company is losing money rapidly")
    assert result["label"] == "negative"
    assert result["score"] > 0.5


def test_analyze_sentiment_batch():
    texts = [
        "Revenue exceeded expectations",
        "The stock crashed badly",
        "The meeting is at 3pm",
    ]
    results = analyze_sentiment_batch(texts)
    assert len(results) == 3
    for r in results:
        assert "label" in r
        assert "score" in r


# ── Sentiment Trajectories ──

def test_trajectories_basic():
    segments = [
        {"text": "Hello, how can I help you?", "speaker": "agent"},
        {"text": "I have a problem with my account", "speaker": "customer"},
        {"text": "Let me check that for you", "speaker": "agent"},
    ]
    customer_traj, agent_traj = compute_sentiment_trajectories(segments)
    assert len(customer_traj) >= 1
    assert len(agent_traj) >= 1
    for score in customer_traj + agent_traj:
        assert -1.0 <= score <= 1.0


def test_trajectories_empty_segments():
    customer_traj, agent_traj = compute_sentiment_trajectories([])
    assert customer_traj == [0.0]
    assert agent_traj == [0.0]


def test_trajectories_non_english():
    """Multilingual model should handle non-English text without crashing."""
    segments = [
        {"text": "Здравствуйте, чем могу помочь?", "speaker": "agent"},
        {"text": "У меня проблема с аккаунтом", "speaker": "customer"},
    ]
    customer_traj, agent_traj = compute_sentiment_trajectories(segments, lang="ru")
    assert len(customer_traj) >= 1
    assert len(agent_traj) >= 1


# ── LLM Routing ──

def test_routing_skip_trivial():
    assert classify_for_llm_routing("okay") == "skip"
    assert classify_for_llm_routing("yes") == "skip"
    assert classify_for_llm_routing("thank you") == "skip"
    assert classify_for_llm_routing("hello") == "skip"


def test_routing_returns_valid():
    result = classify_for_llm_routing("I want to dispute this charge on my account")
    assert result in ("skip", "negative_subset", "full")


# ── Dominant Emotion ──

def test_dominant_emotion_positive():
    assert get_dominant_emotion([0.5, 0.6, 0.4]) == "positive"


def test_dominant_emotion_angry():
    assert get_dominant_emotion([-0.8, -0.7, -0.9]) == "angry"


def test_dominant_emotion_neutral():
    assert get_dominant_emotion([0.1, -0.1, 0.0]) == "neutral"


def test_dominant_emotion_empty():
    assert get_dominant_emotion([]) == "neutral"
