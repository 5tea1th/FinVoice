"""Emotion2vec — Audio-based emotion recognition from speech waveforms.

Uses iic/emotion2vec_plus_large (FunASR) to detect emotions directly from audio:
  angry, disgusted, fearful, happy, neutral, sad, surprised, other

Unlike FinBERT (text-only sentiment), emotion2vec captures vocal tone, pitch
patterns, and prosody — it detects anger even in politely-worded sentences.

VRAM: ~1GB. In the 6GB VRAM lifecycle:
  Stage 3: WhisperX (~3GB) → unload
  Stage 4 (new): emotion2vec (~1GB) → unload
  Stage 4E: Ollama/Qwen3 (~5GB) → unload

Optimized for batch processing — segments are grouped into mini-batches
to reduce per-segment file I/O overhead.
"""

import os
import tempfile
import traceback
import numpy as np
from pathlib import Path
from loguru import logger
from pydantic import BaseModel, Field

try:
    from funasr import AutoModel
    HAS_EMOTION2VEC = True
except ImportError:
    HAS_EMOTION2VEC = False
    logger.warning("funasr not installed — emotion2vec disabled")


EMOTION_LABELS = [
    "angry", "disgusted", "fearful", "happy",
    "neutral", "sad", "surprised", "other",
]

_model = None


class SegmentEmotion(BaseModel):
    """Emotion classification for a single transcript segment."""
    segment_id: int
    speaker: str
    emotion: str = Field(description="Primary emotion: angry, happy, sad, neutral, etc.")
    emotion_score: float = Field(ge=0, le=1, description="Confidence of primary emotion")
    all_scores: dict[str, float] = Field(description="Scores for all 8 emotions")


def _get_model():
    """Load emotion2vec model (lazy, cached).

    PyTorch 2.8 breaks FunASR's model.to("cuda") due to meta tensors.
    Fix: monkey-patch Module.to() during load to handle meta tensors via to_empty().
    This lets FunASR initialize fully on GPU with correct device placement for inputs.
    """
    global _model
    if _model is not None:
        return _model

    if not HAS_EMOTION2VEC:
        return None

    import torch

    # Meta tensor patch is applied globally in app.py at process startup

    if torch.cuda.is_available():
        logger.info("Loading emotion2vec_plus_large on GPU...")
        try:
            _model = AutoModel(
                model="iic/emotion2vec_plus_large",
                trust_remote_code=True,
                device="cuda",
                disable_update=True,
            )
            logger.info("emotion2vec loaded on GPU (~1GB VRAM)")
        except Exception as e:
            logger.error(f"emotion2vec GPU load failed: {e}")
            _model = None

        if _model is not None:
            return _model

    # Fallback to CPU
    logger.info("Loading emotion2vec_plus_large on CPU...")
    try:
        _model = AutoModel(
            model="iic/emotion2vec_plus_large",
            trust_remote_code=True,
            device="cpu",
            disable_update=True,
        )
        logger.info("emotion2vec loaded on CPU")
    except Exception as e:
        logger.error(f"emotion2vec failed to load: {e}")
        _model = None

    return _model


def unload_model():
    """Free emotion2vec from GPU memory."""
    global _model
    if _model is not None:
        del _model
        _model = None
        try:
            import torch
            torch.cuda.empty_cache()
            logger.info("emotion2vec unloaded, VRAM freed")
        except Exception:
            pass


def _parse_emotion_result(result_item, segment_id: int, speaker: str) -> SegmentEmotion | None:
    """Parse a single emotion2vec result into a SegmentEmotion object.

    Handles dict, object, and nested list formats from different FunASR versions.
    """
    labels = None
    scores = None

    if isinstance(result_item, dict):
        labels = result_item.get("labels", result_item.get("label", EMOTION_LABELS))
        scores = result_item.get("scores", result_item.get("score", []))
        # FunASR may nest labels/scores as lists of lists
        if isinstance(labels, list) and labels and isinstance(labels[0], list):
            labels = labels[0]
        if isinstance(scores, list) and scores and isinstance(scores[0], list):
            scores = scores[0]
    elif hasattr(result_item, "labels"):
        labels = result_item.labels
        scores = result_item.scores if hasattr(result_item, "scores") else []
    else:
        return None

    if not scores or not labels:
        return None

    # Build scores dict, mapping FunASR labels to our standard labels
    score_dict = {}
    for label, score_val in zip(labels, scores):
        label_clean = label.strip("/").lower() if isinstance(label, str) else str(label)
        matched = False
        for emotion in EMOTION_LABELS:
            if emotion in label_clean:
                score_dict[emotion] = float(score_val)
                matched = True
                break
        if not matched:
            score_dict[label_clean] = float(score_val)

    if not score_dict:
        return None

    primary = max(score_dict, key=score_dict.get)
    primary_score = score_dict[primary]

    return SegmentEmotion(
        segment_id=segment_id,
        speaker=speaker,
        emotion=primary,
        emotion_score=round(primary_score, 3),
        all_scores={k: round(v, 3) for k, v in score_dict.items()},
    )


def analyze_emotions(
    wav_path: str,
    segments: list,
    granularity: str = "utterance",
    audio_array: np.ndarray | None = None,
    sample_rate: int = 16000,
) -> list[SegmentEmotion]:
    """Analyze emotions per segment from audio waveforms.

    Passes numpy arrays directly to FunASR — no temp file I/O.
    Falls back to temp files only if direct array input fails.

    Args:
        wav_path: Path to 16kHz mono WAV file (used as fallback)
        segments: Transcript segments with 'start', 'end', 'speaker' fields
        granularity: 'utterance' for per-segment analysis
        audio_array: Pre-loaded audio numpy array (avoids redundant disk read)
        sample_rate: Sample rate of audio_array (default 16000)

    Returns:
        List of SegmentEmotion objects
    """
    model = _get_model()
    if model is None:
        logger.warning("emotion2vec not available — skipping emotion analysis")
        return []

    if audio_array is not None:
        audio_data = audio_array
        sr = sample_rate
    else:
        import soundfile as sf
        try:
            audio_data, sr = sf.read(wav_path)
        except Exception as e:
            logger.error(f"Could not read audio for emotion2vec: {e}")
            return []

    # Convert to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    total_samples = len(audio_data)

    # Pre-slice all valid segments into chunks
    valid_segments = []  # (segment_index, segment_dict, audio_chunk)
    for i, seg in enumerate(segments):
        start_time = seg.get("start", 0)
        end_time = seg.get("end", start_time + 1)

        if end_time - start_time < 0.5:
            continue

        start_sample = int(start_time * sr)
        end_sample = min(int(end_time * sr), total_samples)

        if end_sample <= start_sample:
            continue

        chunk = audio_data[start_sample:end_sample]
        if len(chunk) < sr * 0.5:
            continue

        valid_segments.append((i, seg, chunk))

    if not valid_segments:
        logger.warning("emotion2vec: no valid segments to analyze (all too short)")
        return []

    # Subsample for very long files (>100 segments) — process every 2nd segment
    if len(valid_segments) > 100:
        original_count = len(valid_segments)
        valid_segments = valid_segments[::2]
        logger.info(f"emotion2vec: subsampling {original_count} → {len(valid_segments)} segments (long file)")

    logger.info(f"emotion2vec: processing {len(valid_segments)} segments (direct numpy, no temp files)")
    results = []
    BATCH_SIZE = 40  # Larger batches since no file I/O overhead

    for batch_start in range(0, len(valid_segments), BATCH_SIZE):
        batch = valid_segments[batch_start:batch_start + BATCH_SIZE]

        # Pass numpy arrays directly to FunASR (no temp file I/O)
        # Ensure float32 — emotion2vec model weights are float32, numpy default is float64
        import numpy as np
        chunks = [chunk.astype(np.float32) if chunk.dtype != np.float32 else chunk for _, _, chunk in batch]
        try:
            batch_results = model.generate(
                input=chunks,
                output_dir=None,
                granularity=granularity,
            )

            for idx, (seg_idx, seg, _) in enumerate(batch):
                if idx < len(batch_results):
                    parsed = _parse_emotion_result(
                        batch_results[idx], seg_idx, seg.get("speaker", "unknown")
                    )
                    if parsed:
                        results.append(parsed)
                    if not results and idx == 0:
                        logger.debug(f"emotion2vec raw result: {str(batch_results[idx])[:300]}")

        except Exception as batch_err:
            # Fallback: write temp files for this batch (older FunASR may not accept arrays)
            if batch_start == 0:
                logger.warning(f"Direct numpy failed ({batch_err}), falling back to temp files")
            import soundfile as sf
            temp_paths = []
            try:
                for _, _, chunk in batch:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_path = tmp.name
                    sf.write(tmp_path, chunk, sr)
                    temp_paths.append(tmp_path)

                fallback_results = model.generate(
                    input=temp_paths,
                    output_dir=None,
                    granularity=granularity,
                )
                for idx, (seg_idx, seg, _) in enumerate(batch):
                    if idx < len(fallback_results):
                        parsed = _parse_emotion_result(
                            fallback_results[idx], seg_idx, seg.get("speaker", "unknown")
                        )
                        if parsed:
                            results.append(parsed)
            except Exception as e:
                logger.warning(f"emotion2vec batch {batch_start} failed entirely: {e}")
            finally:
                for tmp_path in temp_paths:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    if results:
        emotion_counts = {}
        for r in results:
            emotion_counts[r.emotion] = emotion_counts.get(r.emotion, 0) + 1
        dist = ", ".join(f"{k}={v}" for k, v in sorted(emotion_counts.items(), key=lambda x: -x[1]))
        logger.info(f"emotion2vec: {len(results)} segments analyzed — {dist}")
    else:
        logger.warning("emotion2vec: no segments could be analyzed")

    return results


def get_emotion_summary(emotions: list[SegmentEmotion]) -> dict:
    """Summarize emotions across the call.

    Uses score-weighted averaging for dominant emotion (not just count).
    30 segments of "happy" at 0.51 won't beat 20 segments of "angry" at 0.95.

    Returns:
        Dict with dominant_emotion, emotion_distribution, escalation_moments
    """
    if not emotions:
        return {
            "dominant_emotion": "neutral",
            "emotion_distribution": {},
            "escalation_moments": [],
        }

    # Score-weighted dominant: average confidence per emotion
    weighted_scores = {}
    weighted_counts = {}
    for e in emotions:
        weighted_scores.setdefault(e.emotion, 0.0)
        weighted_counts.setdefault(e.emotion, 0)
        weighted_scores[e.emotion] += e.emotion_score
        weighted_counts[e.emotion] += 1

    weighted_avg = {k: weighted_scores[k] / weighted_counts[k] for k in weighted_scores}
    dominant = max(weighted_avg, key=weighted_avg.get)

    # Distribution by count (for visualization)
    total = len(emotions)
    distribution = {k: round(v / total, 3) for k, v in weighted_counts.items()}

    # Find escalation moments (angry/fearful with high confidence)
    escalation_moments = []
    for e in emotions:
        if e.emotion in ("angry", "fearful") and e.emotion_score > 0.6:
            escalation_moments.append({
                "segment_id": e.segment_id,
                "speaker": e.speaker,
                "emotion": e.emotion,
                "score": e.emotion_score,
            })

    return {
        "dominant_emotion": dominant,
        "emotion_distribution": distribution,
        "escalation_moments": escalation_moments,
    }


def get_speaker_emotion_breakdown(emotions: list[SegmentEmotion]) -> dict:
    """Split emotions by speaker for agent vs customer separate profiles.

    Returns:
        {
            "agent": {"dominant": "neutral", "distribution": {"neutral": 0.7, ...}, "total_segments": 15},
            "customer": {"dominant": "frustrated", "distribution": {...}, "total_segments": 10},
        }
    """
    if not emotions:
        return {}

    by_speaker: dict[str, list[SegmentEmotion]] = {}
    for e in emotions:
        speaker = e.speaker.lower()
        by_speaker.setdefault(speaker, []).append(e)

    result = {}
    for speaker, speaker_emotions in by_speaker.items():
        # Score-weighted dominant per speaker
        weighted_scores = {}
        weighted_counts = {}
        for e in speaker_emotions:
            weighted_scores.setdefault(e.emotion, 0.0)
            weighted_counts.setdefault(e.emotion, 0)
            weighted_scores[e.emotion] += e.emotion_score
            weighted_counts[e.emotion] += 1

        weighted_avg = {k: weighted_scores[k] / weighted_counts[k] for k in weighted_scores}
        dominant = max(weighted_avg, key=weighted_avg.get)

        total = len(speaker_emotions)
        result[speaker] = {
            "dominant": dominant,
            "distribution": {k: round(v / total, 3) for k, v in weighted_counts.items()},
            "total_segments": total,
        }

    return result
