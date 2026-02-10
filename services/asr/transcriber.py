"""Stage 3: Financial Transcription — WhisperX with VRAM-aware loading/unloading."""

import re
import json
import torch
from pathlib import Path
from loguru import logger

# PyTorch 2.6+ changed weights_only default to True in torch.load().
# pyannote/speechbrain checkpoints use omegaconf + typing objects that aren't
# in the safe list, and lightning_fabric passes weights_only=None (which
# PyTorch 2.8 treats as True). Since these are trusted HuggingFace models,
# we patch torch.load to treat None as False.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if kwargs.get("weights_only") is None:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


# Financial term correction dictionary — data-driven, not guessed.
# Run scripts/discover_corrections.py on Earnings-21 to build the full version.
# These are seed corrections; the discovery script will expand this.
FINANCIAL_CORRECTIONS = {
    "emmy": "EMI", "emi": "EMI",
    "kayak": "KYC", "kayc": "KYC",
    "civil": "CIBIL", "sibyl": "CIBIL", "sybil": "CIBIL",
    "hdfc": "HDFC", "icici": "ICICI", "sbi": "SBI",
    "nach": "NACH", "natch": "NACH",
    "upi": "UPI", "gst": "GST", "pan": "PAN", "tan": "TAN",
    "rbi": "RBI", "sebi": "SEBI", "irdai": "IRDAI",
    "nbfc": "NBFC", "npa": "NPA",
    "crore": "crore", "lakh": "lakh",
    "demat": "demat", "d-mat": "demat",
}

# Amount normalization patterns
AMOUNT_PATTERNS = [
    (r"(\d+)\s*thousand", lambda m: str(int(m.group(1)) * 1000)),
    (r"(\d+)\s*lakh", lambda m: str(int(m.group(1)) * 100000)),
    (r"(\d+)\s*crore", lambda m: str(int(m.group(1)) * 10000000)),
    (r"(\d+)\s*k\b", lambda m: str(int(m.group(1)) * 1000)),
]


def load_correction_dictionary(path: str = "data/models/financial_corrections.json") -> dict:
    """Load data-driven corrections if available, otherwise use seed dict."""
    corrections = dict(FINANCIAL_CORRECTIONS)
    if Path(path).exists():
        with open(path) as f:
            discovered = json.load(f)
        corrections.update(discovered)
        logger.info(f"Loaded {len(discovered)} discovered corrections from {path}")
    return corrections


def transcribe_audio(
    wav_path: str,
    batch_size: int = 4,
    language: str | None = None,
    hf_token: str | None = None,
) -> dict:
    """Transcribe audio with WhisperX, apply financial corrections, flag low confidence.

    IMPORTANT: This function loads WhisperX onto GPU (~3GB), transcribes, then
    explicitly frees VRAM. On 6GB cards, nothing else can use GPU while this runs.

    Args:
        wav_path: Path to normalized 16kHz mono WAV
        batch_size: GPU batch size (use 4 for 6GB VRAM, not 16)
        language: Language code or None for auto-detection
        hf_token: HuggingFace token for pyannote diarization models

    Returns:
        dict with segments, low_confidence_segments, overall_confidence, language
    """
    import whisperx

    # Meta tensor patch is applied globally in app.py at process startup

    logger.info(f"Loading WhisperX (large-v3-turbo, INT8) — expect ~3GB VRAM")
    load_kwargs = {
        "whisper_arch": "large-v3-turbo",
        "device": "cuda",
        "compute_type": "int8",
    }
    if language:
        load_kwargs["language"] = language
    model = whisperx.load_model(**load_kwargs)

    # NOTE: Do NOT restore Module.to yet — alignment and diarization also load
    # models that hit the same meta tensor issue. Restored after all GPU work.

    # Transcribe
    audio = whisperx.load_audio(wav_path)
    result = model.transcribe(audio, batch_size=batch_size)

    # Capture detected language
    detected_language = result.get("language", language or "en")
    logger.info(f"Transcribed {len(result['segments'])} segments (language={detected_language})")

    # P2 FIX: Save per-segment confidence from avg_logprob BEFORE alignment
    # (alignment and diarization may discard word-level scores)
    _saved_segment_conf = {}
    _saved_word_scores = {}
    for seg in result.get("segments", []):
        logprob = seg.get("avg_logprob", -0.5)
        # Map avg_logprob (-1.0=low, 0.0=perfect) to 0-1 confidence
        conf = max(0.0, min(1.0, 1.0 + logprob))
        _saved_segment_conf[round(seg.get("start", 0), 1)] = conf
        for w in seg.get("words", []):
            score = w.get("score", w.get("conf"))
            if score is not None:
                key = (w.get("word", "").strip().lower(), round(w.get("start", 0), 2))
                _saved_word_scores[key] = score

    # Word-level alignment
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_language, device="cuda"
        )
        result = whisperx.align(
            result["segments"], align_model, align_metadata, audio, device="cuda"
        )
        del align_model
    except Exception as e:
        logger.warning(f"Word alignment failed (continuing without): {e}")

    # Speaker diarization
    if hf_token:
        try:
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(
                use_auth_token=hf_token, device="cuda"
            )
            diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=6)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            # P1 FIX: Split segments at speaker boundaries
            result["segments"] = _split_segments_by_speaker(result.get("segments", []))
            del diarize_model
            logger.info("Speaker diarization complete")
        except Exception as e:
            logger.warning(f"Diarization failed (continuing without): {e}")
    else:
        logger.warning("No HF token — skipping diarization")

    # FREE VRAM — critical for 6GB cards
    del model
    torch.cuda.empty_cache()
    logger.info(
        f"WhisperX unloaded. VRAM freed: "
        f"{torch.cuda.memory_allocated() / 1024**2:.0f} MB allocated"
    )

    # Apply financial term corrections
    corrections = load_correction_dictionary()
    segments = _apply_corrections(result.get("segments", []), corrections)

    # Flag low-confidence segments (P2 FIX: use saved scores if word-level lost)
    low_confidence = []
    confidences = []
    for i, seg in enumerate(segments):
        words = seg.get("words", [])
        if words:
            # Try word-level scores first
            word_scores = []
            for w in words:
                s = w.get("score", w.get("conf"))
                if s is not None:
                    word_scores.append(s)
                else:
                    # Look up from pre-alignment saved scores
                    key = (w.get("word", "").strip().lower(), round(w.get("start", 0), 2))
                    saved = _saved_word_scores.get(key)
                    if saved is not None:
                        word_scores.append(saved)
                        w["score"] = saved  # reattach for downstream use

            if word_scores:
                avg_conf = sum(word_scores) / len(word_scores)
            else:
                # Fall back to segment-level logprob confidence
                seg_start = round(seg.get("start", 0), 1)
                avg_conf = _saved_segment_conf.get(seg_start, 0.5)
        else:
            seg_start = round(seg.get("start", 0), 1)
            avg_conf = _saved_segment_conf.get(seg_start, 0.5)
        seg["confidence"] = round(avg_conf, 3)
        confidences.append(avg_conf)
        if avg_conf < 0.7:
            low_confidence.append(i)
            seg["flagged"] = True
        else:
            seg["flagged"] = False

    overall_confidence = sum(confidences) / max(len(confidences), 1)

    return {
        "segments": segments,
        "low_confidence_segments": low_confidence,
        "overall_confidence": round(overall_confidence, 3),
        "num_segments": len(segments),
        "language": detected_language,
    }


def _split_segments_by_speaker(segments: list) -> list:
    """Split segments that contain multiple speakers into sub-segments.

    After whisperx.assign_word_speakers(), each word has a 'speaker' field.
    WhisperX only assigns one speaker per SEGMENT (the majority), but the
    word-level labels reveal mid-segment speaker changes. This function
    splits those segments so each sub-segment has exactly one speaker.
    """
    new_segments = []
    for seg in segments:
        words = seg.get("words", [])
        if not words:
            new_segments.append(seg)
            continue

        # Check if multiple speakers exist in this segment
        speakers_in_seg = set()
        for w in words:
            sp = w.get("speaker")
            if sp:
                speakers_in_seg.add(sp)

        if len(speakers_in_seg) <= 1:
            new_segments.append(seg)
            continue

        # Multiple speakers — split into sub-segments at speaker boundaries
        current_speaker = None
        current_words = []

        for w in words:
            word_speaker = w.get("speaker", current_speaker)

            if current_speaker is None:
                current_speaker = word_speaker

            if word_speaker != current_speaker and word_speaker is not None:
                # Speaker changed — emit current sub-segment
                if current_words:
                    new_segments.append(_words_to_segment(current_words, current_speaker, seg))
                current_words = [w]
                current_speaker = word_speaker
            else:
                current_words.append(w)

        # Emit final sub-segment
        if current_words:
            new_segments.append(_words_to_segment(current_words, current_speaker, seg))

    split_count = len(new_segments) - len(segments)
    if split_count > 0:
        logger.info(f"Diarization split: {len(segments)} segments → {len(new_segments)} (+{split_count} from speaker boundaries)")

    return new_segments


def _words_to_segment(words: list, speaker: str, original_seg: dict) -> dict:
    """Create a new segment dict from a group of words with the same speaker."""
    text = " ".join(w.get("word", "") for w in words).strip()
    start = words[0].get("start", original_seg.get("start", 0))
    end = words[-1].get("end", original_seg.get("end", 0))
    return {
        "start": start,
        "end": end,
        "text": text,
        "speaker": speaker,
        "words": words,
    }


def _apply_corrections(segments: list, corrections: dict) -> list:
    """Apply financial term corrections to transcript segments."""
    for seg in segments:
        text = seg.get("text", "")
        original = text

        # Term corrections (case-insensitive)
        for wrong, right in corrections.items():
            text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)

        # Amount normalization
        for pattern, replacement in AMOUNT_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Currency symbol normalization
        text = re.sub(r"\b(rupees?|rs\.?)\b", "₹", text, flags=re.IGNORECASE)

        seg["text"] = text
        if text != original:
            seg["corrected"] = True
            seg["original_text"] = original
        else:
            seg["corrected"] = False

    return segments
