"""Pipeline Orchestrator — coordinates all 5 stages with VRAM lifecycle management.

On 6GB VRAM, the orchestrator enforces strict sequential GPU usage:
  Stage 1-2: CPU only (no GPU needed)
  Stage 3: WhisperX loads → transcribes → unloads (~3GB VRAM)
  Stage 4: CPU extraction first (FinBERT, regex, spaCy, compliance, fraud)
            Then Ollama/Qwen3 loads → extracts → unloads (~5GB VRAM)
  Stage 5: CPU only (export)

No two GPU-heavy stages ever overlap.
"""

import os
import re
import json
import uuid
import time
import torch
from pathlib import Path
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from services.audio.normalizer import normalize_audio
from services.audio.quality import assess_audio_quality
from services.audio.cleanup import cleanup_audio
from services.asr.transcriber import transcribe_audio, unload_whisperx, reload_whisperx
from services.llm.client import extract_structured, extract_raw, unload_ollama_model
from analysis.intelligence import extract_all_entities_layer1
from analysis.compliance import run_compliance_checks
from analysis.fraud_detection import run_fraud_detection, refine_fraud_with_emotions
from analysis.pii_detection import detect_pii
from analysis.profanity import detect_profanity
from analysis.tamper_detection import run_tamper_detection
from analysis.sentiment import (
    compute_sentiment_trajectories,
    interpret_sentiment_context,
    classify_for_llm_routing,
    classify_intent,
    has_intent_model,
    get_dominant_emotion,
)
from config.schemas import (
    CallRecord, UtteranceIntent, FinancialEntity, Obligation,
    ComplianceCheck, FraudSignal, CallIntent, RiskLevel,
)
from services.backboard.client import (
    is_configured as backboard_configured,
    store_call_record_sync,
)

def _write_progress(call_id: str, output_dir: str, stage: str, stage_name: str,
                     completed: list[str], status: str = "processing", error: str | None = None,
                     stage_times: dict | None = None, pipeline_start: float | None = None,
                     extra: dict | None = None):
    """Write pipeline progress to a JSON file for frontend polling."""
    progress = {
        "call_id": call_id,
        "status": status,
        "current_stage": stage,
        "current_stage_name": stage_name,
        "stages_completed": completed,
        "timestamp": time.time(),
        "error": error,
    }
    if stage_times:
        progress["stage_times"] = stage_times
    if pipeline_start:
        progress["elapsed"] = round(time.time() - pipeline_start, 1)
    if extra:
        progress.update(extra)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    progress_path = f"{output_dir}/{call_id}_progress.json"
    with open(progress_path, "w") as f:
        json.dump(progress, f, default=lambda o: float(o) if hasattr(o, 'item') else str(o))


# Language name map for LLM prompt injection
LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil",
    "ru": "Russian", "pl": "Polish", "fr": "French",
    "de": "German", "es": "Spanish", "pt": "Portuguese",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "ar": "Arabic", "bn": "Bengali", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu",
}


def _get_llm_model(lang: str) -> str:
    """Select LLM model based on detected language.

    English → qwen2.5:3b (faster, lighter, sufficient since FinBERT handles most work).
    Non-English → qwen3:8b (stronger multilingual, better code-switching understanding).
    """
    if lang == "en":
        return "qwen2.5:3b"
    return "qwen3:8b"


# Few-shot examples for Hindi/Tamil code-switched financial text
_HINDI_FEW_SHOT = (
    "\nExamples of Hindi/code-switched financial text:\n"
    "  [customer]: EMI due date 15 March hai, payment UPI se karna hoga → intent: info_request\n"
    "  [customer]: Haan ji, main Friday tak 5000 rupaye jama kar dunga → intent: payment_promise\n"
    "  [customer]: Mujhe yeh loan nahi chahiye, cancel kar dijiye → intent: refusal\n"
    "  [agent]: Aapka outstanding balance 25,000 hai, aur late fee 500 lagi hai → intent: info_request\n"
    "  [customer]: Mujhe manager se baat karni hai → intent: escalation\n"
)

_TAMIL_FEW_SHOT = (
    "\nExamples of Tamil/code-switched financial text:\n"
    "  [customer]: EMI amount evvalavu? Next due date enna? → intent: info_request\n"
    "  [customer]: Sari, naan Friday kulla pay panren → intent: payment_promise\n"
    "  [customer]: Enakku indha loan venaam, cancel pannunga → intent: refusal\n"
    "  [agent]: Ungal outstanding balance 25,000 rupees, late fee 500 → intent: info_request\n"
    "  [customer]: Manager kitta pesunum → intent: escalation\n"
)

_HINDI_ENTITY_FEW_SHOT = (
    "\nExamples of Hindi/code-switched entities:\n"
    "  'paanch hazaar ki EMI' → ENTITY|currency_amount|5000|paanch hazaar ki EMI\n"
    "  'pandrah March tak' → ENTITY|date|March 15|pandrah March tak\n"
    "  'home loan ka interest rate' → ENTITY|product_name|home_loan|home loan ka interest rate\n"
)

_TAMIL_ENTITY_FEW_SHOT = (
    "\nExamples of Tamil/code-switched entities:\n"
    "  'aimbadhu aayiram EMI' → ENTITY|currency_amount|50000|aimbadhu aayiram EMI\n"
    "  'March pathinarndhu kulla' → ENTITY|date|March 15|March pathinarndhu kulla\n"
    "  'home loan interest rate evvalavu' → ENTITY|product_name|home_loan|home loan interest rate\n"
)


def _get_few_shot_examples(lang: str, task: str = "intent") -> str:
    """Get language-specific few-shot examples for LLM prompts."""
    if lang == "en":
        return ""
    if task == "intent":
        if lang == "hi":
            return _HINDI_FEW_SHOT
        elif lang == "ta":
            return _TAMIL_FEW_SHOT
        # For other non-English languages, use Hindi examples as closest reference
        return _HINDI_FEW_SHOT
    elif task == "entity":
        if lang == "hi":
            return _HINDI_ENTITY_FEW_SHOT
        elif lang == "ta":
            return _TAMIL_ENTITY_FEW_SHOT
        return _HINDI_ENTITY_FEW_SHOT
    return ""


def process_call(
    input_audio_path: str,
    call_type: str = "general",
    hf_token: str | None = None,
    output_dir: str = "data/processed",
    call_id: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> CallRecord:
    """Process a single call through all 5 pipeline stages.

    Args:
        input_audio_path: Path to raw audio file (any format)
        call_type: One of 'collections', 'kyc', 'onboarding', 'complaint', 'consent', 'general'
        hf_token: HuggingFace token for pyannote diarization
        output_dir: Directory for output files
        call_id: Unique identifier for this call (generated if not provided)
        start_time: Optional start time in seconds (trim before processing)
        end_time: Optional end time in seconds (trim before processing)

    Returns:
        Complete CallRecord with all extracted data
    """
    call_id = call_id or str(uuid.uuid4())[:8]
    hf_token = hf_token or os.getenv("HF_TOKEN")
    logger.info(f"[{call_id}] Starting pipeline for {input_audio_path}")
    completed: list[str] = []
    stage_times: dict[str, float] = {}
    pipeline_start = time.perf_counter()

    _accumulated_extra: dict = {}
    pipeline_start_wall = time.time()

    def _progress(stage: str, name: str, status: str = "processing", error: str | None = None, extra: dict | None = None):
        if extra:
            _accumulated_extra.update(extra)
        _write_progress(call_id, output_dir, stage, name, completed, status, error,
                        stage_times=stage_times, pipeline_start=pipeline_start_wall,
                        extra=_accumulated_extra if _accumulated_extra else None)

    def _stage_timer(stage_name: str):
        """Log and record time for current stage, start next."""
        now = time.perf_counter()
        if hasattr(_stage_timer, '_last'):
            elapsed = now - _stage_timer._last
            stage_times[_stage_timer._name] = round(elapsed, 1)
            logger.info(f"[{call_id}] ⏱ {_stage_timer._name}: {elapsed:.1f}s")
        _stage_timer._last = now
        _stage_timer._name = stage_name

    _progress("1", "Normalizing audio")
    _stage_timer("Stage 1: Normalize")

    # ── STAGE 1: RAW AUDIO INTAKE (CPU) ──
    logger.info(f"[{call_id}] Stage 1: Normalizing audio")
    normalized_path = f"data/transcripts/{call_id}_normalized.wav"
    audio_meta = normalize_audio(input_audio_path, normalized_path, start_time=start_time, end_time=end_time)
    wav_path = audio_meta["output_path"]

    completed.append("1")
    _stage_timer("Stage 2: Quality")
    _progress("2", "Assessing audio quality")

    # ── STAGE 2: CLEAN & ANALYZE AUDIO (CPU) ──
    logger.info(f"[{call_id}] Stage 2: Assessing audio quality")
    quality = assess_audio_quality(wav_path)

    completed.append("2")
    _stage_timer("Stage 2.5: Cleanup")
    _progress("2.5", "Audio cleanup (dead air, hold music)", extra={
        "audio_quality_score": quality.get("quality_score", 0),
        "audio_quality_flag": quality.get("quality_flag", "UNKNOWN"),
        "audio_snr_db": round(quality.get("snr_db", 0), 1),
        "audio_speech_pct": round(quality.get("speech_percentage", 0), 1),
        "audio_duration": round(quality.get("duration", 0), 1),
        "audio_quality_components": quality.get("component_scores", {}),
    })

    # ── STAGE 2.5: AUDIO CLEANUP (CPU) ──
    logger.info(f"[{call_id}] Stage 2.5: Audio cleanup")
    cleanup_path = f"data/transcripts/{call_id}_cleaned.wav"
    cleanup_result = cleanup_audio(
        wav_path,
        cleanup_path,
        speech_timestamps=quality.get("speech_timestamps"),
    )
    if cleanup_result.get("cleanup_applied"):
        wav_path = cleanup_result["output_path"]
        logger.info(
            f"[{call_id}] Cleanup applied: {cleanup_result['original_duration']:.1f}s → "
            f"{cleanup_result['cleaned_duration']:.1f}s"
        )

    completed.append("2.5")
    _stage_timer("Stage 3: WhisperX")
    _progress("3", "Transcribing with WhisperX")

    # ── STAGE 3: FINANCIAL TRANSCRIPTION (GPU — ~3GB VRAM) ──
    # WhisperX is pre-loaded at server startup — transcription starts instantly.
    logger.info(f"[{call_id}] Stage 3: Transcribing with WhisperX")
    transcript = transcribe_audio(
        wav_path,
        batch_size=8,  # INT8 on 6GB VRAM handles batch_size=8 fine
        language=None,  # Auto-detect language
        hf_token=hf_token,
    )
    # Unload WhisperX to free VRAM for emotion2vec + Ollama
    unload_whisperx()
    logger.info(f"[{call_id}] WhisperX unloaded — VRAM free for analysis stages")

    segments = transcript["segments"]
    detected_lang = transcript.get("language", "en")
    logger.info(f"[{call_id}] Transcribed {len(segments)} segments (language={detected_lang})")

    # Auto-detect call type from transcript if user didn't specify
    if call_type == "general":
        detected_type = _auto_detect_call_type(segments)
        if detected_type != "general":
            logger.info(f"[{call_id}] Auto-detected call type: {detected_type} (was 'general')")
            call_type = detected_type

    # Map speaker roles (SPEAKER_00 → agent, SPEAKER_01 → customer)
    segments = _map_speaker_roles(segments, call_type)

    # Smart speaker identification — match speaker IDs to real names from transcript
    segments = _identify_speakers_by_name(segments)

    # Per-segment language tagging (detects code-switching)
    from analysis.language_tagging import tag_segment_languages
    lang_metadata = tag_segment_languages(segments, fallback_lang=detected_lang)
    has_code_switching = lang_metadata.get("has_code_switching", False)
    if has_code_switching:
        logger.info(f"[{call_id}] Code-switching detected: {lang_metadata['language_distribution']}")

    completed.append("3")
    _stage_timer("Stage 4: Analysis (parallel)")
    _progress("4A", "CPU + GPU analysis (parallel)", extra={
        "transcript_segments": len(segments),
        "transcript_language": detected_lang,
        "has_code_switching": has_code_switching,
        "language_distribution": lang_metadata.get("language_distribution", {}),
    })

    # ── STAGE 4: UNDERSTAND FINANCIAL MEANING ──
    # Architecture: CPU stages + emotion2vec (GPU, ~1GB) run simultaneously.
    # emotion2vec doesn't need CPU stage results, and CPU stages don't need GPU.
    # After both finish, Ollama loads for LLM extraction.
    logger.info(f"[{call_id}] Stage 4: Extracting intelligence (parallel CPU + GPU emotion2vec)")

    # Pre-load audio once for all stages that need it (avoids 7+ redundant disk reads)
    import soundfile as sf
    audio_array, audio_sr = sf.read(wav_path)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    # Start Ollama model preload in background
    from services.llm.client import preload_ollama_model
    ollama_preload_future = None

    def _preload_ollama():
        llm_model = _get_llm_model(detected_lang)
        preload_ollama_model(llm_model, keep_alive="5m")

    # emotion2vec wrapper — runs on GPU thread, unloads when done
    def _run_emotion2vec():
        from services.emotion.emotion2vec_analyzer import analyze_emotions, get_emotion_summary, unload_model as unload_emotion2vec
        emotions = analyze_emotions(wav_path, segments, audio_array=audio_array, sample_rate=audio_sr)
        summary = get_emotion_summary(emotions)
        # Free emotion2vec VRAM immediately so Ollama can preload
        unload_emotion2vec()
        torch.cuda.empty_cache()
        return emotions, summary

    # ── PARALLEL: CPU stages (4A-4G) + GPU emotion2vec (4H) + Ollama preload ──
    with ThreadPoolExecutor(max_workers=10) as pool:
        logger.info(f"[{call_id}] Running stages 4A-4H in parallel (CPU + GPU)")

        # CPU stages
        f_sentiment = pool.submit(compute_sentiment_trajectories, segments, detected_lang)
        f_entities = pool.submit(extract_all_entities_layer1, segments, detected_lang)
        f_compliance = pool.submit(run_compliance_checks, segments, call_type, detected_lang)
        f_fraud = pool.submit(run_fraud_detection, wav_path, segments)
        f_pii = pool.submit(detect_pii, segments, 0.5, detected_lang)
        f_profanity = pool.submit(detect_profanity, segments)
        f_tamper = pool.submit(run_tamper_detection, wav_path, source_codec=audio_meta.get("original_format", ""))

        # GPU stage — runs simultaneously since CPU stages don't use GPU
        f_emotion = pool.submit(_run_emotion2vec)

        # Preload Ollama AFTER emotion2vec finishes — they share GPU VRAM.
        # Use a callback on the emotion2vec future to trigger the preload.
        def _chain_ollama_preload(emotion_future):
            try:
                emotion_future.result()  # wait for emotion2vec to finish + unload
            except Exception:
                pass  # emotion2vec failure shouldn't block LLM
            _preload_ollama()

        ollama_preload_future = pool.submit(_chain_ollama_preload, f_emotion)

        # Collect results — each stage is independent
        try:
            customer_sentiment, agent_sentiment = f_sentiment.result()
            customer_emotion = get_dominant_emotion(customer_sentiment)
            sentiment_context = interpret_sentiment_context(
                customer_sentiment, agent_sentiment, call_type
            )
        except Exception as e:
            logger.error(f"[{call_id}] Sentiment failed: {e}")
            customer_sentiment, agent_sentiment = [], []
            customer_emotion = "neutral"
            sentiment_context = {}

        try:
            layer1_entities = f_entities.result()
        except Exception as e:
            logger.error(f"[{call_id}] Entity extraction failed: {e}")
            layer1_entities = []

        try:
            compliance_checks = f_compliance.result()
        except Exception as e:
            logger.error(f"[{call_id}] Compliance checks failed: {e}")
            compliance_checks = []

        try:
            fraud_signals = f_fraud.result()
        except Exception as e:
            logger.error(f"[{call_id}] Fraud detection failed: {e}")
            fraud_signals = []

        try:
            pii_entities = f_pii.result()
        except Exception as e:
            logger.error(f"[{call_id}] PII detection failed: {e}")
            pii_entities = []

        try:
            toxicity_flags = f_profanity.result()
        except Exception as e:
            logger.error(f"[{call_id}] Profanity detection failed: {e}")
            toxicity_flags = []

        try:
            tamper_signals = f_tamper.result()
        except Exception as e:
            logger.error(f"[{call_id}] Tamper detection failed: {e}")
            tamper_signals = []

        # Collect emotion2vec results (GPU, ran in parallel with CPU stages)
        segment_emotions = []
        emotion_distribution = {}
        speaker_emotion_breakdown = {}
        emotion_transitions = []
        try:
            segment_emotions, emotion_summary = f_emotion.result()
            emotion_distribution = emotion_summary.get("emotion_distribution", {})
            # Per-speaker emotion breakdown + transition detection
            if segment_emotions:
                from services.emotion.emotion2vec_analyzer import get_speaker_emotion_breakdown, detect_emotion_transitions
                speaker_emotion_breakdown = get_speaker_emotion_breakdown(segment_emotions)
                emotion_transitions = detect_emotion_transitions(segment_emotions)
        except Exception as e:
            logger.warning(f"[{call_id}] emotion2vec failed (continuing without): {e}")

    logger.info(f"[{call_id}] Parallel stages complete (CPU + GPU)")
    completed.extend(["4A", "4B", "4C", "4D", "4E", "4F", "4G", "4H"])

    # Free pre-loaded audio array
    del audio_array

    # 4H.5: Refine fraud signals with emotion2vec data (CPU, no VRAM)
    if segment_emotions:
        logger.info(f"[{call_id}] Stage 4H.5: Fraud + emotion convergence")
        fraud_signals = refine_fraud_with_emotions(fraud_signals, segment_emotions, segments)

    _stage_timer("Stage 4I: LLM extraction")
    _progress("4I", "LLM intelligence extraction")

    # Wait for Ollama preload to finish
    if ollama_preload_future:
        try:
            ollama_preload_future.result(timeout=30)
        except Exception:
            pass

    # Ensure GPU memory is clear for Ollama
    torch.cuda.empty_cache()

    # 4I: GPU — LLM extraction via Ollama
    logger.info(f"[{call_id}] Stage 4I: LLM intelligence extraction")

    # Intent classification (FinBERT pre-filter — often skips Ollama entirely)
    try:
        intents = _classify_intents_with_prefilter(segments, lang=detected_lang, call_type=call_type)
    except Exception as e:
        logger.error(f"[{call_id}] Intent classification failed: {e}")
        intents = []

    # Run 3 LLM calls in parallel (model selected by language)
    from concurrent.futures import ThreadPoolExecutor as _LLMPool
    with _LLMPool(max_workers=3) as llm_pool:
        f_ent = llm_pool.submit(_extract_entities_llm, segments, detected_lang)
        f_obl = llm_pool.submit(_detect_obligations, segments, detected_lang)
        f_sum = llm_pool.submit(_generate_summary, segments, call_type, detected_lang)

    try:
        llm_entities = f_ent.result(timeout=90)
    except Exception as e:
        logger.warning(f"[{call_id}] LLM entity extraction failed: {e}")
        llm_entities = []
    try:
        obligations = f_obl.result(timeout=90)
    except Exception as e:
        logger.warning(f"[{call_id}] Obligation detection failed: {e}")
        obligations = []
    try:
        call_summary, key_outcomes, next_actions = f_sum.result(timeout=90)
    except Exception as e:
        logger.warning(f"[{call_id}] Summary generation failed: {e}")
        call_summary = "Summary generation failed — see extracted fields."
        key_outcomes, next_actions = ["See analysis fields"], ["Review if needed"]

    # Merge Layer 1 + Layer 2 entities
    all_entities = _merge_entities(layer1_entities, llm_entities)

    # Free Ollama VRAM, then reload WhisperX for next call
    try:
        unload_ollama_model(_get_llm_model(detected_lang))
    except Exception:
        pass

    # Reload WhisperX into GPU — ready for the next call instantly.
    # This runs in a background thread so it doesn't block Stage 5 (CPU-only).
    _reload_future = None
    with ThreadPoolExecutor(max_workers=1) as reload_pool:
        _reload_future = reload_pool.submit(reload_whisperx)

    completed.append("4I")
    _stage_timer("Stage 5: Output")
    _progress("5", "Generating output")

    # ── STAGE 5: PRODUCE OUTPUT (CPU) ──
    logger.info(f"[{call_id}] Stage 5: Generating output")

    # Compute speaker talk percentages (handles both agent/customer and Speaker A/B labels)
    agent_segs = [s for s in segments if s.get("speaker", "").lower() in ("agent", "speaker_00", "speaker a")]
    customer_segs = [s for s in segments if s.get("speaker", "").lower() in ("customer", "speaker_01", "speaker b")]
    total_dur = max(audio_meta["original_duration"], 0.01)
    agent_dur = sum(s.get("end", 0) - s.get("start", 0) for s in agent_segs)
    customer_dur = sum(s.get("end", 0) - s.get("start", 0) for s in customer_segs)

    # Risk assessment — weighted by call type
    num_violations = len([c for c in compliance_checks if not c.passed])
    num_fraud = len(fraud_signals)

    # Call-type weighting: regulated calls escalate faster, general calls de-escalate
    call_type_weights = {
        "collections": 1.5, "kyc": 2.0, "consent": 1.5,
        "complaint": 1.2, "onboarding": 1.0, "general": 0.3,
    }
    weight = call_type_weights.get(call_type, 1.0)
    weighted_violations = num_violations * weight
    weighted_fraud = num_fraud * weight

    has_critical = any(
        c.severity == "critical" for c in compliance_checks if not c.passed
    )

    if (has_critical and weight >= 1.0) or weighted_violations >= 3 or weighted_fraud >= 2:
        risk_level = RiskLevel.CRITICAL
    elif weighted_violations >= 2 or weighted_fraud >= 1:
        risk_level = RiskLevel.HIGH
    elif weighted_violations >= 1:
        risk_level = RiskLevel.MEDIUM
    else:
        risk_level = RiskLevel.LOW

    # Compliance score: penalty per violation, scaled by call type weight
    violation_penalty = int(15 * weight)
    compliance_score = max(0, 100 - (num_violations * violation_penalty))

    # Escalation detection
    escalation = any(
        i.intent == CallIntent.ESCALATION for i in intents
    ) or customer_emotion in ("angry", "frustrated")

    # Tamper risk assessment — require corroboration across signal types
    high_tamper = sum(1 for t in tamper_signals if t.severity == "high")
    tamper_signal_types = set(t.signal_type for t in tamper_signals)
    num_types = len(tamper_signal_types)
    if num_types >= 3 and high_tamper >= 2:
        tamper_risk = "high"
    elif num_types >= 2 and (high_tamper >= 1 or len(tamper_signals) >= 4):
        tamper_risk = "medium"
    elif tamper_signals:
        tamper_risk = "low"
    else:
        tamper_risk = "none"

    # Review reasons
    review_reasons = [c.check_name for c in compliance_checks if not c.passed]
    if fraud_signals:
        review_reasons.extend([f.signal_type for f in fraud_signals])
    if transcript["low_confidence_segments"]:
        review_reasons.append("low_confidence_transcript")
    if toxicity_flags:
        agent_toxic = [f for f in toxicity_flags if f.is_agent]
        if agent_toxic:
            review_reasons.append("agent_toxicity")
    if tamper_risk in ("medium", "high"):
        review_reasons.append("tamper_risk")
    if pii_entities:
        review_reasons.append(f"pii_detected_{len(pii_entities)}")

    # Unique speakers count
    unique_speakers = set()
    for seg in segments:
        sp = seg.get("speaker")
        if sp:
            unique_speakers.add(sp)

    # Build clean transcript segments for output
    transcript_segments = []
    for i, seg in enumerate(segments):
        seg_out = {
            "id": i,
            "start": round(seg.get("start", 0), 2),
            "end": round(seg.get("end", 0), 2),
            "text": seg.get("text", ""),
            "speaker": seg.get("speaker", "unknown"),
            "confidence": seg.get("confidence", 0.5),
        }
        # Include per-word timestamps from WhisperX alignment
        if seg.get("words"):
            seg_out["words"] = [
                {
                    "word": w.get("word", ""),
                    "start": round(w.get("start", 0), 3),
                    "end": round(w.get("end", 0), 3),
                    "score": round(w.get("score", 0), 3) if w.get("score") is not None else None,
                }
                for w in seg["words"]
            ]
        transcript_segments.append(seg_out)

    # Use language detected earlier (line ~89)
    detected_language = detected_lang

    call_record = CallRecord(
        call_id=call_id,
        audio_file=input_audio_path,
        duration_seconds=audio_meta["original_duration"],
        language=detected_language,
        call_type=call_type,
        audio_quality_score=quality["quality_score"],
        audio_quality_flag=quality["quality_flag"],
        snr_db=quality["snr_db"],
        audio_quality_components=quality.get("component_scores", {}),
        speech_percentage=quality.get("speech_percentage", 0.0),
        num_speakers=max(len(unique_speakers), 1),
        agent_talk_percentage=round(agent_dur / total_dur * 100, 1),
        customer_talk_percentage=round(customer_dur / total_dur * 100, 1),
        transcript_segments=transcript_segments,
        overall_transcript_confidence=transcript["overall_confidence"],
        num_low_confidence_segments=len(transcript["low_confidence_segments"]),
        intents=intents,
        financial_entities=all_entities,
        obligations=obligations,
        compliance_checks=compliance_checks,
        fraud_signals=fraud_signals,
        pii_entities=[e.model_dump() for e in pii_entities],
        pii_count=len(pii_entities),
        toxicity_flags=[f.model_dump() for f in toxicity_flags],
        tamper_signals=[t.model_dump() for t in tamper_signals],
        tamper_risk=tamper_risk,
        cleanup_metadata=cleanup_result,
        segment_emotions=[e.model_dump() for e in segment_emotions],
        emotion_distribution=emotion_distribution,
        speaker_emotion_breakdown=speaker_emotion_breakdown,
        emotion_transitions=emotion_transitions,
        pipeline_timings=stage_times,
        detected_language=detected_language,
        has_code_switching=has_code_switching,
        language_distribution=lang_metadata.get("language_distribution", {}),
        customer_sentiment_trajectory=customer_sentiment,
        agent_sentiment_trajectory=agent_sentiment,
        customer_emotion_dominant=customer_emotion,
        sentiment_context=sentiment_context,
        escalation_detected=escalation,
        overall_risk_level=risk_level,
        compliance_score=compliance_score,
        requires_human_review=risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL),
        review_priority=_compute_review_priority(risk_level, num_fraud, num_violations),
        review_reasons=review_reasons,
        call_summary=call_summary,
        key_outcomes=key_outcomes,
        next_actions=next_actions,
    )

    # Save output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = f"{output_dir}/{call_id}_record.json"
    with open(output_path, "w") as f:
        json.dump(call_record.model_dump(), f, indent=2, default=str)
    logger.info(f"[{call_id}] Output saved to {output_path}")
    completed.append("5")
    _stage_timer("done")

    # Log timing summary
    total_elapsed = time.perf_counter() - pipeline_start
    timing_str = " | ".join(f"{k}: {v}s" for k, v in stage_times.items())
    logger.info(f"[{call_id}] Pipeline complete in {total_elapsed:.0f}s — {timing_str}")
    _progress("done", "Complete", status="complete")

    # Wait for WhisperX reload to finish (runs in background during Stage 5)
    if _reload_future:
        try:
            _reload_future.result(timeout=180)
            logger.info(f"[{call_id}] WhisperX reloaded — ready for next call")
        except Exception as e:
            logger.warning(f"[{call_id}] WhisperX reload failed: {e}")

    # ── BACKBOARD: Audit trail + customer memory (non-blocking) ──
    if backboard_configured():
        logger.info(f"[{call_id}] Storing in Backboard audit trail")
        store_call_record_sync(call_record.model_dump())

    return call_record


def _classify_intents_with_prefilter(segments: list, lang: str = "en", call_type: str = "general") -> list[UtteranceIntent]:
    """Classify intents using FinBERT pre-filter + raw LLM call (no Instructor).

    ~40% of utterances skip the LLM entirely (greetings, acknowledgments).
    Remaining utterances are batched into a SINGLE raw LLM call with line-based
    output parsing — no Instructor, no JSON mode, no retry loops.

    Call-type context shapes the prompt: general calls bias toward info_request,
    collections toward payment_promise/refusal/dispute.
    """
    VALID_INTENTS = {
        "agreement", "refusal", "request_extension", "payment_promise",
        "complaint", "consent_given", "consent_denied", "info_request",
        "negotiation", "escalation", "dispute", "acknowledgment", "greeting",
        "unknown",
    }

    intents = []
    llm_queue = []  # (segment_index, text, speaker, route)

    # Phase 1: Pre-filter — only skip trivial/greeting utterances.
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if not text:
            continue

        route = classify_for_llm_routing(text, lang=lang)

        if route == "skip":
            lower = text.lower().strip().rstrip(".")
            from analysis.vocab_loader import get_greetings
            greetings = get_greetings(lang)
            if lower in greetings:
                intent = CallIntent.GREETING
            else:
                intent = CallIntent.ACKNOWLEDGMENT
            intents.append(UtteranceIntent(
                segment_id=i,
                speaker=seg.get("speaker", "unknown"),
                intent=intent,
                confidence=0.90,
            ))
        else:
            llm_queue.append((i, text, seg.get("speaker", "unknown"), route))

    logger.info(f"Intent pre-filter: {len(intents)} trivial, {len(llm_queue)} → LLM")

    if not llm_queue:
        return intents

    # Phase 2: Build compressed conversation context
    full_conversation = []
    prev_speaker = None
    merged_text = ""
    merged_seg_start = 0
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "unknown")
        if not text:
            continue
        if speaker == prev_speaker and merged_text:
            merged_text += " " + text
        else:
            if merged_text:
                full_conversation.append(f"  [{prev_speaker}] (seg {merged_seg_start}): {merged_text}")
            merged_text = text
            merged_seg_start = i
            prev_speaker = speaker
    if merged_text:
        full_conversation.append(f"  [{prev_speaker}] (seg {merged_seg_start}): {merged_text}")

    lang_name = LANGUAGE_NAMES.get(lang, lang)
    lang_context = f"The call transcript is in {lang_name}. Respond in English.\n" if lang != "en" else ""

    conv_text = "\n".join(full_conversation)
    if len(conv_text) > 6000:
        conv_text = conv_text[:6000] + "\n... (truncated)"

    # Batch LLM calls in groups of 50 utterances
    LLM_BATCH = 50
    total_classified = 0

    for batch_start in range(0, len(llm_queue), LLM_BATCH):
        batch = llm_queue[batch_start:batch_start + LLM_BATCH]
        classify_list = []
        for idx, (seg_i, text, speaker, route) in enumerate(batch):
            classify_list.append(f"{idx}. [{speaker}]: {text}")

        # Call-type context shapes which intents are most likely
        if call_type == "collections":
            type_hint = (
                "This is a COLLECTIONS call (debt recovery). "
                "Common intents: payment_promise, refusal, request_extension, dispute, complaint, negotiation."
            )
        elif call_type in ("kyc", "onboarding", "consent"):
            type_hint = (
                f"This is a {call_type.upper()} call. "
                "Common intents: consent_given, consent_denied, info_request, agreement."
            )
        elif call_type == "complaint":
            type_hint = (
                "This is a COMPLAINT call. "
                "Common intents: complaint, escalation, dispute, info_request."
            )
        else:
            type_hint = (
                "This is a GENERAL financial call (earnings, advisory, or informational). "
                "Common intents: info_request, acknowledgment, agreement. "
                "Avoid labeling financial discussion as dispute/complaint unless clearly adversarial."
            )

        prompt = (
            f"Classify the intent of each utterance in a financial phone call.\n"
            f"{lang_context}"
            f"{type_hint}\n\n"
            f"CONVERSATION CONTEXT:\n{conv_text}\n\n"
            f"CLASSIFY THESE UTTERANCES:\n"
            + "\n".join(classify_list) + "\n\n"
            f"Valid intents: agreement, refusal, request_extension, payment_promise, "
            f"complaint, consent_given, consent_denied, info_request, negotiation, "
            f"escalation, dispute, acknowledgment, greeting, unknown\n\n"
            f"For EACH utterance, output exactly one line:\n"
            f"IDX|intent|confidence\n"
            f"Where confidence is 0.0-1.0 (how certain you are).\n"
            f"Example: 0|agreement|0.90\n"
            f"Example: 1|info_request|0.75\n"
            f"Example: 2|unknown|0.40\n\n"
            f"Use conversation context: 'Yes' after payment question = agreement. "
            f"Agent giving info = info_request. Customer confirming identity = consent_given.\n"
            f"Use 'unknown' if the intent is genuinely ambiguous.\n"
            f"{_get_few_shot_examples(lang, 'intent')}"
            f"Output ONLY the IDX|intent|confidence lines, nothing else."
        )

        try:
            from services.llm.client import extract_raw
            raw = extract_raw(prompt, model=_get_llm_model(lang), timeout=45)

            # Parse line-based output: "0|agreement|0.85", "1|info_request|0.70", etc.
            classified_indices = set()
            for line in raw.strip().split("\n"):
                line = line.strip().strip("-").strip("*").strip()
                if "|" not in line:
                    continue
                parts = line.split("|")
                try:
                    idx = int(parts[0].strip())
                    intent_str = parts[1].strip().lower().replace(" ", "_")
                    if idx < 0 or idx >= len(batch):
                        continue
                    if intent_str not in VALID_INTENTS:
                        intent_str = "unknown"

                    # Parse LLM-provided confidence (3rd field), fallback to 0.70
                    conf = 0.70
                    if len(parts) >= 3:
                        try:
                            conf = float(parts[2].strip())
                            conf = max(0.0, min(1.0, conf))
                        except ValueError:
                            conf = 0.70

                    seg_i, _, speaker, _ = batch[idx]
                    intents.append(UtteranceIntent(
                        segment_id=seg_i,
                        speaker=speaker,
                        intent=CallIntent(intent_str),
                        confidence=round(conf, 2),
                    ))
                    classified_indices.add(idx)
                except (ValueError, KeyError):
                    continue

            # Fill any missed utterances with "unknown" (not "acknowledgment")
            for idx, (seg_i, _, speaker, _) in enumerate(batch):
                if idx not in classified_indices:
                    intents.append(UtteranceIntent(
                        segment_id=seg_i,
                        speaker=speaker,
                        intent=CallIntent.UNKNOWN,
                        confidence=0.3,
                    ))

            total_classified += len(classified_indices)

        except Exception as e:
            logger.warning(f"Batch intent classification failed (batch {batch_start//LLM_BATCH + 1}): {e}")
            for seg_i, _, speaker, _ in batch:
                intents.append(UtteranceIntent(
                    segment_id=seg_i,
                    speaker=speaker,
                    intent=CallIntent.UNKNOWN,
                    confidence=0.1,
                ))

    logger.info(f"Intent classification: {total_classified} classified via LLM in {(len(llm_queue) + LLM_BATCH - 1) // LLM_BATCH} batch(es)")

    # Sort by segment_id for consistent ordering
    intents.sort(key=lambda x: x.segment_id)
    return intents


def _extract_all_llm_batched(
    segments: list, call_type: str, lang: str = "en"
) -> tuple[list[FinancialEntity], list[Obligation], str, list[str], list[str]]:
    """Combined LLM call: entities + obligations + summary in ONE request.

    Reduces 3 Ollama roundtrips to 1, saving ~60% of LLM inference time.
    Falls back to individual calls if the combined call fails.
    """
    from pydantic import BaseModel, Field

    class CombinedExtraction(BaseModel):
        financial_entities: list[FinancialEntity] = Field(
            default_factory=list,
            description="Contextual financial entities (amounts, products, dates mentioned indirectly)"
        )
        obligations: list[Obligation] = Field(
            default_factory=list,
            description="Verbal commitments, promises, authorizations, disputes"
        )
        summary: str = Field(
            default="Call processed.",
            description="2-3 sentence natural language summary"
        )
        key_outcomes: list[str] = Field(
            default_factory=list,
            description="3-5 bullet points of call outcomes"
        )
        next_actions: list[str] = Field(
            default_factory=list,
            description="Required follow-up actions"
        )

    # Compress transcript: merge consecutive same-speaker segments to reduce LLM tokens
    full_text = _compress_transcript_for_llm(segments)

    lang_name = LANGUAGE_NAMES.get(lang, lang)
    lang_context = f"The call transcript is in {lang_name}. Respond in English.\n" if lang != "en" else ""

    prompt = (
        f"Analyze this {call_type} financial call transcript and extract ALL of the following:\n"
        f"{lang_context}\n"
        f"1. FINANCIAL ENTITIES: Amounts referenced indirectly ('the monthly installment'), "
        f"product names, tenure, contextual dates. Skip obvious numbers already in the text.\n\n"
        f"2. OBLIGATIONS: Payment promises, consent given/denied, authorizations, disputes. "
        f"Classify strength as: binding, conditional, promise, vague, or denial.\n\n"
        f"3. SUMMARY: A 2-3 sentence summary, key outcomes (3-5 bullets), and next actions.\n\n"
        f"TRANSCRIPT:\n{full_text[:6000]}"
    )

    try:
        result = extract_structured(
            prompt=prompt,
            response_model=CombinedExtraction,
            model=_get_llm_model(lang),
            system_prompt=(
                "You are a financial call analyst. Extract entities, obligations, and summary "
                "in a SINGLE response. The transcript may be in any language. "
                "Always return structured fields in English."
            ),
            timeout=60,
        )
        return (
            result.financial_entities,
            result.obligations,
            result.summary,
            result.key_outcomes,
            result.next_actions,
        )
    except Exception as e:
        logger.warning(f"Combined LLM extraction failed ({e}), trying individual calls...")
        # Fallback to individual calls
        llm_entities = _extract_entities_llm(segments, lang=lang)
        obligations = _detect_obligations(segments, lang=lang)
        summary, outcomes, actions = _generate_summary(segments, call_type, lang=lang)
        return llm_entities, obligations, summary, outcomes, actions


def _extract_entities_llm(segments: list, lang: str = "en") -> list[FinancialEntity]:
    """Layer 2: LLM-based contextual entity extraction using raw completion."""
    from services.llm.client import extract_raw
    full_text = _compress_transcript_for_llm(segments)

    lang_name = LANGUAGE_NAMES.get(lang, lang)
    lang_context = f"The call transcript is in {lang_name}. Respond in English.\n" if lang != "en" else ""

    prompt = (
        f"Extract financial entities from this call transcript.\n"
        f"{lang_context}"
        f"For each entity, output ONE LINE in this exact format:\n"
        f"ENTITY|type|value|raw_text\n"
        f"Types: currency_amount, date, account_number, product_name, interest_rate, tenure, penalty\n"
        f"Example: ENTITY|currency_amount|5000|the monthly installment of five thousand\n\n"
        f"Only extract entities that require context to understand (not obvious numbers).\n"
        f"{_get_few_shot_examples(lang, 'entity')}"
        f"If none found, output: NONE\n\n"
        f"{full_text[:4000]}"
    )

    try:
        raw = extract_raw(prompt, model=_get_llm_model(lang), timeout=45)
        entities = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line.startswith("ENTITY|"):
                parts = line.split("|", 3)
                if len(parts) >= 4:
                    entities.append(FinancialEntity(
                        entity_type=parts[1].strip(),
                        normalized_value=parts[2].strip(),
                        raw_text=parts[3].strip(),
                        confidence=0.7,
                    ))
        return entities
    except Exception as e:
        logger.warning(f"LLM entity extraction failed: {e}")
        return []


def _detect_obligations(segments: list, lang: str = "en") -> list[Obligation]:
    """Detect verbal commitments and obligations using raw completion."""
    from services.llm.client import extract_raw
    full_text = _compress_transcript_for_llm(segments)

    lang_name = LANGUAGE_NAMES.get(lang, lang)
    lang_context = f"The call transcript is in {lang_name}. Respond in English.\n" if lang != "en" else ""

    prompt = (
        f"Extract verbal commitments and obligations from this call.\n"
        f"{lang_context}"
        f"For each obligation, output ONE LINE in this exact format:\n"
        f"OBLIGATION|speaker|strength|text\n"
        f"Strength: binding, conditional, promise, vague, denial\n"
        f"Speaker: agent or customer\n"
        f"Example: OBLIGATION|customer|promise|I will pay five thousand by Friday\n\n"
        f"If none found, output: NONE\n\n"
        f"{full_text[:4000]}"
    )

    try:
        raw = extract_raw(prompt, model=_get_llm_model(lang), timeout=45)
        obligations = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line.startswith("OBLIGATION|"):
                parts = line.split("|", 3)
                if len(parts) >= 4:
                    obligations.append(Obligation(
                        speaker=parts[1].strip(),
                        strength=parts[2].strip(),
                        text=parts[3].strip(),
                    ))
        return obligations
    except Exception as e:
        logger.warning(f"Obligation detection failed: {e}")
        return []


def _generate_summary(segments: list, call_type: str, lang: str = "en") -> tuple[str, list[str], list[str]]:
    """Generate call summary using raw completion."""
    from services.llm.client import extract_raw
    compressed = _compress_transcript_for_llm(segments)

    lang_name = LANGUAGE_NAMES.get(lang, lang)
    lang_context = f"The call transcript is in {lang_name}. Respond in English.\n" if lang != "en" else ""

    prompt = (
        f"Summarize this {call_type} call.\n"
        f"{lang_context}"
        f"Use this EXACT format:\n"
        f"SUMMARY: [2-3 sentence summary]\n"
        f"OUTCOMES:\n- [outcome 1]\n- [outcome 2]\n- [outcome 3]\n"
        f"ACTIONS:\n- [action 1]\n- [action 2]\n\n"
        f"{compressed[:4000]}"
    )

    try:
        raw = extract_raw(prompt, model=_get_llm_model(lang), timeout=45)
        summary = ""
        outcomes = []
        actions = []
        section = None
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line.startswith("SUMMARY:"):
                summary = line[8:].strip()
                section = "summary"
            elif line.startswith("OUTCOMES:"):
                section = "outcomes"
            elif line.startswith("ACTIONS:"):
                section = "actions"
            elif line.startswith("- ") or line.startswith("* "):
                item = line[2:].strip()
                if section == "outcomes":
                    outcomes.append(item)
                elif section == "actions":
                    actions.append(item)
            elif section == "summary" and line and not summary:
                summary = line

        return (
            summary or "Call processed — see extracted fields.",
            outcomes or ["See analysis fields"],
            actions or ["Review if needed"],
        )
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
        return "Call processed — see extracted fields.", ["See analysis fields"], ["Review if needed"]


def _merge_entities(
    layer1: list[FinancialEntity], layer2: list[FinancialEntity]
) -> list[FinancialEntity]:
    """Merge Layer 1 (regex/spaCy) and Layer 2 (LLM) entities, deduplicating."""
    # Layer 1 has higher confidence for structured patterns
    merged = list(layer1)
    l1_keys = {(e.entity_type, e.segment_id) for e in layer1}

    for e in layer2:
        if (e.entity_type, e.segment_id) not in l1_keys:
            merged.append(e)

    return merged


def _compute_review_priority(
    risk_level: RiskLevel, num_fraud: int, num_violations: int
) -> int:
    """Compute review priority (1=urgent, 5=routine)."""
    if risk_level == RiskLevel.CRITICAL:
        return 1
    elif risk_level == RiskLevel.HIGH:
        return 2
    elif num_fraud > 0 or num_violations > 0:
        return 3
    else:
        return 5


def _compress_transcript_for_llm(segments: list, max_chars: int = 6000) -> str:
    """Compress transcript for LLM by merging consecutive same-speaker segments.

    Reduces token count by ~30-50% for long transcripts, proportionally
    reducing LLM inference time without losing any content.
    """
    lines = []
    prev_speaker = None
    merged_text = ""
    merged_start = 0.0

    for seg in segments:
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "?")
        if not text:
            continue

        if speaker == prev_speaker and merged_text:
            merged_text += " " + text
        else:
            if merged_text:
                lines.append(f"[{prev_speaker}] ({merged_start:.0f}s): {merged_text}")
            merged_text = text
            merged_start = seg.get("start", 0)
            prev_speaker = speaker
    if merged_text:
        lines.append(f"[{prev_speaker}] ({merged_start:.0f}s): {merged_text}")

    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... (truncated)"
    return result


def _auto_detect_call_type(segments: list) -> str:
    """Auto-detect call type from first ~30s of transcript using keyword signals.

    Returns detected call type or 'general' if no strong signal.
    """
    # Collect first ~30s of text
    opening_text = []
    for seg in segments:
        if seg.get("start", 0) > 45:  # generous 45s window
            break
        opening_text.append(seg.get("text", "").lower())
    text = " ".join(opening_text)

    if not text or len(text) < 20:
        return "general"

    # Score each call type by keyword matches
    type_signals = {
        "collections": [
            "overdue", "outstanding", "emi", "payment due", "recovery",
            "balance due", "default", "pay now", "bakaya", "vasuli", "kist",
            "installment", "repay", "dues", "delinquent", "late payment",
            "settlement", "one-time settlement", "ots",
        ],
        "kyc": [
            "kyc", "verification", "verify your", "identity", "aadhaar",
            "pan card", "date of birth", "confirm your", "mother's maiden",
            "know your customer", "video kyc", "re-kyc",
        ],
        "complaint": [
            "complaint", "grievance", "not satisfied", "escalate",
            "supervisor", "manager", "resolve", "dissatisfied",
            "file a complaint", "lodge a complaint", "unhappy",
            "shikayat", "problem",
        ],
        "consent": [
            "consent", "authorize", "auto-debit", "nach", "mandate",
            "standing instruction", "agree to", "permission",
            "do you agree", "recording consent",
        ],
        "onboarding": [
            "welcome", "new account", "opening", "onboarding",
            "terms and conditions", "loan application", "approved",
            "congratulations", "sanction", "disburse",
        ],
    }

    scores: dict[str, int] = {}
    for ctype, keywords in type_signals.items():
        score = sum(1 for kw in keywords if kw in text)
        if score >= 2:  # require at least 2 keyword matches
            scores[ctype] = score

    if not scores:
        return "general"

    best = max(scores, key=scores.get)
    logger.debug(f"Call type auto-detection scores: {scores}, selected: {best}")
    return best


def _map_speaker_roles(segments: list, call_type: str = "general") -> list:
    """Map diarization speaker labels (SPEAKER_00/01/02) to semantic roles.

    For regulated call types (collections, kyc, onboarding, consent, complaint):
        First speaker → "agent", second → "customer", extras → "other"

    For general/unregulated calls (earnings calls, investor presentations):
        First speaker → "Speaker A", second → "Speaker B", extras → "Speaker C" etc.
        (No agent/customer assumption — avoids wrong role labels on non-call-center audio)
    """
    # Collect unique speakers and their stats
    speaker_stats = {}
    for i, seg in enumerate(segments):
        sp = seg.get("speaker", "unknown")
        if sp == "unknown" or not sp:
            continue
        if sp not in speaker_stats:
            speaker_stats[sp] = {"first_seen": i, "total_duration": 0, "count": 0}
        speaker_stats[sp]["total_duration"] += seg.get("end", 0) - seg.get("start", 0)
        speaker_stats[sp]["count"] += 1

    if len(speaker_stats) < 2:
        # Can't determine roles with fewer than 2 speakers
        return segments

    # Sort speakers by first appearance
    speakers_by_appearance = sorted(speaker_stats.keys(), key=lambda s: speaker_stats[s]["first_seen"])

    # Regulated call types: assign agent/customer roles
    regulated_types = {"collections", "kyc", "onboarding", "consent", "complaint"}
    use_agent_customer = call_type in regulated_types

    if use_agent_customer:
        first_speaker = speakers_by_appearance[0]
        agent_speaker = first_speaker
        customer_speaker = speakers_by_appearance[1] if speakers_by_appearance[1] != agent_speaker else (
            speakers_by_appearance[0] if len(speakers_by_appearance) > 1 else None
        )

        role_map = {agent_speaker: "agent"}
        if customer_speaker:
            role_map[customer_speaker] = "customer"
        for sp in speaker_stats:
            if sp not in role_map:
                role_map[sp] = "other"

        logger.info(
            f"Speaker role mapping (regulated): {role_map} "
            f"(agent={speaker_stats.get(agent_speaker, {}).get('total_duration', 0):.1f}s, "
            f"customer={speaker_stats.get(customer_speaker, {}).get('total_duration', 0):.1f}s)"
        )
    else:
        # General/unknown calls: neutral labels (Speaker A, B, C...)
        labels = [f"Speaker {chr(65 + i)}" for i in range(len(speakers_by_appearance))]
        role_map = dict(zip(speakers_by_appearance, labels))

        dur_parts = [f"{v}={speaker_stats[k]['total_duration']:.1f}s" for k, v in role_map.items()]
        logger.info(
            f"Speaker role mapping (general): {role_map} "
            f"(durations: {', '.join(dur_parts)})"
        )

    # Apply mapping to segments
    for seg in segments:
        original_speaker = seg.get("speaker", "unknown")
        if original_speaker in role_map:
            seg["speaker"] = role_map[original_speaker]
            seg["original_speaker_id"] = original_speaker

    return segments


def _identify_speakers_by_name(segments: list) -> list:
    """Identify speakers by name from transcript text.

    Scans for introduction patterns like "I'm [Name]", "This is [Name]",
    "turn the call over to [Name]", etc. Maps speaker IDs to real names.
    Only applies names when confidence is high (multiple confirmations
    or clear introduction pattern).
    """
    # Patterns that reveal speaker identity (group 1 captures the name)
    SELF_INTRO_PATTERNS = [
        r"(?:I'm|I am|my name is|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:speaking|here)\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    ]

    # Patterns where someone introduces the NEXT speaker
    HANDOFF_PATTERNS = [
        r"(?:turn (?:the )?(?:call|conference|floor) over to|hand (?:it )?over to|introduce)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:joined by|welcome)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    ]

    # Patterns where someone addresses a speaker by name (reveals the OTHER speaker)
    ADDRESS_PATTERNS = [
        r"(?:Thank you|Thanks),?\s+([A-Z][a-z]+)",
        r"(?:over to you|go ahead),?\s+([A-Z][a-z]+)",
    ]

    # Common false positives to skip
    SKIP_NAMES = {
        "Sir", "Madam", "Ma", "Mr", "Mrs", "Ms", "Dr", "The", "This",
        "Good", "Well", "Yes", "No", "Hi", "Hello", "Thank", "Thanks",
        "Sure", "Ok", "Okay", "Now", "So", "Also", "But", "And",
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    }

    # Collect name evidence: speaker_id -> {name: count}
    speaker_name_evidence: dict[str, dict[str, int]] = {}

    for i, seg in enumerate(segments):
        text = seg.get("text", "")
        speaker_id = seg.get("original_speaker_id", seg.get("speaker", ""))
        if not text or not speaker_id:
            continue

        # Self-introduction: "I'm [Name]" -> this segment's speaker is [Name]
        for pattern in SELF_INTRO_PATTERNS:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if name.split()[0] not in SKIP_NAMES and len(name) > 2:
                    speaker_name_evidence.setdefault(speaker_id, {})
                    speaker_name_evidence[speaker_id][name] = (
                        speaker_name_evidence[speaker_id].get(name, 0) + 2  # Strong signal
                    )

        # Handoff: "turn it over to [Name]" -> NEXT different speaker is [Name]
        for pattern in HANDOFF_PATTERNS:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if name.split()[0] not in SKIP_NAMES and len(name) > 2:
                    # Find next segment with different speaker
                    for j in range(i + 1, min(i + 5, len(segments))):
                        next_sp = segments[j].get("original_speaker_id", segments[j].get("speaker", ""))
                        if next_sp and next_sp != speaker_id:
                            speaker_name_evidence.setdefault(next_sp, {})
                            speaker_name_evidence[next_sp][name] = (
                                speaker_name_evidence[next_sp].get(name, 0) + 2
                            )
                            break

        # Address: "Thank you, [Name]" -> the previous different speaker is [Name]
        for pattern in ADDRESS_PATTERNS:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if name.split()[0] not in SKIP_NAMES and len(name) > 2:
                    # Find previous segment with different speaker
                    for j in range(i - 1, max(i - 5, -1), -1):
                        prev_sp = segments[j].get("original_speaker_id", segments[j].get("speaker", ""))
                        if prev_sp and prev_sp != speaker_id:
                            speaker_name_evidence.setdefault(prev_sp, {})
                            speaker_name_evidence[prev_sp][name] = (
                                speaker_name_evidence[prev_sp].get(name, 0) + 1
                            )
                            break

    if not speaker_name_evidence:
        return segments

    # Resolve: pick the name with the most evidence for each speaker
    speaker_names: dict[str, str] = {}
    used_names: set[str] = set()
    for sp_id, name_counts in sorted(
        speaker_name_evidence.items(),
        key=lambda x: max(x[1].values()),
        reverse=True,
    ):
        best_name = max(name_counts, key=name_counts.get)
        best_count = name_counts[best_name]
        # Require at least 1 strong signal (self-intro) or 2 weak signals
        if best_count >= 2 and best_name not in used_names:
            speaker_names[sp_id] = best_name
            used_names.add(best_name)

    if not speaker_names:
        return segments

    logger.info(f"Speaker identification: {speaker_names}")

    # Apply names to segments
    for seg in segments:
        sp_id = seg.get("original_speaker_id", "")
        if sp_id in speaker_names:
            seg["speaker_name"] = speaker_names[sp_id]

    return segments
