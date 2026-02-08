# FinSight v2: Financial Call Intelligence Pipeline
## Pivoted to Match Judge Requirements — Bank/Lender Call Processing

---

## THE PROBLEM (As Judges Defined It)

Fintech companies get messy, unstructured data (calls, documents). Humans understand it easily. AI can't — it needs numerical, structured data. Banks record MILLIONS of calls:
- Customer onboarding
- KYC verification
- Payment reminders / collections
- Complaints
- Legal consent calls

These calls contain **legal landmines**: "I promise I'll pay tomorrow" — is that a binding verbal commitment? Was proper consent language used? Did the agent follow regulatory scripts?

**Our job:** Turn raw audio into clean, structured, ML-trainable, legally auditable data.

---

## THE PIPELINE (Matching Judge's 5-Stage Framework)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  STAGE 1: RAW AUDIO INTAKE                                              │
│  ┌─────────────────────────────────────────────────────────┐            │
│  │ • Accept ANY audio/video format (mp3/wav/mp4/webm/ogg)  │            │
│  │ • Normalize to 16kHz mono WAV                           │            │
│  │ • FFmpeg-based universal converter                       │            │
│  └─────────────────────────────────────────────────────────┘            │
│      │                                                                   │
│      ▼                                                                   │
│  STAGE 2: CLEAN & ANALYZE AUDIO                                         │
│  ┌─────────────────────────────────────────────────────────┐            │
│  │ • Noise detection & SNR scoring (silero-vad + openSMILE) │            │
│  │ • Language detection (whisper auto-detect or langid)     │            │
│  │ • Speaker diarization: Agent vs Customer (pyannote)      │            │
│  │ • Audio quality scoring (0-100 trustworthiness flag)     │            │
│  │ • Fraud voice pattern detection (emotion2vec + stress)   │            │
│  │                                                          │            │
│  │ OUTPUT: Clean audio stream + audio_metadata.json         │            │
│  │ {                                                        │            │
│  │   "format": "wav", "sample_rate": 16000,                │            │
│  │   "duration_sec": 342.5,                                │            │
│  │   "snr_db": 28.4,                                       │            │
│  │   "quality_score": 87,                                  │            │
│  │   "quality_flag": "TRUSTWORTHY",                        │            │
│  │   "language": "en",                                     │            │
│  │   "language_confidence": 0.97,                          │            │
│  │   "num_speakers": 2,                                    │            │
│  │   "speakers": [                                         │            │
│  │     {"id": "SPEAKER_0", "role": "agent", "talk_pct": 62},│           │
│  │     {"id": "SPEAKER_1", "role": "customer", "talk_pct": 38}│         │
│  │   ],                                                    │            │
│  │   "fraud_flags": [],                                    │            │
│  │   "noise_segments": [{"start": 45.2, "end": 47.8}]     │            │
│  │ }                                                        │            │
│  └─────────────────────────────────────────────────────────┘            │
│      │                                                                   │
│      ▼                                                                   │
│  STAGE 3: FINANCIAL TRANSCRIPTION (Speech → Text)                       │
│  ┌─────────────────────────────────────────────────────────┐            │
│  │ • WhisperX (faster-whisper large-v3-turbo)               │            │
│  │ • Word-level timestamps + speaker alignment              │            │
│  │ • Per-word confidence scores                             │            │
│  │ • Financial term correction (post-processing layer)      │            │
│  │   "EMI" not "Emmy", "KYC" not "kayak",                  │            │
│  │   "CIBIL" not "civil", "₹" amounts preserved            │            │
│  │ • Low-confidence segment flagging for human review       │            │
│  │                                                          │            │
│  │ OUTPUT: Time-aligned transcript with confidence          │            │
│  │ {                                                        │            │
│  │   "segments": [                                          │            │
│  │     {                                                    │            │
│  │       "speaker": "agent",                               │            │
│  │       "start": 0.0, "end": 4.2,                        │            │
│  │       "text": "Good morning, this is Priya from HDFC",  │            │
│  │       "confidence": 0.96,                               │            │
│  │       "words": [                                        │            │
│  │         {"word":"Good","start":0.0,"end":0.3,"conf":0.99},│          │
│  │         {"word":"morning","start":0.3,"end":0.8,"conf":0.98}│        │
│  │       ],                                                │            │
│  │       "flagged": false                                  │            │
│  │     }                                                    │            │
│  │   ],                                                    │            │
│  │   "low_confidence_segments": [12, 47, 89],              │            │
│  │   "overall_confidence": 0.93                            │            │
│  │ }                                                        │            │
│  └─────────────────────────────────────────────────────────┘            │
│      │                                                                   │
│      ▼                                                                   │
│  STAGE 4: UNDERSTAND FINANCIAL MEANING (Intelligence Layer)             │
│  ┌─────────────────────────────────────────────────────────┐            │
│  │ A) INTENT CLASSIFICATION (per utterance)                 │            │
│  │    • agreement / refusal / request_extension /           │            │
│  │      complaint / consent_given / consent_denied /        │            │
│  │      information_request / negotiation / escalation      │            │
│  │                                                          │            │
│  │ B) FINANCIAL ENTITY EXTRACTION                           │            │
│  │    • EMI amount, loan amount, interest rate              │            │
│  │    • Account numbers, reference numbers                  │            │
│  │    • Dates (payment due, promise date, next call)        │            │
│  │    • Customer name, agent name, product name             │            │
│  │    • Currency amounts with normalization                 │            │
│  │                                                          │            │
│  │ C) OBLIGATION & COMMITMENT DETECTION                     │            │
│  │    • "I will pay by Friday" → VERBAL_COMMITMENT          │            │
│  │    • "I promise to transfer ₹5000" → PAYMENT_PROMISE    │            │
│  │    • "I authorize the auto-debit" → CONSENT_GIVEN        │            │
│  │    • "I never agreed to this" → CONSENT_DISPUTED         │            │
│  │    • Strength: binding / conditional / vague             │            │
│  │                                                          │            │
│  │ D) REGULATORY COMPLIANCE CHECKING                        │            │
│  │    • Did agent state required disclosures?               │            │
│  │    • Was consent language properly delivered?             │            │
│  │    • Were prohibited phrases used? (threats, coercion)   │            │
│  │    • RBI/SEBI guidelines adherence scoring               │            │
│  │    • Fair debt collection practices compliance           │            │
│  │                                                          │            │
│  │ E) SENTIMENT & EMOTION (per speaker per turn)            │            │
│  │    • Customer frustration escalation tracking            │            │
│  │    • Agent tone compliance (professional/aggressive)     │            │
│  │    • Emotional trajectory across the call                │            │
│  │                                                          │            │
│  │ F) FRAUD / RISK SIGNALS                                  │            │
│  │    • Voice stress anomalies during identity verification │            │
│  │    • Inconsistent information detection                  │            │
│  │    • Coached/rehearsed speech patterns                   │            │
│  │    • Third-party-on-call detection                       │            │
│  └─────────────────────────────────────────────────────────┘            │
│      │                                                                   │
│      ▼                                                                   │
│  STAGE 5: PRODUCE REVIEWABLE, AUDITABLE OUTPUT                          │
│  ┌─────────────────────────────────────────────────────────┐            │
│  │ A) STRUCTURED JSON (ML-TRAINABLE)                        │            │
│  │    Complete call record with all extracted fields         │            │
│  │    Every field traceable to exact timestamp in audio      │            │
│  │    Exportable as CSV/Parquet for model training           │            │
│  │                                                          │            │
│  │ B) AUDIT REPORT                                          │            │
│  │    Compliance scorecard per call                         │            │
│  │    Flagged segments with source highlighting             │            │
│  │    Risk classification: LOW / MEDIUM / HIGH / CRITICAL   │            │
│  │                                                          │            │
│  │ C) SEARCHABLE KNOWLEDGE BASE (Backboard.io)              │            │
│  │    All calls indexed with persistent memory              │            │
│  │    Cross-call pattern detection                          │            │
│  │    "Show me all calls where customer disputed consent"   │            │
│  │                                                          │            │
│  │ D) HUMAN REVIEW QUEUE                                    │            │
│  │    Auto-flagged calls needing human attention            │            │
│  │    Priority scoring based on risk level                  │            │
│  │    One-click approve/reject/escalate workflow            │            │
│  └─────────────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## WHAT CHANGES VS WHAT STAYS FROM OUR ORIGINAL PLAN

### STAYS (Reuse 100%)
- Multi-GPU architecture (Alpha 3060 + Bravo 4050 over network)
- WhisperX for ASR (faster-whisper large-v3-turbo + pyannote diarization)
- Ollama with Qwen3 8B for LLM extraction
- Instructor + Pydantic for structured output
- emotion2vec for emotion classification
- openSMILE + Parselmouth for vocal analysis
- FinBERT for sentiment
- Backboard.io for memory, RAG, LLM routing, thread management
- Team structure (5 roles, same GPU assignments)
- React frontend architecture

### PIVOTS (New Domain Logic)
- Earnings calls → Bank/lender calls (KYC, collections, complaints, consent)
- Trust Score → Compliance Scorecard
- Evasion detection → Intent classification (agree/refuse/stall)
- Forward guidance → Obligation & commitment detection
- CEO-CFO divergence → Agent vs Customer analysis
- SEC EDGAR → RBI/SEBI regulatory compliance
- Knowledge graph → Searchable call intelligence (Backboard RAG)
- Dashboard → Audit-ready structured output + review queue

### NEW ADDITIONS
- Audio normalization + quality scoring pipeline
- Financial term correction (post-processing on transcription)
- Regulatory phrase library (required disclosures, prohibited language)
- ML-trainable export formats (CSV, Parquet, JSON Lines)
- Human review queue with priority scoring

---

## REVISED PYDANTIC SCHEMAS (Instructor + Qwen3 8B)

These are the structured output definitions that make our extraction reliable:

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

# ── INTENT CLASSIFICATION ──

class CallIntent(str, Enum):
    AGREEMENT = "agreement"                    # Customer agrees to terms/payment
    REFUSAL = "refusal"                        # Customer refuses
    REQUEST_EXTENSION = "request_extension"     # "Can I get more time?"
    PAYMENT_PROMISE = "payment_promise"         # "I'll pay by Friday"
    COMPLAINT = "complaint"                    # Customer complaining
    CONSENT_GIVEN = "consent_given"            # Explicit consent provided
    CONSENT_DENIED = "consent_denied"          # Consent explicitly refused
    INFORMATION_REQUEST = "info_request"        # Asking for details
    NEGOTIATION = "negotiation"               # Negotiating terms
    ESCALATION = "escalation"                 # Requesting supervisor
    DISPUTE = "dispute"                        # Disputing charges/terms
    ACKNOWLEDGMENT = "acknowledgment"          # Neutral acknowledgment
    GREETING = "greeting"                      # Call opening/closing

class UtteranceIntent(BaseModel):
    """Intent classification for a single utterance"""
    segment_id: int = Field(description="Index of the transcript segment")
    speaker: str = Field(description="'agent' or 'customer'")
    intent: CallIntent
    confidence: float = Field(ge=0, le=1)
    sub_intent: Optional[str] = Field(None, 
        description="More specific intent, e.g., 'partial_agreement', 'conditional_refusal'")


# ── FINANCIAL ENTITY EXTRACTION ──

class CurrencyAmount(BaseModel):
    value: float
    currency: str = Field(default="INR", description="ISO currency code")
    raw_text: str = Field(description="Original text, e.g., '₹5,000' or 'five thousand rupees'")

class FinancialEntity(BaseModel):
    """A single financial entity extracted from the call"""
    entity_type: str = Field(description="One of: emi_amount, loan_amount, interest_rate, "
        "penalty_amount, payment_amount, account_number, reference_number, "
        "due_date, promise_date, next_call_date, product_name, tenure_months")
    value: str = Field(description="Normalized value")
    raw_text: str = Field(description="Exact text from transcript")
    segment_id: int = Field(description="Which transcript segment this came from")
    start_time: float = Field(description="Timestamp in audio (seconds)")
    confidence: float = Field(ge=0, le=1)


# ── OBLIGATION & COMMITMENT DETECTION ──

class ObligationStrength(str, Enum):
    BINDING = "binding"             # "I authorize", "I agree", "Yes, debit my account"
    CONDITIONAL = "conditional"     # "I'll pay if you waive the penalty"
    PROMISE = "promise"            # "I'll pay by Friday"
    VAGUE = "vague"               # "I'll try to arrange something"
    DENIAL = "denial"             # "I never agreed to this"

class Obligation(BaseModel):
    """A verbal commitment or obligation detected in the call"""
    text: str = Field(description="Exact quote from transcript")
    speaker: str = Field(description="'agent' or 'customer'")
    obligation_type: str = Field(description="One of: payment_promise, consent, "
        "authorization, commitment, denial, dispute")
    strength: ObligationStrength
    amount: Optional[CurrencyAmount] = None
    date_referenced: Optional[str] = Field(None, description="ISO date if a date was mentioned")
    segment_id: int
    start_time: float
    legally_significant: bool = Field(
        description="True if this could have legal/compliance implications")


# ── REGULATORY COMPLIANCE ──

class ComplianceViolationType(str, Enum):
    MISSING_DISCLOSURE = "missing_disclosure"         # Required disclosure not given
    PROHIBITED_LANGUAGE = "prohibited_language"        # Threats, coercion, abuse
    CONSENT_NOT_OBTAINED = "consent_not_obtained"      # Action without proper consent
    IMPROPER_COLLECTION = "improper_collection"        # Violates fair debt practices
    PRIVACY_VIOLATION = "privacy_violation"            # Discussed account with wrong person
    MISLEADING_INFO = "misleading_information"         # Agent gave wrong info
    CALL_TIME_VIOLATION = "call_time_violation"        # Called outside permitted hours

class ComplianceCheck(BaseModel):
    """A single regulatory compliance check result"""
    check_name: str = Field(description="What was checked, e.g., 'opening_disclosure'")
    passed: bool
    violation_type: Optional[ComplianceViolationType] = None
    evidence_text: Optional[str] = Field(None, description="Relevant quote from transcript")
    segment_id: Optional[int] = None
    regulation: str = Field(description="Which regulation, e.g., 'RBI_Fair_Practice_Code'")
    severity: str = Field(description="'low', 'medium', 'high', 'critical'")


# ── CALL-LEVEL RISK ASSESSMENT ──

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"

class FraudSignal(BaseModel):
    signal_type: str = Field(description="e.g., 'voice_stress_anomaly', "
        "'inconsistent_info', 'coached_speech', 'third_party_detected'")
    description: str
    segment_id: int
    confidence: float = Field(ge=0, le=1)


# ── MASTER OUTPUT: THE COMPLETE STRUCTURED CALL RECORD ──

class CallRecord(BaseModel):
    """Complete structured output for one call — ML-trainable and audit-ready"""
    
    # Metadata
    call_id: str
    audio_file: str
    duration_seconds: float
    language: str
    call_type: str = Field(description="'collections', 'kyc', 'onboarding', "
        "'complaint', 'consent', 'general'")
    
    # Audio Quality
    audio_quality_score: int = Field(ge=0, le=100)
    audio_quality_flag: str = Field(description="'TRUSTWORTHY', 'DEGRADED', 'UNRELIABLE'")
    snr_db: float
    
    # Speakers
    num_speakers: int
    agent_talk_percentage: float
    customer_talk_percentage: float
    
    # Transcript Summary
    overall_transcript_confidence: float
    num_low_confidence_segments: int
    
    # Intelligence Extraction
    intents: list[UtteranceIntent]
    financial_entities: list[FinancialEntity]
    obligations: list[Obligation]
    compliance_checks: list[ComplianceCheck]
    fraud_signals: list[FraudSignal]
    
    # Sentiment Trajectory
    customer_sentiment_trajectory: list[float] = Field(
        description="Sentiment score per segment for customer turns")
    agent_sentiment_trajectory: list[float] = Field(
        description="Sentiment score per segment for agent turns")
    customer_emotion_dominant: str = Field(description="Most frequent customer emotion")
    escalation_detected: bool
    
    # Risk Assessment
    overall_risk_level: RiskLevel
    compliance_score: int = Field(ge=0, le=100, 
        description="100 = fully compliant, 0 = severe violations")
    requires_human_review: bool
    review_priority: int = Field(ge=1, le=5, description="1=urgent, 5=routine")
    review_reasons: list[str]
    
    # Call Summary
    call_summary: str = Field(description="2-3 sentence natural language summary")
    key_outcomes: list[str] = Field(description="Bullet points of call outcomes")
    next_actions: list[str] = Field(description="Required follow-up actions")
```

---

## REVISED BACKBOARD.IO INTEGRATION (Deeper Than Before)

### Role 1: PERSISTENT CALL MEMORY — The Customer Brain

Every customer gets a Backboard thread. Every call with that customer adds to that thread. The system remembers:
- "Customer X promised ₹5000 by Jan 15 on call #47. Today is Jan 20. This is call #48."
- "Customer Y has disputed consent 3 times in the last 6 months."
- "Customer Z's frustration level has been escalating over the last 4 calls."

```python
# After processing each call, store the structured record
async def store_call_to_backboard(customer_id: str, call_record: CallRecord):
    """Store call analysis in customer's persistent thread"""
    
    # Get or create customer thread
    thread_id = await get_or_create_customer_thread(customer_id)
    
    # Store the structured analysis (send_to_llm=False — just memory storage)
    await backboard.send_message(
        thread_id=thread_id,
        content=f"""CALL RECORD — {call_record.call_id}
        Date: {call_record.timestamp}
        Type: {call_record.call_type}
        Duration: {call_record.duration_seconds}s
        
        OBLIGATIONS DETECTED:
        {json.dumps([o.dict() for o in call_record.obligations], indent=2)}
        
        COMPLIANCE SCORE: {call_record.compliance_score}/100
        RISK LEVEL: {call_record.overall_risk_level}
        
        KEY OUTCOMES: {call_record.key_outcomes}
        NEXT ACTIONS: {call_record.next_actions}
        
        CUSTOMER SENTIMENT: {call_record.customer_emotion_dominant}
        ESCALATION: {call_record.escalation_detected}""",
        metadata={
            "custom_timestamp": call_record.timestamp,
            "call_type": call_record.call_type,
            "risk_level": call_record.overall_risk_level,
            "compliance_score": str(call_record.compliance_score)
        },
        send_to_llm=False
    )
```

### Role 2: CROSS-CALL INTELLIGENCE — Pattern Detection

```python
# Query across ALL customer calls
async def detect_cross_call_patterns(assistant_id: str, thread_id: str):
    """Use Backboard's memory to find patterns across calls"""
    
    patterns = await backboard.query_thread(
        thread_id=thread_id,
        assistant_id=assistant_id,
        question="""Analyze all calls in this thread and identify:
        1. Unfulfilled payment promises (promised but no follow-up confirming payment)
        2. Escalating frustration pattern (sentiment getting worse over calls)
        3. Repeated compliance issues
        4. Consent disputes or reversals
        Return as structured JSON."""
    )
    return patterns
```

### Role 3: RAG FOR REGULATORY KNOWLEDGE BASE

Upload regulatory documents (RBI guidelines, fair practice codes, SEBI circulars) to Backboard RAG. When the system needs to check compliance, it searches the actual regulations.

```python
# At setup: upload regulatory documents
async def setup_regulatory_rag(assistant_id: str):
    """Upload regulatory docs for compliance checking"""
    docs = [
        "rbi_fair_practice_code_2024.pdf",
        "sebi_kyc_guidelines.pdf", 
        "rbi_digital_lending_guidelines.pdf",
        "fair_debt_collection_practices.pdf",
        "telecom_recording_consent_rules.pdf"
    ]
    for doc in docs:
        await backboard.upload_document(assistant_id, doc)

# During analysis: query regulations
async def check_against_regulations(thread_id, assistant_id, agent_statement):
    """Check if agent's statement complies with regulations"""
    result = await backboard.query_thread(
        thread_id=thread_id,
        assistant_id=assistant_id,
        question=f"""The agent said: "{agent_statement}"
        
        Does this comply with RBI Fair Practice Code and fair debt collection guidelines?
        Check for: threats, coercion, misleading information, missing disclosures.
        Cite the specific regulation section if there's a violation."""
    )
    return result
```

### Role 4: LLM ROUTING — Local for Speed, Cloud for Complexity

```python
# Simple intent classification → LOCAL Qwen3 8B (fast, free)
# Complex compliance reasoning → CLOUD via Backboard (GPT-4o)
# Cross-call pattern analysis → CLOUD via Backboard (needs full context)

async def smart_route(task_type: str, content: str):
    if task_type in ["intent", "entity_extraction", "sentiment"]:
        return await local_qwen3(content)        # ~200ms, free
    elif task_type in ["compliance_check", "cross_call_pattern", "risk_synthesis"]:
        return await backboard_cloud(content)     # ~2s, accurate
```

### Role 5: THREAD MANAGEMENT — Audit Trail

Every analysis creates a Backboard thread that serves as an immutable audit trail. If regulators ask "how did you classify this call?" — you can replay the entire analysis chain.

```python
# Create audit thread for each call
async def create_audit_trail(call_id: str, call_record: CallRecord):
    thread = await backboard.create_thread(metadata={
        "type": "audit_trail",
        "call_id": call_id,
        "risk_level": call_record.overall_risk_level
    })
    
    # Log each stage
    await backboard.send_message(thread["id"],
        content=f"STAGE 1 - Audio Analysis: SNR={call_record.snr_db}dB, "
                f"Quality={call_record.audio_quality_score}/100",
        send_to_llm=False)
    
    await backboard.send_message(thread["id"],
        content=f"STAGE 2 - Transcription: Confidence={call_record.overall_transcript_confidence}, "
                f"Low-conf segments: {call_record.num_low_confidence_segments}",
        send_to_llm=False)
    
    # ... and so on for each stage
    # This creates a complete, searchable, timestamped audit log
```

---

## REVISED TECHNICAL IMPLEMENTATION

### Stage 1: Audio Intake & Normalization

```python
# services/audio/normalizer.py
import subprocess
import json

def normalize_audio(input_path: str, output_path: str) -> dict:
    """Accept ANY audio/video format, normalize to 16kHz mono WAV"""
    
    # Probe input format
    probe = subprocess.run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", input_path
    ], capture_output=True, text=True)
    info = json.loads(probe.stdout)
    
    # Normalize: 16kHz, mono, WAV, PCM 16-bit
    subprocess.run([
        "ffmpeg", "-i", input_path,
        "-ar", "16000",       # 16kHz sample rate (Whisper optimal)
        "-ac", "1",           # Mono
        "-c:a", "pcm_s16le",  # 16-bit PCM
        "-y", output_path
    ], check=True)
    
    return {
        "original_format": info["format"]["format_name"],
        "original_duration": float(info["format"]["duration"]),
        "original_sample_rate": int(info["streams"][0].get("sample_rate", 0)),
        "original_channels": int(info["streams"][0].get("channels", 0)),
        "normalized_to": "16kHz_mono_wav"
    }
```

### Stage 2: Audio Quality & Analysis

```python
# services/audio/quality.py
import numpy as np
import opensmile
from silero_vad import load_silero_vad, get_speech_timestamps

def assess_audio_quality(wav_path: str) -> dict:
    """Comprehensive audio quality assessment"""
    
    # 1. SNR estimation via Silero VAD
    model = load_silero_vad()
    wav, sr = torchaudio.load(wav_path)
    speech_timestamps = get_speech_timestamps(wav, model)
    
    # Estimate SNR from speech vs non-speech segments
    speech_power = compute_power(wav, speech_timestamps)
    noise_power = compute_power(wav, get_noise_segments(speech_timestamps, len(wav[0])))
    snr_db = 10 * np.log10(speech_power / max(noise_power, 1e-10))
    
    # 2. openSMILE quality features
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    features = smile.process_file(wav_path)
    
    # 3. Quality scoring (0-100)
    quality_score = compute_quality_score(snr_db, features)
    
    # 4. Quality flag
    if quality_score >= 70:
        flag = "TRUSTWORTHY"
    elif quality_score >= 40:
        flag = "DEGRADED"
    else:
        flag = "UNRELIABLE"
    
    return {
        "snr_db": round(snr_db, 1),
        "quality_score": quality_score,
        "quality_flag": flag,
        "clipping_detected": detect_clipping(wav),
        "noise_segments": get_noise_segments_timeranges(speech_timestamps, sr),
        "silence_percentage": compute_silence_pct(speech_timestamps, len(wav[0]), sr)
    }


def detect_language(wav_path: str) -> dict:
    """Detect language using Whisper's built-in language detection"""
    import whisper
    model = whisper.load_model("base")  # Tiny model just for lang detection
    
    # Load first 30 seconds
    audio = whisper.load_audio(wav_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    _, probs = model.detect_language(mel)
    top_lang = max(probs, key=probs.get)
    
    return {
        "language": top_lang,
        "confidence": round(probs[top_lang], 3),
        "top_3": sorted(probs.items(), key=lambda x: -x[1])[:3]
    }
```

### Stage 3: Financial Transcription with Term Correction

```python
# services/asr/financial_transcription.py

# Financial term correction dictionary
FINANCIAL_CORRECTIONS = {
    # Common ASR mistakes for financial terms
    "emmy": "EMI", "emi": "EMI",
    "kayak": "KYC", "kayc": "KYC",
    "civil": "CIBIL", "sibyl": "CIBIL", "sybil": "CIBIL",
    "nifty": "NIFTY", "sensex": "SENSEX",
    "hdfc": "HDFC", "icici": "ICICI", "sbi": "SBI",
    "nach": "NACH", "natch": "NACH",
    "upi": "UPI", "upa": "UPA",
    "gst": "GST", "pan": "PAN", "tan": "TAN",
    "rbi": "RBI", "sebi": "SEBI", "irdai": "IRDAI",
    "nbfc": "NBFC", "npa": "NPA",
    "repo rate": "repo rate", "reverse repo": "reverse repo",
    "crore": "crore", "lakh": "lakh",
    "demat": "demat", "d-mat": "demat",
    # Indian currency normalization
    "rupees": "₹", "rupee": "₹", "rs": "₹", "rs.": "₹",
}

# Amount pattern normalization
AMOUNT_PATTERNS = [
    (r"(\d+)\s*thousand", lambda m: str(int(m.group(1)) * 1000)),
    (r"(\d+)\s*lakh", lambda m: str(int(m.group(1)) * 100000)),
    (r"(\d+)\s*crore", lambda m: str(int(m.group(1)) * 10000000)),
    (r"(\d+)\s*k\b", lambda m: str(int(m.group(1)) * 1000)),
]

def apply_financial_corrections(transcript_segments: list) -> list:
    """Post-process ASR output with financial term corrections"""
    corrected = []
    for seg in transcript_segments:
        text = seg["text"]
        
        # Apply term corrections (case-insensitive)
        for wrong, right in FINANCIAL_CORRECTIONS.items():
            text = re.sub(rf'\b{wrong}\b', right, text, flags=re.IGNORECASE)
        
        # Normalize amounts
        for pattern, replacement in AMOUNT_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        seg["text"] = text
        seg["corrected"] = text != seg.get("original_text", text)
        corrected.append(seg)
    
    return corrected
```

### Stage 4: Intelligence Layer — Instructor Extraction

```python
# analysis/intelligence.py
import instructor
from openai import OpenAI

# Connect to local Ollama
client = instructor.from_openai(
    OpenAI(base_url="http://192.168.x.100:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON
)

def classify_intents(transcript_segments: list) -> list[UtteranceIntent]:
    """Classify intent for each customer/agent utterance"""
    
    intents = []
    for seg in transcript_segments:
        result = client.chat.completions.create(
            model="qwen3:8b",
            response_model=UtteranceIntent,
            messages=[{
                "role": "system",
                "content": """You are a financial call analyst. Classify the intent of this 
                utterance from a bank customer service call. Be precise about whether the 
                customer is agreeing, refusing, promising payment, or disputing."""
            }, {
                "role": "user", 
                "content": f"Speaker: {seg['speaker']}\nUtterance: \"{seg['text']}\""
            }],
            max_retries=2
        )
        intents.append(result)
    
    return intents


def extract_obligations(transcript_text: str) -> list[Obligation]:
    """Extract all verbal commitments and obligations from the call"""
    
    result = client.chat.completions.create(
        model="qwen3:8b",
        response_model=list[Obligation],
        messages=[{
            "role": "system",
            "content": """You are a legal analyst reviewing a bank customer call transcript.
            Extract ALL verbal commitments, promises, consents, and obligations.
            Pay special attention to:
            - Payment promises with dates and amounts
            - Consent given or denied for auto-debit, data sharing, recording
            - Disputes about prior agreements
            - Conditional promises ("I'll pay IF you waive the fee")
            Mark each as legally_significant if it could matter in a dispute."""
        }, {
            "role": "user",
            "content": transcript_text
        }],
        max_retries=2
    )
    
    return result


# For COMPLEX compliance checking → route to Backboard cloud
def check_compliance_cloud(transcript_text: str, backboard_client) -> list[ComplianceCheck]:
    """Use Backboard → cloud LLM for nuanced regulatory compliance"""
    
    # This needs GPT-4o level reasoning — local 8B isn't reliable enough
    # for legal compliance determinations
    response = backboard_client.query_thread(
        question=f"""Review this bank customer call transcript for regulatory compliance.

        Check against:
        1. RBI Fair Practice Code for lending
        2. Fair debt collection practices
        3. Telecom recording consent requirements
        4. Customer privacy protection
        5. Required disclosure statements
        
        For each violation found, cite the specific regulation.
        
        TRANSCRIPT:
        {transcript_text}""",
        response_format="json"
    )
    
    return parse_compliance_response(response)
```

### Stage 5: Auditable Output Generation

```python
# pipeline/output_generator.py
import pandas as pd
import json

def generate_ml_trainable_output(call_record: CallRecord) -> dict:
    """Generate multiple output formats for ML training"""
    
    outputs = {}
    
    # 1. Flat CSV row (one row per call — for classification models)
    flat_record = {
        "call_id": call_record.call_id,
        "duration_sec": call_record.duration_seconds,
        "language": call_record.language,
        "call_type": call_record.call_type,
        "audio_quality_score": call_record.audio_quality_score,
        "transcript_confidence": call_record.overall_transcript_confidence,
        "num_speakers": call_record.num_speakers,
        "agent_talk_pct": call_record.agent_talk_percentage,
        "customer_talk_pct": call_record.customer_talk_percentage,
        "num_obligations": len(call_record.obligations),
        "num_binding_obligations": len([o for o in call_record.obligations 
                                         if o.strength == "binding"]),
        "num_compliance_violations": len([c for c in call_record.compliance_checks 
                                           if not c.passed]),
        "num_fraud_signals": len(call_record.fraud_signals),
        "compliance_score": call_record.compliance_score,
        "risk_level": call_record.overall_risk_level,
        "customer_dominant_emotion": call_record.customer_emotion_dominant,
        "escalation_detected": call_record.escalation_detected,
        "requires_review": call_record.requires_human_review,
    }
    outputs["csv_row"] = flat_record
    
    # 2. JSON Lines (one JSON per utterance — for NER/intent training)
    utterance_records = []
    for intent in call_record.intents:
        utterance_records.append({
            "text": get_segment_text(call_record, intent.segment_id),
            "speaker_role": intent.speaker,
            "intent": intent.intent,
            "intent_confidence": intent.confidence,
            "entities": [e.dict() for e in call_record.financial_entities 
                        if e.segment_id == intent.segment_id],
            "obligations": [o.dict() for o in call_record.obligations
                          if o.segment_id == intent.segment_id],
        })
    outputs["jsonl_utterances"] = utterance_records
    
    # 3. Full structured JSON (complete call record)
    outputs["full_json"] = call_record.dict()
    
    # 4. Audit report summary
    outputs["audit_summary"] = {
        "call_id": call_record.call_id,
        "compliance_score": call_record.compliance_score,
        "risk_level": call_record.overall_risk_level,
        "violations": [c.dict() for c in call_record.compliance_checks if not c.passed],
        "obligations": [o.dict() for o in call_record.obligations if o.legally_significant],
        "review_reasons": call_record.review_reasons,
        "requires_human_review": call_record.requires_human_review,
        "review_priority": call_record.review_priority,
    }
    
    return outputs


def export_batch_to_parquet(call_records: list[CallRecord], output_path: str):
    """Export batch of call records as Parquet for ML training"""
    rows = [generate_ml_trainable_output(r)["csv_row"] for r in call_records]
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
```

---

## REVISED REGULATORY COMPLIANCE LIBRARY

```python
# analysis/compliance.py

# Required disclosure phrases (agent MUST say something equivalent)
REQUIRED_DISCLOSURES = {
    "collections": [
        {"check": "caller_identification", 
         "description": "Agent must identify themselves and their organization",
         "keywords": ["calling from", "my name is", "this is", "speaking from"],
         "regulation": "RBI_Fair_Practice_Code_Section_4"},
        {"check": "purpose_disclosure",
         "description": "Agent must state the purpose of the call",
         "keywords": ["regarding your", "about your", "in reference to", "concerning"],
         "regulation": "RBI_Fair_Practice_Code_Section_4"},
        {"check": "recording_consent",
         "description": "Agent must inform call is being recorded",
         "keywords": ["call is being recorded", "recorded for", "this call may be"],
         "regulation": "IT_Act_2000_Section_43A"},
        {"check": "mini_miranda",
         "description": "Debt collection disclosure (this is an attempt to collect a debt)",
         "keywords": ["collect", "outstanding", "due", "payment", "overdue"],
         "regulation": "Fair_Debt_Collection"},
    ],
    "kyc": [
        {"check": "identity_verification",
         "description": "Must verify customer identity before sharing account info",
         "keywords": ["verify", "confirm", "date of birth", "last four digits", "mother's name"],
         "regulation": "RBI_KYC_Master_Direction"},
        {"check": "data_consent",
         "description": "Must obtain consent for data collection/processing",
         "keywords": ["consent", "agree", "authorize", "permission"],
         "regulation": "Digital_Personal_Data_Protection_Act_2023"},
    ],
    "consent": [
        {"check": "clear_terms",
         "description": "Must clearly state what customer is consenting to",
         "keywords": ["you are agreeing to", "this means", "by saying yes", "consent to"],
         "regulation": "RBI_Digital_Lending_Guidelines"},
        {"check": "right_to_refuse",
         "description": "Must inform customer they can refuse",
         "keywords": ["right to", "not obligated", "can decline", "optional", "your choice"],
         "regulation": "Consumer_Protection_Act_2019"},
    ]
}

# Prohibited phrases (agent must NEVER say these)
PROHIBITED_PHRASES = {
    "threats": [
        "we will send police", "legal action will be taken", "you will go to jail",
        "we will seize your", "we will come to your house", "we will tell your employer",
        "we will inform your", "you will be blacklisted", "we will auction",
        "your reputation will", "everyone will know"
    ],
    "coercion": [
        "you have no choice", "you must pay now", "this is your last chance",
        "we will not stop calling", "you cannot escape", "there is no way out",
        "pay or else", "you better pay", "don't make me"
    ],
    "misleading": [
        "your CIBIL score will become zero", "you can never get a loan again",
        "interest will become 100%", "your salary will be stopped",
        "your passport will be cancelled"
    ],
    "unprofessional": [
        # Detect via sentiment/emotion analysis rather than keyword matching
        # Flag if agent sentiment falls below professional threshold
    ]
}

def run_compliance_checks(transcript_segments, call_type: str) -> list[ComplianceCheck]:
    """Check transcript against regulatory requirements"""
    
    checks = []
    disclosures = REQUIRED_DISCLOSURES.get(call_type, [])
    
    # Check required disclosures (agent speech only, first 2 minutes)
    agent_opening = " ".join([
        s["text"] for s in transcript_segments 
        if s["speaker"] == "agent" and s["start"] < 120  # First 2 min
    ]).lower()
    
    for disclosure in disclosures:
        found = any(kw in agent_opening for kw in disclosure["keywords"])
        checks.append(ComplianceCheck(
            check_name=disclosure["check"],
            passed=found,
            violation_type=ComplianceViolationType.MISSING_DISCLOSURE if not found else None,
            evidence_text=agent_opening[:200] if not found else None,
            regulation=disclosure["regulation"],
            severity="high" if not found else "low"
        ))
    
    # Check prohibited phrases (entire call)
    full_agent_text = " ".join([
        s["text"] for s in transcript_segments if s["speaker"] == "agent"
    ]).lower()
    
    for category, phrases in PROHIBITED_PHRASES.items():
        if category == "unprofessional":
            continue  # Handled by emotion analysis
        for phrase in phrases:
            if phrase.lower() in full_agent_text:
                # Find the exact segment
                seg_id = find_segment_containing(transcript_segments, phrase)
                checks.append(ComplianceCheck(
                    check_name=f"prohibited_{category}",
                    passed=False,
                    violation_type=ComplianceViolationType.PROHIBITED_LANGUAGE,
                    evidence_text=f'Agent said: "{phrase}"',
                    segment_id=seg_id,
                    regulation="RBI_Fair_Practice_Code",
                    severity="critical"
                ))
    
    return checks
```

---

## REVISED DEMO SCRIPT (5 minutes)

### Minute 0:00 — The Hook
> "Banks record millions of calls every day — collections, KYC, complaints, consent. Right now, these calls sit in storage as unstructured audio. Nobody knows if agents followed regulations. Nobody catches verbal commitments that could be legally binding. Nobody spots fraud patterns across calls. We built FinSight to change that."

### Minute 0:30 — Architecture (1 slide)
> "Raw audio goes in. Clean, structured, ML-trainable, audit-ready data comes out. Five stages — all powered by local AI running on two laptops plus Backboard.io for persistent intelligence."

### Minute 1:00 — Live Demo
> Upload a collections call audio file.
> Show processing stages: Audio Quality ✓ → Transcription ✓ → Intelligence ✓ → Compliance ✓
> "42 seconds. From raw audio to full structured analysis."

### Minute 1:30 — The Structured Output
> Show the CallRecord JSON: entities extracted, intents classified, obligations detected.
> "Every field is traceable to an exact timestamp in the audio. Click any entity — hear the original audio."
> Show: "I'll pay five thousand by Friday" → tagged as PAYMENT_PROMISE, CONDITIONAL, ₹5000, 2026-02-13

### Minute 2:00 — Compliance Scorecard
> Show compliance checks: ✅ Caller ID, ✅ Purpose disclosed, ❌ Recording consent missing, ❌ Prohibited language detected
> "The agent said 'we will send police to your house.' That's an automatic critical violation of RBI Fair Practice Code Section 4."
> Click the violation → audio jumps to that exact moment.

### Minute 2:30 — Fraud Detection
> Show vocal stress chart spiking during identity verification.
> "Voice stress analysis detected anomalies when the caller was asked for their date of birth. Combined with inconsistent account details, this call is flagged for fraud review."

### Minute 3:00 — The Backboard Moment
> Open chat: "Show me all calls with this customer where payment promises were made."
> Backboard retrieves from persistent memory: 3 calls, 3 promises, zero payments.
> "That's Backboard.io — not just memory, but a complete intelligence layer. Five capabilities deep: persistent memory, hybrid RAG search, cloud LLM routing, embeddings, and audit trails."

### Minute 3:30 — ML-Trainable Output
> Show: Export as Parquet. "10,000 calls → structured dataset → train a model to predict which customers will default on their payment promises."
> Show: JSON Lines format for NER/intent training data.
> "Every call we process makes the next model better."

### Minute 4:00 — Scale Story
> "Two laptops today. The same pipeline scales to millions of calls. Local AI for speed and privacy — no customer data leaves the bank's network. Backboard.io for intelligence that compounds over time."

### Minute 4:30 — Close
> "FinSight: from raw audio to auditable intelligence. Structured. Searchable. ML-trainable. Compliant. Questions?"

---

## REVISED TEAM ASSIGNMENTS (What Changes)

The team structure STAYS THE SAME — 5 people, same GPU roles. Only the domain logic changes:

### PERSON 1 (Architect) — CHANGES:
- Replace earnings call schemas with CallRecord schemas above
- Backboard threads per CUSTOMER instead of per COMPANY
- Add audit trail thread creation
- Add regulatory document upload to Backboard RAG

### PERSON 2 (Audio Engineer) — ADDS:
- Audio normalization (FFmpeg, Stage 1) — NEW
- Audio quality scoring (SNR, Silero VAD) — NEW  
- Language detection — NEW
- Agent vs Customer role assignment from diarization — ENHANCED
- Voice stress for fraud detection stays same, just different context

### PERSON 3 (NLP Scientist) — PIVOTS:
- Evasion detection → Intent classification (agree/refuse/stall)
- Forward guidance → Obligation detection (payment promises, consent)
- Trust Score → Compliance Scorecard
- CEO-CFO divergence → Agent tone compliance monitoring
- NEW: Regulatory compliance checking (keyword + LLM)
- NEW: Financial term correction dictionary
- NEW: Prohibited phrase detection

### PERSON 4 (Frontend Wizard) — PIVOTS:
- Evasion heatmap → Compliance scorecard visualization
- Tone divergence chart → Customer sentiment trajectory
- Trust gauge → Risk level indicator
- NEW: Human review queue with priority sorting
- NEW: Click-to-audio source linking
- NEW: Export buttons (JSON, CSV, Parquet)
- Chat stays same (Backboard-powered Q&A)

### PERSON 5 (Integrator) — PIVOTS:
- Earnings call datasets → Generate synthetic bank call data
  (Use TTS to create collections calls, KYC calls, complaint calls)
- LightRAG stays but indexes call records instead of earnings transcripts  
- Demo data: create 5 compelling synthetic calls showing different scenarios
  (compliant call, violation call, fraud call, payment promise call, consent dispute)
- Presentation pivot to match new narrative

---

## SYNTHETIC DEMO DATA STRATEGY

You don't have real bank call recordings. Generate compelling demo data:

```python
# scripts/generate_demo_calls.py
# Use a TTS model to generate realistic multi-speaker call audio

DEMO_SCENARIOS = [
    {
        "name": "compliant_collection",
        "description": "Model collection call — agent follows all rules",
        "script": [
            ("agent", "Good morning, my name is Priya calling from HDFC Bank. "
                      "This call is being recorded for quality purposes. "
                      "I'm calling regarding your personal loan account ending 4532."),
            ("customer", "Yes, I know about the EMI. I've been meaning to pay."),
            ("agent", "Your EMI of rupees twelve thousand five hundred was due on January 5th. "
                      "We noticed it hasn't been received yet. Would you like to make the payment today?"),
            ("customer", "I can pay five thousand by this Friday and the rest by month end."),
            ("agent", "Thank you. So to confirm, you'll pay rupees five thousand by February 14th "
                      "and rupees seven thousand five hundred by February 28th. Is that correct?"),
            ("customer", "Yes, that's correct."),
        ],
        "expected_flags": ["payment_promise", "partial_agreement"]
    },
    {
        "name": "violation_collection",
        "description": "Agent uses prohibited language — compliance violations",
        "script": [
            ("agent", "Hello, this is regarding your loan."),  # Missing name + recording disclosure
            ("customer", "I already told you I can't pay right now."),
            ("agent", "Sir, if you don't pay immediately, we will have to send police to your house. "
                      "Your CIBIL score will become zero and you can never get a loan again."),
            ("customer", "You can't threaten me like that!"),
            ("agent", "This is your last chance. Pay or else we will tell your employer about this debt."),
        ],
        "expected_flags": ["prohibited_threat", "missing_disclosure", "coercion", "misleading_info"]
    },
    {
        "name": "fraud_kyc",
        "description": "Suspicious KYC call with potential identity fraud",
        "script": [
            ("agent", "This is Amit from SBI. This call is recorded. "
                      "I need to verify your identity for the account update request. "
                      "Can you please confirm your date of birth?"),
            ("customer", "Uh... it's... March... 15th... 1990."),  # Hesitation = stress flag
            ("agent", "And your mother's maiden name?"),
            ("customer", "Um... let me think... it's... Sharma."),
            ("agent", "The name on file doesn't match. Can you verify your PAN number?"),
            ("customer", "I don't have it right now, can you just process it anyway?"),
        ],
        "expected_flags": ["voice_stress_anomaly", "inconsistent_info", "verification_failure"]
    },
]
```

---

## CRITICAL TIMELINE ADJUSTMENT

The pivot doesn't cost you much time because the INFRASTRUCTURE stays the same. Here's what changes in the timeline:

### Hours 0-8: Same foundation setup + NEW schemas
### Hours 8-14: Person 3 builds compliance library + intent classifier (instead of evasion detector)  
### Hours 8-14: Person 5 generates synthetic demo calls (instead of downloading earnings data)
### Hours 14-25: Same integration work, different domain logic
### Hours 25-45: Same polish cycle
### Hours 45-65: Demo rehearsal with new narrative

**Net time impact: ~4-6 hours of rework, mostly in Person 3 and Person 5's domain logic.**
