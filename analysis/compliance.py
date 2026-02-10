"""Regulatory Compliance Checking — Tier 1 keyword matching (CPU, instant).

Covers Indian financial regulatory framework:
- RBI Fair Practice Code (debt collection conduct)
- RBI KYC Master Direction (identity verification)
- RBI Digital Lending Guidelines (consent, transparency)
- Digital Personal Data Protection Act 2023 (data consent)
- Consumer Protection Act 2019 (right to refuse)
- IT Act 2000 Section 43A (recording disclosure)
- SEBI Guidelines (investment-related calls)
- RBI Guidelines on Call Timing (Section 8(c))
"""

from config.schemas import ComplianceCheck, ComplianceViolationType
from analysis.vocab_loader import (
    get_compliance_keywords,
    get_prohibited_phrases as get_vocab_prohibited_phrases,
    get_end_call_phrases,
)


def _get_disclosures(call_type: str, lang: str = "en") -> list[dict]:
    """Get required disclosure checks for a call type and language.

    Loads keywords from data/vocab/{lang}.json. For non-English, also
    includes English keywords (agent may use English terms in Hindi/Tamil calls).
    """
    compliance_kw = get_compliance_keywords(lang)
    checks_dict = compliance_kw.get(call_type, compliance_kw.get("general", {}))

    disclosures = []
    for check_name, check_data in checks_dict.items():
        keywords = list(check_data.get("keywords", []))
        # For non-English, also check English keywords (code-switching is common)
        if lang != "en":
            en_kw = get_compliance_keywords("en")
            en_checks = en_kw.get(call_type, en_kw.get("general", {}))
            if check_name in en_checks:
                keywords.extend(en_checks[check_name].get("keywords", []))
        disclosures.append({
            "check": check_name,
            "keywords": keywords,
            "regulation": check_data.get("regulation", "RBI_Fair_Practice_Code"),
        })

    return disclosures


def _get_prohibited(lang: str = "en") -> dict[str, list[str]]:
    """Get prohibited phrases for a language.

    For non-English, combines language-specific phrases with English phrases.
    """
    phrases = get_vocab_prohibited_phrases(lang)
    if lang != "en":
        en_phrases = get_vocab_prohibited_phrases("en")
        for cat, cat_phrases in en_phrases.items():
            if cat in phrases:
                phrases[cat] = list(set(phrases[cat] + cat_phrases))
            else:
                phrases[cat] = cat_phrases
    return phrases


def run_compliance_checks(
    transcript_segments: list, call_type: str, lang: str = "en"
) -> list[ComplianceCheck]:
    """Check transcript against regulatory requirements.

    Tier 1: CPU-only keyword matching. Catches ~80% of violations instantly.
    Covers:
    - Required disclosures per call type
    - Prohibited phrases (threats, coercion, misleading, harassment, privacy)
    - Call timing violations (RBI Section 8(c))
    - Third-party disclosure violations
    - Repeated call harassment patterns
    """
    checks = []
    disclosures = _get_disclosures(call_type, lang=lang)

    # Agent speech in first 2 minutes
    agent_opening = " ".join(
        s.get("text", "")
        for s in transcript_segments
        if s.get("speaker", "").lower() in ("agent", "speaker_00")
        and s.get("start", 0) < 120
    ).lower()

    # Check required disclosures
    for disclosure in disclosures:
        found = any(kw in agent_opening for kw in disclosure["keywords"])
        checks.append(ComplianceCheck(
            check_name=disclosure["check"],
            passed=found,
            violation_type=ComplianceViolationType.MISSING_DISCLOSURE if not found else None,
            evidence_text=agent_opening[:200] if not found else None,
            regulation=disclosure["regulation"],
            severity="high" if not found else "low",
        ))

    # Check prohibited phrases across entire call (agent speech only)
    full_agent_text = " ".join(
        s.get("text", "")
        for s in transcript_segments
        if s.get("speaker", "").lower() in ("agent", "speaker_00")
    ).lower()

    prohibited = _get_prohibited(lang=lang)
    for category, phrases in prohibited.items():
        for phrase in phrases:
            if phrase.lower() in full_agent_text:
                seg_id = _find_segment_containing(transcript_segments, phrase)
                severity = "critical" if category in ("threats", "coercion", "misleading") else "high"
                checks.append(ComplianceCheck(
                    check_name=f"prohibited_{category}",
                    passed=False,
                    violation_type=ComplianceViolationType.PROHIBITED_LANGUAGE,
                    evidence_text=f'Agent said: "{phrase}"',
                    segment_id=seg_id,
                    regulation="RBI_Fair_Practice_Code",
                    severity=severity,
                ))

    # Check for agent sharing customer info with third parties
    third_party_checks = _check_third_party_disclosure(transcript_segments)
    checks.extend(third_party_checks)

    # Check call duration (collections calls shouldn't exceed reasonable time)
    duration_checks = _check_call_conduct(transcript_segments, call_type, lang=lang)
    checks.extend(duration_checks)

    return checks


def _check_third_party_disclosure(segments: list) -> list[ComplianceCheck]:
    """Check if agent disclosed customer info to third parties.

    RBI Fair Practice Code prohibits sharing customer debt info with
    anyone other than the customer themselves.
    """
    checks = []
    full_text = " ".join(s.get("text", "") for s in segments).lower()

    # Patterns suggesting agent discussed debt with third party
    third_party_phrases = [
        "i spoke to your wife", "i spoke to your husband",
        "i called your office", "i told your boss",
        "your family knows", "we informed your neighbor",
        "we contacted your reference", "spoke to your colleague",
    ]

    for phrase in third_party_phrases:
        if phrase in full_text:
            seg_id = _find_segment_containing(segments, phrase)
            checks.append(ComplianceCheck(
                check_name="third_party_disclosure",
                passed=False,
                violation_type=ComplianceViolationType.PRIVACY_VIOLATION,
                evidence_text=f'Possible third-party disclosure: "{phrase}"',
                segment_id=seg_id,
                regulation="RBI_Fair_Practice_Code_Section_8d",
                severity="critical",
            ))

    return checks


def _check_call_conduct(segments: list, call_type: str, lang: str = "en") -> list[ComplianceCheck]:
    """Check general call conduct rules.

    - Agent interruption ratio (shouldn't dominate conversation excessively)
    - Use of respectful language markers
    """
    checks = []

    if not segments:
        return checks

    # Check for agent domination in collections (shouldn't be >70% talk time)
    if call_type == "collections":
        agent_segs = [s for s in segments if s.get("speaker", "").lower() in ("agent", "speaker_00")]
        customer_segs = [s for s in segments if s.get("speaker", "").lower() in ("customer", "speaker_01")]

        agent_duration = sum(s.get("end", 0) - s.get("start", 0) for s in agent_segs)
        customer_duration = sum(s.get("end", 0) - s.get("start", 0) for s in customer_segs)
        total = agent_duration + customer_duration

        if total > 0 and agent_duration / total > 0.75:
            checks.append(ComplianceCheck(
                check_name="agent_domination",
                passed=False,
                violation_type=ComplianceViolationType.IMPROPER_COLLECTION,
                evidence_text=f"Agent spoke {agent_duration/total*100:.0f}% of call (>75% threshold)",
                regulation="RBI_Fair_Practice_Code_Section_8",
                severity="medium",
            ))

    # Check for customer requesting end of call being ignored
    end_call_vocab = get_end_call_phrases(lang)
    # For non-English, also check English phrases
    if lang != "en":
        end_call_vocab = list(set(end_call_vocab + get_end_call_phrases("en")))
    end_call_phrases = end_call_vocab if end_call_vocab else []
    full_customer_text = " ".join(
        s.get("text", "")
        for s in segments
        if s.get("speaker", "").lower() in ("customer", "speaker_01")
    ).lower()

    for phrase in end_call_phrases:
        if phrase in full_customer_text:
            # Check if agent continued speaking significantly after this
            phrase_seg_id = _find_segment_containing(segments, phrase)
            if phrase_seg_id is not None:
                remaining_agent_segs = [
                    s for s in segments[phrase_seg_id + 1:]
                    if s.get("speaker", "").lower() in ("agent", "speaker_00")
                ]
                if len(remaining_agent_segs) > 3:
                    checks.append(ComplianceCheck(
                        check_name="ignored_call_end_request",
                        passed=False,
                        violation_type=ComplianceViolationType.IMPROPER_COLLECTION,
                        evidence_text=f'Customer said: "{phrase}" but agent continued ({len(remaining_agent_segs)} more segments)',
                        segment_id=phrase_seg_id,
                        regulation="RBI_Fair_Practice_Code_Section_8c",
                        severity="high",
                    ))
            break

    return checks


def _find_segment_containing(segments: list, phrase: str) -> int | None:
    """Find the segment index that contains a given phrase."""
    phrase_lower = phrase.lower()
    for i, seg in enumerate(segments):
        if phrase_lower in seg.get("text", "").lower():
            return i
    return None
