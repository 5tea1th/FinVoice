"""Discover Financial Term Corrections — data-driven, not guessed.

Runs WhisperX on the Earnings-22 dataset (119 hours of earnings calls with
human-transcribed ground truth, 125 files, 27 countries) and diffs the output
against ground truth to discover consistent misrecognitions of financial terms.

Dataset: revdotcom/earnings22 on HuggingFace (free, no auth needed)
Output: data/models/financial_corrections.json

Usage:
    python scripts/discover_corrections.py [--max-files 5] [--min-count 3]
"""

import os
import sys
import json
import re
import difflib
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _cleanup_stale_locks(cache_dir: str):
    """Remove stale .lock files that prevent dataset downloads."""
    cache_path = Path(cache_dir)
    if cache_path.exists():
        for lock_file in cache_path.rglob("*.lock"):
            try:
                lock_file.unlink()
                logger.info(f"Removed stale lock: {lock_file}")
            except OSError:
                pass


def download_earnings21(cache_dir: str = "data/datasets/earnings21", max_retries: int = 3) -> list[dict]:
    """Download Earnings-21 dataset from HuggingFace.

    Uses streaming=True to avoid bulk download timeout.
    Retries with exponential backoff (10s, 20s, 40s).

    Returns list of dicts with audio and transcript keys.
    """
    import time as _time
    from datasets import load_dataset

    # Clean up stale lock files from previous failed downloads
    _cleanup_stale_locks(cache_dir)

    for attempt in range(max_retries):
        try:
            logger.info(f"Loading Earnings-21 dataset (attempt {attempt + 1}/{max_retries})...")
            ds = load_dataset(
                "Revai/earnings21",
                split="test",
                cache_dir=cache_dir,
                streaming=True,
            )
            # Convert streaming dataset to list (we only need a few samples)
            logger.info("Streaming Earnings-21 — collecting samples...")
            samples = []
            for sample in ds:
                samples.append(sample)
            logger.info(f"Loaded {len(samples)} samples from Earnings-21")
            return samples
        except Exception as e:
            wait = 10 * (2 ** attempt)
            logger.warning(f"Earnings-21 load failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {wait}s...")
                _time.sleep(wait)
                _cleanup_stale_locks(cache_dir)
            else:
                logger.error(f"All {max_retries} attempts failed for Earnings-21")
                raise


def transcribe_sample(audio_path: str, language: str = "en") -> str:
    """Transcribe a single audio file with WhisperX."""
    import whisperx
    import torch

    model = whisperx.load_model(
        whisper_arch="large-v3-turbo",
        device="cuda",
        compute_type="int8",
        language=language,
    )

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=4)

    # Free VRAM
    del model
    torch.cuda.empty_cache()

    # Combine all segment texts
    text = " ".join(seg["text"].strip() for seg in result["segments"])
    return text


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def diff_transcripts(whisper_text: str, ground_truth: str, min_ratio: float = 0.6) -> list[tuple[str, str]]:
    """Find word-level differences between Whisper output and ground truth.

    Returns list of (whisper_word, ground_truth_word) mismatches.
    """
    w_words = normalize_text(whisper_text).split()
    gt_words = normalize_text(ground_truth).split()

    matcher = difflib.SequenceMatcher(None, w_words, gt_words, autojunk=False)
    mismatches = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            # One-to-one replacements are most informative
            w_chunk = w_words[i1:i2]
            gt_chunk = gt_words[j1:j2]

            if len(w_chunk) == len(gt_chunk):
                for w, g in zip(w_chunk, gt_chunk):
                    if w != g:
                        mismatches.append((w, g))
            elif len(w_chunk) == 1 and len(gt_chunk) == 1:
                mismatches.append((w_chunk[0], gt_chunk[0]))

    return mismatches


# Known financial terms to watch for in corrections
FINANCIAL_TERMS = {
    # Acronyms
    "emi", "kyc", "cibil", "hdfc", "icici", "sbi", "rbi", "sebi", "irdai",
    "nbfc", "npa", "nach", "upi", "gst", "pan", "tan", "neft", "rtgs",
    "imps", "ecs", "ebitda", "eps", "pe", "roe", "roa", "cagr", "nav",
    "aum", "sip", "nfo", "etf", "ipo", "fpo", "ofs", "qip", "esop",
    # Financial terms
    "demat", "repo", "crore", "lakh", "rupee", "rupees",
    # Common earnings call terms
    "revenue", "margin", "guidance", "consensus", "dividend", "buyback",
    "capex", "opex", "amortization", "depreciation", "goodwill",
    "receivable", "payable", "leverage", "covenant", "maturity",
    "yield", "spread", "basis points", "bps",
}


def is_financial_correction(whisper_word: str, gt_word: str) -> bool:
    """Check if a correction involves a financial term."""
    return gt_word.lower() in FINANCIAL_TERMS or whisper_word.lower() in FINANCIAL_TERMS


def discover_corrections(
    max_files: int = 10,
    min_count: int = 3,
    output_path: str = "data/models/financial_corrections.json",
) -> dict:
    """Main discovery pipeline.

    1. Download Earnings-21
    2. Transcribe with WhisperX
    3. Diff against ground truth
    4. Find consistent misrecognitions
    5. Save corrections dictionary
    """
    ds = download_earnings22()

    all_mismatches = Counter()
    financial_mismatches = Counter()
    correction_examples = defaultdict(list)

    num_to_process = min(max_files, len(ds))
    logger.info(f"Processing {num_to_process} files from Earnings-21...")

    for idx in range(num_to_process):
        sample = ds[idx]

        # Get audio and ground truth
        audio = sample.get("audio")
        ground_truth = sample.get("transcript", sample.get("text", ""))

        if not ground_truth:
            logger.warning(f"Sample {idx} has no ground truth, skipping")
            continue

        # Get audio file path (decode=False gives us raw bytes or path)
        temp_path = None
        if audio and isinstance(audio, dict):
            if "path" in audio and audio["path"]:
                temp_path = audio["path"]
            elif "bytes" in audio and audio["bytes"]:
                # Write raw bytes to temp file
                temp_path = f"data/transcripts/_earnings21_temp_{idx}.wav"
                Path(temp_path).parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "wb") as af:
                    af.write(audio["bytes"])
            elif "array" in audio:
                import soundfile as sf
                import numpy as _np
                temp_path = f"data/transcripts/_earnings21_temp_{idx}.wav"
                Path(temp_path).parent.mkdir(parents=True, exist_ok=True)
                sf.write(temp_path, _np.array(audio["array"]), audio["sampling_rate"])

        if not temp_path:
            logger.warning(f"Sample {idx} has no audio data, skipping")
            continue

        logger.info(f"[{idx+1}/{num_to_process}] Transcribing ({len(ground_truth)} chars GT)...")

        try:
            whisper_text = transcribe_sample(temp_path)
        except Exception as e:
            logger.warning(f"Transcription failed for sample {idx}: {e}")
            continue

        # Diff and collect mismatches
        mismatches = diff_transcripts(whisper_text, ground_truth)
        logger.info(f"  Found {len(mismatches)} word-level differences")

        for w_word, gt_word in mismatches:
            key = (w_word, gt_word)
            all_mismatches[key] += 1

            if is_financial_correction(w_word, gt_word):
                financial_mismatches[key] += 1

            correction_examples[key].append({
                "sample_idx": idx,
                "whisper": w_word,
                "ground_truth": gt_word,
            })

        # Clean up temp file
        if temp_path.startswith("data/"):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    # Filter to consistent corrections (appearing min_count+ times)
    corrections = {}

    # Priority 1: Financial term corrections
    for (w_word, gt_word), count in financial_mismatches.items():
        if count >= min_count:
            corrections[w_word] = gt_word
            logger.info(f"  FINANCIAL: '{w_word}' → '{gt_word}' ({count} occurrences)")

    # Priority 2: High-frequency general corrections (might be financial-adjacent)
    for (w_word, gt_word), count in all_mismatches.most_common(200):
        if count >= min_count and w_word not in corrections:
            # Only include if the correction is meaningful (not just minor spelling)
            if len(w_word) >= 3 and len(gt_word) >= 3:
                ratio = difflib.SequenceMatcher(None, w_word, gt_word).ratio()
                if ratio < 0.8:  # Significantly different words
                    corrections[w_word] = gt_word
                    logger.info(f"  GENERAL: '{w_word}' → '{gt_word}' ({count} occurrences)")

    # Save corrections
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(corrections, f, indent=2, sort_keys=True)

    logger.info(f"\nDiscovery complete!")
    logger.info(f"  Total mismatches found: {sum(all_mismatches.values())}")
    logger.info(f"  Financial corrections: {len([k for k in corrections if k in dict(financial_mismatches)])}")
    logger.info(f"  Total corrections saved: {len(corrections)}")
    logger.info(f"  Output: {output_path}")

    # Also save detailed report
    report_path = output_path.replace(".json", "_report.json")
    report = {
        "total_samples_processed": num_to_process,
        "total_mismatches": sum(all_mismatches.values()),
        "unique_mismatches": len(all_mismatches),
        "corrections_saved": len(corrections),
        "top_50_mismatches": [
            {"whisper": w, "ground_truth": g, "count": c}
            for (w, g), c in all_mismatches.most_common(50)
        ],
        "corrections": corrections,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Report: {report_path}")

    return corrections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover Whisper financial term corrections")
    parser.add_argument("--max-files", type=int, default=3,
                        help="Max audio files to process (default: 3)")
    parser.add_argument("--min-count", type=int, default=3,
                        help="Min occurrences for a correction to be saved (default: 3)")
    parser.add_argument("--output", type=str, default="data/models/financial_corrections.json",
                        help="Output path for corrections JSON")
    args = parser.parse_args()

    corrections = discover_corrections(
        max_files=args.max_files,
        min_count=args.min_count,
        output_path=args.output,
    )

    print(f"\n{'='*60}")
    print(f"Corrections discovered: {len(corrections)}")
    print(f"{'='*60}")
    for wrong, right in sorted(corrections.items()):
        print(f"  '{wrong}' → '{right}'")
