"""Batch process all judge dataset calls through the FinVoice pipeline.

Usage:
    python scripts/batch_process.py [--data-dir data/judge_dataset] [--output-dir data/processed/batch_results]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Apply PyTorch 2.8 torch.load patch before any model imports
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if kwargs.get("weights_only") is None:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from pipeline.orchestrator import process_call

# Language detection from folder names
LANGUAGE_MAP = {
    "EN": "en",
    "RU": "ru",
    "PL": "pl",
    "FR": "fr",
    "DE": "de",
    "ES": "es",
    "Portuguese": "pt",
}


def detect_language(folder_name: str) -> str:
    """Detect language code from judge dataset folder name."""
    for key, code in LANGUAGE_MAP.items():
        if f"({key} " in folder_name or f"({key})" in folder_name:
            return code
    return "en"  # Default to English


def detect_call_type(folder_name: str) -> str:
    """Detect call type from judge dataset folder name."""
    lower = folder_name.lower()
    if "billing" in lower or "account" in lower:
        return "collections"
    elif "sale" in lower:
        return "general"
    elif "support" in lower or "customer service" in lower:
        return "complaint"
    elif "pharma" in lower:
        return "general"
    elif "finance" in lower:
        return "general"
    return "general"


def find_audio_files(data_dir: str) -> list[dict]:
    """Find all MP3/WAV files in the judge dataset with metadata."""
    files = []
    base = Path(data_dir) / "Call center data samples"
    if not base.exists():
        base = Path(data_dir)

    audio_extensions = ("*.mp3", "*.wav", "*.m4a", "*.ogg", "*.flac", "*.wma", "*.aac")
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        for ext in audio_extensions:
            for audio_file in folder.glob(ext):
                files.append({
                    "path": str(audio_file),
                    "folder": folder.name,
                    "language": detect_language(folder.name),
                    "call_type": detect_call_type(folder.name),
                })

    return files


def main():
    parser = argparse.ArgumentParser(description="Batch process judge dataset calls")
    parser.add_argument("--data-dir", default="data/judge_dataset", help="Judge dataset directory")
    parser.add_argument("--output-dir", default="data/processed/batch_results", help="Output directory")
    parser.add_argument("--language", default=None, help="Filter by language code (e.g., 'en')")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set — diarization will be skipped")

    audio_files = find_audio_files(args.data_dir)
    if args.language:
        audio_files = [f for f in audio_files if f["language"] == args.language]

    logger.info(f"Found {len(audio_files)} audio files to process")
    for f in audio_files:
        logger.info(f"  {f['folder']} — lang={f['language']}, type={f['call_type']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_start = time.time()

    for i, file_info in enumerate(audio_files):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {i+1}/{len(audio_files)}: {file_info['folder']}")
        logger.info(f"{'='*60}")

        start = time.time()
        try:
            record = process_call(
                input_audio_path=file_info["path"],
                call_type=file_info["call_type"],
                hf_token=hf_token,
                output_dir=str(output_dir),
            )
            elapsed = time.time() - start

            summary = {
                "call_id": record.call_id,
                "folder": file_info["folder"],
                "language": file_info["language"],
                "call_type": file_info["call_type"],
                "duration_s": record.duration_seconds,
                "processing_time_s": round(elapsed, 1),
                "num_speakers": record.num_speakers,
                "num_segments": len(record.intents),
                "num_entities": len(record.financial_entities),
                "num_obligations": len(record.obligations),
                "compliance_score": record.compliance_score,
                "risk_level": record.overall_risk_level.value,
                "transcript_confidence": record.overall_transcript_confidence,
                "status": "success",
            }
            results.append(summary)

            logger.info(f"Completed in {elapsed:.1f}s — risk={record.overall_risk_level.value}, "
                       f"compliance={record.compliance_score}/100")

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"FAILED after {elapsed:.1f}s: {e}")
            results.append({
                "folder": file_info["folder"],
                "language": file_info["language"],
                "status": "failed",
                "error": str(e),
                "processing_time_s": round(elapsed, 1),
            })

    total_elapsed = time.time() - total_start

    # Save summary
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print results table
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE — {len(results)} calls in {total_elapsed:.0f}s")
    print(f"{'='*80}")
    print(f"{'Folder':<45} {'Lang':>4} {'Time':>6} {'Risk':>8} {'Score':>5} {'Status':>7}")
    print("-" * 80)
    for r in results:
        folder = r["folder"][:44]
        lang = r.get("language", "?")
        t = f"{r['processing_time_s']:.0f}s"
        risk = r.get("risk_level", "?")
        score = str(r.get("compliance_score", "?"))
        status = r["status"]
        print(f"{folder:<45} {lang:>4} {t:>6} {risk:>8} {score:>5} {status:>7}")

    print(f"\nResults saved to: {summary_path}")
    succeeded = sum(1 for r in results if r["status"] == "success")
    print(f"Success: {succeeded}/{len(results)}")


if __name__ == "__main__":
    main()
