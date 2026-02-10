"""Build a curated test dataset of ~10 financial audio files.

Downloads individual files from free sources. No bulk downloads.

Target: ~10 files covering different financial call types and languages:
- 2x English earnings calls (Revai/earnings21 via HF API rows endpoint)
- 2x English call center (judge dataset v1 — local copy)
- 3x Hindi speech (ai4bharat/IndicVoices via streaming)
- 2x Tamil speech (ai4bharat/IndicVoices via streaming)
- 1x Indian English (ai4bharat/IndicVoices via streaming)

Usage:
    python scripts/build_test_dataset.py
"""

import io
import os
import sys
import json
import shutil
import requests
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path("data/test_dataset")


def _save_audio_from_dict(audio_data, output_path: str, target_sr: int = 16000) -> bool:
    """Save HuggingFace audio dict (with array + sampling_rate) to WAV."""
    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        logger.error("soundfile/numpy required")
        return False

    try:
        if isinstance(audio_data, dict):
            if "array" in audio_data and audio_data["array"] is not None:
                arr = np.array(audio_data["array"], dtype=np.float32)
                sr = audio_data.get("sampling_rate", target_sr)
                if sr != target_sr:
                    try:
                        import librosa
                        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
                        sr = target_sr
                    except ImportError:
                        pass
                sf.write(output_path, arr, sr)
                return True
            elif "bytes" in audio_data and audio_data["bytes"]:
                with open(output_path, "wb") as f:
                    f.write(audio_data["bytes"])
                return True
            elif "path" in audio_data and audio_data["path"]:
                shutil.copy2(audio_data["path"], output_path)
                return True
        elif isinstance(audio_data, (bytes, bytearray)):
            with open(output_path, "wb") as f:
                f.write(audio_data)
            return True
    except Exception as e:
        logger.error(f"Failed to save audio to {output_path}: {e}")
    return False


def _get_duration(path: str) -> float:
    """Get audio duration in seconds."""
    try:
        import soundfile as sf
        return sf.info(path).duration
    except Exception:
        return 0.0


def _trim_audio(path: str, max_seconds: float):
    """Trim audio to max_seconds."""
    try:
        import soundfile as sf
        data, sr = sf.read(path)
        max_samples = int(max_seconds * sr)
        if len(data) > max_samples:
            sf.write(path, data[:max_samples], sr)
            logger.info(f"    Trimmed to {max_seconds:.0f}s")
    except Exception as e:
        logger.warning(f"Could not trim: {e}")


def download_earnings_via_api(n: int = 2) -> list[dict]:
    """Download earnings call audio via HuggingFace rows API (bypasses torchcodec).

    Uses the dataset viewer API to get audio URLs directly.
    """
    files = []
    logger.info(f"Fetching {n} earnings call samples via HF API...")

    try:
        # Use the rows API to get audio file URLs
        url = "https://datasets-server.huggingface.co/rows"
        params = {
            "dataset": "Revai/earnings21",
            "config": "default",
            "split": "test",
            "offset": 0,
            "length": n,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for i, row_data in enumerate(data.get("rows", [])):
            row = row_data.get("row", {})
            company = row.get("company_name", f"company_{i}")
            sector = row.get("sector", "unknown")

            # Audio comes as a list with src URL
            audio_info = row.get("audio")
            if not audio_info:
                logger.warning(f"  Sample {i}: no audio field")
                continue

            # The API returns audio as [{"src": "url", "type": "audio/wav"}]
            audio_url = None
            if isinstance(audio_info, list) and audio_info:
                audio_url = audio_info[0].get("src")
            elif isinstance(audio_info, dict):
                audio_url = audio_info.get("src")

            if not audio_url:
                logger.warning(f"  Sample {i}: no audio URL in response")
                continue

            safe_name = company.replace(" ", "_").replace("/", "_")[:30]
            filename = f"earnings_{safe_name}.wav"
            output_path = str(OUTPUT_DIR / filename)

            logger.info(f"  Downloading: {company} ({sector})...")
            try:
                audio_resp = requests.get(audio_url, timeout=120, stream=True)
                audio_resp.raise_for_status()

                with open(output_path, "wb") as f:
                    for chunk in audio_resp.iter_content(chunk_size=8192):
                        f.write(chunk)

                duration = _get_duration(output_path)

                # Trim to 5 min max
                if duration > 300:
                    _trim_audio(output_path, 300)
                    duration = 300.0

                files.append({
                    "filename": filename,
                    "source": "Revai/earnings21",
                    "type": "earnings_call",
                    "language": "en",
                    "description": f"{company} ({sector}) earnings call",
                    "duration_s": round(duration, 1),
                })
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"  Saved: {filename} ({duration:.0f}s, {size_mb:.1f}MB)")

            except Exception as e:
                logger.error(f"  Download failed for {company}: {e}")

    except Exception as e:
        logger.error(f"HF API request failed: {e}")

    return files


def download_indicvoices_samples(languages: dict[str, int]) -> list[dict]:
    """Download speech samples from ai4bharat/IndicVoices via streaming.

    Supports Hindi (hi) and Tamil (ta).
    """
    from datasets import load_dataset

    lang_names = {"hi": "Hindi", "ta": "Tamil", "en": "Indian English"}
    files = []

    for lang, n in languages.items():
        lang_name = lang_names.get(lang, lang)
        logger.info(f"Streaming {n} {lang_name} samples from ai4bharat/IndicVoices...")

        try:
            ds = load_dataset(
                "ai4bharat/IndicVoices",
                lang,
                split="train",
                streaming=True,
            )
            count = 0
            for i, sample in enumerate(ds):
                if count >= n:
                    break
                if i > 300:  # Safety limit
                    break

                audio = sample.get("audio")
                text = sample.get("transcript", sample.get("text", sample.get("sentence", "")))

                # Skip very short
                if text and len(text) < 15:
                    continue

                filename = f"{lang_name.lower().replace(' ', '_')}_indicvoices_{count+1}.wav"
                output_path = str(OUTPUT_DIR / filename)

                if audio and _save_audio_from_dict(audio, output_path):
                    duration = _get_duration(output_path)
                    if duration < 2:
                        os.remove(output_path)
                        continue

                    files.append({
                        "filename": filename,
                        "source": "ai4bharat/IndicVoices",
                        "type": "speech_sample",
                        "language": lang,
                        "description": f"{lang_name}: {text[:80]}" if text else f"{lang_name} speech",
                        "duration_s": round(duration, 1),
                    })
                    logger.info(f"  Saved: {filename} ({duration:.0f}s)")
                    count += 1

        except Exception as e:
            logger.error(f"IndicVoices {lang} failed: {e}")

            # Fallback: try Shrutilipi
            logger.info(f"  Trying fallback: ai4bharat/Shrutilipi ({lang})...")
            try:
                ds2 = load_dataset(
                    "ai4bharat/Shrutilipi",
                    lang,
                    split="train",
                    streaming=True,
                )
                count2 = len([f for f in files if f["language"] == lang])
                for j, sample in enumerate(ds2):
                    if count2 >= n:
                        break
                    if j > 300:
                        break

                    audio = sample.get("audio")
                    text = sample.get("transcript", sample.get("text", ""))
                    if text and len(text) < 15:
                        continue

                    filename = f"{lang_name.lower().replace(' ', '_')}_shrutilipi_{count2+1}.wav"
                    output_path = str(OUTPUT_DIR / filename)

                    if audio and _save_audio_from_dict(audio, output_path):
                        duration = _get_duration(output_path)
                        if duration < 2:
                            os.remove(output_path)
                            continue
                        files.append({
                            "filename": filename,
                            "source": "ai4bharat/Shrutilipi",
                            "type": "speech_sample",
                            "language": lang,
                            "description": f"{lang_name}: {text[:80]}" if text else f"{lang_name} speech",
                            "duration_s": round(duration, 1),
                        })
                        logger.info(f"  Saved: {filename} ({duration:.0f}s)")
                        count2 += 1

            except Exception as e2:
                logger.error(f"  Shrutilipi fallback also failed: {e2}")

    return files


def copy_judge_english_files() -> list[dict]:
    """Copy the usable English files from judge v1 dataset."""
    files = []
    judge_v1 = Path("data/judge_dataset/Call center data samples")

    en_dirs = [
        ("1755884171.51632 (EN Support-Billing)", "en_support_billing", "collections"),
        ("1735404531.458927 (EN customer support )", "en_customer_support", "complaint"),
    ]

    for dirname, safe_name, call_type in en_dirs:
        src_dir = judge_v1 / dirname
        if not src_dir.exists():
            continue
        for mp3 in src_dir.glob("*.mp3"):
            filename = f"judge_{safe_name}.mp3"
            dest = OUTPUT_DIR / filename
            shutil.copy2(str(mp3), str(dest))
            duration = _get_duration(str(dest))
            files.append({
                "filename": filename,
                "source": "judge_dataset_v1",
                "type": call_type,
                "language": "en",
                "description": f"Judge dataset: {dirname.split('(')[1].rstrip(')')}",
                "duration_s": round(duration, 1),
            })
            logger.info(f"  Copied: {filename}")
            break

    return files


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_files = []

    # 1. English earnings calls (2 files via HF API)
    logger.info("=" * 60)
    logger.info("Step 1: Earnings calls (English) via HF API")
    earnings = download_earnings_via_api(n=2)
    all_files.extend(earnings)

    # 2. English judge dataset files (2 files — local copy)
    logger.info("=" * 60)
    logger.info("Step 2: Judge dataset English files")
    judge = copy_judge_english_files()
    all_files.extend(judge)

    # 3. Hindi speech (3 files)
    logger.info("=" * 60)
    logger.info("Step 3: Hindi speech samples")
    hindi = download_indicvoices_samples({"hi": 3})
    all_files.extend(hindi)

    # 4. Tamil speech (2 files)
    logger.info("=" * 60)
    logger.info("Step 4: Tamil speech samples")
    tamil = download_indicvoices_samples({"ta": 2})
    all_files.extend(tamil)

    # 5. Indian English (1 file)
    logger.info("=" * 60)
    logger.info("Step 5: Indian English speech")
    indian_en = download_indicvoices_samples({"en": 1})
    all_files.extend(indian_en)

    # Save manifest
    manifest = {
        "description": "FinVoice test dataset — curated financial audio samples for pipeline testing",
        "total_files": len(all_files),
        "languages": sorted(set(f["language"] for f in all_files)),
        "types": sorted(set(f["type"] for f in all_files)),
        "files": all_files,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"TEST DATASET BUILT — {len(all_files)} files")
    print(f"{'='*60}")
    print(f"{'#':<3} {'Filename':<40} {'Lang':>4} {'Type':<16} {'Dur':>6}")
    print("-" * 72)
    for i, f in enumerate(all_files, 1):
        dur = f"{f['duration_s']:.0f}s" if f['duration_s'] else "?"
        print(f"{i:<3} {f['filename']:<40} {f['language']:>4} {f['type']:<16} {dur:>6}")
    print(f"\nOutput: {OUTPUT_DIR}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
