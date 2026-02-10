"""Stage 1: Raw Audio Intake — normalize any audio/video to 16kHz mono WAV."""

import subprocess
import json
from pathlib import Path
from loguru import logger


def normalize_audio(
    input_path: str, output_path: str,
    start_time: float | None = None, end_time: float | None = None,
) -> dict:
    """Accept ANY audio/video format, normalize to 16kHz mono WAV.

    Args:
        input_path: Path to input audio/video file (mp3, wav, mp4, webm, ogg, m4a, flac, etc.)
        output_path: Path for normalized WAV output
        start_time: Optional start time in seconds to trim from
        end_time: Optional end time in seconds to trim to

    Returns:
        dict with original format metadata and normalization info
    """
    input_path = str(input_path)
    output_path = str(output_path)

    # Probe input format
    probe = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", input_path
        ],
        capture_output=True, text=True
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {input_path}: {probe.stderr}")

    info = json.loads(probe.stdout)

    # Find audio stream
    audio_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "audio"),
        None
    )
    if audio_stream is None:
        raise ValueError(f"No audio stream found in {input_path}")

    # Normalize: 16kHz, mono, WAV, PCM 16-bit
    # Optional trimming via -ss (start) and -to (end)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = ["ffmpeg"]
    if start_time is not None and start_time > 0:
        ffmpeg_cmd += ["-ss", str(start_time)]
    ffmpeg_cmd += ["-i", input_path]
    if end_time is not None and end_time > 0:
        ffmpeg_cmd += ["-to", str(end_time - (start_time or 0))]
    ffmpeg_cmd += [
        "-af", "loudnorm=I=-23:TP=-1.5:LRA=11",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-y", output_path,
    ]

    if start_time or end_time:
        logger.info(f"Trimming audio: start={start_time}s, end={end_time}s")

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg normalization failed: {result.stderr}")

    # Probe actual output duration (may differ from original if trimmed)
    try:
        out_info = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", output_path],
            capture_output=True, text=True,
        )
        output_duration = float(json.loads(out_info.stdout)["format"].get("duration", 0))
    except Exception:
        output_duration = float(info["format"].get("duration", 0))

    metadata = {
        "original_format": info["format"]["format_name"],
        "original_duration": output_duration,
        "source_duration": float(info["format"].get("duration", 0)),
        "original_sample_rate": int(audio_stream.get("sample_rate", 0)),
        "original_channels": int(audio_stream.get("channels", 0)),
        "normalized_to": "16kHz_mono_wav",
        "output_path": output_path,
    }

    logger.info(
        f"Normalized {input_path} → {output_path} "
        f"({metadata['original_format']}, {metadata['original_sample_rate']}Hz, "
        f"{metadata['original_channels']}ch → 16kHz mono)"
    )
    return metadata
