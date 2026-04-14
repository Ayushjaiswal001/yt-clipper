#!/usr/bin/env python3
"""
FFmpeg Processor – cut video, scale to 9:16 (1080×1920), burn ASS captions.
"""

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


# ── Timestamp helpers ─────────────────────────────────────────────────────────
def _ts_to_sec(t: str) -> float:
    """Convert "MM:SS", "HH:MM:SS", or a plain float-string to seconds."""
    parts = str(t).strip().split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, IndexError):
        raise ValueError(f"Cannot parse timestamp: {t!r}")


def _safe_filename(text: str, max_len: int = 30) -> str:
    """Sanitize text into a safe filesystem fragment."""
    cleaned = re.sub(r"[^\w\s-]", "", text).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:max_len] or "clip"


# ── SRT generation ────────────────────────────────────────────────────────────
def _fmt_srt_time(sec: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s_int = int(sec % 60)
    ms = int(round((sec % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s_int:02d},{ms:03d}"


def _generate_srt(caption_lines: list[str], duration: float) -> str:
    """
    Build SRT content distributing caption_lines evenly across clip duration.
    Each line occupies an equal time slot.
    """
    if not caption_lines:
        return ""
    slot = duration / len(caption_lines)
    entries = []
    for i, line in enumerate(caption_lines):
        t_start = i * slot
        t_end = (i + 1) * slot
        entries.append(
            f"{i + 1}\n{_fmt_srt_time(t_start)} --> {_fmt_srt_time(t_end)}\n{line}"
        )
    return "\n\n".join(entries) + "\n"


# ── ASS subtitle style ────────────────────────────────────────────────────────
_SUBTITLE_STYLE = ",".join([
    "FontName=DejaVu Sans",
    "FontSize=52",
    "PrimaryColour=&H00FFFFFF",   # white text
    "OutlineColour=&H00000000",   # black outline
    "BackColour=&H00000000",
    "Outline=3",
    "Shadow=1",
    "Alignment=2",                # bottom-centre (ASS numpad alignment)
    "MarginV=120",
])


# ── Public API ────────────────────────────────────────────────────────────────
def process_clip(
    video_path: str,
    spec,           # gemini_analyzer.ClipSpec
    video_id: str,
    clip_num: int,
    output_dir: str,
) -> str | None:
    """
    Cut one clip from video_path, scale to 1080×1920, burn captions.
    Returns the output file path, or None on failure (caller should skip).
    """
    # ── Validate timestamps ──────────────────────────────────────────────────
    try:
        start_sec = _ts_to_sec(spec.start_time)
        end_sec = _ts_to_sec(spec.end_time)
    except ValueError as exc:
        log.error(f"Clip {clip_num}: bad timestamp – {exc}")
        return None

    duration = end_sec - start_sec
    if duration <= 0:
        log.error(f"Clip {clip_num}: non-positive duration ({duration:.1f}s) – skipping")
        return None

    # ── Output path ──────────────────────────────────────────────────────────
    safe_hook = _safe_filename(spec.hook_text)
    out_name = f"{video_id}_{clip_num:02d}_{safe_hook}.mp4"
    out_path = os.path.join(output_dir, out_name)

    # ── Write temp SRT ───────────────────────────────────────────────────────
    srt_content = _generate_srt(spec.caption_lines, duration)
    srt_fd, srt_path = tempfile.mkstemp(suffix=".srt", prefix="yt_cap_")
    try:
        with os.fdopen(srt_fd, "w", encoding="utf-8") as f:
            f.write(srt_content)

        # On Linux paths are already forward-slash; escaping colons not needed
        # but we must wrap in single-quotes within the filter string if path
        # could contain spaces. Using temp paths avoids this.
        vf = (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            f"subtitles='{srt_path}':force_style='{_SUBTITLE_STYLE}'"
        )

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-i", video_path,
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            out_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            log.error(
                f"FFmpeg clip {clip_num} failed (exit {result.returncode}):\n"
                f"{result.stderr[-600:]}"
            )
            return None

        size_mb = Path(out_path).stat().st_size / 1_048_576
        log.info(f"Clip {clip_num}: {out_name} ({size_mb:.1f} MB, {duration:.0f}s)")
        return out_path

    except subprocess.TimeoutExpired:
        log.error(f"FFmpeg clip {clip_num} timed out (>300 s)")
        return None
    except Exception as exc:
        log.error(f"FFmpeg clip {clip_num} unexpected error: {exc}")
        return None
    finally:
        if os.path.exists(srt_path):
            os.unlink(srt_path)
