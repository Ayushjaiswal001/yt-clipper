#!/usr/bin/env python3
"""
YouTube Clipper – Main Orchestrator
Downloads videos, extracts clips via Gemini AI, burns captions, uploads to Drive.
"""

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import requests

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("run_log.txt", mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Utility helpers ───────────────────────────────────────────────────────────
def safe_url(url: str) -> str:
    """Truncate URL to ≤40 chars to avoid leaking video IDs in public logs."""
    return url[:40] + "…" if len(url) > 40 else url


def load_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── GitHub Issue creation (self-heal alert) ───────────────────────────────────
def create_github_issue(title: str, body: str) -> None:
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not token or not repo:
        log.warning("Skipping issue creation – GITHUB_TOKEN or GITHUB_REPOSITORY not set")
        return
    try:
        resp = requests.post(
            f"https://api.github.com/repos/{repo}/issues",
            json={"title": title, "body": body, "labels": ["clipper-alert"]},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30,
        )
        if resp.status_code == 201:
            log.info(f"GitHub issue created: {resp.json().get('html_url', 'n/a')}")
        else:
            log.warning(f"Issue creation failed ({resp.status_code}): {resp.text[:200]}")
    except Exception as exc:
        log.warning(f"Could not create GitHub issue: {exc}")


# ── yt-dlp download ───────────────────────────────────────────────────────────
def download_video(url: str, work_dir: str) -> tuple[str, str | None, str, str]:
    """
    Download video + auto-subtitles via yt-dlp.
    Returns: (video_path, srt_path_or_None, video_id, title)
    """
    cmd = [
        "yt-dlp",
        "--write-auto-sub",
        "--write-sub",
        "--sub-lang", "en",
        "--convert-subs", "srt",
        "--write-info-json",
        "-f",
        "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]"
        "/bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", f"{work_dir}/%(id)s.%(ext)s",
        "--no-playlist",
        "--socket-timeout", "60",
        "--retries", "3",
        "--no-warnings",
        # tv_simply is designed for TV apps — bypasses bot detection on
        # datacenter IPs (GitHub Actions runs on flagged Azure IPs).
        # mweb + default layered as fallbacks.
        "--extractor-args", "youtube:player_client=tv_simply,mweb,default",
        "--extractor-args", "youtubepot-bgutilhttp:base_url=http://127.0.0.1:4416",
        "--sleep-interval", "3",
        "--max-sleep-interval", "6",
    ]
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp exit {result.returncode}: {result.stderr[:300]}")

    mp4_files = sorted(Path(work_dir).glob("*.mp4"))
    if not mp4_files:
        raise RuntimeError("yt-dlp succeeded but no .mp4 found in work directory")

    video_path = str(mp4_files[0])
    video_id = mp4_files[0].stem

    # Extract title from info JSON
    title = video_id
    for info_file in Path(work_dir).glob("*.info.json"):
        try:
            with open(info_file, encoding="utf-8") as f:
                info = json.load(f)
            title = info.get("title", video_id)
            break
        except Exception:
            pass

    # Find any SRT (.en.srt, .en-auto.srt, etc.)
    srt_files = sorted(Path(work_dir).glob("*.srt"))
    srt_path = str(srt_files[0]) if srt_files else None

    return video_path, srt_path, video_id, title


def parse_srt(srt_path: str | None) -> str:
    """Convert SRT file → plain timestamped transcript string for Gemini."""
    if not srt_path or not os.path.exists(srt_path):
        return ""
    try:
        with open(srt_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as exc:
        log.warning(f"Could not read SRT file: {exc}")
        return ""

    lines = []
    for block in content.strip().split("\n\n"):
        parts = block.strip().splitlines()
        if len(parts) < 3:
            continue
        # parts[1]: "00:01:23,456 --> 00:01:45,678"
        try:
            start_raw = parts[1].split(" --> ")[0].replace(",", ".")
            h, m, s = start_raw.split(":")
            total = int(h) * 3600 + int(m) * 60 + float(s)
            mm, ss = divmod(int(total), 60)
            stamp = f"[{mm:02d}:{ss:02d}]"
        except Exception:
            stamp = "[??:??]"

        # Strip SRT timing tags like <00:00:01.234>
        text = " ".join(parts[2:]).strip()
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text:
            lines.append(f"{stamp} {text}")

    return "\n".join(lines)


# ── Per-URL pipeline ──────────────────────────────────────────────────────────
def process_url(url: str, config: dict) -> tuple[int, str]:
    """
    Full pipeline for one URL: download → analyse → cut → upload.
    Returns (clips_generated, drive_folder_path).
    Raises on unrecoverable failure.
    """
    from gemini_analyzer import analyze_transcript
    from ffmpeg_processor import process_clip
    from rclone_uploader import upload_clips

    folder_id = os.environ.get("GDRIVE_FOLDER_ID", "")

    with tempfile.TemporaryDirectory(prefix="yt_clipper_") as work_dir:
        clips_dir = os.path.join(work_dir, "clips")
        os.makedirs(clips_dir)

        log.info(f"Downloading → {safe_url(url)}")
        video_path, srt_path, video_id, title = download_video(url, work_dir)
        log.info(f"Downloaded: {title!r} ({video_id})")

        transcript = parse_srt(srt_path)
        if not transcript:
            raise RuntimeError("No subtitles/transcript found – cannot identify clips")
        log.info(f"Transcript: {len(transcript)} chars from {srt_path}")

        clip_specs = analyze_transcript(transcript, title, config)
        if not clip_specs:
            raise RuntimeError("Gemini returned no usable clip specifications")

        clips_generated = 0
        for i, spec in enumerate(clip_specs, start=1):
            try:
                out = process_clip(video_path, spec, video_id, i, clips_dir)
                if out:
                    clips_generated += 1
                    log.info(f"  Clip {i}/{len(clip_specs)} → {Path(out).name}")
            except Exception as exc:
                log.error(f"  Clip {i} FFmpeg error (skipping): {exc}")

        if clips_generated == 0:
            raise RuntimeError("All clips failed in FFmpeg processing")

        drive_folder = upload_clips(clips_dir, title, folder_id)
        return clips_generated, drive_folder


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("=" * 60)
    log.info("YouTube Clipper starting")

    config: dict = load_json("config.json", {
        "clips_per_video": 4,
        "min_clip_duration": 30,
        "max_clip_duration": 90,
    })
    queue: list[dict] = load_json("queue.json", [])
    processed: list[dict] = load_json("processed.json", [])
    processed_urls: set[str] = {p["url"] for p in processed}

    pending = [item for item in queue if item["url"] not in processed_urls]
    log.info(
        f"Queue: {len(queue)} total | {len(pending)} pending | "
        f"{len(processed_urls)} already done"
    )

    if not pending:
        log.info("Nothing new to process – exiting.")
        return

    consecutive_failures = 0
    issue_raised = False

    for item in pending:
        url = item["url"]
        safe = safe_url(url)
        success = False

        for attempt in range(1, 4):  # up to 3 attempts per URL
            try:
                log.info(f"── {safe}  (attempt {attempt}/3) ──")
                clips_generated, drive_folder = process_url(url, config)

                processed.append({
                    "url": url,
                    "clips_generated": clips_generated,
                    "processed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                    "drive_folder": drive_folder,
                })
                save_json("processed.json", processed)
                processed_urls.add(url)
                consecutive_failures = 0
                success = True
                log.info(f"✓ {safe} → {clips_generated} clips → {drive_folder}")
                break

            except Exception as exc:
                log.error(f"Attempt {attempt}/3 failed for {safe}: {exc}")
                if attempt < 3:
                    log.info("Retrying in 10 s…")
                    time.sleep(10)

        if not success:
            consecutive_failures += 1
            log.error(f"✗ All 3 attempts failed for {safe}")

            if consecutive_failures >= 3 and not issue_raised:
                issue_raised = True
                repo = os.environ.get("GITHUB_REPOSITORY", "your-repo")
                create_github_issue(
                    "🚨 YouTube Clipper: 3+ consecutive video failures",
                    f"Three or more consecutive video URLs have failed to process.\n\n"
                    f"**Last failed URL (truncated):** `{safe}`\n\n"
                    f"**Actions run:** "
                    f"https://github.com/{repo}/actions\n\n"
                    f"**Common causes:**\n"
                    f"- YouTube bot-detection / rate limiting\n"
                    f"- Gemini API quota exceeded\n"
                    f"- rclone / Drive service account permissions\n"
                    f"- Video has no auto-generated subtitles",
                )

    # Remove successfully-processed URLs from queue and commit
    new_queue = [item for item in queue if item["url"] not in processed_urls]
    save_json("queue.json", new_queue)
    removed = len(queue) - len(new_queue)
    log.info(f"Queue updated: removed {removed}, {len(new_queue)} remaining")
    log.info("Pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.critical(f"Fatal unhandled error: {exc}", exc_info=True)
        sys.exit(1)
