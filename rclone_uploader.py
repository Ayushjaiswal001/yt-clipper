#!/usr/bin/env python3
"""
rclone Uploader – copies finished clips to Google Drive via service account.
Folder structure on Drive: YouTubeClips/{video_title}/{clip files}
"""

import logging
import os
import re
import subprocess
import time

log = logging.getLogger(__name__)

_SA_FILE = "/tmp/sa.json"
_REMOTE = "gdrive"
_MAX_RETRIES = 3
_RETRY_DELAY_SEC = 30


def _safe_folder_name(title: str) -> str:
    """Sanitize video title to a valid Google Drive folder name (max 80 chars)."""
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title).strip()
    return safe[:80] or "Untitled"


def upload_clips(clips_dir: str, video_title: str, folder_id: str) -> str:
    if os.environ.get("DRIVE_CONFIGURED", "true").lower() != "true":
        log.warning("Drive not configured (GDRIVE_SERVICE_ACCOUNT_JSON not set) – skipping upload")
        return "local-only"
    if not os.path.exists(_SA_FILE):
        log.warning(f"Service account file {_SA_FILE} missing – skipping upload")
        return "local-only"
    """
    Upload all .mp4 files in clips_dir to Google Drive.

    Drive path:  {folder_id root}/YouTubeClips/{safe_title}/
    Uses --drive-root-folder-id to set the shared Drive folder as root,
    then creates YouTubeClips/{title} as a subfolder within it.

    Returns the rclone destination string.
    Raises RuntimeError after _MAX_RETRIES consecutive failures.
    """
    safe_title = _safe_folder_name(video_title)
    subfolder = f"YouTubeClips/{safe_title}"
    dest = f"{_REMOTE}:{subfolder}"

    # Count clips to upload
    mp4s = [f for f in os.listdir(clips_dir) if f.endswith(".mp4")]
    log.info(f"Uploading {len(mp4s)} clip(s) → Drive:{subfolder}")

    # Build base command
    base_cmd = [
        "rclone", "copy",
        clips_dir,
        dest,
        "--drive-service-account-file", _SA_FILE,
        "--progress",
        "--transfers", "4",
        "--retries", "3",
        "--stats", "15s",
    ]

    # If a root folder ID is provided, scope the remote to that folder
    if folder_id:
        base_cmd += ["--drive-root-folder-id", folder_id]

    last_stderr = ""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = subprocess.run(
                base_cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                log.info(f"Upload complete → {dest}")
                return dest

            last_stderr = result.stderr[-400:]
            log.warning(
                f"rclone attempt {attempt}/{_MAX_RETRIES} "
                f"exit {result.returncode}: {last_stderr}"
            )

        except subprocess.TimeoutExpired:
            last_stderr = "process timed out"
            log.warning(f"rclone attempt {attempt}/{_MAX_RETRIES} timed out (>600 s)")

        except Exception as exc:
            last_stderr = str(exc)
            log.warning(f"rclone attempt {attempt}/{_MAX_RETRIES} error: {exc}")

        if attempt < _MAX_RETRIES:
            log.info(f"Retrying upload in {_RETRY_DELAY_SEC} s…")
            time.sleep(_RETRY_DELAY_SEC)

    raise RuntimeError(
        f"rclone upload failed after {_MAX_RETRIES} attempts for '{safe_title}'. "
        f"Last error: {last_stderr}"
    )
