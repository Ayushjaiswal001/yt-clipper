#!/usr/bin/env python3
"""
Gemini Analyzer – sends transcript to Gemini 2.0 Flash and returns ClipSpec list.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

import google.generativeai as genai

log = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class ClipSpec:
    start_time: str           # "MM:SS"
    end_time: str             # "MM:SS"
    viral_score: int          # 1-10
    hook_text: str
    caption_lines: list[str] = field(default_factory=list)


# ── Prompts ───────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a viral short-form video editor. "
    "Given a YouTube video transcript with timestamps, "
    "identify the {n} best moments to clip into Shorts ({min_dur}-{max_dur} seconds each). "
    "Score each on: emotional hooks, surprising facts, quotable moments, "
    "story climaxes, actionable tips. "
    "Return ONLY valid JSON array. No preamble. No markdown. Schema:\n"
    '[{{"start_time":"MM:SS","end_time":"MM:SS","viral_score":1-10,'
    '"hook_text":"string","caption_lines":["4-word chunks"]}}]'
)

_STRICT_PROMPT = (
    "Output ONLY a raw JSON array — zero other text, no markdown fences, no explanation.\n"
    "The array must contain exactly {n} objects.\n"
    "Required keys per object (exact types):\n"
    '  "start_time": string in "MM:SS" format\n'
    '  "end_time":   string in "MM:SS" format\n'
    '  "viral_score": integer 1-10\n'
    '  "hook_text":  string\n'
    '  "caption_lines": array of strings, ~4 words each\n'
    "Each clip must be {min_dur}-{max_dur} seconds long.\n\n"
    "Example (single item):\n"
    '[{{"start_time":"01:23","end_time":"02:10","viral_score":8,'
    '"hook_text":"This will shock you","caption_lines":["This will shock","you completely right","now watch closely"]}}]'
)


def _build_user_message(
    transcript: str, title: str, n: int, min_dur: int, max_dur: int
) -> str:
    return (
        f"Video title: {title}\n\n"
        f"Transcript (with timestamps):\n{transcript}\n\n"
        f"Find the {n} best moments. Each clip must be {min_dur}–{max_dur} seconds."
    )


# ── JSON parser ───────────────────────────────────────────────────────────────
def _parse_response(text: str) -> list[ClipSpec]:
    """Strip markdown wrappers, locate the JSON array, and parse into ClipSpecs."""
    # Remove code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text).strip()

    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end <= 0:
        raise ValueError("No JSON array found in model output")

    data = json.loads(text[start:end])
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Parsed result is empty or not a list")

    specs: list[ClipSpec] = []
    for item in data:
        specs.append(ClipSpec(
            start_time=str(item["start_time"]),
            end_time=str(item["end_time"]),
            viral_score=int(item.get("viral_score", 5)),
            hook_text=str(item.get("hook_text", "")),
            caption_lines=[str(x) for x in item.get("caption_lines", [])],
        ))
    return specs


# ── Public API ────────────────────────────────────────────────────────────────
def analyze_transcript(transcript: str, video_title: str, config: dict) -> list[ClipSpec]:
    """
    Send transcript to Gemini 2.0 Flash and return a list of ClipSpec.
    Retries once with a stricter prompt if the first response is unparseable.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    n = config.get("clips_per_video", 4)
    min_dur = config.get("min_clip_duration", 30)
    max_dur = config.get("max_clip_duration", 90)

    user_msg = _build_user_message(transcript, video_title, n, min_dur, max_dur)

    prompts = [
        _SYSTEM_PROMPT.format(n=n, min_dur=min_dur, max_dur=max_dur),
        _STRICT_PROMPT.format(n=n, min_dur=min_dur, max_dur=max_dur),
    ]

    last_exc: Exception | None = None
    for attempt, system_prompt in enumerate(prompts, start=1):
        try:
            log.info(f"Gemini request (attempt {attempt}/2)…")
            response = model.generate_content(
                f"{system_prompt}\n\n{user_msg}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
                ),
            )
            raw = response.text
            specs = _parse_response(raw)
            log.info(f"Gemini returned {len(specs)} clip specs")
            return specs

        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            last_exc = exc
            log.warning(f"Gemini parse error (attempt {attempt}): {exc}")
            if attempt < len(prompts):
                log.info("Retrying Gemini with stricter prompt in 3 s…")
                time.sleep(3)

        except Exception as exc:
            # API-level error – propagate immediately
            log.error(f"Gemini API error (attempt {attempt}): {exc}")
            raise

    log.error(f"Both Gemini attempts failed. Last error: {last_exc}")
    return []
