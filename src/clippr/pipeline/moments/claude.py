"""Optional paid picker using the Claude API. Only used when ANTHROPIC_API_KEY is set."""

from __future__ import annotations

import logging
import math

import anthropic

from clippr.config import Settings
from clippr.pipeline.moments.errors import PickerError
from clippr.pipeline.moments.prompt import SYSTEM_PROMPT, format_transcript, request_text
from clippr.schemas import Moment, MomentList, Signals, Transcript, VideoInfo

log = logging.getLogger(__name__)

MAX_INPUT_TOKENS = 900_000  # stay well inside the 1M-token context window
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def pick(transcript: Transcript, signals: Signals, info: VideoInfo, settings: Settings, *,
         n: int, min_s: float, max_s: float, focus: str | None = None) -> list[Moment]:
    if not settings.anthropic_api_key:
        raise PickerError("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    transcript_text = format_transcript(transcript, signals, info)
    system = [{"type": "text", "text": SYSTEM_PROMPT}]
    # The transcript is the stable, cached prefix; the request after it can change on "regenerate".
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": transcript_text, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": request_text(math.ceil(n * 1.5), min_s, max_s, focus)},
        ],
    }]

    try:
        if len(transcript_text) > 2_000_000:  # only pay the round trip when it could matter
            count = client.messages.count_tokens(model=settings.anthropic_model, system=system, messages=messages)
            if count.input_tokens > MAX_INPUT_TOKENS:
                raise PickerError(f"Transcript is too long for one request ({count.input_tokens} tokens)")

        response = client.beta.messages.parse(
            model=settings.anthropic_model,
            max_tokens=16000,
            system=system,
            messages=messages,
            output_format=MomentList,
            output_config={"effort": settings.anthropic_effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
    except anthropic.AuthenticationError as e:
        raise PickerError("Claude rejected the API key") from e
    except anthropic.RateLimitError as e:
        raise PickerError("Claude rate limit reached; try again shortly") from e
    except anthropic.APIStatusError as e:
        raise PickerError(f"Claude API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise PickerError("Could not reach the Claude API") from e

    if response.stop_reason == "refusal":
        raise PickerError("Claude declined to pick clips for this video")
    if response.stop_reason == "max_tokens":
        raise PickerError("Claude's answer was cut off before it finished")
    if response.parsed_output is None:
        raise PickerError("Claude's answer did not match the expected format")

    usage = response.usage
    log.info(
        "claude picked %d moments (input=%s cached=%s output=%s)",
        len(response.parsed_output.moments), usage.input_tokens,
        getattr(usage, "cache_read_input_tokens", None), usage.output_tokens,
    )
    return response.parsed_output.moments
