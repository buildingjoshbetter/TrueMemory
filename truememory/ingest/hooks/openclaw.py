"""Native OpenClaw recall transport without capture or transcript admission."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import TextIO

MAX_PROMPT_BYTES = 128 * 1024
MAX_INPUT_BYTES = MAX_PROMPT_BYTES + 4096
_SESSION_RE = re.compile(r"openclaw_[0-9a-f]{48}\Z")
_RUN_RE = re.compile(r"[0-9a-f]{64}\Z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--event", choices=("before_prompt_build",), required=True)
    parser.add_argument("--user", default="")
    parser.add_argument("--db", default="")
    return parser.parse_args()


def _request(stream: TextIO) -> tuple[str, str] | None:
    try:
        text = stream.read(MAX_INPUT_BYTES + 1)
        if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
            return None
        payload = json.loads(text)
    except (OSError, ValueError, TypeError, UnicodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"session_id", "run_id", "prompt"}:
        return None
    session_id, run_id, prompt = (payload[key] for key in ("session_id", "run_id", "prompt"))
    if not isinstance(session_id, str) or not _SESSION_RE.fullmatch(session_id):
        return None
    if not isinstance(run_id, str) or not _RUN_RE.fullmatch(run_id):
        return None
    if not isinstance(prompt, str) or len(prompt.strip()) < 3 or "\0" in prompt:
        return None
    try:
        if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            return None
    except UnicodeError:
        return None
    return session_id, prompt


def main() -> None:
    if os.environ.get("TRUEMEMORY_EXTRACTION"):
        return
    args = _parse_args()
    request = _request(sys.stdin)
    if request is None:
        return
    session_id, prompt = request
    from truememory.ingest.hooks.user_prompt_submit import _try_auto_recall

    # Native run correlation supplies no admitted-turn cadence or capture proof.
    context = _try_auto_recall(prompt, args.user, args.db, session_id, debounced=False)
    if context is not None and not isinstance(context, str):
        return
    sys.stdout.write(json.dumps({"context": context or ""}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
