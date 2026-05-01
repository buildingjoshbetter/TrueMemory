"""
LLM Model Adapters
==================

Unified interface for fact extraction across multiple backends:
- Ollama (local, zero cost)
- OpenRouter (cloud — Haiku 4.5, GPT-4.1-mini)
- Anthropic direct (cloud — Claude models)

The extraction runs in the background AFTER conversations end (the "cold path"),
so latency is not critical. Prefer local models for zero-cost operation.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import socket
import subprocess
import time
import urllib.request
import urllib.error
from dataclasses import dataclass

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when an LLM call fails for any reason (network, parsing, auth)."""
    pass


# Retry configuration for transient network failures
_MAX_RETRIES = 3
_BASE_BACKOFF_SEC = 1.0
# HTTP status codes that indicate a transient server-side issue worth retrying
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
# Exception types that indicate transient network issues
_RETRYABLE_EXCEPTIONS = (
    urllib.error.URLError,
    socket.timeout,
    TimeoutError,
    ConnectionError,
    OSError,
)


def _should_retry(exc: Exception) -> bool:
    """Return True if an exception is worth retrying."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _RETRYABLE_STATUS
    return isinstance(exc, _RETRYABLE_EXCEPTIONS)


def _retry_backoff(attempt: int) -> float:
    """Exponential backoff with jitter: 1s, 2s, 4s (+/- 25%)."""
    base = _BASE_BACKOFF_SEC * (2 ** attempt)
    jitter = base * 0.25
    return base + random.uniform(-jitter, jitter)


@dataclass
class LLMConfig:
    """Configuration for an LLM backend."""
    provider: str = "auto"       # auto, ollama, openrouter, anthropic, openai
    model: str = ""              # Model name (auto-detected if empty)
    base_url: str = ""           # API base URL
    api_key: str = ""            # API key
    temperature: float = 0.0     # Deterministic by default
    max_tokens: int = 2000       # Sufficient for fact extraction


def hydrate_config(config: LLMConfig) -> LLMConfig:
    """Fill in provider-specific defaults on an existing config.

    When a user passes ``--provider anthropic`` (or similar) via the CLI,
    they get a bare ``LLMConfig(provider="anthropic", model="...")`` with
    no ``api_key`` or ``base_url``. This helper centralizes the
    provider-to-env-var and provider-to-default-url mapping so explicit
    and auto-detected configs both get the same treatment, and we don't
    silently fire off requests with empty auth headers.

    Mutates and returns the same config for convenience.
    """
    provider = (config.provider or "").lower()

    if provider == "anthropic":
        if not config.api_key:
            config.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not config.model:
            config.model = "claude-haiku-4-5-20251001"

    elif provider == "openrouter":
        if not config.api_key:
            config.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not config.base_url:
            config.base_url = "https://openrouter.ai/api/v1"
        if not config.model:
            config.model = "anthropic/claude-haiku-4-5-20251001"

    elif provider == "openai":
        if not config.api_key:
            config.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not config.base_url:
            config.base_url = "https://api.openai.com/v1"
        if not config.model:
            config.model = "gpt-4o-mini"

    elif provider == "ollama":
        if not config.base_url:
            config.base_url = "http://localhost:11434/v1"
        if not config.model:
            # Prefer qwen if present, else whatever's available
            available = _ollama_models()
            config.model = "qwen2.5:7b-instruct"
            if available and config.model not in available:
                config.model = available[0]

    elif provider in ("claude_cli", "claude-cli"):
        # Normalize to underscore form for downstream dispatch
        config.provider = "claude_cli"
        # No api_key, no base_url — the CLI handles auth and routing.
        # Leave model empty by default so the CLI picks the user's
        # configured default (usually Opus); callers can override.

    return config


def auto_detect() -> LLMConfig:
    """
    Detect the best available LLM backend.

    Priority:
    1. Ollama (free, local, no API key, fully offline)
    2. Claude CLI (free for subscribers, uses OAuth — no API key)
    3. OpenRouter (one key for many models)
    4. Anthropic (direct API)
    """
    # 1. Ollama — fully offline, no cost, first choice
    if _ollama_available():
        cfg = hydrate_config(LLMConfig(provider="ollama"))
        log.info("Auto-detected Ollama with model %s", cfg.model)
        return cfg

    # 2. Claude CLI — zero additional cost for subscribers, no key mgmt
    if _claude_cli_available():
        log.info("Auto-detected Claude CLI (subscription auth)")
        return hydrate_config(LLMConfig(provider="claude_cli"))

    # 3. OpenRouter
    if os.environ.get("OPENROUTER_API_KEY", ""):
        log.info("Auto-detected OpenRouter API key")
        return hydrate_config(LLMConfig(provider="openrouter"))

    # 4. Anthropic
    if os.environ.get("ANTHROPIC_API_KEY", ""):
        log.info("Auto-detected Anthropic API key")
        return hydrate_config(LLMConfig(provider="anthropic"))

    raise RuntimeError(
        "No LLM backend found for fact extraction. Options:\n"
        "  1. Run Ollama locally: ollama serve && ollama pull qwen2.5:7b-instruct\n"
        "  2. Install Claude Code (provides `claude` CLI + subscription auth)\n"
        "  3. Set OPENROUTER_API_KEY environment variable\n"
        "  4. Set ANTHROPIC_API_KEY environment variable"
    )


def complete(config: LLMConfig, prompt: str, system: str = "") -> str:
    """
    Get a completion from the configured LLM.

    Uses the OpenAI-compatible API for Ollama and OpenRouter.
    Uses the Anthropic SDK for direct Anthropic calls.
    Uses the local ``claude`` CLI binary for the claude_cli provider.
    """
    if config.provider == "anthropic":
        return _complete_anthropic(config, prompt, system)
    if config.provider in ("claude_cli", "claude-cli"):
        return _complete_claude_cli(config, prompt, system)
    return _complete_openai_compat(config, prompt, system)


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    """Check if Ollama is running locally."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ollama_models() -> list[str]:
    """List available Ollama models."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# OpenAI-compatible completion (Ollama, OpenRouter, generic)
# ---------------------------------------------------------------------------

def _complete_openai_compat(config: LLMConfig, prompt: str, system: str) -> str:
    """Complete using the OpenAI-compatible chat API.

    Raises LLMError on network failure, HTTP error, or malformed response.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = json.dumps({
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }).encode()

    url = f"{config.base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    # Retry loop with exponential backoff for transient failures
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            break  # success
        except urllib.error.HTTPError as e:
            last_exc = e
            if attempt < _MAX_RETRIES and _should_retry(e):
                wait = _retry_backoff(attempt)
                log.info("%s HTTP %d (attempt %d/%d), retrying in %.1fs",
                         config.provider, e.code, attempt + 1, _MAX_RETRIES, wait)
                time.sleep(wait)
                continue
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            raise LLMError(f"HTTP {e.code} from {config.provider}: {detail or e.reason}") from e
        except _RETRYABLE_EXCEPTIONS as e:
            last_exc = e
            if attempt < _MAX_RETRIES:
                wait = _retry_backoff(attempt)
                log.info("%s network error (attempt %d/%d): %s, retrying in %.1fs",
                         config.provider, attempt + 1, _MAX_RETRIES, e, wait)
                time.sleep(wait)
                continue
            if isinstance(e, urllib.error.URLError):
                raise LLMError(f"Network error calling {config.provider}: {e.reason}") from e
            if isinstance(e, (socket.timeout, TimeoutError)):
                raise LLMError(f"Timeout calling {config.provider}") from e
            raise LLMError(f"Connection error calling {config.provider}: {e}") from e
    else:
        # Loop exited without break — last_exc should be set
        raise LLMError(f"All retries exhausted for {config.provider}: {last_exc}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"Invalid JSON from {config.provider}: {e}") from e

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"Unexpected response shape from {config.provider}: {data}") from e


# ---------------------------------------------------------------------------
# Anthropic completion
# ---------------------------------------------------------------------------

def _complete_anthropic(config: LLMConfig, prompt: str, system: str) -> str:
    """Complete using the Anthropic API directly (no SDK dependency).

    Raises LLMError on network failure, HTTP error, or malformed response.
    """
    body = json.dumps({
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "messages": [{"role": "user", "content": prompt}],
        **({"system": system} if system else {}),
    }).encode()

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
    }

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            break
        except urllib.error.HTTPError as e:
            last_exc = e
            if attempt < _MAX_RETRIES and _should_retry(e):
                wait = _retry_backoff(attempt)
                log.info("Anthropic HTTP %d (attempt %d/%d), retrying in %.1fs",
                         e.code, attempt + 1, _MAX_RETRIES, wait)
                time.sleep(wait)
                continue
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            raise LLMError(f"Anthropic HTTP {e.code}: {detail or e.reason}") from e
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as e:
            last_exc = e
            if attempt < _MAX_RETRIES:
                wait = _retry_backoff(attempt)
                log.info("Anthropic %s (attempt %d/%d), retrying in %.1fs",
                         type(e).__name__, attempt + 1, _MAX_RETRIES, wait)
                time.sleep(wait)
                continue
            if isinstance(e, urllib.error.URLError):
                raise LLMError(f"Anthropic network error: {e.reason}") from e
            raise LLMError(f"Anthropic connection error: {e}") from e
    else:
        raise LLMError("Anthropic: max retries exceeded") from last_exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"Anthropic returned invalid JSON: {e}") from e

    try:
        return data["content"][0]["text"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"Unexpected Anthropic response shape: {data}") from e


# ---------------------------------------------------------------------------
# Claude CLI completion (uses the local `claude` binary + subscription auth)
# ---------------------------------------------------------------------------

def _claude_cli_available() -> bool:
    """Return True if the `claude` CLI binary is on PATH."""
    return shutil.which("claude") is not None


def _complete_claude_cli(config: LLMConfig, prompt: str, system: str) -> str:
    """Complete using the local ``claude`` CLI in one-shot print mode.

    This backend requires zero API keys — it uses the user's existing
    Claude Code subscription auth (OAuth/keychain). Ideal for offline-first
    deployments where the user already has Claude Code installed.

    Importantly: we **unset ANTHROPIC_API_KEY** before invoking the CLI
    because ``claude --bare`` and some other modes will prefer that env
    var if set, and a stale key would cause the CLI to return auth errors
    instead of using the working OAuth path.

    Raises LLMError on CLI failure or malformed output.
    """
    if not _claude_cli_available():
        raise LLMError(
            "`claude` CLI not found on PATH. Install Claude Code or "
            "choose a different --provider."
        )

    # Claude CLI supports a system prompt via --append-system-prompt; we fold
    # any system content into the user prompt for simplicity (extractors
    # embed their system prompt in the user message anyway).
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    cmd = ["claude", "-p", "--output-format", "json"]
    if config.model:
        cmd.extend(["--model", config.model])

    # Strip ANTHROPIC_API_KEY so the CLI uses OAuth/keychain auth rather
    # than a potentially stale key from the parent environment.
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    try:
        proc = subprocess.run(
            cmd,
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise LLMError("claude CLI timed out after 120s") from e
    except OSError as e:
        raise LLMError(f"claude CLI invocation failed: {e}") from e

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()[:500]
        raise LLMError(f"claude CLI exit {proc.returncode}: {stderr or 'no stderr'}")

    # Parse the --output-format json envelope: {type, subtype, is_error, result, ...}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise LLMError(f"claude CLI returned non-JSON: {proc.stdout[:300]}") from e

    if data.get("is_error"):
        raise LLMError(f"claude CLI reported error: {data.get('result', 'unknown')}")

    result = data.get("result")
    if not isinstance(result, str):
        raise LLMError(f"claude CLI response missing 'result' string: {data}")

    return result
