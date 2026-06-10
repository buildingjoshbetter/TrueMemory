"""Issue #595 — Fable-5 / Opus-4.8 param-safety shim + model ID hygiene.

Tests that:
  1. ``sanitize_model_params`` strips unsupported keys for Fable-5 / Opus-4.8.
  2. ``sanitize_model_params`` preserves all keys for older / non-Claude models.
  3. ``thinking: {type: disabled}`` is stripped for strict models but kept otherwise.
  4. ``output_format`` is stripped for strict models.
  5. Model IDs referenced in the codebase are well-formed.
"""

from __future__ import annotations

import re
import pytest

from truememory.ingest.models import sanitize_model_params, _STRICT_PARAM_MODEL_RE


# ---------------------------------------------------------------------------
# Fixtures — representative model IDs
# ---------------------------------------------------------------------------

STRICT_MODELS = [
    "claude-fable-5-20260301",
    "claude-opus-4-8-20260501",
    "anthropic/claude-fable-5-20260301",
    "anthropic/claude-opus-4-8-20260501",
    # Case variations
    "Claude-Fable-5-20260301",
    "CLAUDE-OPUS-4-8-20260501",
]

PERMISSIVE_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "anthropic/claude-haiku-4-5-20251001",
    "gpt-4o-mini",
    "qwen2.5:7b-instruct",
    "",
]


def _sample_params() -> dict:
    """Full param dict including every key the shim should consider."""
    return {
        "model": "placeholder",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 300,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "budget_tokens": 1000,
        "thinking": {"type": "disabled"},
        "output_format": "json",
    }


# ---------------------------------------------------------------------------
# 1. Strips unsupported keys for strict models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id", STRICT_MODELS)
def test_strips_temperature_for_strict_models(model_id: str):
    out = sanitize_model_params(model_id, _sample_params())
    assert "temperature" not in out


@pytest.mark.parametrize("model_id", STRICT_MODELS)
def test_strips_top_p_for_strict_models(model_id: str):
    out = sanitize_model_params(model_id, _sample_params())
    assert "top_p" not in out


@pytest.mark.parametrize("model_id", STRICT_MODELS)
def test_strips_top_k_for_strict_models(model_id: str):
    out = sanitize_model_params(model_id, _sample_params())
    assert "top_k" not in out


@pytest.mark.parametrize("model_id", STRICT_MODELS)
def test_strips_budget_tokens_for_strict_models(model_id: str):
    out = sanitize_model_params(model_id, _sample_params())
    assert "budget_tokens" not in out


@pytest.mark.parametrize("model_id", STRICT_MODELS)
def test_strips_thinking_disabled_for_strict_models(model_id: str):
    out = sanitize_model_params(model_id, _sample_params())
    assert "thinking" not in out


@pytest.mark.parametrize("model_id", STRICT_MODELS)
def test_strips_output_format_for_strict_models(model_id: str):
    out = sanitize_model_params(model_id, _sample_params())
    assert "output_format" not in out


@pytest.mark.parametrize("model_id", STRICT_MODELS)
def test_preserves_safe_keys_for_strict_models(model_id: str):
    out = sanitize_model_params(model_id, _sample_params())
    assert "model" in out
    assert "messages" in out
    assert "max_tokens" in out


# ---------------------------------------------------------------------------
# 2. Preserves all keys for permissive models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id", PERMISSIVE_MODELS)
def test_preserves_temperature_for_permissive_models(model_id: str):
    out = sanitize_model_params(model_id, _sample_params())
    assert "temperature" in out
    assert out["temperature"] == 0.3


@pytest.mark.parametrize("model_id", PERMISSIVE_MODELS)
def test_preserves_all_keys_for_permissive_models(model_id: str):
    params = _sample_params()
    out = sanitize_model_params(model_id, params)
    assert set(out.keys()) == set(params.keys())


# ---------------------------------------------------------------------------
# 3. thinking: {type: enabled} is kept even for strict models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_id", STRICT_MODELS)
def test_keeps_thinking_enabled_for_strict_models(model_id: str):
    params = {"thinking": {"type": "enabled", "budget_tokens": 5000}}
    out = sanitize_model_params(model_id, params)
    # thinking itself is kept (only stripped when disabled); budget_tokens
    # inside thinking is a sub-key so not stripped by the top-level filter.
    assert "thinking" in out
    assert out["thinking"]["type"] == "enabled"


# ---------------------------------------------------------------------------
# 4. Returns a copy (does not mutate the original)
# ---------------------------------------------------------------------------

def test_does_not_mutate_original():
    params = _sample_params()
    original_keys = set(params.keys())
    sanitize_model_params("claude-fable-5-20260301", params)
    assert set(params.keys()) == original_keys


# ---------------------------------------------------------------------------
# 5. Model ID format — OpenRouter IDs in the codebase
# ---------------------------------------------------------------------------

# OpenRouter Anthropic model IDs must follow: anthropic/claude-<family>-<version>-YYYYMMDD
_OPENROUTER_ANTHROPIC_RE = re.compile(
    r"^anthropic/claude-[a-z0-9]+-[0-9]+-[0-9]+-\d{8}$"
)


def test_openrouter_model_id_in_mcp_server():
    """The OpenRouter model ID in mcp_server.py must be well-formed."""
    import truememory.mcp_server as mod
    import inspect

    source = inspect.getsource(mod._build_openrouter_llm)
    # Extract model ID from the source
    match = re.search(r'"(anthropic/claude-[^"]+)"', source)
    assert match, "Could not find OpenRouter model ID in _build_openrouter_llm"
    model_id = match.group(1)
    assert _OPENROUTER_ANTHROPIC_RE.match(model_id), (
        f"Malformed OpenRouter model ID: {model_id}"
    )


def test_openrouter_default_in_models_py():
    """The default OpenRouter model in ingest/models.py must be well-formed."""
    import truememory.ingest.models as mod
    import inspect

    source = inspect.getsource(mod.hydrate_config)
    matches = re.findall(r'"(anthropic/claude-[^"]+)"', source)
    for model_id in matches:
        assert _OPENROUTER_ANTHROPIC_RE.match(model_id), (
            f"Malformed OpenRouter model ID: {model_id}"
        )


# ---------------------------------------------------------------------------
# 6. Regex coverage — edge cases
# ---------------------------------------------------------------------------

def test_regex_does_not_match_fable_4():
    """A hypothetical claude-fable-4 should not be treated as strict."""
    assert not _STRICT_PARAM_MODEL_RE.search("claude-fable-4-20250101")


def test_regex_does_not_match_opus_4():
    """Standard opus-4 should not be treated as strict."""
    assert not _STRICT_PARAM_MODEL_RE.search("claude-opus-4-20250514")


def test_regex_matches_bare_fable_5():
    """Bare fable-5 without date suffix should still match."""
    assert _STRICT_PARAM_MODEL_RE.search("claude-fable-5")
