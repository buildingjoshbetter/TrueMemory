"""Shared transcript-extraction cache for the MEMORIST gate-eval harness.

Most Phase 6 candidates (#1, #2, #5, #7, #8, #11-17) all start by calling
`truememory.ingest.extractor.extract_facts(transcript, llm_config)` to
LLM-extract atomic facts from a session transcript. The candidates differ
in WHAT they do with those facts (gate, dedup, decay, prune) — not in HOW
they extract.

Without sharing, a 6-candidate × 48-session sweep would cost:
    6 candidates × 48 sessions × $0.06 per extraction call ≈ $17

Which would blow the spec's $10 local-sweep budget on its own.

This module provides an optional `install()` that monkey-patches
`truememory.ingest.extractor.extract_facts` to consult a content-hash-keyed
cache at `~/.cache/memorist_extracts/<sha256-prefix>.json` before calling
the real LLM. First candidate to see a transcript pays the LLM cost;
subsequent candidates with the same transcript hit the cache and pay $0.

Cache key: sha256 of (transcript_text + llm_provider + llm_model). The
provider+model components ensure that switching extractor backends
invalidates the cache instead of mixing results from different LLMs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, fields
from pathlib import Path

log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "memorist_extracts"
_INSTALLED = False


def _cache_key(transcript: str, llm_config) -> str:
    """Stable key per (transcript, provider, model) tuple."""
    h = hashlib.sha256()
    h.update(transcript.encode("utf-8", errors="replace"))
    h.update(b"|")
    h.update((getattr(llm_config, "provider", "") or "").encode())
    h.update(b"|")
    h.update((getattr(llm_config, "model", "") or "").encode())
    return h.hexdigest()[:24]


def _facts_to_jsonable(facts) -> list[dict]:
    out = []
    for f in facts:
        try:
            out.append(asdict(f))
        except TypeError:
            # Not a dataclass — fall back to attribute scrape
            out.append({fld.name: getattr(f, fld.name, None) for fld in fields(f)})
    return out


def install() -> bool:
    """Monkey-patch truememory.ingest.extractor.extract_facts to use the cache.

    Returns True if installation succeeded, False if the underlying
    function couldn't be located. Safe to call multiple times — only the
    first call does anything.
    """
    global _INSTALLED
    if _INSTALLED:
        return True

    try:
        from truememory.ingest import extractor as _extractor
        from truememory.ingest.extractor import ExtractedFact
    except ImportError:
        log.warning("install(): truememory.ingest.extractor not importable; cache disabled")
        return False

    _real_extract_facts = _extractor.extract_facts

    def _cached_extract_facts(transcript: str, config, max_facts: int = 50,
                                max_chunks: int = 20):
        if not transcript or not transcript.strip():
            return []

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = _cache_key(transcript, config)
        cache_path = CACHE_DIR / f"{key}.json"

        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    return [ExtractedFact(
                        content=it.get("content", ""),
                        category=it.get("category", "general"),
                        confidence=it.get("confidence", "medium"),
                        source_role=it.get("source_role", "user"),
                    ) for it in payload[:max_facts]]
            except (json.JSONDecodeError, OSError, KeyError, TypeError) as e:
                log.warning("Cache hit but parse failed (%s) — re-extracting", e)

        # Cache miss — call the real extractor
        facts = _real_extract_facts(transcript, config, max_facts=max_facts,
                                      max_chunks=max_chunks)
        try:
            cache_path.write_text(
                json.dumps(_facts_to_jsonable(facts), ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            log.warning("Cache write failed: %s — extraction still returned", e)
        return facts

    _extractor.extract_facts = _cached_extract_facts

    # Also patch any module that already imported it under its original name
    try:
        from truememory.ingest import pipeline as _pipeline
        if getattr(_pipeline, "extract_facts", None) is _real_extract_facts:
            _pipeline.extract_facts = _cached_extract_facts
    except ImportError:
        pass

    _INSTALLED = True
    log.info("Extract-cache installed (cache dir: %s)", CACHE_DIR)
    return True


def cache_stats() -> dict:
    """Inspect cache state for journaling / Phase 9 reporting."""
    if not CACHE_DIR.exists():
        return {"n_entries": 0, "total_bytes": 0, "cache_dir": str(CACHE_DIR)}
    files = list(CACHE_DIR.glob("*.json"))
    return {
        "n_entries": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "cache_dir": str(CACHE_DIR),
    }


def clear() -> int:
    """Remove all cache entries. Returns the number of files deleted."""
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for p in CACHE_DIR.glob("*.json"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        import json as _j
        print(_j.dumps(cache_stats(), indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "clear":
        print(f"Cleared {clear()} entries")
    else:
        print("Usage: python _extract_cache.py [stats|clear]")
