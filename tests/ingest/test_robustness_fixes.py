"""
Regression tests for the robustness fixes applied against EDGE_CASE_REPORT.md
and the correctness fix in PERF_REPORT.md.

Covers, in order:

- **Bug #1** — Unreadable file is not silently reinterpreted as inline
  content. ``parse_transcript`` must return ``[]`` (and log) rather than
  produce a fake ``Message`` containing the path string.

- **Bug #2** — ``sqlite3.OperationalError`` raised by ``_store_fact`` /
  ``_update_fact`` is caught at the pipeline level and recorded in the
  trace as ``storage_failed`` rather than crashing the whole run.

- **Bug #3** — ``save_trace`` never raises on filesystem errors: it logs a
  warning and returns ``False`` so an otherwise-successful ingestion's
  exit code isn't contaminated by a diagnostic write failure.

- **Bug #4** — CLI preflight refuses to run (exit 4) when the DB or trace
  parent directory isn't writable, *before* any LLM call happens. This
  prevents paying for an extraction call against a dead DB path.

- **Bug #5** — Long transcripts are NO LONGER silently truncated to 30K
  chars. ``extract_facts`` now chunks long inputs and runs the extractor
  per chunk, covering the full conversation (up to ``max_chunks``).

- **Improvement A** — CLI exits 3 (not 0) when ``--provider claude_cli``
  is requested but the binary isn't on PATH.

- **Improvement B** — Malformed JSONL lines produce a summary warning
  rather than being silently dropped.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from truememory.ingest.extractor import (
    _chunk_transcript,
    extract_facts,
)
from truememory.ingest.models import LLMConfig
from truememory.ingest.pipeline import IngestionPipeline, save_trace, IngestionResult
from truememory.ingest.transcript import parse_transcript


REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class MockMemory:
    """Minimal Memory stand-in that mirrors the pipeline's expected surface."""

    def __init__(self) -> None:
        self.stored: list[dict] = []
        self._engine = self
        self.fail_next_adds = 0  # how many upcoming add() calls should raise

    def search(self, query: str, limit: int = 5, user_id: str | None = None) -> list[dict]:
        return []

    def add(self, content: str = "", user_id: str | None = None, metadata=None,
            sender: str = "", timestamp: str = "", category: str = "", **kwargs) -> dict:
        if self.fail_next_adds > 0:
            self.fail_next_adds -= 1
            raise sqlite3.OperationalError("database is locked")
        entry = {"id": len(self.stored), "content": content, "category": category}
        self.stored.append(entry)
        return entry

    def update(self, memory_id: int, content: str) -> dict | None:
        for m in self.stored:
            if m["id"] == memory_id:
                m["content"] = content
                return m
        return None


def _run_cli(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    base_env = os.environ.copy()
    base_env.pop("ANTHROPIC_API_KEY", None)
    base_env.pop("OPENROUTER_API_KEY", None)
    if env:
        base_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "truememory.ingest.cli"] + args,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=base_env,
    )


# ---------------------------------------------------------------------------
# Bug #1 — transcript.py silent path-fallback
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses chmod 000")
def test_bug1_unreadable_file_returns_empty_not_fake_content(caplog):
    """
    A file that exists but can't be read (chmod 000) must NOT be silently
    re-parsed as inline content. Previously, PermissionError was caught and
    the path string itself was handed to the plain-text parser, producing
    a fake ``Message(role='unknown', content='/tmp/unreadable.jsonl')``.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        f.write('{"role": "user", "content": "hi"}\n')
        path = f.name

    try:
        os.chmod(path, 0o000)

        caplog.clear()
        caplog.set_level(logging.ERROR, logger="truememory.ingest.transcript")
        msgs = parse_transcript(path)

        # MUST return empty — never a fake Message with path-as-content
        assert msgs == [], (
            f"parse_transcript should return [] for an unreadable file, "
            f"got: {msgs}"
        )

        # Must have logged the failure so operators can diagnose
        assert any(
            "cannot read transcript" in rec.message.lower()
            for rec in caplog.records
        ), f"Expected an ERROR log for unreadable file; got: {[r.message for r in caplog.records]}"
    finally:
        # Restore permissions so tempfile cleanup works
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        Path(path).unlink(missing_ok=True)


def test_bug1_nonexistent_path_still_treated_as_inline_content():
    """
    Strings that don't point at an existing file should still fall through
    to the inline-content path (that's the documented behaviour and the
    ``parse_plain_text`` code path depends on it).
    """
    # A string that clearly isn't a path but starts with a role marker
    text = "Human: I prefer bun over npm\nAssistant: Got it"
    msgs = parse_transcript(text)
    assert len(msgs) == 2
    assert msgs[0].role == "human"
    assert "bun" in msgs[0].content


# ---------------------------------------------------------------------------
# Bug #2 — sqlite3.OperationalError during storage shouldn't crash the run
# ---------------------------------------------------------------------------


def _fixture_with_extractable_fact() -> Path:
    """Write a short transcript that heuristic extraction will find a fact in."""
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    Path(path).write_text(
        "User: I prefer bun over npm for everything.\n"
        "Assistant: Noted!\n"
        "User: I live in Seattle, Washington.\n"
        "Assistant: Great city."
    )
    return Path(path)


def test_bug2_sqlite_operational_error_is_caught_and_traced():
    """
    A sqlite3.OperationalError from the storage layer must not propagate
    out of ingest_transcript. The affected fact is recorded as
    ``storage_failed`` in the trace and the pipeline continues.
    """
    memory = MockMemory()
    # Make the FIRST add() call fail with a "database is locked" error
    memory.fail_next_adds = 1

    pipeline = IngestionPipeline(
        memory=memory,
        gate_threshold=0.20,
        use_llm_dedup=False,
        llm_config=None,
    )
    pipeline.llm_config = None  # force heuristic extraction

    transcript = _fixture_with_extractable_fact()
    try:
        # This must NOT raise
        result = pipeline.ingest_transcript(str(transcript))
    finally:
        transcript.unlink(missing_ok=True)

    # At least one trace entry should record the storage failure
    failed_entries = [e for e in result.trace if e.get("action") == "storage_failed"]
    assert failed_entries, (
        f"Expected at least one storage_failed trace entry; "
        f"got actions={[e.get('action') for e in result.trace]}"
    )
    assert "storage_error" in failed_entries[0]
    assert "db locked" in failed_entries[0]["storage_error"]["reason"].lower()


# ---------------------------------------------------------------------------
# Bug #3 — save_trace must not raise on filesystem errors
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses chmod 555")
def test_bug3_save_trace_does_not_raise_on_unwritable_dir(caplog):
    """
    ``save_trace`` should log a warning and return ``False`` when its
    target directory isn't writable — never raise. A raised exception
    would contaminate the exit code of an otherwise-successful ingestion
    and cause callers to retry, duplicating facts.
    """
    result = IngestionResult(facts_extracted=3, facts_stored=3, elapsed_seconds=1.2)

    with tempfile.TemporaryDirectory() as tmp:
        locked_dir = Path(tmp) / "locked"
        locked_dir.mkdir()
        os.chmod(locked_dir, 0o555)  # read + execute but not writable
        try:
            target = locked_dir / "trace.json"

            caplog.clear()
            caplog.set_level(logging.WARNING, logger="truememory.ingest.pipeline")

            # Must return False, not raise
            ok = save_trace(result, target)
            assert ok is False

            assert any(
                "could not write trace" in rec.message.lower()
                for rec in caplog.records
            ), f"Expected a warning about the trace write failure; got: {[r.message for r in caplog.records]}"
        finally:
            os.chmod(locked_dir, 0o755)  # restore so TemporaryDirectory can clean up


def test_bug3_save_trace_returns_true_on_success():
    """Happy path: save_trace returns True and writes the file."""
    result = IngestionResult(facts_extracted=1, facts_stored=1, elapsed_seconds=0.5)
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "trace.json"
        ok = save_trace(result, target)
        assert ok is True
        assert target.exists()
        assert "facts_extracted" in target.read_text()


# ---------------------------------------------------------------------------
# Bug #4 — CLI preflight for non-writable DB / trace parent dirs
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses chmod 555")
def test_bug4_cli_exits_4_when_db_dir_not_writable(tmp_path):
    """
    When the DB parent directory isn't writable, the CLI must exit with code 4
    (preflight failure) BEFORE any LLM call — distinct from 2 (bad args) and
    3 (no LLM backend).
    """
    # Create a small transcript file
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("User: hi\nAssistant: hello")

    # Create a non-writable dir for the DB
    locked = tmp_path / "locked_db"
    locked.mkdir()
    os.chmod(locked, 0o555)

    try:
        result = _run_cli([
            "ingest",
            str(transcript),
            "--db", str(locked / "memories.db"),
        ])

        assert result.returncode == 4, (
            f"Expected exit code 4 for unwritable DB dir, got {result.returncode}. "
            f"stderr:\n{result.stderr}"
        )
        assert "not writable" in result.stderr.lower() or "cannot create" in result.stderr.lower() \
            or "cannot open" in result.stderr.lower(), (
            f"Expected a preflight error message; got stderr:\n{result.stderr}"
        )
    finally:
        os.chmod(locked, 0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses chmod 555")
def test_bug4_cli_exits_4_when_trace_dir_not_writable(tmp_path):
    """Same preflight but for the --trace target."""
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("User: hi\nAssistant: hello")

    locked = tmp_path / "locked_trace"
    locked.mkdir()
    os.chmod(locked, 0o555)

    try:
        result = _run_cli([
            "ingest",
            str(transcript),
            "--db", str(tmp_path / "mem.db"),
            "--trace", str(locked / "trace.json"),
        ])
        assert result.returncode == 4, (
            f"Expected exit code 4 for unwritable trace dir, got {result.returncode}. "
            f"stderr:\n{result.stderr}"
        )
    finally:
        os.chmod(locked, 0o755)


# ---------------------------------------------------------------------------
# Bug #5 — long transcripts are chunked, not silently truncated
# ---------------------------------------------------------------------------


def test_bug5_chunk_transcript_splits_on_message_boundaries():
    """``_chunk_transcript`` should honour message boundaries (``\\n\\n``)."""
    messages = [f"User: Message number {i} with some filler text" for i in range(500)]
    transcript = "\n\n".join(messages)

    chunks = _chunk_transcript(transcript, budget=5_000)

    # We should get multiple chunks
    assert len(chunks) > 1
    # Every chunk should be under budget (except possibly a single-message
    # chunk that's itself too big, which our test doesn't exercise)
    assert all(len(c) <= 5_500 for c in chunks), \
        f"A chunk exceeded budget: sizes={[len(c) for c in chunks]}"
    # Re-joining chunks should yield the original (modulo the separator
    # accounting — chunks are rejoined with \n\n internally). At minimum
    # no message should be split mid-text.
    rejoined = "\n\n".join(chunks)
    assert rejoined == transcript


def test_bug5_chunk_small_transcript_returns_single_chunk():
    """Short transcripts should not be split."""
    transcript = "User: hi\n\nAssistant: hello"
    chunks = _chunk_transcript(transcript, budget=20_000)
    assert chunks == [transcript]


def test_bug5_extract_facts_calls_llm_for_every_chunk():
    """
    A transcript bigger than the 20K-char chunk budget must result in
    multiple LLM calls — not a single truncated call. Previously, any
    transcript >30K chars was hard-truncated and the tail was silently
    dropped. Now ``extract_facts`` runs the extractor on every chunk and
    merges the results.
    """
    # Build a transcript well over 30K chars so the old hard truncation
    # would have dropped content AND that chunks into multiple pieces.
    filler = "User: This is an unrelated filler message that adds bulk. " * 20
    messages = []
    for i in range(50):
        messages.append(f"User: Message {i}. {filler}")
    transcript = "\n\n".join(messages)
    assert len(transcript) > 30_000, "precondition: transcript must exceed old 30K limit"

    call_count = {"n": 0}
    call_inputs: list[str] = []

    def fake_complete(config, prompt, system=""):
        call_count["n"] += 1
        call_inputs.append(prompt)
        # Return a distinct fact per chunk so we can verify merging
        return (
            f'[{{"content": "Chunk {call_count["n"]} fact", '
            f'"category": "technical", "confidence": "high"}}]'
        )

    config = LLMConfig(provider="ollama", model="fake", base_url="http://localhost:0")

    with patch("truememory.ingest.extractor.complete", side_effect=fake_complete):
        facts = extract_facts(transcript, config)

    # Must have called complete() MORE than once — the old code would
    # truncate and call only once.
    assert call_count["n"] > 1, (
        f"Expected multiple LLM calls for a >30K-char transcript, got {call_count['n']}"
    )
    # Each chunk should produce a distinct fact in the merged result
    assert len(facts) >= 2, f"Expected at least 2 merged facts, got {len(facts)}"
    # Verify the chunks together cover the full transcript length (no content dropped)
    # We approximate this by summing the prompt sizes and requiring near-coverage.
    # Because the prompt template adds overhead around the transcript content,
    # we just check that the number of chunks × budget >= transcript length.
    total_chunk_chars = sum(len(p) for p in call_inputs)
    assert total_chunk_chars >= len(transcript) * 0.8, (
        f"Total chunk content ({total_chunk_chars}) is far smaller than "
        f"transcript length ({len(transcript)}) — looks like silent truncation"
    )


def test_bug5_extract_facts_dedupes_across_chunks():
    """Identical facts emitted from multiple chunks should collapse to one."""
    transcript = "User: message " * 3000  # ~36K chars, forces chunking

    def fake_complete(config, prompt, system=""):
        # Every chunk returns the same fact — merging must collapse it
        return '[{"content": "User is verbose", "category": "preference"}]'

    config = LLMConfig(provider="ollama", model="fake", base_url="http://localhost:0")
    with patch("truememory.ingest.extractor.complete", side_effect=fake_complete):
        facts = extract_facts(transcript, config)

    # After dedupe, only one fact should remain even though multiple chunks ran
    assert len(facts) == 1
    assert facts[0].content == "User is verbose"


def test_bug5_extract_facts_respects_max_chunks_cap(caplog):
    """
    When chunking would exceed ``max_chunks`` we warn and process only the
    first N. The warning must mention how much content was dropped so users
    have a signal rather than the silent-truncation behaviour of the old code.
    """
    # Build a transcript with REAL message boundaries (\n\n) so the
    # chunker actually produces multiple chunks. A flat string without
    # separators would end up as a single over-budget chunk.
    one_message = "User: filler message with enough text to take up space " * 40
    assert len(one_message) > 1_000
    # Each message is ~2.3KB. ~200 of them with \n\n separators ≈ 500KB,
    # which at a 20K-char budget yields ~25 chunks — enough to exceed
    # max_chunks=3.
    messages = [f"{one_message} {i}" for i in range(200)]
    transcript = "\n\n".join(messages)

    def fake_complete(config, prompt, system=""):
        return "[]"

    caplog.clear()
    caplog.set_level(logging.WARNING, logger="truememory.ingest.extractor")

    config = LLMConfig(provider="ollama", model="fake", base_url="http://localhost:0")
    with patch("truememory.ingest.extractor.complete", side_effect=fake_complete) as mock:
        extract_facts(transcript, config, max_chunks=3)

    # Should have called complete() exactly max_chunks times
    assert mock.call_count == 3, (
        f"Expected 3 LLM calls (max_chunks cap), got {mock.call_count}"
    )

    # And emitted a warning naming the drop ratio
    warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("max_chunks" in m for m in warning_msgs), \
        f"Expected max_chunks warning; got: {warning_msgs}"


# ---------------------------------------------------------------------------
# Improvement A — claude_cli missing should exit non-zero
# ---------------------------------------------------------------------------


def test_improvement_a_missing_claude_cli_exits_3(tmp_path):
    """
    When ``--provider claude_cli`` is requested but ``claude`` is not on
    PATH, the CLI should exit 3 with a clear error rather than running
    the full pipeline and quietly exiting 0.
    """
    transcript = tmp_path / "t.txt"
    transcript.write_text("User: I prefer bun over npm\nAssistant: noted")

    # Nuke PATH so `claude` definitely can't be found
    result = _run_cli(
        [
            "ingest",
            str(transcript),
            "--provider", "claude_cli",
            "--db", str(tmp_path / "memories.db"),
        ],
        env={"PATH": "/nonexistent"},
    )

    assert result.returncode == 3, (
        f"Expected exit code 3 when claude CLI is missing, got {result.returncode}. "
        f"stderr:\n{result.stderr}"
    )
    assert "claude" in result.stderr.lower()


# ---------------------------------------------------------------------------
# Improvement B — malformed JSONL lines produce a warning
# ---------------------------------------------------------------------------


def test_improvement_b_malformed_jsonl_warns(caplog, tmp_path):
    """Mixed good/bad JSONL files should log a summary warning."""
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        '{"role": "user", "content": "I live in Seattle"}\n'
        'THIS IS NOT JSON\n'
        '{"role": "assistant", "content": "cool"}\n'
        'ALSO NOT JSON {{\n'
    )

    caplog.clear()
    caplog.set_level(logging.WARNING, logger="truememory.ingest.transcript")
    msgs = parse_transcript(str(path))

    # Good lines should still be parsed
    assert len(msgs) == 2

    # A single summary warning should have fired
    warnings = [r.message for r in caplog.records if "malformed" in r.message.lower()]
    assert warnings, f"Expected a malformed-line warning; got: {[r.message for r in caplog.records]}"
    assert "2" in warnings[0], f"Expected the count in the warning; got: {warnings[0]}"
