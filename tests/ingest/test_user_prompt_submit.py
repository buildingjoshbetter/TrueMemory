"""Tests for _try_capture_email and _detect_recall in user_prompt_submit hook.

Covers:
  - #275: email capture hardened against injection / bogus emails
  - #288: recall regex expanded to high-confidence patterns
"""
from __future__ import annotations

import importlib
import json
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from truememory.ingest.hooks import user_prompt_submit as ups


@pytest.fixture(autouse=True)
def _reload():
    importlib.reload(ups)


# ── helpers ──────────────────────────────────────────────────────────


def _capture_email(prompt: str) -> str | None:
    """Run _try_capture_email in a temp dir and return the captured email."""
    tmp = tempfile.mkdtemp()
    try:
        with patch.object(Path, "home", return_value=Path(tmp)):
            tm = Path(tmp) / ".truememory"
            tm.mkdir()
            cfg = tm / "config.json"
            cfg.write_text(json.dumps({"tier": "base"}))
            ups._try_capture_email(prompt)
            return json.loads(cfg.read_text()).get("email")
    finally:
        import shutil
        shutil.rmtree(tmp)


# ── #275: email capture ─────────────────────────────────────────────


class TestEmailCapture:
    """Email capture should accept real emails and reject injection/noise."""

    def test_bare_email(self):
        assert _capture_email("josh@sauron.network") == "josh@sauron.network"

    def test_my_email_is(self):
        assert _capture_email("my email is josh@sauron.network") == "josh@sauron.network"

    def test_email_colon(self):
        assert _capture_email("email: josh@example.com") == "josh@example.com"

    def test_reach_me_at(self):
        assert _capture_email("reach me at josh@company.com") == "josh@company.com"

    def test_casual_wrapper(self):
        assert _capture_email("yeah josh@sauron.network") == "josh@sauron.network"

    def test_cctld(self):
        assert _capture_email("josh@mail.co.uk") == "josh@mail.co.uk"

    def test_cctld_com_au(self):
        assert _capture_email("my email is josh@example.com.au") == "josh@example.com.au"

    def test_rejects_sql_injection(self):
        assert _capture_email("'; DROP TABLE users; -- admin@evil.com") is None

    def test_rejects_select(self):
        assert _capture_email("SELECT * FROM users WHERE e='x@y.com'") is None

    def test_rejects_backtick(self):
        assert _capture_email("var x = `user@evil.com`") is None

    def test_rejects_long_prompt(self):
        assert _capture_email("x" * 185 + " user@example.com") is None

    def test_rejects_url_email(self):
        assert _capture_email("check https://site.com?u=test@evil.com") is None

    def test_rejects_someone_elses_email(self):
        assert _capture_email("email alice@company.com about the meeting") is None

    def test_rejects_config_snippet(self):
        assert _capture_email("SMTP_FROM=noreply@myapp.com") is None

    def test_rejects_code_comment(self):
        assert _capture_email("# Default: admin@example.com") is None

    def test_rejects_log_line(self):
        assert _capture_email("log shows auth failed for admin@corp.com") is None

    def test_already_set_skips(self):
        tmp = tempfile.mkdtemp()
        try:
            with patch.object(Path, "home", return_value=Path(tmp)):
                tm = Path(tmp) / ".truememory"
                tm.mkdir()
                cfg = tm / "config.json"
                cfg.write_text(json.dumps({"tier": "base", "email": "existing@x.com"}))
                ups._try_capture_email("new@example.com")
                assert json.loads(cfg.read_text())["email"] == "existing@x.com"
        finally:
            import shutil
            shutil.rmtree(tmp)


class TestEmailRegex:
    """_EMAIL_RE should match valid emails and reject bad TLDs."""

    def test_ascii_flag(self):
        assert ups._EMAIL_RE.flags & re.ASCII

    def test_valid_emails(self):
        for email in ["josh@sauron.network", "user@example.com", "a+b@c.co"]:
            assert ups._EMAIL_RE.search(email), f"Should match: {email}"

    def test_cctld_full_match(self):
        m = ups._EMAIL_RE.search("user@mail.co.uk")
        assert m and m.group(0) == "user@mail.co.uk"

    def test_rejects_numeric_tld(self):
        m = ups._EMAIL_RE.search("user@host.123")
        assert m is None

    def test_rejects_1char_tld(self):
        assert ups._EMAIL_RE.search("user@host.x") is None

    def test_rejects_11char_tld(self):
        assert ups._EMAIL_RE.search("user@host.abcdefghijk") is None

    def test_tld_segment_cap(self):
        m = ups._EMAIL_RE.search("user@a.com.au.uk.nz.xx")
        assert m and m.group(0).count(".") <= 4


# ── #288: recall detection ──────────────────────────────────────────


class TestRecallDetection:
    """_detect_recall should catch recall prompts and reject code/noise."""

    @pytest.mark.parametrize("prompt", [
        "do you remember my name",
        "can you recall what we decided",
        "you told me something about Python",
        "you said it was ready",
        "you mentioned a deadline",
        "we discussed deploying last week",
        "we decided to use React",
        "we agreed on the architecture",
        "last time we worked on the API",
        "last session we talked about auth",
        "earlier you mentioned something",
        "previously we discussed this",
        "yesterday you said it was done",
        "previous session notes",
        "previous conversation about the PR",
        "what is my favorite food",
        "what's my name",
        "who is the CEO",
        "when was the meeting",
        "where is the office",
        "remind me about the deadline",
        "have we ever discussed this",
        "my favorite editor is vim",
    ])
    def test_recall_matches(self, prompt):
        assert ups._detect_recall(prompt), f"Should match: {prompt}"

    @pytest.mark.parametrize("prompt", [
        "fix the bug in parser.py",
        "deploy to production",
        "npm install express",
        "git commit -m 'update'",
        "refactor the database layer",
        "run the test suite",
        "tell me about photosynthesis",
        "tell me about the Python GIL",
    ])
    def test_recall_rejects_non_recall(self, prompt):
        assert not ups._detect_recall(prompt), f"Should not match: {prompt}"

    def test_rejects_code(self):
        assert not ups._detect_recall("def recall_memories(): pass")

    def test_rejects_code_block(self):
        assert not ups._detect_recall("```python\nremember = True\n```")

    def test_rejects_short(self):
        assert not ups._detect_recall("hi")

    def test_rejects_long(self):
        assert not ups._detect_recall("x" * 501)
