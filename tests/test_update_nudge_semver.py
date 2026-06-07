"""Regression lock for issue #424 — update nudge must never advertise a downgrade.

The telemetry server returns an advisory ``update_available`` flag plus a
``latest_version`` string. The client used to trust that flag blindly, so a
stale/rolled-back server (or any mismatch) could nudge the user to "upgrade" to
a version older than or equal to the one already installed.

`telemetry._is_real_upgrade` is the client-side semver gate: the notice is only
surfaced when ``latest_version`` parses to a strictly newer version than the
currently installed one. These tests pin that behavior.
"""
from __future__ import annotations

from truememory import telemetry
from truememory.telemetry import _is_real_upgrade


class TestIsRealUpgrade:
    def test_newer_latest_is_upgrade(self):
        assert _is_real_upgrade("0.8.0", "0.7.1.3") is True
        assert _is_real_upgrade("0.7.2", "0.7.1.3") is True
        assert _is_real_upgrade("1.0.0", "0.7.1.3") is True
        assert _is_real_upgrade("0.7.1.4", "0.7.1.3") is True

    def test_equal_version_is_not_upgrade(self):
        # The pre-fix bug: equal version still nudged. Must be suppressed.
        assert _is_real_upgrade("0.7.1.3", "0.7.1.3") is False

    def test_older_latest_is_downgrade_suppressed(self):
        # The core #424 bug: server advertises an older "latest". Never nudge.
        assert _is_real_upgrade("0.7.0", "0.7.1.3") is False
        assert _is_real_upgrade("0.6.9", "0.7.1.3") is False
        assert _is_real_upgrade("0.7.1.2", "0.7.1.3") is False

    def test_missing_or_unknown_version_suppressed(self):
        assert _is_real_upgrade(None, "0.7.1.3") is False
        assert _is_real_upgrade("0.8.0", None) is False
        assert _is_real_upgrade("0.8.0", "unknown") is False
        assert _is_real_upgrade("", "0.7.1.3") is False

    def test_latest_unknown_suppressed(self):
        # latest == "unknown" must suppress, even with a valid current version.
        # A naive string compare ("unknown" != "0.7.1.3") would wrongly nudge.
        assert _is_real_upgrade("unknown", "0.7.1.3") is False

    def test_unparseable_version_suppressed(self):
        # Conservative: garbage version -> suppress rather than risk a downgrade.
        assert _is_real_upgrade("not-a-version", "0.7.1.3") is False
        assert _is_real_upgrade("0.8.0", "also-not-a-version") is False

    def test_packaging_missing_suppressed(self, monkeypatch):
        # When ``packaging`` is unavailable at import time, the module-level
        # ``_version_parse`` is left as None. We must fail conservative and
        # suppress the nudge -- never fall back to a naive string compare that
        # could advertise a downgrade (e.g. "0.7.0" != "0.7.1.3" -> True).
        monkeypatch.setattr(telemetry, "_version_parse", None)
        # A genuinely newer latest would normally pass, but with no parser
        # available we cannot trust the comparison, so suppress.
        assert _is_real_upgrade("0.8.0", "0.7.1.3") is False
        # And we definitely never advertise a downgrade via string compare.
        assert _is_real_upgrade("0.7.0", "0.7.1.3") is False


class TestFlushSyncGate:
    """`_flush_sync` must apply the semver gate to the server response."""

    def _post(self, monkeypatch, payload):
        class _Resp:
            def json(self_inner):
                return payload

        class _FakeHttpx:
            @staticmethod
            def post(*args, **kwargs):
                return _Resp()

        import sys
        monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx)
        # Ensure there is at least one event to flush.
        with telemetry._lock:
            telemetry._session_events.clear()
            telemetry._session_events.append({"event": "x"})

    def test_downgrade_flag_is_dropped(self, monkeypatch):
        monkeypatch.setattr(telemetry, "_get_version", lambda: "0.7.1.3")
        self._post(
            monkeypatch,
            {"update_available": True, "latest_version": "0.6.0", "message": "old"},
        )
        # Pre-fix this returned the dict and a downgrade nudge was written.
        assert telemetry._flush_sync() is None

    def test_equal_flag_is_dropped(self, monkeypatch):
        monkeypatch.setattr(telemetry, "_get_version", lambda: "0.7.1.3")
        self._post(
            monkeypatch,
            {"update_available": True, "latest_version": "0.7.1.3"},
        )
        assert telemetry._flush_sync() is None

    def test_real_upgrade_passes_through(self, monkeypatch):
        monkeypatch.setattr(telemetry, "_get_version", lambda: "0.7.1.3")
        self._post(
            monkeypatch,
            {"update_available": True, "latest_version": "0.9.0", "message": "go"},
        )
        info = telemetry._flush_sync()
        assert info is not None
        assert info["latest_version"] == "0.9.0"
