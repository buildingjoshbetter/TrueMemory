"""Regression test for M11 #508: timezone mixing in detect_episodes.

Timestamps with negative UTC offsets (e.g., -05:00) create aware datetimes
while Z-suffixed ones become naive after the strip logic, causing TypeError
on comparison.
"""

import sqlite3
import unittest


def _make_db_with_messages(timestamps):
    """Create an in-memory DB with messages at the given timestamps."""
    from truememory.storage import create_db
    conn = create_db(":memory:")
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS episodes "
            "(id INTEGER PRIMARY KEY, start_time TEXT, end_time TEXT, message_count INTEGER)"
        )
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE messages ADD COLUMN episode_id INTEGER")
    except Exception:
        pass
    for i, ts in enumerate(timestamps):
        conn.execute(
            "INSERT INTO messages (content, sender, recipient, timestamp, category, modality) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (f"message {i}", "alice", "bob", ts, "", ""),
        )
    conn.commit()
    return conn


class TestDetectEpisodesTimezone(unittest.TestCase):

    def test_mixed_tz_correctly_groups_episodes(self):
        """Mixed timezones must produce correct episode groupings, not silent misgroups."""
        from truememory.temporal import detect_episodes
        conn = _make_db_with_messages([
            "2026-01-15T10:00:00Z",
            "2026-01-15T11:00:00-05:00",
            "2026-01-16T23:00:00+03:00",
        ])
        count = detect_episodes(conn)
        self.assertEqual(count, 2,
                         "Messages 12+ hours apart should be in separate episodes even with mixed TZs")

    def test_negative_offset_not_silently_kept(self):
        """Negative-offset timestamps must not create aware datetimes."""
        from truememory.temporal import detect_episodes
        conn = _make_db_with_messages([
            "2026-01-15T10:00:00Z",
            "2026-01-15T10:30:00-05:00",
        ])
        count = detect_episodes(conn)
        self.assertGreater(count, 0)

    def test_all_naive_still_works(self):
        from truememory.temporal import detect_episodes
        conn = _make_db_with_messages([
            "2026-01-15T10:00:00",
            "2026-01-15T10:30:00",
            "2026-01-16T20:00:00",
        ])
        count = detect_episodes(conn)
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
