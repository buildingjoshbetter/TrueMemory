#!/usr/bin/env python3
"""Populate ~/.truememory/backlog/ with markers pointing to the last N session
transcripts from ~/.claude/projects/. Used to stress-test the spawn cap fix.

Usage:
    python scripts/populate_backlog_stress_test.py [--count 300] [--dry-run]
"""
import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    claude_projects = Path.home() / ".claude" / "projects"
    backlog_dir = Path.home() / ".truememory" / "backlog"

    transcripts = sorted(
        (p for p in claude_projects.rglob("*.jsonl") if p.stat().st_size > 1024),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:args.count]

    print(f"Found {len(transcripts)} transcripts (requested {args.count})")

    if not args.dry_run:
        backlog_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    for t in transcripts:
        session_id = t.stem
        marker = {
            "transcript_path": str(t),
            "session_id": session_id,
            "user_id": "",
            "db_path": "",
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "reason": "stress_test_backfill",
        }
        marker_path = backlog_dir / f"{session_id}.json"
        if args.dry_run:
            print(f"  [dry-run] would create {marker_path.name}")
        else:
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
        created += 1

    print(f"{'Would create' if args.dry_run else 'Created'} {created} backlog markers in {backlog_dir}")
    if not args.dry_run:
        print(f"\nNow re-enable TrueMemory hooks and watch:")
        print(f"  watch -n1 'pgrep -f truememory.ingest.cli | wc -l'")
        print(f"  # Should stay <= SPAWN_CAP (default 2)")


if __name__ == "__main__":
    main()
