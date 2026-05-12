"""
Migrate MEMORY.md facts into TrueMemory.

Parses Claude Code's ~/.claude/projects/*/memory/MEMORY.md into atomic
facts and stores them via Memory().add(). The encoding gate handles
deduplication automatically.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_PURE_LINK_RE = re.compile(r"^\s*-\s*\[([^\]]*)\]\([^)]*\)\s*$")
_BULLET_RE = re.compile(r"^(\s*)- (.+)$")
_HEADER_RE = re.compile(r"^##\s+(.+)$")
_CODE_FENCE_RE = re.compile(r"^```")

_SLIM_TEMPLATE = """\
# Memory

This file is for **session-specific working notes only** — task state,
scratch plans, and conversation context. Long-horizon user facts live in
TrueMemory (search with `truememory_search`).
"""


def auto_detect_memory_md() -> Path | None:
    """Find MEMORY.md in the standard Claude Code memory directories."""
    candidates = [
        Path.home() / ".claude" / "projects",
    ]
    for base in candidates:
        if not base.exists():
            continue
        for memory_md in sorted(base.rglob("MEMORY.md")):
            if memory_md.stat().st_size > 0:
                return memory_md
    return None


def parse_memory_md(path: Path) -> list[dict]:
    """Parse a MEMORY.md file into a list of atomic facts.

    Returns list of dicts with keys: content, category, source_header.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    facts: list[dict] = []
    current_header = ""
    in_code_block = False
    parent_bullet: str | None = None

    for line in lines:
        if _CODE_FENCE_RE.match(line):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            continue

        header_match = _HEADER_RE.match(line)
        if header_match:
            current_header = header_match.group(1).strip()
            parent_bullet = None
            continue

        if not line.strip():
            parent_bullet = None
            continue

        if line.startswith(">"):
            continue

        if line.startswith("# ") and not line.startswith("## "):
            continue

        if _PURE_LINK_RE.match(line):
            continue

        bullet_match = _BULLET_RE.match(line)
        if not bullet_match:
            continue

        indent = bullet_match.group(1)
        raw_text = bullet_match.group(2).strip()

        if not raw_text:
            continue

        content = _LINK_RE.sub(r"\1", raw_text)
        content = content.strip()

        if not content:
            continue

        is_nested = len(indent) >= 2

        if is_nested and parent_bullet:
            fact_text = f"{current_header}: {parent_bullet} — {content}"
        elif current_header:
            fact_text = f"{current_header}: {content}"
            parent_bullet = content
        else:
            fact_text = content
            parent_bullet = content

        if not is_nested:
            parent_bullet = content

        facts.append({
            "content": fact_text,
            "category": current_header,
            "source_header": current_header,
        })

    return facts


def migrate(
    memory_md_path: Path,
    db_path: str | None = None,
    dry_run: bool = False,
    backup: bool = True,
    write_slim_template: bool = False,
) -> dict:
    """Migrate MEMORY.md facts into TrueMemory.

    Returns dict with keys: migrated, skipped, duplicates, errors, backup_path.
    """
    facts = parse_memory_md(memory_md_path)

    result = {
        "migrated": 0,
        "skipped": 0,
        "duplicates": 0,
        "errors": 0,
        "backup_path": None,
    }

    if not facts:
        return result

    if backup and not dry_run:
        backup_path = memory_md_path.with_suffix(".md.bak")
        shutil.copy2(memory_md_path, backup_path)
        result["backup_path"] = str(backup_path)

    if dry_run:
        for fact in facts:
            content = fact["content"]
            category = fact["category"]
            print(f"  [{category}] {content}")
        result["migrated"] = len(facts)
        return result

    from truememory import Memory

    kwargs = {}
    if db_path:
        kwargs["path"] = db_path
    mem = Memory(**kwargs)

    for fact in facts:
        try:
            mem.add(
                content=fact["content"],
                user_id="MEMORY.md migration",
            )
            result["migrated"] += 1
        except Exception as e:
            print(f"  ERROR storing fact: {e}")
            result["errors"] += 1

    if write_slim_template:
        memory_md_path.write_text(_SLIM_TEMPLATE, encoding="utf-8")

    return result
