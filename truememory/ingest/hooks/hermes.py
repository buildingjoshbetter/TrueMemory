#!/usr/bin/env python3
"""Native Hermes shell hooks with private, retained content snapshots.

The installed argv selects the event and account. Native input supplies
content and correlation identifiers, never a transcript filename. Worker
return, context delivery and complete extraction are separate observations.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import logging
import os
import re
import stat
import subprocess
import sys
import time
import uuid
import contextlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

log = logging.getLogger(__name__)
EVENTS = ("on_session_start", "pre_llm_call", "post_llm_call",
          "on_session_end", "on_session_finalize")
MAX_INPUT_BYTES = 4 * 1024 * 1024
MAX_SCOPE_BYTES = 64 * 1024 * 1024
MAX_SNAPSHOTS = 1024
MAX_CONTEXT_BYTES = 32 * 1024
MAX_STATE_BYTES = 8 * 1024 * 1024
CONTEXT_WAIT_SECONDS = 2.0
MAX_ATTEMPTS = 5
_HEX = re.compile(r"[a-f0-9]{64}\Z")
_SNAPSHOT = re.compile(r"[a-f0-9]{64}\.json\Z")
# Capture states that never need their snapshot for extraction again.
_SETTLED = {"extraction_complete", "deferred", "superseded"}
# Context states that need no further recall work or delivery.
_CONTEXT_DONE = {"ready", "written", "expired"}
# Released capture entries kept per session, counted in turns from the latest.
RETAINED_TURNS = 64


class BridgeError(ValueError):
    """A rejected native event or unavailable owned state."""


def _encoded(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def _identity(value: object, name: str) -> str:
    if (not isinstance(value, str) or not value.strip()
            or value in {"unknown", "undefined", "null"}
            or len(value.encode("utf-8")) > 1024
            or any(ord(char) < 32 for char in value)):
        raise BridgeError(f"invalid {name}")
    return value


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise BridgeError("unsupported native text")
    parts: list[str] = []
    for part in value:
        if not isinstance(part, dict):
            raise BridgeError("invalid native content part")
        kind = part.get("type")
        if kind in {"text", "input_text", "output_text"}:
            text = part.get("text")
            if not isinstance(text, str):
                raise BridgeError("invalid native text part")
            parts.append(text)
        elif kind not in {"image", "image_url", "input_image", "audio", "input_audio"}:
            raise BridgeError("unsupported native content part")
    return "\n".join(parts)


def _messages(extra: dict[str, object], event: str) -> list[dict[str, str]]:
    history = extra.get("conversation_history", [])
    if not isinstance(history, list):
        raise BridgeError("invalid native conversation history")
    messages: list[dict[str, str]] = []
    for row in history:
        if not isinstance(row, dict) or not isinstance(row.get("role"), str):
            raise BridgeError("invalid native history row")
        role = row["role"]
        if role in {"system", "developer", "tool", "tool_use", "tool_result"}:
            continue
        if role not in {"user", "human", "assistant"}:
            raise BridgeError("unsupported native history role")
        content = _text(row.get("content"))
        messages.append({"role": "human" if role in {"user", "human"} else role,
                         "content": content})
    user = _text(extra.get("user_message"))
    # The current user is the final user row in the native producer's history.
    # Never discover the current prompt by searching historical text.
    last_user = next((i for i in range(len(messages) - 1, -1, -1)
                      if messages[i]["role"] == "human"), None)
    if last_user is None:
        messages.append({"role": "human", "content": user})
    elif messages[last_user]["content"] != user:
        messages.append({"role": "human", "content": user})
    if event == "post_llm_call":
        assistant = _text(extra.get("assistant_response"))
        if not messages or messages[-1] != {"role": "assistant", "content": assistant}:
            messages.append({"role": "assistant", "content": assistant})
    return messages


def _check_stat(value: os.stat_result, *, directory: bool = False) -> None:
    valid = stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode)
    if not valid or (not directory and value.st_nlink != 1):
        raise BridgeError("owned path is not a private regular file")
    if hasattr(os, "getuid") and value.st_uid != os.getuid():
        raise BridgeError("owned path has a foreign owner")
    if directory and os.name != "nt" and value.st_mode & 0o077:
        raise BridgeError("owned directory is not private")


def _directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    _check_stat(path.lstat(), directory=True)


def _open_owned(path: Path, flags: int) -> int:
    try:
        before = path.lstat()
    except FileNotFoundError:
        before = None
    if before is not None:
        _check_stat(before)
    fd = os.open(str(path), flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        actual = os.fstat(fd)
        _check_stat(actual)
        after = path.lstat()
        _check_stat(after)
        if (actual.st_dev, actual.st_ino) != (after.st_dev, after.st_ino):
            raise BridgeError("owned file changed during open")
        if before is not None and (before.st_dev, before.st_ino) != (actual.st_dev, actual.st_ino):
            raise BridgeError("owned file substituted during open")
        return fd
    except (OSError, BridgeError):
        os.close(fd)
        raise


def _read_owned(path: Path, limit: int = MAX_STATE_BYTES) -> dict[str, object]:
    fd = _open_owned(path, os.O_RDONLY)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if os.name == "nt":
        final_fd = _open_owned(path, os.O_RDONLY)
        try:
            final = os.fstat(final_fd)
        finally:
            os.close(final_fd)
    else:
        final = path.lstat()
    _check_stat(final)
    def version(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if (version(before) != version(after) or (after.st_dev, after.st_ino, after.st_size,
                           after.st_mtime_ns, after.st_ctime_ns) !=
            (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns)):
        raise BridgeError("owned file changed during read")
    if len(raw) > limit:
        raise BridgeError("owned state exceeds bound")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise BridgeError("invalid owned state")
    return data


def _write_owned(path: Path, value: object, *, immutable: bool = False) -> None:
    raw = _encoded(value)
    if len(raw) > MAX_STATE_BYTES:
        raise BridgeError("owned state exceeds bound")
    if immutable:
        try:
            existing = _read_owned(path)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if _encoded(existing) != raw:
                raise BridgeError("immutable capture mismatch")
            return
    elif path.exists() or path.is_symlink():
        _check_stat(path.lstat())
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = _open_owned(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    # The scope lock serializes replacement. Failed temporary files remain.
    os.replace(temporary, path)
    if os.name != "nt":
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


@contextmanager
def _lock(path: Path, wait: float = 0.0) -> Iterator[bool]:
    fd = _open_owned(path, os.O_RDWR | os.O_CREAT)
    acquired = False
    deadline = time.monotonic() + wait
    try:
        while True:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
        yield acquired
    finally:
        if acquired:
            if sys.platform == "win32":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _birth(pid: int) -> float | None:
    from truememory._platform import process_birth
    return process_birth(pid)


def _owner() -> dict[str, object]:
    return {"pid": os.getpid(), "born": _birth(os.getpid())}


def _alive(identity: object) -> bool:
    if not isinstance(identity, dict):
        return False
    pid, born = identity.get("pid"), identity.get("born")
    if type(pid) is not int or pid <= 0:
        return False
    try:
        actual = _birth(pid)
    except OSError:
        return True
    # Unknown birth identity is never permission to replace a live owner.
    return actual is not None and (born is None or born == actual)


@dataclass
class Store:
    root: Path
    user: str
    db: str
    session: str

    def __post_init__(self) -> None:
        self.db = (self.db if self.db == ":memory:" else
                   str(Path(self.db or Path.home() / ".truememory/memories.db").expanduser().resolve()))
        self.account = _digest([self.user, self.db])
        self.key = _digest([self.account, self.session])
        if self.root.is_symlink():
            raise BridgeError("spool root is a symlink")
        self.root.parent.mkdir(parents=True, exist_ok=True)
        _directory(self.root)
        self.account_dir = self.root / self.account
        _directory(self.account_dir)
        self.path = self.account_dir / self.key
        _directory(self.path)
        self.capture_lock = self.path / "capture.lock"
        self.worker_lock = self.path / "worker.lock"
        self.state_path = self.path / "state.json"

    def state(self) -> dict[str, object]:
        self.validate_paths()
        try:
            data = _read_owned(self.state_path)
        except FileNotFoundError:
            return {"version": 1, "account": self.account, "session_id": self.session,
                    "turns": {}, "captures": {}, "worker": {}}
        if (data.get("version") != 1 or data.get("account") != self.account
                or data.get("session_id") != self.session
                or not all(isinstance(data.get(name), dict) for name in ("turns", "captures", "worker"))):
            raise BridgeError("owned state identity mismatch")
        return data

    def save(self, state: dict[str, object], *, quota_held: bool = False) -> None:
        self.validate_paths()
        self.write(self.state_path, state, quota_held=quota_held)

    def usage(self) -> tuple[int, int]:
        size = snapshots = 0
        for directory, subdirs, files in os.walk(self.account_dir, followlinks=False):
            for subdir in subdirs:
                _check_stat((Path(directory) / subdir).lstat(), directory=True)
            for filename in files:
                value = (Path(directory) / filename).lstat()
                _check_stat(value)
                size += value.st_size
                snapshots += bool(re.fullmatch(r"[a-f0-9]{64}\.json", filename))
        return size, snapshots

    def write(self, path: Path, value: object, *, quota_held: bool = False,
              immutable: bool = False) -> None:
        if not quota_held:
            with _lock(self.root / "quota.lock", 0.25) as allowed:
                if not allowed:
                    raise BridgeError("state backpressure: quota lock busy")
                self.write(path, value, quota_held=True, immutable=immutable)
            return
        self.validate_paths()
        size, _snapshots = self.usage()
        # Reserve the temporary write too. Denial preserves all old bytes.
        if size + len(_encoded(value)) > MAX_SCOPE_BYTES:
            raise BridgeError("state backpressure: retained account quota")
        _write_owned(path, value, immutable=immutable)

    def validate_paths(self) -> None:
        for path in (self.root, self.account_dir, self.path):
            _check_stat(path.lstat(), directory=True)

    def snapshot(self, name: str) -> dict[str, object]:
        self.validate_paths()
        if not re.fullmatch(r"[a-f0-9]{64}\.json", name):
            raise BridgeError("invalid owned snapshot key")
        data = _read_owned(self.path / name)
        expected = _digest({key: value for key, value in data.items() if key != "sha256"})
        if (data.get("sha256") != expected or name != f"{expected}.json"
                or data.get("account") != self.account or data.get("session_id") != self.session
                or data.get("user") != self.user or data.get("db") != self.db):
            raise BridgeError("owned snapshot identity mismatch")
        return data

    def capture(self, payload: dict[str, object], event: str) -> str:
        if event not in {"pre_llm_call", "post_llm_call"}:
            raise BridgeError("invalid capture event")
        if _identity(payload.get("session_id"), "session_id") != self.session:
            raise BridgeError("native session identity mismatch")
        extra = payload.get("extra")
        if not isinstance(extra, dict):
            raise BridgeError("invalid native extra")
        turn = _identity(extra.get("turn_id"), "turn_id")
        if "task_id" in extra:
            _identity(extra["task_id"], "task_id")
        messages = _messages(extra, event)
        data: dict[str, object] = {
            "version": 1, "account": self.account, "session_id": self.session,
            "turn_id": turn, "task_id": extra.get("task_id", ""),
            "user": self.user, "db": self.db, "event": event, "messages": messages,
            "native_user_message": extra.get("user_message"),
            "native_assistant_response": extra.get("assistant_response"),
            "native_history": extra.get("conversation_history", []),
        }
        digest = _digest(data)
        data["sha256"] = digest
        name, turn_key = f"{digest}.json", _digest([self.account, self.session, turn])
        with _lock(self.root / "quota.lock", 0.25) as quota_lock:
            if not quota_lock:
                raise BridgeError("capture backpressure: quota lock busy")
            with _lock(self.capture_lock, 0.25) as capture_lock:
                if not capture_lock:
                    raise BridgeError("capture backpressure: scope busy")
                state = self.state()
                captures, turns = state["captures"], state["turns"]
                # Free settled snapshots first so a full account can still admit.
                self.release(state)
                replaced = []
                if name not in captures:
                    size, count = self.usage()
                    if size + len(_encoded(data)) > MAX_SCOPE_BYTES // 2 or count >= MAX_SNAPSHOTS // 2:
                        # Other sessions' settled snapshots and finished sessions count
                        # toward the same account; reclaim them well before it is full.
                        self.release_account()
                        size, count = self.usage()
                    if count >= MAX_SNAPSHOTS:
                        raise BridgeError("capture backpressure: retained snapshot quota")
                    self.write(self.path / name, data, immutable=True, quota_held=True)
                    replaced = self.superseded(state, messages)
                    # A superseded flushed snapshot hands its extraction to this one.
                    flush = event == "post_llm_call" or any(captures[older].get("flush") for older in replaced)
                    captures[name] = {"turn": turn_key, "state": "captured", "attempts": 0,
                                      "retry_at": 0.0, "flush": flush}
                else:
                    self.snapshot(name)
                if turn_key not in turns:
                    turns[turn_key] = {"ordinal": len(turns) + 1, "snapshot": name,
                                       "context_state": "pending", "context_attempts": 0,
                                       "context_retry_at": 0.0, "context_requested": False}
                turn_state = turns[turn_key]
                if event == "pre_llm_call":
                    turn_state["context_requested"] = True
                    # Recall always uses original current input, never a post history row.
                    turn_state["snapshot"] = name
                for older in replaced:
                    # The new snapshot begins with these exact messages and covers them.
                    captures[older].update(state="superseded", superseded_by=name)
                self.save(state, quota_held=True)
                if replaced:
                    self.release(state)
                    self.save(state, quota_held=True)
        return turn_key

    def superseded(self, state: dict[str, object], messages: list[dict[str, str]]) -> list[str]:
        """Unsettled captures whose messages begin the incoming messages."""
        names = []
        for name, item in state["captures"].items():
            if item.get("state") in _SETTLED | {"processing"} or item.get("released"):
                continue
            try:
                older = self.snapshot(name)["messages"]
            except (OSError, ValueError):
                continue
            if isinstance(older, list) and older == messages[:len(older)]:
                names.append(name)
        return names

    def release_account(self) -> None:
        """Release settled snapshots of this account's other sessions; quota.lock is held."""
        for entry in os.scandir(self.account_dir):
            if entry.name == self.key or not _HEX.match(entry.name) or not entry.is_dir(follow_symlinks=False):
                continue
            try:
                session = _identity(_read_owned(Path(entry.path) / "state.json").get("session_id"), "session_id")
                other = Store(self.root, self.user, self.db, session)
                if other.key != entry.name:
                    continue
                with _lock(other.capture_lock, 0.25) as held:
                    if held:
                        state = other.state()
                        other.release(state)
                        if other.idle(state) and other.state_path.stat().st_mtime < time.time() - 3600:
                            # A finished session's residual state is not evidence anyone reads.
                            other.remove()
                        else:
                            other.save(state, quota_held=True)
            except (OSError, ValueError):
                continue

    def idle(self, state: dict[str, object]) -> bool:
        """Every snapshot released, no context owed and no live worker."""
        worker = state["worker"]
        return (all(item.get("released") for item in state["captures"].values())
                and all(turn.get("context_state") in _CONTEXT_DONE or not turn.get("context_requested")
                        or not _due(turn, "context_") for turn in state["turns"].values())
                and not (worker.get("state") in {"reserved", "running"}
                         and (_alive(worker.get("child")) or _alive(worker.get("launcher")))))

    def remove(self) -> None:
        for entry in os.scandir(self.path):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(entry.path)
        with contextlib.suppress(OSError):
            os.rmdir(self.path)

    def release(self, state: dict[str, object]) -> None:
        """Delete settled snapshots, delivered contexts and unadmitted files.

        Entries stay as evidence. A snapshot whose turn still awaits its context
        is kept for recall. Only persisted settled states are released, so a
        crash before the next save never strands live work without its bytes.
        """
        turns = state["turns"]
        latest = max((turn.get("ordinal", 0) for turn in turns.values()), default=0)
        for turn in turns.values():
            if turn.get("ordinal", 0) < latest and turn.get("context_state") not in {"written", "expired"}:
                # Only the hook of the latest turn can still deliver a context.
                turn["context_state"] = "expired"
        pending = {turn.get("snapshot") for turn in turns.values()
                   if turn.get("context_requested") and turn.get("context_state") not in _CONTEXT_DONE
                   and _due(turn, "context_")}
        for name, item in state["captures"].items():
            exhausted = item.get("state") != "processing" and not _due({"attempts": item.get("attempts", 0)})
            if (item.get("state") in _SETTLED or exhausted) and name not in pending and not item.get("released"):
                with contextlib.suppress(FileNotFoundError):
                    (self.path / name).unlink()
                item["released"] = True
        for key, turn in turns.items():
            if turn.get("context_state") in {"written", "expired"}:
                with contextlib.suppress(FileNotFoundError):
                    (self.path / f"{key}.context.json").unlink()
        horizon = latest - RETAINED_TURNS
        for name in [name for name, item in state["captures"].items()
                     if item.get("released") and turns.get(item.get("turn"), {}).get("ordinal", 0) < horizon]:
            # Old released entries are bounded; their bytes are already gone.
            del state["captures"][name]
        for entry in os.scandir(self.path):
            # A snapshot written by a capture whose state save then failed.
            if _SNAPSHOT.match(entry.name) and entry.name not in state["captures"]:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(entry.path)

    def flush(self) -> None:
        with _lock(self.capture_lock, 0.25) as allowed:
            if not allowed:
                raise BridgeError("flush backpressure: scope busy")
            state = self.state()
            for item in state["captures"].values():
                item["flush"] = True
            self.save(state)


def _recall(snapshot: dict[str, object], ordinal: int) -> str:
    from truememory.ingest.hooks import session_start, user_prompt_submit
    user, db = snapshot["user"], snapshot["db"]
    prompt = _text(snapshot["native_user_message"])
    if not prompt.strip():
        return ""
    intensity, _store_intensity = user_prompt_submit._get_intensity_config()
    if ordinal == 1:
        return session_start.recall_memories(
            {}, user_id=user, db_path=db, budget=session_start._intensity_budget(intensity),
            intensity=intensity,
        )
    context, searched = user_prompt_submit._try_proactive_recall(
        prompt, user, db, snapshot["session_id"], intensity, ordinal, debounced=False,
    )
    if not context and not searched:
        context = user_prompt_submit._try_auto_recall(
            prompt, user, db, snapshot["session_id"], debounced=False,
        )
    return context or ""


def _run_pipeline(snapshot: dict[str, object]) -> dict[str, object]:
    from truememory.ingest import extractor, pipeline
    from truememory.ingest.hooks.stop import GATE_THRESHOLD, MIN_MESSAGES
    from truememory.ingest.transcript import Message, format_for_extraction
    messages = [Message(role=row["role"], content=row["content"])
                for row in snapshot["messages"]]
    text = format_for_extraction(messages)
    if len(messages) < MIN_MESSAGES or len(text.strip()) < 50:
        return {"state": "deferred", "reason": "original_minimum"}
    outcomes: list[object] = []
    original, simple = pipeline.extract_facts, pipeline.extract_facts_simple
    outcome_api = getattr(extractor, "extract_facts_outcome", None)
    coverage_supported = callable(outcome_api)

    def observe(*args: object, **kwargs: object) -> list[object]:
        outcome = outcome_api(*args, **kwargs)
        outcomes.append(outcome)
        return outcome.facts

    heuristic_completed = False

    def observe_simple(*args: object, **kwargs: object) -> list[object]:
        nonlocal heuristic_completed
        facts = simple(*args, **kwargs)
        heuristic_completed = True
        return facts

    worker = pipeline.IngestionPipeline(user_id=snapshot["user"], db_path=snapshot["db"],
                                        gate_threshold=GATE_THRESHOLD)
    # This process runs one pipeline on one thread. Restore bindings even if
    # extraction or storage throws; the underlying selection is unchanged.
    if coverage_supported:
        pipeline.extract_facts, pipeline.extract_facts_simple = observe, observe_simple
    try:
        result = worker.ingest_text(text, session_id=f"hermes:{snapshot['session_id']}")
    finally:
        pipeline.extract_facts, pipeline.extract_facts_simple = original, simple
    fields = ("facts_extracted", "facts_stored", "facts_updated", "facts_skipped_gate",
              "facts_skipped_dedup", "facts_encoded")
    counts = {name: getattr(result, name, None) for name in fields}
    valid = all(type(value) is int and value >= 0 for value in counts.values())
    accounted = (valid and counts["facts_extracted"] == sum(counts[name] for name in fields[1:5])
                 and counts["facts_encoded"] == sum(counts[name] for name in
                                                      ("facts_stored", "facts_updated", "facts_skipped_dedup")))
    extraction_complete = coverage_supported and len(outcomes) == 1 and outcomes[0].complete
    coverage = ("coverage_unproven" if not coverage_supported else
                outcomes[0].status if len(outcomes) == 1 else
                "heuristic_returned" if heuristic_completed else "unobserved")
    return {"state": "extraction_complete" if extraction_complete and accounted else "worker_returned",
            "extraction_status": coverage, "terminal_accounting_complete": bool(accounted),
            "counts": counts}


def _retry(item: dict[str, object], prefix: str = "") -> None:
    attempts = int(item.get(f"{prefix}attempts", 0))
    item[f"{prefix}retry_at"] = time.time() + min(300, 2 ** min(attempts, 9))


def _due(item: dict[str, object], prefix: str = "") -> bool:
    return (int(item.get(f"{prefix}attempts", 0)) < MAX_ATTEMPTS
            and float(item.get(f"{prefix}retry_at", 0)) <= time.time())


def _work_available(state: dict[str, object]) -> bool:
    return (any(item.get("context_requested") and item.get("context_state") not in _CONTEXT_DONE
                and _due(item, "context_") for item in state["turns"].values())
            or any(item.get("flush") and item.get("state") not in _SETTLED and not item.get("released")
                   and _due(item) for item in state["captures"].values()))


def _spawn(store: Store) -> None:
    from truememory.hooks.core import register_spawned_pid, spawn_gate
    from truememory.ingest.hooks._shared import check_extraction_budget, refund_extraction_budget
    with _lock(store.capture_lock, 0.25) as scope_lock:
        if not scope_lock:
            return
        state = store.state()
        if not _work_available(state):
            return
        with _lock(store.worker_lock) as worker_free:
            if not worker_free:
                return
            reservation = state["worker"]
            if (reservation.get("state") in {"reserved", "running"}
                    and (_alive(reservation.get("child")) or _alive(reservation.get("launcher")))):
                return
            with _lock(store.root / "budget.lock", 0.25) as budget_lock:
                if not budget_lock or not check_extraction_budget():
                    return
                spawned = False
                try:
                    with spawn_gate() as allowed:
                        if not allowed:
                            refund_extraction_budget()
                            return
                        token = uuid.uuid4().hex
                        state["worker"] = {"state": "reserved", "token": token,
                                           "launcher": _owner(), "child": None}
                        store.save(state)
                        args = [sys.executable, str(Path(__file__).resolve()), "--worker", store.key,
                                "--token", token, "--session", store.session,
                                "--user", store.user, "--db", store.db]
                        env = dict(os.environ, TRUEMEMORY_EXTRACTION="1",
                                   TRUEMEMORY_HERMES_SPOOL_DIR=str(store.root))
                        kwargs: dict[str, object] = {"stdin": subprocess.DEVNULL,
                                                     "stdout": subprocess.DEVNULL,
                                                     "stderr": subprocess.DEVNULL, "env": env}
                        if sys.platform == "win32":
                            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                        else:
                            kwargs["start_new_session"] = True
                        child = subprocess.Popen(args, **kwargs)
                        spawned = True
                        register_spawned_pid(child.pid)
                        state["worker"]["child"] = {"pid": child.pid, "born": _birth(child.pid)}
                        store.save(state)
                except (OSError, BridgeError):
                    if not spawned:
                        refund_extraction_budget()
                    state["worker"]["state"] = "spawn_failed"
                    store.save(state)
                    raise


def _worker(store: Store, token: str) -> None:
    from truememory.ingest.hooks._shared import check_extraction_budget, refund_extraction_budget
    with _lock(store.worker_lock) as allowed:
        if not allowed:
            return
        with _lock(store.capture_lock, 2.0) as scope_lock:
            if not scope_lock:
                raise BridgeError("worker scope lock busy")
            state = store.state()
            if state["worker"].get("token") != token:
                raise BridgeError("worker reservation mismatch")
            state["worker"].update({"state": "running", "child": _owner(), "launcher": None})
            store.save(state)
        # Refresh after every attempt so content arriving during extraction
        # stays eligible. One launch handles at most 32 bounded attempts.
        extraction_attempts = 0
        for _attempt in range(32):
            with _lock(store.capture_lock, 2.0) as scope_lock:
                if not scope_lock:
                    raise BridgeError("worker queue claim busy")
                current = store.state()
                contexts = [(key, item) for key, item in current["turns"].items()
                            if item.get("context_requested") and item.get("context_state") not in _CONTEXT_DONE
                            and _due(item, "context_")]
                captures = [(key, item) for key, item in current["captures"].items()
                            if item.get("flush") and item.get("state") not in _SETTLED
                            and not item.get("released") and _due(item)]
                if not contexts and not captures:
                    break
                is_context = bool(contexts)
                key, target = (contexts or captures)[0]
                if not is_context and extraction_attempts:
                    with _lock(store.root / "budget.lock", 0.25) as budget_lock:
                        if not budget_lock or not check_extraction_budget():
                            break
                target = dict(target)
                selected = current["turns" if is_context else "captures"][key]
                prefix = "context_" if is_context else ""
                selected[f"{prefix}attempts"] += 1
                selected[f"{prefix}state"] = "processing"
                _retry(selected, prefix)
                store.save(current)
            try:
                if is_context:
                    context = _recall(store.snapshot(target["snapshot"]), int(target["ordinal"]))
                    if not isinstance(context, str) or len(context.encode("utf-8")) > MAX_CONTEXT_BYTES:
                        raise BridgeError("context output exceeds bound")
                    result = {"account": store.account, "turn": key, "context": context,
                              "sha256": _digest(context)}
                    store.write(store.path / f"{key}.context.json", result)
                    observation = {"context_state": "ready", "context_error": ""}
                else:
                    extraction_attempts += 1
                    observation = _run_pipeline(store.snapshot(key))
            except Exception as error:
                observation = {f"{prefix}state": "failed", f"{prefix}error_category": type(error).__name__}
            with _lock(store.capture_lock, 2.0) as scope_lock:
                if not scope_lock:
                    raise BridgeError("worker observation commit busy")
                current = store.state()
                selected = current["turns" if is_context else "captures"][key]
                selected.update(observation)
                if selected[f"{prefix}state"] not in {"ready", "extraction_complete", "deferred"}:
                    _retry(selected, prefix)
                store.save(current)
                # Persisted settled snapshots and ready contexts' sources are released.
                store.release(current)
                store.save(current)
        with _lock(store.capture_lock, 2.0) as scope_lock:
            if not scope_lock:
                raise BridgeError("worker final observation busy")
            current = store.state()
            current["worker"]["state"] = "returned"
            store.save(current)
        if not extraction_attempts:
            with _lock(store.root / "budget.lock", 0.25) as budget_lock:
                if budget_lock:
                    refund_extraction_budget()


def _emit_context(store: Store, turn_key: str, output: TextIO) -> bool:
    deadline = time.monotonic() + CONTEXT_WAIT_SECONDS
    while True:
        with _lock(store.capture_lock, 0.1) as scope_lock:
            if scope_lock:
                state = store.state()
                turn = state["turns"][turn_key]
                if turn["context_state"] in {"ready", "written"}:
                    result = _read_owned(store.path / f"{turn_key}.context.json", MAX_CONTEXT_BYTES * 6)
                    context = result.get("context")
                    if (not isinstance(context, str) or len(context.encode("utf-8")) > MAX_CONTEXT_BYTES
                            or result.get("account") != store.account or result.get("turn") != turn_key
                            or result.get("sha256") != _digest(context)):
                        raise BridgeError("context cache identity mismatch")
                    rendered = json.dumps({"context": context}, ensure_ascii=False)
                    written = output.write(rendered + "\n")
                    if written is not None and written != len(rendered) + 1:
                        raise BridgeError("incomplete context output write")
                    output.flush()
                    turn["context_state"] = "written"
                    store.save(state)
                    return bool(context)
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


def _emit_direct_context(store: Store, payload: dict[str, object], output: TextIO) -> bool:
    """Recall in the hook when the spool cannot admit this turn's snapshot."""
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        return False
    try:
        ordinal = len(store.state()["turns"]) + 1
    except (OSError, ValueError):
        ordinal = 2
    snapshot = {"user": store.user, "db": store.db, "session_id": store.session,
                "native_user_message": extra.get("user_message")}
    context = _recall(snapshot, ordinal)
    if not isinstance(context, str) or not context or len(context.encode("utf-8")) > MAX_CONTEXT_BYTES:
        return False
    rendered = json.dumps({"context": context}, ensure_ascii=False)
    written = output.write(rendered + "\n")
    if written is not None and written != len(rendered) + 1:
        raise BridgeError("incomplete context output write")
    output.flush()
    return True


def _input(stream: TextIO) -> dict[str, object]:
    source = getattr(stream, "buffer", stream)
    raw = source.read(MAX_INPUT_BYTES + 1)
    encoded = raw if isinstance(raw, bytes) else raw.encode("utf-8")
    if len(encoded) > MAX_INPUT_BYTES:
        raise BridgeError("native input exceeds bound")
    value = json.loads(encoded, parse_constant=lambda _value: (_ for _ in ()).throw(BridgeError("invalid JSON number")))
    if not isinstance(value, dict):
        raise BridgeError("native input is not an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", choices=EVENTS)
    parser.add_argument("--user", default=os.environ.get("TRUEMEMORY_USER_ID", ""))
    parser.add_argument("--db", default=os.environ.get("TRUEMEMORY_DB_PATH", ""))
    parser.add_argument("--worker")
    parser.add_argument("--token")
    parser.add_argument("--session")
    args = parser.parse_args()
    root = Path(os.environ.get("TRUEMEMORY_HERMES_SPOOL_DIR",
                               str(Path.home() / ".truememory/hermes-native-v1")))
    try:
        if args.worker is not None:
            if (not _HEX.fullmatch(args.worker) or not args.token
                    or not re.fullmatch(r"[a-f0-9]{32}", args.token)):
                raise BridgeError("invalid worker reservation")
            store = Store(root, args.user, args.db, _identity(args.session, "session_id"))
            if store.key != args.worker:
                raise BridgeError("worker account mismatch")
            _worker(store, args.token)
            return 0
        if os.environ.get("TRUEMEMORY_EXTRACTION"):
            return 0
        if args.event is None:
            raise BridgeError("installed event required")
        payload = _input(sys.stdin)
        session = _identity(payload.get("session_id"), "session_id")
        store = Store(root, args.user, args.db, session)
        if args.event in {"pre_llm_call", "post_llm_call"}:
            try:
                turn_key = store.capture(payload, args.event)
            except BridgeError as error:
                if args.event != "pre_llm_call" or "backpressure" not in str(error):
                    raise
                # A full spool must not cost the turn its recall; the worker
                # still runs so settled snapshots can be released.
                with contextlib.suppress(OSError, ValueError):
                    _spawn(store)
                _emit_direct_context(store, payload, sys.stdout)
                return 0
        else:
            turn_key = ""
            if args.event != "on_session_start":
                store.flush()
        _spawn(store)
        if args.event == "pre_llm_call":
            _emit_context(store, turn_key, sys.stdout)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        log.warning("Hermes native hook rejected or deferred (%s)", type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
