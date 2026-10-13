"""Strict config projection of an already committed database serving decision.

This is a cooperating-writer CAS, not atomicity across JSON and SQLite. Database
selection remains authoritative after any config write or acknowledgement fault.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from truememory.tier_switch.activation import ConfigGuard, LegacyTierPolicy, TierSelection

CONFIG_WRITE_LOCK = threading.Lock()
_MAX_CONFIG_BYTES = 1024 * 1024
_GENERATION_KEY = "tier_activation_generation"


class ConfigProjectionConflict(RuntimeError):
    """Config is unavailable or has changed since the durable intent."""


class ConfigFileLock:
    """Shared blocking lock; old writers retain optional best-effort behavior."""

    def __init__(self, path: Path, *, strict: bool = False) -> None:
        self.path = path
        self.strict = strict
        self._fd = None

    def __enter__(self) -> ConfigFileLock:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = open(self.path, "a+b")
            if os.name == "nt":
                import msvcrt
                self._fd.seek(0)
                msvcrt.locking(self._fd.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fd, fcntl.LOCK_EX)
        except OSError as error:
            if self._fd is not None:
                try:
                    self._fd.close()
                except OSError:
                    pass
                self._fd = None
            if self.strict:
                raise ConfigProjectionConflict("Config lock is unavailable") from error
        return self

    def __exit__(self, *exc: object) -> bool:
        if self._fd is None:
            return False
        try:
            if os.name == "nt":
                import msvcrt
                self._fd.seek(0)
                try:
                    msvcrt.locking(self._fd.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    if self.strict:
                        raise
            else:
                import fcntl
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            try:
                self._fd.close()
            except OSError:
                if self.strict:
                    raise
            finally:
                self._fd = None
        return False


def _paths(config_path: Path | None, lock_path: Path | None) -> tuple[Path, Path]:
    config = Path.home() / ".truememory" / "config.json" if config_path is None else Path(config_path)
    return config, config.with_name(config.name + ".lock") if lock_path is None else Path(lock_path)


def _unique(pairs: list[tuple[str, object]]) -> dict:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ConfigProjectionConflict("Duplicate config field")
    return value


def _read_config(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_CONFIG_BYTES + 1)
    except FileNotFoundError:
        return {}
    if len(raw) > _MAX_CONFIG_BYTES:
        raise ConfigProjectionConflict("Config exceeds the projection byte bound")
    try:
        value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite config")))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ConfigProjectionConflict("Config is not a bounded JSON object") from error
    if type(value) is not dict:
        raise ConfigProjectionConflict("Config is not a JSON object")
    return value


def _guard(config: dict) -> ConfigGuard:
    from truememory.tier_switch.activation import ConfigGuard, TierActivationError

    try:
        return ConfigGuard("tier" in config, config.get("tier", "edge"), config.get(_GENERATION_KEY))
    except TierActivationError as error:
        raise ConfigProjectionConflict("Config has an invalid tier or generation") from error


def _replace(src: str, dst: str) -> None:
    for attempt in range(10):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.03)


def _write_config(path: Path, config: dict) -> None:
    try:
        raw = (json.dumps(config, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    except (ValueError, TypeError, RecursionError) as error:
        raise ConfigProjectionConflict("Config fields are not serializable") from error
    if len(raw) > _MAX_CONFIG_BYTES:
        raise ConfigProjectionConflict("Config exceeds the projection byte bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".config.tmp.", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        _replace(temporary, str(path))
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def capture_config_guard(*, config_path: Path | None = None,
                         lock_path: Path | None = None) -> ConfigGuard:
    """Capture actual prior tier and seed a generation without changing policy."""
    config_path, lock_path = _paths(config_path, lock_path)
    with ConfigFileLock(lock_path, strict=True), CONFIG_WRITE_LOCK:
        config = _read_config(config_path)
        guard = _guard(config)
        if guard.generation is None:
            config[_GENERATION_KEY] = uuid.uuid4().hex
            _write_config(config_path, config)
            guard = _guard(config)
        return guard


def patch_config_fields(fields: dict, *, config_path: Path | None = None,
                        lock_path: Path | None = None) -> dict:
    """Merge independent settings under the same lock as activation projection."""
    if type(fields) is not dict or any(type(key) is not str for key in fields):
        raise ConfigProjectionConflict("Config patch must be an object with string keys")
    if {"tier", _GENERATION_KEY}.intersection(fields):
        raise ConfigProjectionConflict("Policy fields require a committed activation")
    config_path, lock_path = _paths(config_path, lock_path)
    with ConfigFileLock(lock_path, strict=True), CONFIG_WRITE_LOCK:
        config = _read_config(config_path)
        config.update(fields)
        _write_config(config_path, config)
        return config.copy()


def mirror_activation(result: TierSelection | LegacyTierPolicy, *,
                      expected_config: ConfigGuard | None,
                      config_path: Path | None = None, lock_path: Path | None = None) -> ConfigGuard:
    """CAS the tier/generation only; UNKNOWN old intents cannot authorize a write.

    An already-applied exact result is idempotent. Other settings come from the
    fresh locked file. This does not reconcile native runtime or acknowledge DB.
    """
    from truememory.tier_switch.activation import ConfigGuard, LegacyTierPolicy, TierSelection

    if type(result) not in (TierSelection, LegacyTierPolicy):
        raise ConfigProjectionConflict("A typed committed serving result is required")
    if expected_config is not None and type(expected_config) is not ConfigGuard:
        raise ConfigProjectionConflict("Invalid expected config guard")
    config_path, lock_path = _paths(config_path, lock_path)
    with ConfigFileLock(lock_path, strict=True), CONFIG_WRITE_LOCK:
        config = _read_config(config_path)
        actual = _guard(config)
        desired = ConfigGuard(True, result.target.tier, result.generation)
        if actual == desired:
            return actual
        if expected_config is None or actual != expected_config:
            raise ConfigProjectionConflict("Config changed since the activation intent")
        config["tier"] = result.target.tier
        config[_GENERATION_KEY] = result.generation
        _write_config(config_path, config)
        return desired
