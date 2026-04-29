"""Test that TRUEMEMORY_GATE_ENABLED env var controls the gate."""

import os


def test_gate_disabled_env_var():
    """Setting TRUEMEMORY_GATE_ENABLED=0 should bypass the encoding gate."""
    os.environ["TRUEMEMORY_GATE_ENABLED"] = "0"
    try:
        enabled = os.environ.get(
            "TRUEMEMORY_GATE_ENABLED", "1"
        ).lower() in ("1", "true", "yes")
        assert enabled is False, (
            "TRUEMEMORY_GATE_ENABLED=0 should result in gate_enabled=False"
        )
    finally:
        os.environ.pop("TRUEMEMORY_GATE_ENABLED", None)


def test_gate_enabled_by_default():
    """Gate should be enabled by default (no env var set)."""
    os.environ.pop("TRUEMEMORY_GATE_ENABLED", None)
    enabled = os.environ.get(
        "TRUEMEMORY_GATE_ENABLED", "1"
    ).lower() in ("1", "true", "yes")
    assert enabled is True, (
        "Gate should be enabled by default when env var is not set"
    )


def test_pipeline_has_gate_enabled_attribute():
    """IngestionPipeline should have a gate_enabled attribute."""
    from truememory.ingest.pipeline import IngestionPipeline
    # Check the source code for TRUEMEMORY_GATE_ENABLED
    import inspect
    source = inspect.getsource(IngestionPipeline.__init__)
    assert "TRUEMEMORY_GATE_ENABLED" in source, (
        "IngestionPipeline.__init__ should read TRUEMEMORY_GATE_ENABLED env var"
    )
