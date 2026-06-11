"""Core Memory class tests using in-memory database."""
import pytest
from truememory import Memory


@pytest.fixture
def mem():
    """Fresh in-memory Memory instance for each test."""
    m = Memory(":memory:")
    yield m
    m.close()


def test_add_returns_dict(mem):
    result = mem.add("Prefers dark mode", user_id="alice")
    assert isinstance(result, dict)
    assert "id" in result
    assert result["content"] == "Prefers dark mode"


def test_add_and_search(mem):
    mem.add("Alice likes coffee in the morning", user_id="alice")
    results = mem.search("coffee", user_id="alice")
    assert len(results) >= 1
    assert any("coffee" in r["content"].lower() for r in results)


def test_search_empty_db(mem):
    results = mem.search("anything", user_id="alice")
    assert results == []


def test_get_by_id(mem):
    added = mem.add("Test memory", user_id="alice")
    fetched = mem.get(added["id"])
    assert fetched is not None
    assert fetched["content"] == "Test memory"


def test_metadata_round_trips_through_crud_and_search(mem):
    metadata = {"session_id": "sess-123", "source": "unit-test", "nested": {"n": 1}}
    added = mem.add("Alice prefers metadata-aware memories", user_id="alice", metadata=metadata)

    assert added["metadata"] == metadata

    fetched = mem.get(added["id"])
    assert fetched is not None
    assert fetched["metadata"] == metadata

    all_mems = mem.get_all(user_id="alice")
    assert all_mems[0]["metadata"] == metadata

    results = mem.search("aware", user_id="alice")
    assert results
    assert results[0]["metadata"] == metadata


def test_metadata_defaults_to_empty_dict(mem):
    added = mem.add("No metadata here", user_id="alice")
    assert added["metadata"] == {}
    fetched = mem.get(added["id"])
    assert fetched["metadata"] == {}


def test_metadata_must_be_dict(mem):
    with pytest.raises(TypeError, match="metadata must be a dict or None"):
        mem.add("Bad metadata", metadata=["not", "a", "dict"])


def test_metadata_must_be_json_serializable(mem):
    with pytest.raises(TypeError, match="metadata must be JSON-serializable"):
        mem.add("Bad metadata", metadata={"bad": object()})


def test_delete(mem):
    added = mem.add("To be deleted", user_id="alice")
    mem.delete(added["id"])
    fetched = mem.get(added["id"])
    assert fetched is None


def test_update(mem):
    added = mem.add("Original content", user_id="alice")
    mem.update(added["id"], "Updated content")
    fetched = mem.get(added["id"])
    assert fetched["content"] == "Updated content"


def test_get_all(mem):
    mem.add("Memory one", user_id="alice")
    mem.add("Memory two", user_id="alice")
    all_mems = mem.get_all(user_id="alice")
    assert len(all_mems) >= 2


def test_user_id_filtering(mem):
    mem.add("Alice memory", user_id="alice")
    mem.add("Bob memory", user_id="bob")
    alice_results = mem.search("memory", user_id="alice")
    # Should not return Bob's memory
    for r in alice_results:
        assert r.get("user_id") != "bob" or "bob" not in r.get("content", "").lower()


def test_context_manager():
    with Memory(":memory:") as m:
        m.add("Context manager test", user_id="alice")
        results = m.search("context", user_id="alice")
        assert len(results) >= 1
