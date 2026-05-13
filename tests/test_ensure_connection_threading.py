"""Test that _ensure_connection() is thread-safe (#307).

Without the _init_lock, concurrent threads calling add() on a fresh engine
race through _ensure_connection(), creating orphaned connections and
potential deadlocks.
"""
import threading

from truememory.engine import TrueMemoryEngine


def test_concurrent_add_no_data_loss(tmp_path):
    """5 threads calling add() on a fresh engine should all succeed."""
    db = str(tmp_path / "test_add.db")
    eng = TrueMemoryEngine(db_path=db)

    errors = []
    results = []

    def writer(tid):
        try:
            r = eng.add(f"Thread {tid} message", sender=f"t{tid}")
            results.append(r)
        except Exception as e:
            errors.append((tid, e))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Threads failed: {errors}"
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"


def test_concurrent_search_on_fresh_engine(tmp_path):
    """Multiple threads searching before any add() should not crash."""
    db = str(tmp_path / "test_search.db")
    eng = TrueMemoryEngine(db_path=db)

    errors = []

    def searcher(tid):
        try:
            eng.search(f"query {tid}")
        except Exception as e:
            errors.append((tid, e))

    threads = [threading.Thread(target=searcher, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Search threads failed: {errors}"
