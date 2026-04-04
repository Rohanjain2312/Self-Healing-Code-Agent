"""
Tests for LessonStore cross-session memory (Fix 10).

The first test always runs — it verifies graceful degradation when chromadb
is not installed or the collection is unavailable. The roundtrip tests are
skipped automatically when chromadb is not present, since they need a real
persistent client to exercise semantic retrieval.
"""

import pytest

try:
    import chromadb  # noqa: F401
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False


def test_lesson_store_degrades_gracefully(tmp_path):
    """LessonStore never raises — all methods are safe no-ops when unavailable."""
    from agent.memory_store import LessonStore

    store = LessonStore(persist_dir=str(tmp_path / "lessons"))
    # These must never raise regardless of chromadb availability
    store.store_lesson("test lesson", task_category="test", failure_category="runtime_error")
    results = store.retrieve_relevant_lessons("test query")
    assert isinstance(results, list)
    assert isinstance(store.total_lessons, int)
    assert store.total_lessons >= 0


@pytest.mark.skipif(not _HAS_CHROMADB, reason="chromadb not installed")
def test_lesson_store_roundtrip(tmp_path):
    """Store a lesson then retrieve it by semantic similarity."""
    from agent.memory_store import LessonStore

    store = LessonStore(persist_dir=str(tmp_path / "lessons"))
    lesson = "Always handle None inputs before indexing a list to avoid TypeError"
    store.store_lesson(lesson, task_category="indexing", failure_category="runtime_error")

    assert store.total_lessons == 1

    results = store.retrieve_relevant_lessons("list index out of range with None value", n_results=3)
    assert len(results) >= 1
    assert results[0] == lesson


@pytest.mark.skipif(not _HAS_CHROMADB, reason="chromadb not installed")
def test_lesson_store_idempotent_upsert(tmp_path):
    """Storing the same lesson twice does not create duplicate entries."""
    from agent.memory_store import LessonStore

    store = LessonStore(persist_dir=str(tmp_path / "lessons"))
    lesson = "Check for empty list before accessing first element"
    store.store_lesson(lesson)
    store.store_lesson(lesson)  # duplicate upsert

    assert store.total_lessons == 1


@pytest.mark.skipif(not _HAS_CHROMADB, reason="chromadb not installed")
def test_lesson_store_multiple_lessons(tmp_path):
    """Store multiple distinct lessons and verify count and retrieval."""
    from agent.memory_store import LessonStore

    store = LessonStore(persist_dir=str(tmp_path / "lessons"))
    lessons = [
        "Sort the list before applying binary search",
        "Use defaultdict to avoid KeyError on missing keys",
        "Strip whitespace from strings before comparison",
    ]
    for lesson in lessons:
        store.store_lesson(lesson, task_category="algorithms")

    assert store.total_lessons == len(lessons)

    results = store.retrieve_relevant_lessons("sorted list search", n_results=5)
    assert len(results) >= 1
