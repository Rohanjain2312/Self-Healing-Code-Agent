"""
Cross-session learning via ChromaDB vector store (Fix 10).

LessonStore provides persistent, semantically-searchable storage for agent lessons
accumulated across multiple runs. When enabled via AgentConfig.enable_cross_session_memory:

  1. At iteration 0, generate_solution retrieves past lessons relevant to the current
     task and prepends them to the learning_log so the generator benefits from prior
     repair experience even on first attempt.
  2. After each successful (or max-iterations) run, run_agent() persists the final
     learning_log entries back to the store.

Design decisions:
  - All ChromaDB operations are wrapped in try/except — store failures never crash
    the agent (graceful degradation to in-memory-only behavior).
  - Uses PersistentClient so lessons survive process restarts.
  - Document IDs are SHA-256 hashes of the lesson text (truncated to 16 hex chars)
    enabling idempotent upserts without duplicates.
  - Without chromadb installed, all methods are silent no-ops.
"""

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LessonStore:
    """
    Persistent vector store for agent lessons using ChromaDB.

    Provides semantic retrieval of past lessons to inform future repairs.
    Gracefully degrades to a no-op when chromadb is not installed or unavailable.
    """

    def __init__(self, persist_dir: str = ".agent_memory") -> None:
        """
        Initialize and connect to the ChromaDB "lessons" collection.

        Args:
            persist_dir: Directory where ChromaDB stores its data on disk.
        """
        self._collection = None
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(
                "lessons",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "LessonStore initialized at '%s' — %d lessons in store",
                persist_dir,
                self.total_lessons,
            )
        except ImportError:
            logger.warning(
                "chromadb not installed — cross-session memory disabled. "
                "Install with: pip install chromadb"
            )
        except Exception as exc:
            logger.warning("LessonStore init failed (non-fatal): %s", exc)

    def store_lesson(
        self,
        lesson: str,
        task_category: str = "",
        failure_category: str = "",
        task_id: str = "",
    ) -> None:
        """
        Persist a single lesson string with metadata for future retrieval.

        Uses upsert so storing the same lesson twice is idempotent.

        Args:
            lesson: The lesson text (e.g. "Always handle None inputs before indexing").
            task_category: Optional category label for filtering (e.g. "text_processing").
            failure_category: The failure type that produced this lesson.
            task_id: Short identifier for the source task.
        """
        if self._collection is None or not lesson.strip():
            return
        try:
            doc_id = hashlib.sha256(lesson.encode()).hexdigest()[:16]
            self._collection.upsert(
                ids=[doc_id],
                documents=[lesson],
                metadatas=[{
                    "task_category": task_category,
                    "failure_category": failure_category,
                    "task_id": task_id[:100],
                }],
            )
            logger.debug("LessonStore: stored lesson id=%s", doc_id)
        except Exception as exc:
            logger.warning("LessonStore.store_lesson failed (non-fatal): %s", exc)

    def retrieve_relevant_lessons(
        self,
        query: str,
        task_category: str = "",
        n_results: int = 5,
    ) -> list[str]:
        """
        Semantically retrieve the most relevant past lessons for a query.

        Args:
            query: The task description or failure message to search against.
            task_category: Optional filter — only return lessons from this category.
            n_results: Maximum number of lessons to return.

        Returns:
            List of lesson strings ranked by relevance. Empty on any failure.
        """
        if self._collection is None or not query.strip():
            return []
        try:
            count = self.total_lessons
            if count == 0:
                return []
            actual_n = min(n_results, count)
            kwargs: dict[str, Any] = {
                "query_texts": [query],
                "n_results": actual_n,
            }
            if task_category:
                kwargs["where"] = {"task_category": task_category}
            results = self._collection.query(**kwargs)
            return results.get("documents", [[]])[0]
        except Exception as exc:
            logger.warning("LessonStore.retrieve failed (non-fatal): %s", exc)
            return []

    @property
    def total_lessons(self) -> int:
        """Return total number of lessons stored in the collection."""
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0
