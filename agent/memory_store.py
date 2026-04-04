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
import os
from typing import Any

logger = logging.getLogger(__name__)

# Semantic similarity threshold: lessons with cosine similarity above this
# value are treated as paraphrases of an existing lesson and skipped.
_SEMANTIC_DUP_THRESHOLD = 0.92


class LessonStore:
    """
    Persistent vector store for agent lessons using ChromaDB.

    Provides semantic retrieval of past lessons to inform future repairs.
    Gracefully degrades to a no-op when chromadb is not installed or unavailable.
    """

    def __init__(self, persist_dir: str = ".agent_memory") -> None:
        """
        Initialize and connect to the ChromaDB "lessons" collection.

        If HF_LESSONS_REPO and HF_TOKEN environment variables are set, existing
        lessons are downloaded from HuggingFace and pre-populated into the
        collection so the agent benefits from cross-session history on startup.

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

        # Pre-populate from HuggingFace if credentials are available
        if self._collection is not None:
            repo_id = os.environ.get("HF_LESSONS_REPO", "")
            token = os.environ.get("HF_TOKEN", "")
            if repo_id and token:
                self._prepopulate_from_hf(repo_id, token)

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

    def _prepopulate_from_hf(self, repo_id: str, token: str) -> None:
        """
        Download lessons from HuggingFace and upsert them into ChromaDB.

        Called once at init when HF_LESSONS_REPO and HF_TOKEN are set.
        Uses store_lesson() (idempotent upsert) so re-running never creates
        duplicates in the local collection.
        """
        try:
            from agent.hf_memory_sync import load_lessons_from_hf
            hf_lessons = load_lessons_from_hf(repo_id, token)
            for lesson_dict in hf_lessons:
                self.store_lesson(
                    lesson_dict["lesson"],
                    task_category=lesson_dict.get("task_id", ""),
                    failure_category=lesson_dict.get("failure_category", ""),
                    task_id=lesson_dict.get("task_id", ""),
                )
            if hf_lessons:
                logger.info(
                    "LessonStore: pre-populated %d lessons from HF repo '%s'.",
                    len(hf_lessons),
                    repo_id,
                )
        except Exception as exc:
            logger.warning("LessonStore: HF pre-population failed (non-fatal): %s", exc)

    def _is_semantic_duplicate(self, lesson: str) -> bool:
        """
        Return True if lesson is a paraphrase of any existing ChromaDB lesson.

        Uses ChromaDB's cosine distance (stored as 1 - cosine_similarity).
        A distance < (1 - threshold) means similarity > threshold.
        """
        if self._collection is None or self.total_lessons == 0:
            return False
        try:
            results = self._collection.query(
                query_texts=[lesson],
                n_results=1,
                include=["distances"],
            )
            distances = results.get("distances", [[]])[0]
            if distances:
                similarity = 1.0 - distances[0]
                return similarity > _SEMANTIC_DUP_THRESHOLD
        except Exception as exc:
            logger.warning("LessonStore._is_semantic_duplicate failed (non-fatal): %s", exc)
        return False

    async def sync_to_hf(
        self,
        new_lessons: list[dict],
        router: Any = None,
    ) -> None:
        """
        Push new lessons to HuggingFace with semantic + fingerprint deduplication.

        Deduplication layers:
          1. Semantic (here): drop lessons with cosine similarity > 0.92 to any
             existing lesson already in the ChromaDB collection.
          2. Fingerprint (in save_lessons_to_hf): drop lessons whose SHA-256
             hash matches an existing JSONL entry.

        No-ops silently if HF_LESSONS_REPO or HF_TOKEN are not set.

        Args:
            new_lessons: Lesson dicts with keys: lesson, task_id,
                         failure_category, timestamp, fingerprint.
            router: Optional LLMRouter forwarded to save_lessons_to_hf for
                    LLM-based compaction when the file exceeds 200 entries.
        """
        repo_id = os.environ.get("HF_LESSONS_REPO", "")
        token = os.environ.get("HF_TOKEN", "")
        if not repo_id or not token:
            return

        # Layer 1: semantic dedup against the local ChromaDB collection
        semantically_new = [
            ld for ld in new_lessons
            if not self._is_semantic_duplicate(ld["lesson"])
        ]

        if not semantically_new:
            logger.info("HF sync: all %d lessons are semantic duplicates — skipping.", len(new_lessons))
            return

        logger.info(
            "HF sync: %d/%d lessons passed semantic dedup.",
            len(semantically_new),
            len(new_lessons),
        )

        try:
            from agent.hf_memory_sync import save_lessons_to_hf
            await save_lessons_to_hf(semantically_new, repo_id, token, router=router)
        except Exception as exc:
            logger.warning("LessonStore.sync_to_hf failed (non-fatal): %s", exc)

    @property
    def total_lessons(self) -> int:
        """Return total number of lessons stored in the collection."""
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0
