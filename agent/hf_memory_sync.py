"""
HuggingFace Datasets backend for cross-session lesson persistence.

Provides upload/download of lessons.jsonl to/from a HuggingFace dataset repo
with two-layer deduplication and automatic size-cap compaction:

  Layer 1 — Fingerprint dedup (in save_lessons_to_hf):
    SHA-256 hash of the normalized lesson text (lowercase, stripped, no punctuation).
    Catches exact and near-exact duplicates regardless of formatting.

  Layer 2 — Semantic dedup (in LessonStore.sync_to_hf):
    Cosine similarity check against the ChromaDB collection. Lessons with
    similarity > 0.92 to any existing lesson are dropped before calling
    save_lessons_to_hf. This catches paraphrases like "always handle empty
    inputs" vs "check for empty inputs before processing".

  Compaction:
    When the JSONL file would exceed 200 lessons, group by failure_category,
    keep the 3 most recent per category, and merge the overflow via the LLM
    (memory_summarizer role). Compacted file stays permanently under ~50 KB.

Design:
  - All HF operations are wrapped in try/except — failures never crash the agent.
  - Requires huggingface_hub; degrades silently if not installed.
  - save_lessons_to_hf is async only because compaction may call the LLM.
"""

import hashlib
import json
import logging
import os
import string
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_LESSONS_FILENAME = "lessons.jsonl"
_MAX_LESSONS = 200
_KEEP_PER_CATEGORY = 3


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def compute_fingerprint(lesson: str) -> str:
    """
    Return a SHA-256 hex digest of the normalized lesson text.

    Normalization pipeline: lowercase → strip → remove punctuation → collapse
    whitespace. Identical content with different capitalization or punctuation
    maps to the same fingerprint.
    """
    normalized = lesson.lower().strip()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def load_lessons_from_hf(repo_id: str, token: str) -> list[dict]:
    """
    Download lessons.jsonl from a HuggingFace dataset repo.

    Each line in the file is a JSON object:
      {"lesson": "...", "task_id": "...", "failure_category": "...",
       "timestamp": "...", "fingerprint": "..."}

    Args:
        repo_id: HuggingFace dataset repo ID (e.g. "user/self-healing-lessons").
        token: HuggingFace API token with at least read access.

    Returns:
        List of lesson dicts. Empty list if the file doesn't exist yet or on
        any error (graceful degradation).
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    except ImportError:
        logger.warning("huggingface_hub not installed — HF memory sync disabled.")
        return []

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=_LESSONS_FILENAME,
            repo_type="dataset",
            token=token,
        )
        lessons: list[dict] = []
        with open(local_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lessons.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        logger.info("HF sync: loaded %d lessons from '%s'.", len(lessons), repo_id)
        return lessons

    except (EntryNotFoundError, RepositoryNotFoundError):
        logger.info(
            "HF sync: '%s' has no %s yet — will create on first push.",
            repo_id,
            _LESSONS_FILENAME,
        )
        return []
    except Exception as exc:
        logger.warning("HF sync: load failed (non-fatal): %s", exc)
        return []


async def save_lessons_to_hf(
    new_lessons: list[dict],
    repo_id: str,
    token: str,
    router: Any = None,
) -> None:
    """
    Append new lessons to the HuggingFace dataset repo with deduplication.

    Pipeline:
      1. Load existing lessons from HF.
      2. Fingerprint-dedup: drop any new lesson whose SHA-256 hash matches an
         existing one (exact / near-exact duplicate).
      3. Append the remaining genuinely new lessons.
      4. If combined total > 200, compact: keep _KEEP_PER_CATEGORY most recent
         per failure_category; merge overflow via LLM if router provided,
         otherwise discard overflow.
      5. Serialize to JSONL and upload via HfApi.upload_file().

    Args:
        new_lessons: Lesson dicts produced by the agent run. Expected keys:
                     lesson, task_id, failure_category, timestamp, fingerprint.
        repo_id: HuggingFace dataset repo ID.
        token: HuggingFace API token with write access.
        router: Optional LLMRouter for LLM-based compaction; compaction still
                runs without it (overflow is discarded rather than merged).
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed — HF sync skipped.")
        return

    # ── Step 1: fetch existing ────────────────────────────────────────────────
    existing = load_lessons_from_hf(repo_id, token)
    existing_fps = {ld.get("fingerprint", "") for ld in existing}

    # ── Step 2: fingerprint dedup ─────────────────────────────────────────────
    to_add = []
    for lesson_dict in new_lessons:
        fp = lesson_dict.get("fingerprint") or compute_fingerprint(lesson_dict["lesson"])
        if fp not in existing_fps:
            lesson_dict["fingerprint"] = fp  # ensure field is populated
            to_add.append(lesson_dict)
            existing_fps.add(fp)  # prevent duplicates within the batch itself

    if not to_add:
        logger.info("HF sync: all %d lessons already present — skipping upload.", len(new_lessons))
        return

    combined = existing + to_add
    logger.info(
        "HF sync: appending %d new lessons (total before compaction: %d).",
        len(to_add),
        len(combined),
    )

    # ── Step 3: compaction ────────────────────────────────────────────────────
    if len(combined) > _MAX_LESSONS:
        logger.info(
            "HF sync: %d lessons exceeds cap of %d — running compaction.",
            len(combined),
            _MAX_LESSONS,
        )
        combined = await _compact_lessons(combined, router=router)
        logger.info("HF sync: compacted to %d lessons.", len(combined))

    # ── Step 4: serialize and upload ──────────────────────────────────────────
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp:
            for lesson_dict in combined:
                tmp.write(json.dumps(lesson_dict, ensure_ascii=False) + "\n")
            tmp_path = tmp.name

        HfApi(token=token).upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=_LESSONS_FILENAME,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=(
                f"Update lessons.jsonl (+{len(to_add)} new, total={len(combined)})"
            ),
        )
        logger.info(
            "HF sync: uploaded %d lessons to '%s/%s'.",
            len(combined),
            repo_id,
            _LESSONS_FILENAME,
        )
    except Exception as exc:
        logger.warning("HF sync: upload failed (non-fatal): %s", exc)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Internal compaction helpers
# ---------------------------------------------------------------------------

async def _compact_lessons(
    lessons: list[dict],
    router: Any = None,
) -> list[dict]:
    """
    Reduce lessons to <= _MAX_LESSONS while preserving the most useful content.

    Strategy:
      1. Group by failure_category.
      2. Sort each group by timestamp descending (most recent first).
      3. Keep the _KEEP_PER_CATEGORY most recent per category unconditionally.
      4. Collect the overflow (everything beyond position 3 in each category).
      5. If a router is provided, call the memory_summarizer LLM to merge the
         overflow into consolidated lessons. Otherwise discard the overflow.
      6. Final hard-cap at _MAX_LESSONS (most-recent-first).
    """
    by_category: dict[str, list[dict]] = defaultdict(list)
    for lesson_dict in lessons:
        cat = lesson_dict.get("failure_category", "unknown")
        by_category[cat].append(lesson_dict)

    for cat in by_category:
        by_category[cat].sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    kept: list[dict] = []
    overflow: list[dict] = []

    for cat_lessons in by_category.values():
        kept.extend(cat_lessons[:_KEEP_PER_CATEGORY])
        overflow.extend(cat_lessons[_KEEP_PER_CATEGORY:])

    if overflow and router is not None:
        try:
            merged = await _llm_merge_lessons(overflow, router)
            kept.extend(merged)
            logger.info(
                "Compaction: LLM merged %d overflow lessons → %d.",
                len(overflow),
                len(merged),
            )
        except Exception as exc:
            logger.warning(
                "Compaction: LLM merge failed (non-fatal) — overflow discarded: %s", exc
            )

    # Hard cap: sort by recency and trim
    if len(kept) > _MAX_LESSONS:
        kept.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        kept = kept[:_MAX_LESSONS]

    return kept


async def _llm_merge_lessons(lessons: list[dict], router: Any) -> list[dict]:
    """
    Use the memory_summarizer role to consolidate a batch of overflow lessons.

    Sends lesson texts as a numbered list to the LLM and asks it to produce
    a smaller set of merged/consolidated lessons. Falls back to returning an
    empty list if the LLM output is malformed or unavailable.
    """
    lesson_text = "\n".join(
        f"{i + 1}. [{ld.get('failure_category', 'unknown')}] {ld['lesson']}"
        for i, ld in enumerate(lessons)
    )

    result, used_fallback = await router.call_with_fallback(
        role="memory_summarizer",
        template_key="summarize",
        variables={"learning_log": lesson_text},
    )

    if used_fallback:
        return []

    merged_texts: list[str] = result.get("lessons", [])
    timestamp = datetime.now(timezone.utc).isoformat()

    return [
        {
            "lesson": text.strip(),
            "task_id": "compaction",
            "failure_category": "merged",
            "timestamp": timestamp,
            "fingerprint": compute_fingerprint(text),
        }
        for text in merged_texts
        if isinstance(text, str) and text.strip()
    ]
