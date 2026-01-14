"""Background flashcard generation service.

Pre-generates flashcards for topics in a chapter when the user selects it.
Cards are stored in SQLite via the existing cache system, so subsequent
requests get instant cache hits.
"""

import threading
import logging
from datetime import datetime
from typing import Callable

from langchain_openai import ChatOpenAI

from ..database.connection import get_db
from ..agents.card_generator import generate_cards_batch
from .flashcard_cache import cache_flashcards, get_cached_flashcards

logger = logging.getLogger(__name__)

# Module-level status tracking (per chapter)
prefetch_status: dict[int, dict] = {}

# Concurrency control - max 3 concurrent LLM generations
_generation_semaphore = threading.Semaphore(3)

# Track which chapters are currently being generated
_active_chapters: set[int] = set()
_active_chapters_lock = threading.Lock()


class BackgroundGenerator:
    """Orchestrates background flashcard generation for chapters."""

    def __init__(
        self,
        card_generator_llm: ChatOpenAI,
        search_func: Callable[[str, int], str],
        get_topics_func: Callable[[int, str], list[dict]],
    ):
        """
        Initialize the background generator.

        Args:
            card_generator_llm: LLM for generating flashcards
            search_func: Function to search materials (query, k) -> context
            get_topics_func: Function to get topics (chapter_id, scope) -> topics
        """
        self.card_generator_llm = card_generator_llm
        self.search_func = search_func
        self.get_topics_func = get_topics_func

    def start_prefetch(self, chapter_id: int) -> dict:
        """
        Start background generation for a chapter.

        Returns immediately with status. Generation runs in a daemon thread.

        Args:
            chapter_id: The chapter to generate cards for

        Returns:
            Dict with state ("started", "already_in_progress", "already_completed"),
            chapter_id, total_topics, and message
        """
        with _active_chapters_lock:
            # Check if already in progress
            if chapter_id in _active_chapters:
                status = prefetch_status.get(chapter_id, {})
                return {
                    "chapter_id": chapter_id,
                    "state": "already_in_progress",
                    "total_topics": status.get("total_topics", 0),
                    "message": f"Generation already in progress for chapter {chapter_id}",
                }

            # Check if already completed
            status = prefetch_status.get(chapter_id)
            if status and status.get("state") == "completed":
                return {
                    "chapter_id": chapter_id,
                    "state": "already_completed",
                    "total_topics": status.get("total_topics", 0),
                    "message": f"Chapter {chapter_id} already prefetched ({status.get('cards_generated', 0)} cards)",
                }

            # Get topics for this chapter
            topics = self.get_topics_func(chapter_id, "single")
            if not topics:
                return {
                    "chapter_id": chapter_id,
                    "state": "no_topics",
                    "total_topics": 0,
                    "message": f"No topics found for chapter {chapter_id}",
                }

            # Mark as active
            _active_chapters.add(chapter_id)

        # Initialize status
        prefetch_status[chapter_id] = {
            "state": "in_progress",
            "total_topics": len(topics),
            "completed_topics": 0,
            "current_topic": "",
            "cards_generated": 0,
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "error": None,
        }

        # Start background thread
        thread = threading.Thread(
            target=self._generate_for_chapter,
            args=(chapter_id, topics),
            daemon=True,
        )
        thread.start()

        logger.info(f"Started prefetch for chapter {chapter_id} ({len(topics)} topics)")

        return {
            "chapter_id": chapter_id,
            "state": "started",
            "total_topics": len(topics),
            "message": f"Started generating cards for {len(topics)} topics",
        }

    def get_status(self, chapter_id: int) -> dict | None:
        """
        Get current prefetch status for a chapter.

        Args:
            chapter_id: The chapter to check

        Returns:
            Status dict or None if no status exists
        """
        return prefetch_status.get(chapter_id)

    def _generate_for_chapter(self, chapter_id: int, topics: list[dict]):
        """
        Worker function that generates cards for all topics in a chapter.

        Runs in a background thread.
        """
        try:
            for topic_info in topics:
                section_name = topic_info.get("section_name", "Unknown")
                subtopics = topic_info.get("subtopics", [])

                # Update current topic
                prefetch_status[chapter_id]["current_topic"] = section_name

                # Generate cards for this topic
                cards_count = self._generate_for_topic(
                    topic_name=section_name,
                    subtopics=subtopics,
                    chapter_id=chapter_id,
                )

                # Update progress
                prefetch_status[chapter_id]["completed_topics"] += 1
                prefetch_status[chapter_id]["cards_generated"] += cards_count

            # Mark as completed
            prefetch_status[chapter_id]["state"] = "completed"
            prefetch_status[chapter_id]["completed_at"] = datetime.utcnow().isoformat()
            prefetch_status[chapter_id]["current_topic"] = ""

            logger.info(
                f"Completed prefetch for chapter {chapter_id}: "
                f"{prefetch_status[chapter_id]['cards_generated']} cards"
            )

        except Exception as e:
            logger.error(f"Error during prefetch for chapter {chapter_id}: {e}")
            prefetch_status[chapter_id]["state"] = "error"
            prefetch_status[chapter_id]["error"] = str(e)

        finally:
            with _active_chapters_lock:
                _active_chapters.discard(chapter_id)

    def _generate_for_topic(
        self,
        topic_name: str,
        subtopics: list[str],
        chapter_id: int,
    ) -> int:
        """
        Generate cards for a single topic with semaphore-controlled concurrency.

        Args:
            topic_name: The topic/section name
            subtopics: List of subtopics
            chapter_id: The chapter ID

        Returns:
            Number of cards generated and cached
        """
        # Acquire semaphore (blocks if 3 generations in progress)
        with _generation_semaphore:
            try:
                # Search for context
                context = self.search_func(topic_name, 4)

                # Check if already cached
                with get_db() as db:
                    cached = get_cached_flashcards(topic_name, context, db)
                    if cached:
                        return len(cached)

                # Generate 5-7 cards per topic (default count=6)
                cards = generate_cards_batch(
                    self.card_generator_llm,
                    topic=topic_name,
                    context=context,
                    count=6,
                )

                if not cards:
                    logger.warning(f"No cards generated for topic '{topic_name}'")
                    return 0

                # Skip quality check for background prefetch (speed over perfection)
                # Store in database via cache
                with get_db() as db:
                    cached_cards = cache_flashcards(
                        topic=topic_name,
                        source_context=context,
                        cards_data=cards,
                        db=db,
                        chapter_id=chapter_id,
                    )

                logger.info(f"Generated {len(cached_cards)} cards for topic '{topic_name}'")
                return len(cached_cards)

            except Exception as e:
                logger.error(f"Error generating cards for topic '{topic_name}': {e}")
                return 0
