"""Curriculum generation service for StudyBuddy v10.

Generates learning curricula for any topic using AI.
Users can also upload their own topic lists in markdown format.
"""

import re
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


CURRICULUM_PROMPT = ChatPromptTemplate.from_template("""You are an expert curriculum designer. Create a comprehensive
learning curriculum for the topic: {topic}

Depth level: {depth} (beginner, intermediate, or advanced)

Structure your response as a markdown document with:
- Chapters using # Chapter N: Title format
- Topics using ## Topic Name format
- Subtopics using - Subtopic format

Include {chapter_count} chapters covering the subject systematically,
from foundational concepts to advanced applications.

Focus on practical, actionable learning objectives.
""")


CURRICULUM_FROM_DOCUMENTS_PROMPT = ChatPromptTemplate.from_template("""You are an expert curriculum designer. Based on the following document content, create a structured curriculum that covers ALL the key topics and concepts found in the material.

Program name: {program_name}
Program description: {program_description}

Document content (sampled from all uploaded documents):
{document_content}

Structure your response as a markdown document with:
- Chapters using # Chapter N: Title format
- Topics using ## Topic Name format
- Subtopics using - Subtopic format

IMPORTANT: You MUST include topics for ALL the document content shown above, even if some topics go beyond the program description. The documents represent what the student actually wants to learn. If the documents cover topics not mentioned in the program description, add chapters for those topics too.

Create a logical learning progression. Group related topics into chapters and identify subtopics for each main topic.

Include approximately {chapter_count} chapters, but add more if the documents cover a wider range of topics. Focus on the actual content present in the documents.
""")


async def generate_curriculum(
    topic: str,
    depth: str = "intermediate",
    chapter_count: int = 8,
) -> str:
    """Generate a curriculum for any topic.

    Returns markdown-formatted curriculum text.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    chain = CURRICULUM_PROMPT | llm

    result = await chain.ainvoke({
        "topic": topic,
        "depth": depth,
        "chapter_count": chapter_count,
    })

    return result.content


async def generate_curriculum_from_documents(
    program_name: str,
    program_description: str,
    document_content: str,
    chapter_count: int = 6,
) -> str:
    """Generate a curriculum based on uploaded document content.

    Returns markdown-formatted curriculum text.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    chain = CURRICULUM_FROM_DOCUMENTS_PROMPT | llm

    result = await chain.ainvoke({
        "program_name": program_name,
        "program_description": program_description or program_name,
        "document_content": document_content,
        "chapter_count": chapter_count,
    })

    return result.content


def parse_topic_list(markdown: str) -> dict:
    """Parse a markdown topic list into structured JSON.

    Expected format:
    # Chapter 1: Title
    ## Topic Name
    - Subtopic
    - Subtopic

    Returns:
    {
        "chapters": [
            {
                "number": 1,
                "title": "Title",
                "topics": [
                    {
                        "title": "Topic Name",
                        "subtopics": ["Subtopic", "Subtopic"]
                    }
                ]
            }
        ]
    }
    """
    chapters = []
    current_chapter = None
    current_topic = None

    lines = markdown.strip().split("\n")

    for line in lines:
        line = line.rstrip()

        # Skip empty lines
        if not line:
            continue

        # Chapter header: # Chapter N: Title
        chapter_match = re.match(r"^#\s+Chapter\s+(\d+):\s*(.+)$", line, re.IGNORECASE)
        if chapter_match:
            if current_chapter:
                if current_topic:
                    current_chapter["topics"].append(current_topic)
                    current_topic = None
                chapters.append(current_chapter)

            current_chapter = {
                "number": int(chapter_match.group(1)),
                "title": chapter_match.group(2).strip(),
                "topics": [],
            }
            continue

        # Topic header: ## Topic Name
        topic_match = re.match(r"^##\s+(.+)$", line)
        if topic_match and current_chapter:
            if current_topic:
                current_chapter["topics"].append(current_topic)

            current_topic = {
                "title": topic_match.group(1).strip(),
                "subtopics": [],
            }
            continue

        # Subtopic: - Subtopic
        subtopic_match = re.match(r"^[-*]\s+(.+)$", line)
        if subtopic_match and current_topic:
            current_topic["subtopics"].append(subtopic_match.group(1).strip())
            continue

    # Add final chapter/topic
    if current_chapter:
        if current_topic:
            current_chapter["topics"].append(current_topic)
        chapters.append(current_chapter)

    return {"chapters": chapters}


def topic_list_to_markdown(topic_list: dict) -> str:
    """Convert structured topic list back to markdown."""
    lines = []

    for chapter in topic_list.get("chapters", []):
        lines.append(f"# Chapter {chapter['number']}: {chapter['title']}")
        lines.append("")

        for topic in chapter.get("topics", []):
            lines.append(f"## {topic['title']}")

            for subtopic in topic.get("subtopics", []):
                lines.append(f"- {subtopic}")

            lines.append("")

    return "\n".join(lines)


def get_topics_for_chapter(topic_list: dict, chapter_number: int) -> list[str]:
    """Get all topic titles for a specific chapter."""
    for chapter in topic_list.get("chapters", []):
        if chapter["number"] == chapter_number:
            topics = []
            for topic in chapter.get("topics", []):
                topics.append(topic["title"])
                topics.extend(topic.get("subtopics", []))
            return topics
    return []


def get_all_topics(topic_list: dict) -> list[str]:
    """Get all topic and subtopic titles from a topic list."""
    topics = []
    for chapter in topic_list.get("chapters", []):
        for topic in chapter.get("topics", []):
            topics.append(topic["title"])
            topics.extend(topic.get("subtopics", []))
    return topics


def count_topics(topic_list: dict) -> int:
    """Count total number of topics and subtopics."""
    count = 0
    for chapter in topic_list.get("chapters", []):
        for topic in chapter.get("topics", []):
            count += 1  # The topic itself
            count += len(topic.get("subtopics", []))
    return count
