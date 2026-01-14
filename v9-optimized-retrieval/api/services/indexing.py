"""Semantic chunking service for StudyBuddy v9.

Provides intelligent document chunking that respects markdown structure
and uses embedding-based semantic boundaries for large sections.
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Lazy import for SemanticChunker (may not be available on all platforms)
_semantic_chunker_class = None


def _get_semantic_chunker_class():
    """Lazy load SemanticChunker to avoid import errors on Vercel."""
    global _semantic_chunker_class
    if _semantic_chunker_class is None:
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            _semantic_chunker_class = SemanticChunker
        except ImportError:
            _semantic_chunker_class = False  # Mark as unavailable
    return _semantic_chunker_class


def chunk_reference_document(
    content: str,
    source: str,
    embeddings: OpenAIEmbeddings | None = None,
    large_section_threshold: int = 1500,
    semantic_threshold: int = 90,
) -> list[Document]:
    """
    Chunk reference documents using a hybrid approach:
    1. Split by markdown headers first (preserves document structure)
    2. Apply semantic chunking within large sections (finds natural topic boundaries)

    Args:
        content: The markdown content to chunk
        source: Source filename for metadata
        embeddings: OpenAI embeddings model (created if not provided)
        large_section_threshold: Sections larger than this get semantic chunking
        semantic_threshold: Percentile threshold for semantic breaks (higher = fewer splits)

    Returns:
        List of Document objects with source metadata
    """
    if embeddings is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # First pass: split by markdown headers
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
    )

    header_chunks = header_splitter.split_text(content)

    # Check if semantic chunking is available
    SemanticChunker = _get_semantic_chunker_class()

    # Fallback splitter for when semantic chunking is unavailable
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, length_function=len
    )

    final_chunks = []
    for chunk in header_chunks:
        chunk_content = chunk.page_content

        # If chunk is large, apply semantic chunking (or fallback)
        if len(chunk_content) > large_section_threshold:
            try:
                if SemanticChunker:
                    semantic_chunker = SemanticChunker(
                        embeddings=embeddings,
                        breakpoint_threshold_type="percentile",
                        breakpoint_threshold_amount=semantic_threshold,
                    )
                    sub_chunks = semantic_chunker.create_documents([chunk_content])
                else:
                    # Fallback to recursive splitting
                    sub_chunks = fallback_splitter.create_documents([chunk_content])

                for sub in sub_chunks:
                    # Preserve header metadata from parent chunk
                    sub.metadata.update(chunk.metadata)
                    sub.metadata["source"] = source
                    final_chunks.append(sub)
            except Exception as e:
                # Fall back to keeping the chunk as-is if chunking fails
                print(f"Chunking failed for {source}: {e}")
                chunk.metadata["source"] = source
                final_chunks.append(chunk)
        else:
            chunk.metadata["source"] = source
            final_chunks.append(chunk)

    return final_chunks


def chunk_all_documents(
    documents: list[tuple[str, str]],
    embeddings: OpenAIEmbeddings | None = None,
) -> list[Document]:
    """
    Chunk multiple documents using semantic chunking.

    Args:
        documents: List of (content, source_filename) tuples
        embeddings: Shared embeddings model (created once if not provided)

    Returns:
        List of all Document chunks with source metadata
    """
    if embeddings is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    all_chunks = []
    for content, source in documents:
        chunks = chunk_reference_document(content, source, embeddings)
        all_chunks.extend(chunks)

    return all_chunks
