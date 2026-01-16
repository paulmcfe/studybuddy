"""Document indexing service for StudyBuddy v10.

Handles document parsing, chunking, and indexing into program-specific
Qdrant collections.
"""

import os
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_openai import OpenAIEmbeddings

from api.database.models import Document, LearningProgram
from api.services.retrieval import (
    get_program_vector_store,
    ensure_collection_exists,
    EMBEDDING_MODEL,
)

# Lazy imports for document loaders
_pypdf_loader = None
_text_loader = None


def _get_pdf_loader():
    """Lazy load PDF loader."""
    global _pypdf_loader
    if _pypdf_loader is None:
        try:
            from langchain_community.document_loaders import PyPDFLoader
            _pypdf_loader = PyPDFLoader
        except ImportError:
            _pypdf_loader = False
    return _pypdf_loader if _pypdf_loader else None


def _get_text_loader():
    """Lazy load text loader."""
    global _text_loader
    if _text_loader is None:
        try:
            from langchain_community.document_loaders import TextLoader
            _text_loader = TextLoader
        except ImportError:
            _text_loader = False
    return _text_loader if _text_loader else None


async def load_document(file_path: str, file_type: str) -> list[LCDocument]:
    """Load a document from file path based on type."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    # PDF files
    if file_type == "application/pdf" or path.suffix.lower() == ".pdf":
        PyPDFLoader = _get_pdf_loader()
        if not PyPDFLoader:
            raise ImportError("pypdf not installed for PDF processing")
        loader = PyPDFLoader(str(path))
        return loader.load()

    # Markdown files
    if file_type == "text/markdown" or path.suffix.lower() in [".md", ".markdown"]:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return [LCDocument(page_content=content, metadata={"source": str(path)})]

    # Plain text files
    TextLoader = _get_text_loader()
    if TextLoader:
        loader = TextLoader(str(path))
        return loader.load()

    # Fallback: read as text
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return [LCDocument(page_content=content, metadata={"source": str(path)})]


def chunk_documents(
    documents: list[LCDocument],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[LCDocument]:
    """Chunk documents with markdown-aware splitting.

    Uses header-based splitting for markdown content,
    falls back to recursive character splitting for other content.
    """
    all_chunks = []

    for doc in documents:
        content = doc.page_content

        # Check if it's markdown with headers
        if any(line.startswith("#") for line in content.split("\n")[:20]):
            chunks = _chunk_markdown(doc, chunk_size, chunk_overlap)
        else:
            chunks = _chunk_text(doc, chunk_size, chunk_overlap)

        all_chunks.extend(chunks)

    return all_chunks


def _chunk_markdown(
    doc: LCDocument,
    chunk_size: int,
    chunk_overlap: int,
) -> list[LCDocument]:
    """Chunk markdown document by headers first, then by size."""
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    # Split by headers
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    header_chunks = header_splitter.split_text(doc.page_content)

    # Further split large sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    final_chunks = []
    for chunk in header_chunks:
        if len(chunk.page_content) > chunk_size:
            sub_chunks = text_splitter.split_documents([chunk])
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    # Preserve original metadata
    for chunk in final_chunks:
        chunk.metadata.update(doc.metadata)

    return final_chunks


def _chunk_text(
    doc: LCDocument,
    chunk_size: int,
    chunk_overlap: int,
) -> list[LCDocument]:
    """Chunk text document by size."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents([doc])

    # Preserve original metadata
    for chunk in chunks:
        chunk.metadata.update(doc.metadata)

    return chunks


async def index_document_to_program(
    document: Document,
    program: LearningProgram,
) -> int:
    """Index a document into a program's vector store.

    Returns the number of chunks indexed.
    """
    # Load document
    docs = await load_document(document.file_path, document.file_type)

    # Chunk
    chunks = chunk_documents(docs)

    # Add metadata to chunks
    for i, chunk in enumerate(chunks):
        chunk.metadata["document_id"] = document.id
        chunk.metadata["program_id"] = program.id
        chunk.metadata["filename"] = document.filename
        chunk.metadata["chunk_index"] = i

    # Ensure collection exists
    ensure_collection_exists(program.qdrant_collection)

    # Get vector store and add documents
    vector_store = get_program_vector_store(program)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Add chunks to vector store
    vector_store.add_documents(chunks)

    return len(chunks)


async def delete_document_from_program(
    document: Document,
    program: LearningProgram,
):
    """Remove a document's chunks from a program's vector store."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

    # Delete all points with this document_id
    client.delete(
        collection_name=program.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(value=document.id),
                )
            ]
        ),
    )
