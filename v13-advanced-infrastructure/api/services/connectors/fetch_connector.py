"""Fetch/URL Connector for StudyBuddy v11.

Uses the MCP Fetch server (mcp-server-fetch) to import web page content
as learning materials. Content is fetched as markdown, saved as a document,
and indexed into the program's Qdrant collection via the existing pipeline.

MCP Server: mcp-server-fetch (Anthropic, pip/uvx)
Transport: stdio
"""

import hashlib
import logging
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from ...database.models import Document, ConnectorConfig, LearningProgram
from .mcp_client import get_fetch_server_config

logger = logging.getLogger(__name__)


async def fetch_url_content(url: str) -> str:
    """Fetch URL content using the Fetch MCP server.

    Spawns the mcp-server-fetch subprocess, calls the fetch tool,
    and returns the page content as markdown.

    Args:
        url: The URL to fetch

    Returns:
        Page content as markdown text
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {"fetch": get_fetch_server_config()}
    )
    tools = await client.get_tools()

    # Find the fetch tool
    fetch_tool = None
    for tool in tools:
        if "fetch" in tool.name.lower():
            fetch_tool = tool
            break

    if not fetch_tool:
        raise RuntimeError(
            f"Fetch tool not found. Available tools: {[t.name for t in tools]}"
        )

    # Call the fetch tool
    result = await fetch_tool.ainvoke({"url": url})

    # Extract text content from the result
    if isinstance(result, str):
        return result
    elif hasattr(result, "content"):
        return str(result.content)
    else:
        return str(result)


async def import_url(
    url: str,
    program: LearningProgram,
    connector: ConnectorConfig,
    db: Session,
) -> Document:
    """Import a URL as a document into a learning program.

    Flow:
    1. Fetch URL content via MCP Fetch server
    2. Check for duplicate content
    3. Save content to disk as .md file
    4. Create Document record with source_type="url"
    5. Return document (caller triggers index_document_background)

    Args:
        url: The URL to import
        program: Target learning program
        connector: The fetch connector config
        db: Database session

    Returns:
        The created Document record (status="pending", ready for indexing)

    Raises:
        ValueError: If content is too short or already imported
    """
    logger.info(f"Importing URL: {url} into program {program.id}")

    # Step 1: Fetch content via MCP
    content = await fetch_url_content(url)

    if not content or len(content.strip()) < 50:
        raise ValueError(
            f"Insufficient content extracted from {url} "
            f"({len(content.strip()) if content else 0} chars)"
        )

    # Step 2: Check for duplicates via content hash
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    existing = (
        db.query(Document)
        .filter(
            Document.program_id == program.id,
            Document.content_hash == content_hash,
        )
        .first()
    )

    if existing:
        raise ValueError(
            f"This content has already been imported as '{existing.filename}'"
        )

    # Step 3: Generate filename from URL and save to disk
    parsed = urlparse(url)
    slug = f"{parsed.netloc}{parsed.path}".replace("/", "_").strip("_")
    # Truncate long filenames and ensure .md extension
    filename = slug[:100] + ".md"

    upload_dir = Path("uploads") / str(program.id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename

    # Prepend source URL as a header
    full_content = f"# Source: {url}\n\n{content}"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(full_content)

    # Step 4: Create Document record
    document = Document(
        program_id=program.id,
        filename=filename,
        file_type="text/markdown",
        file_size=len(full_content.encode()),
        file_path=str(file_path),
        content_hash=content_hash,
        status="pending",
        source_type="url",
        source_url=url,
        connector_id=connector.id,
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    logger.info(
        f"Created document {document.id} from URL {url} "
        f"({document.file_size} bytes)"
    )

    return document
