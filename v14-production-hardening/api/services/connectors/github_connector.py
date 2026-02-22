"""GitHub Connector for StudyBuddy v11.

Uses the GitHub REST API to import README files, documentation,
and markdown from GitHub repositories.

Supports:
- Browsing repository files
- Selective file import
- Incremental sync (track changes, re-index only updates)
"""

import base64
import hashlib
import logging
from datetime import datetime
from pathlib import Path

import httpx
from sqlalchemy.orm import Session

from ...database.models import Document, ConnectorConfig, LearningProgram

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"

# File extensions we can import as learning materials
IMPORTABLE_EXTENSIONS = {".md", ".markdown", ".txt", ".rst", ".adoc"}


def _github_headers(token: str) -> dict:
    """Build headers for GitHub API requests."""
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def list_repo_files(
    owner: str,
    repo: str,
    token: str,
    path: str = "",
    branch: str = "main",
) -> list[dict]:
    """List importable files in a GitHub repository.

    Uses the Git Trees API for recursive listing (when browsing from root)
    or the Contents API for a specific subdirectory.

    Args:
        owner: Repository owner (e.g., "langchain-ai")
        repo: Repository name (e.g., "langchain")
        token: GitHub Personal Access Token
        path: Directory path to browse (empty for full recursive listing)
        branch: Branch name (default: "main")

    Returns:
        List of file dicts: [{"path": "docs/intro.md", "type": "file", "size": 0}]
    """
    if not path:
        # Use Git Trees API for a single recursive call
        return await _list_files_recursive(owner, repo, token, branch)

    # For specific subdirectory, use Contents API
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": branch}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            url, headers=_github_headers(token), params=params, timeout=30.0
        )

    if response.status_code == 404:
        raise ValueError(f"Repository or path not found: {owner}/{repo}/{path}")
    response.raise_for_status()

    data = response.json()

    if isinstance(data, list):
        files = [
            {"path": item["path"], "type": item["type"], "size": item.get("size", 0)}
            for item in data
        ]
    else:
        files = [
            {"path": data["path"], "type": data["type"], "size": data.get("size", 0)}
        ]

    importable = [
        f
        for f in files
        if f.get("type") == "file"
        and any(f["path"].lower().endswith(ext) for ext in IMPORTABLE_EXTENSIONS)
    ]

    return importable


async def _list_files_recursive(
    owner: str,
    repo: str,
    token: str,
    branch: str,
) -> list[dict]:
    """List all importable files recursively using the Git Trees API.

    Single API call returns the entire file tree.
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/git/trees/{branch}"
    params = {"recursive": "1"}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            url, headers=_github_headers(token), params=params, timeout=30.0
        )

    if response.status_code == 404:
        raise ValueError(f"Repository or branch not found: {owner}/{repo}@{branch}")
    response.raise_for_status()

    data = response.json()
    tree = data.get("tree", [])

    importable = [
        {"path": item["path"], "type": "file", "size": item.get("size", 0)}
        for item in tree
        if item.get("type") == "blob"
        and any(item["path"].lower().endswith(ext) for ext in IMPORTABLE_EXTENSIONS)
    ]

    return importable


async def _fetch_file_content(
    owner: str,
    repo: str,
    token: str,
    file_path: str,
    branch: str,
) -> str:
    """Fetch a single file's content from GitHub.

    Returns the decoded text content.
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{file_path}"
    params = {"ref": branch}

    async with httpx.AsyncClient() as client:
        response = await client.get(
            url, headers=_github_headers(token), params=params, timeout=30.0
        )

    response.raise_for_status()
    data = response.json()

    if data.get("encoding") == "base64" and "content" in data:
        return base64.b64decode(data["content"]).decode("utf-8")
    elif "content" in data:
        return data["content"]
    else:
        raise ValueError(f"Unexpected response format for {file_path}")


async def import_github_files(
    owner: str,
    repo: str,
    token: str,
    file_paths: list[str],
    branch: str,
    program: LearningProgram,
    connector: ConnectorConfig,
    db: Session,
) -> list[Document]:
    """Import selected files from a GitHub repository.

    For each file:
    1. Fetch content via GitHub REST API
    2. Check sync_state for changes (incremental sync)
    3. Create/update Document records
    4. Save content to disk for indexing

    Args:
        owner: Repository owner
        repo: Repository name
        token: GitHub Personal Access Token
        file_paths: List of file paths to import (e.g., ["README.md", "docs/intro.md"])
        branch: Branch to import from
        program: Target learning program
        connector: The GitHub connector config
        db: Database session

    Returns:
        List of new/updated Document records (status="pending", ready for indexing)
    """
    imported = []
    sync_state = dict(connector.sync_state or {})

    logger.info(
        f"Importing {len(file_paths)} files from {owner}/{repo} "
        f"into program {program.id}"
    )

    for file_path in file_paths:
        try:
            content = await _fetch_file_content(owner, repo, token, file_path, branch)

            if not content or len(content.strip()) < 10:
                logger.warning(f"Skipping {file_path}: insufficient content")
                continue

            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Incremental sync: skip if unchanged
            prev = sync_state.get(file_path)
            if prev and prev.get("content_hash") == content_hash:
                logger.info(f"Skipping {file_path}: unchanged")
                continue

            # If previously imported, delete old document
            if prev and prev.get("document_id"):
                _delete_old_document(prev["document_id"], program, db)

            # Save to disk
            filename = file_path.replace("/", "_")
            upload_dir = Path("uploads") / str(program.id)
            upload_dir.mkdir(parents=True, exist_ok=True)
            disk_path = upload_dir / filename

            source_url = (
                f"https://github.com/{owner}/{repo}"
                f"/blob/{branch}/{file_path}"
            )

            full_content = f"# Source: {source_url}\n\n{content}"
            with open(disk_path, "w", encoding="utf-8") as f:
                f.write(full_content)

            # Create Document record
            document = Document(
                program_id=program.id,
                filename=filename,
                file_type="text/markdown",
                file_size=len(full_content.encode()),
                file_path=str(disk_path),
                content_hash=content_hash,
                status="pending",
                source_type="github",
                source_url=source_url,
                connector_id=connector.id,
            )

            db.add(document)
            db.flush()  # Get the ID without committing
            imported.append(document)

            # Update sync state
            sync_state[file_path] = {
                "content_hash": content_hash,
                "document_id": document.id,
                "synced_at": datetime.utcnow().isoformat(),
            }

            logger.info(f"Imported {file_path} as document {document.id}")

        except Exception as e:
            logger.error(f"Failed to import {file_path}: {e}")
            continue

    # Update connector sync state
    connector.sync_state = sync_state
    connector.last_sync_at = datetime.utcnow()
    connector.status = "synced" if imported else "configured"
    db.commit()

    return imported


def _delete_old_document(
    document_id: str,
    program: LearningProgram,
    db: Session,
) -> None:
    """Delete an old document and its vectors during incremental sync."""
    import os
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    old_doc = db.query(Document).filter(Document.id == document_id).first()
    if not old_doc:
        return

    # Remove vectors from Qdrant
    try:
        client = QdrantClient(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333")
        )
        client.delete(
            collection_name=program.qdrant_collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
    except Exception as e:
        logger.warning(f"Failed to delete vectors for document {document_id}: {e}")

    # Remove file from disk
    if old_doc.file_path:
        try:
            os.unlink(old_doc.file_path)
        except OSError:
            pass

    # Remove database record
    db.delete(old_doc)
