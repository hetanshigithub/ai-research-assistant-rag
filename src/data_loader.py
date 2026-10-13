from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


@dataclass(frozen=True)
class Document:
    """Raw document loaded from disk."""

    source: str
    text: str


def list_document_paths(directory: Path) -> list[Path]:
    """Supported files in a directory (or the file itself), sorted by path."""
    if directory.is_file():
        return [directory]
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_documents(data_dir: Path) -> list[Document]:
    """Load all supported documents from a directory recursively."""
    return load_corpus([data_dir])


def load_corpus(directories: Sequence[Path]) -> list[Document]:
    """Load documents from several directories.

    The first directory is required. Later ones (for example a folder of uploads)
    are optional and skipped when they do not exist yet.
    """
    primary = Path(directories[0])
    if not primary.exists():
        raise FileNotFoundError(f"Data directory does not exist: {primary}")

    paths: list[Path] = []
    for directory in directories:
        directory = Path(directory)
        if directory.exists():
            paths.extend(list_document_paths(directory))

    documents: list[Document] = []
    for path in paths:
        text = load_document(path)
        if text.strip():
            documents.append(Document(source=str(path), text=normalize_text(text)))

    if not documents:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"No supported documents found. Expected: {supported}")

    return documents


def load_document(path: Path) -> str:
    """Load one TXT or PDF file."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return load_pdf(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_pdf(path: Path) -> str:
    """Extract text from a PDF with pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF support requires pypdf. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    paragraphs: list[str] = []
    buffer: list[str] = []

    for line in lines:
        if line:
            buffer.append(line)
        elif buffer:
            paragraphs.append(" ".join(buffer))
            buffer = []

    if buffer:
        paragraphs.append(" ".join(buffer))

    return "\n\n".join(paragraphs)
