"""
File extraction for text attachments.

Shell layer: handles file I/O.
Supports: PDF, Markdown, plain text, code files.
"""

from pathlib import Path
from typing import Literal

from pypdf import PdfReader

from ..core.schemas import FileAttachment


# File extension to content type mapping
EXTENSION_MAP: dict[str, Literal["pdf", "markdown", "text", "code"]] = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".java": "code",
    ".c": "code",
    ".cpp": "code",
    ".h": "code",
    ".go": "code",
    ".rs": "code",
    ".rb": "code",
    ".php": "code",
    ".swift": "code",
    ".kt": "code",
    ".scala": "code",
    ".r": "code",
    ".sql": "code",
    ".sh": "code",
    ".bash": "code",
    ".zsh": "code",
    ".yaml": "code",
    ".yml": "code",
    ".json": "code",
    ".xml": "code",
    ".html": "code",
    ".css": "code",
    ".toml": "code",
    ".ini": "code",
    ".cfg": "code",
    ".conf": "code",
}


def get_content_type(filename: str) -> Literal["pdf", "markdown", "text", "code"]:
    """Determine content type from filename."""
    suffix = Path(filename).suffix.lower()
    return EXTENSION_MAP.get(suffix, "text")


def extract_pdf_text(file_path: Path) -> str:
    """Extract text from PDF file."""
    reader = PdfReader(file_path)
    text_parts = []

    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(f"[Page {page_num}]\n{page_text}")

    return "\n\n".join(text_parts)


def extract_text_file(file_path: Path) -> str:
    """Extract text from text-based file."""
    encodings = ["utf-8", "latin-1", "cp1252"]

    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    return file_path.read_text(encoding="utf-8", errors="replace")


def extract_file_content(
    file_path: Path,
    max_chars: int = 50000,
) -> FileAttachment:
    """
    Extract content from a file.

    Returns FileAttachment with extracted text (truncated if needed).
    """
    filename = file_path.name
    content_type = get_content_type(filename)

    if content_type == "pdf":
        text = extract_pdf_text(file_path)
    else:
        text = extract_text_file(file_path)

    char_count = len(text)

    if char_count > max_chars:
        text = text[:max_chars] + f"\n\n[Truncated: {char_count - max_chars} chars omitted]"

    return FileAttachment(
        filename=filename,
        content_type=content_type,
        extracted_text=text,
        char_count=char_count,
    )


def extract_from_bytes(
    filename: str,
    content: bytes,
    max_chars: int = 50000,
) -> FileAttachment:
    """
    Extract content from file bytes (for Gradio uploads).

    Writes to temp file if needed for PDF processing.
    """
    import tempfile

    content_type = get_content_type(filename)

    if content_type == "pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            text = extract_pdf_text(Path(tmp.name))
            Path(tmp.name).unlink()  # Clean up
    else:
        encodings = ["utf-8", "latin-1", "cp1252"]
        text = None
        for encoding in encodings:
            try:
                text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = content.decode("utf-8", errors="replace")

    char_count = len(text)

    if char_count > max_chars:
        text = text[:max_chars] + f"\n\n[Truncated: {char_count - max_chars} chars omitted]"

    return FileAttachment(
        filename=filename,
        content_type=content_type,
        extracted_text=text,
        char_count=char_count,
    )


def validate_file(
    filename: str,
    size_bytes: int,
    max_size_mb: int = 10,
) -> tuple[bool, str | None]:
    """
    Validate file before processing.

    Returns (is_valid, error_message).
    """
    content_type = get_content_type(filename)

    if content_type not in ("pdf", "markdown", "text", "code"):
        return False, f"Unsupported file type: {Path(filename).suffix}"

    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        return False, f"File too large: {size_bytes / 1024 / 1024:.1f}MB (max {max_size_mb}MB)"

    return True, None
