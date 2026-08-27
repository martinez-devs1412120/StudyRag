"""Text chunking utilities."""
from typing import Generator


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150
) -> Generator[str, None, None]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= chunk_size:
        yield text
        return

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind(". ")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)
            if break_point > chunk_size * 0.5:
                end = start + break_point + 1
                chunk = text[start:end]

        yield chunk.strip()
        start = end - overlap


def chunk_with_metadata(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 150
) -> Generator[dict, None, None]:
    """Yield chunks with metadata."""
    for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
        yield {
            "text": chunk,
            "source": source,
            "chunk_id": i,
        }