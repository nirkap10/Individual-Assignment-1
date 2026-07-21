"""Deterministic token-window chunking.

IMPORTANT: this exact logic is used by BOTH the Phase 2 sweep and the Phase 3
full ingestion, so the sweep results transfer to production. Do not change the
algorithm without re-running the sweep.

chunk_size is measured in tokens (tiktoken cl100k_base, a good proxy for
text-embedding-3-small). Stride = chunk_size * (1 - overlap_ratio).
"""
from dataclasses import dataclass

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    chunk_index: int
    text: str
    n_tokens: int


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def chunk_text(text: str, chunk_size: int, overlap_ratio: float) -> list[Chunk]:
    """Split text into overlapping token windows.

    A chunk_size of 1024 with overlap 0.2 yields a stride of ~819 tokens.
    The final window is whatever remains (may be shorter than chunk_size).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0, 1)")

    tokens = _ENC.encode(text, disallowed_special=())
    n = len(tokens)
    if n == 0:
        return []
    if n <= chunk_size:
        return [Chunk(0, text, n)]

    stride = max(1, int(chunk_size * (1.0 - overlap_ratio)))
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < n:
        end = min(start + chunk_size, n)
        piece = _ENC.decode(tokens[start:end])
        chunks.append(Chunk(idx, piece, end - start))
        idx += 1
        if end == n:
            break
        start += stride
    return chunks
