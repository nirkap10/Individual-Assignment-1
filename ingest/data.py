"""CSV loading and metadata parsing.

article_id == 0-based row index in the CSV. All 7,682 rows are kept
(no deduplication); duplicate content is handled at retrieval time instead.
"""
from __future__ import annotations

import ast

import pandas as pd

from ingest import config


def _parse_list(value: str) -> list[str]:
    """tags and authors are stringified Python lists, e.g. "['A', 'B']"."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def load_df() -> pd.DataFrame:
    df = pd.read_csv(config.CSV_PATH, encoding=config.CSV_ENCODING)
    df["article_id"] = df.index
    return df


def article_record(row: pd.Series) -> dict:
    """Normalized view of one article for chunking + Pinecone metadata."""
    return {
        "article_id": int(row["article_id"]),
        "title": str(row["title"]),
        "text": str(row["text"]),
        "url": str(row["url"]),
        "authors": _parse_list(row["authors"]),
        "timestamp": str(row["timestamp"]),
        "tags": _parse_list(row["tags"]),
    }
