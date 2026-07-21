"""Batched embedding via the course-provided OpenAI-compatible proxy.

Never embeds one chunk at a time. Callers pass a list of texts; this splits
into batches of config.EMBED_BATCH and returns vectors in input order.
"""
from openai import OpenAI

from ingest import config

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        env = config.require_env("ZYRANGG_API_KEY", "ZYRANGG_BASE_URL")
        _client = OpenAI(
            api_key=env["ZYRANGG_API_KEY"],
            base_url=env["ZYRANGG_BASE_URL"],
        )
    return _client


def embed_texts(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """Embed many texts in batches. Returns one vector per input text."""
    batch_size = batch_size or config.EMBED_BATCH
    out: list[list[float]] = []
    c = client()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = c.embeddings.create(model=config.EMBED_MODEL, input=batch)
        # resp.data preserves input order
        out.extend(d.embedding for d in resp.data)
    return out


def embed_one(text: str) -> list[float]:
    """Single query embedding (used by eval and the API, not for bulk ingest)."""
    return embed_texts([text], batch_size=1)[0]
