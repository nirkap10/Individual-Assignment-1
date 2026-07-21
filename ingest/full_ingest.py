"""Phase 3 spend step: chunk + embed + upsert the FULL corpus into one namespace.

Uses the winning sweep config (config.FINAL_CHUNK_SIZE / FINAL_OVERLAP) and
writes to config.MAIN_NAMESPACE. This is the only Phase 3 script that costs
money, and the corpus is embedded exactly once.

Safety:
  - Dry run by default: chunks locally, prints chunk count, token total and
    estimated cost, then exits without calling the API. Spends only with --yes.
  - Resumable per batch. Phase 2 resumed at whole-namespace granularity, so a
    DNS drop during the 1024 upsert forced a full re-embed of that config. Here
    a checkpoint is written after each batch is embedded AND upserted, so a
    crash resumes from the last completed batch rather than from zero.
  - Embed and upsert calls retry with exponential backoff, since the Phase 2
    failure was a transient `getaddrinfo failed`, not a code bug.

Run (dry run) : python -m ingest.full_ingest
Run (spend)   : python -m ingest.full_ingest --yes
Reset resume  : python -m ingest.full_ingest --reset
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from ingest import config
from ingest import data
from ingest.chunker import chunk_text

MAX_RETRIES = 5
BACKOFF_BASE = 2.0  # seconds; doubles each attempt


def build_chunks(chunk_size: int, overlap_ratio: float):
    """Return (vector_id, article_record, Chunk) for every article in the CSV.

    Deterministic: same CSV and params always yield the same list in the same
    order. The resume checkpoint depends on that stability.
    """
    df = data.load_df()
    items = []
    for _, row in df.iterrows():
        rec = data.article_record(row)
        for ch in chunk_text(rec["text"], chunk_size, overlap_ratio):
            items.append((f"{rec['article_id']}#{ch.chunk_index}", rec, ch))
    return items


def preflight(items) -> float:
    toks = sum(ch.n_tokens for _, _, ch in items)
    cost = toks / 1_000_000 * config.EMBED_PRICE_PER_1M
    n_articles = len({rec["article_id"] for _, rec, _ in items})
    print(f"{'articles':<18}{n_articles:>12,}")
    print(f"{'chunks':<18}{len(items):>12,}")
    print(f"{'embed tokens':<18}{toks:>12,}")
    print(f"{'est. cost':<18}{f'${cost:.4f}':>12}")
    return cost


def load_checkpoint() -> int:
    """Number of items already embedded AND upserted."""
    if not config.INGEST_CHECKPOINT.exists():
        return 0
    try:
        return int(json.loads(config.INGEST_CHECKPOINT.read_text(encoding="utf-8"))["done"])
    except (ValueError, KeyError, TypeError):
        return 0


def save_checkpoint(done: int, total: int) -> None:
    config.INGEST_CHECKPOINT.write_text(
        json.dumps({"done": done, "total": total, "namespace": config.MAIN_NAMESPACE}),
        encoding="utf-8",
    )


def with_retry(fn, what: str):
    """Retry a network call with exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - transport errors vary by SDK
            if attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE ** attempt
            print(f"  ! {what} failed ({type(exc).__name__}: {exc}); "
                  f"retry {attempt}/{MAX_RETRIES - 1} in {wait:.0f}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def get_index():
    from pinecone import Pinecone, ServerlessSpec

    env = config.require_env("PINECONE_API_KEY", "PINECONE_INDEX")
    pc = Pinecone(api_key=env["PINECONE_API_KEY"])
    name = env["PINECONE_INDEX"]
    if name not in {ix["name"] for ix in pc.list_indexes()}:
        print(f"creating Pinecone index '{name}' (dim={config.EMBED_DIM}, cosine)...")
        pc.create_index(
            name=name,
            dimension=config.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(name)


def to_vector(vid: str, rec: dict, ch, values: list[float]) -> dict:
    """Pinecone record. Metadata fields are those required by the spec."""
    return {
        "id": vid,
        "values": values,
        "metadata": {
            "article_id": str(rec["article_id"]),
            "title": rec["title"],
            "authors": rec["authors"],
            "url": rec["url"],
            "timestamp": rec["timestamp"],
            "tags": rec["tags"],
            "chunk_text": ch.text,
            "chunk_index": ch.chunk_index,
        },
    }


def ingest(index, items, start: int) -> None:
    from ingest import embed_client

    ns = config.MAIN_NAMESPACE
    total = len(items)
    batch = config.EMBED_BATCH

    for i in range(start, total, batch):
        block = items[i : i + batch]
        texts = [ch.text for _, _, ch in block]

        vecs = with_retry(lambda: embed_client.embed_texts(texts, batch_size=batch),
                          f"embed [{i}:{i + len(block)}]")
        payload = [to_vector(vid, rec, ch, v) for (vid, rec, ch), v in zip(block, vecs)]

        for j in range(0, len(payload), config.UPSERT_BATCH):
            part = payload[j : j + config.UPSERT_BATCH]
            with_retry(lambda: index.upsert(vectors=part, namespace=ns),
                       f"upsert [{i + j}:{i + j + len(part)}]")

        done = i + len(block)
        save_checkpoint(done, total)
        pct = done / total * 100
        print(f"  [{ns}] {done:>7,}/{total:,} ({pct:5.1f}%) embedded + upserted")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually embed & upsert (spends money)")
    ap.add_argument("--reset", action="store_true", help="clear the resume checkpoint and exit")
    args = ap.parse_args()

    if args.reset:
        config.INGEST_CHECKPOINT.unlink(missing_ok=True)
        print("checkpoint cleared.")
        sys.exit(0)

    size, ov = config.FINAL_CHUNK_SIZE, config.FINAL_OVERLAP
    print(f"config: chunk_size={size}, overlap={ov} -> namespace '{config.MAIN_NAMESPACE}'\n")

    items = build_chunks(size, ov)
    cost = preflight(items)

    done = load_checkpoint()
    if done:
        remaining = len(items) - done
        frac = remaining / len(items) if items else 0
        print(f"\nresuming: {done:,} chunks already done, {remaining:,} left "
              f"(~${cost * frac:.4f} of the estimate remains)")

    if not args.yes:
        print("\nDRY RUN. No API calls made. Re-run with --yes to embed & upsert.")
        sys.exit(0)

    if done >= len(items):
        print("\nalready complete; nothing to do.")
        sys.exit(0)

    print("\n--yes given: embedding for real.\n")
    index = get_index()
    ingest(index, items, done)
    print(f"\ndone. namespace '{config.MAIN_NAMESPACE}' holds {len(items):,} vectors.")


if __name__ == "__main__":
    main()
