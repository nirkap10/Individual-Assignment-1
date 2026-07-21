"""Phase 2 spend step: chunk + embed + upsert the subset into 3 namespaces.

This is the ONLY script in Phase 2 that calls the embedding API and costs money.

Safety:
  - By default it runs a DRY RUN: it chunks the subset locally, prints the exact
    chunk counts, token totals and estimated cost, and exits WITHOUT embedding.
  - It only spends money when passed --yes.
  - It is resumable: a namespace that already holds the expected number of
    vectors is skipped, so a crash never forces a full re-embed.

Run (dry run) : python -m ingest.sweep_ingest
Run (spend)   : python -m ingest.sweep_ingest --yes
"""
import argparse
import json
import sys

from ingest import config
from ingest import data
from ingest.chunker import chunk_text


def build_chunks(subset_ids: list[int], chunk_size: int, overlap_ratio: float):
    """Return list of (vector_id, article_record, Chunk) for the subset."""
    df = data.load_df()
    df = df[df["article_id"].isin(set(subset_ids))]
    items = []
    for _, row in df.iterrows():
        rec = data.article_record(row)
        for ch in chunk_text(rec["text"], chunk_size, overlap_ratio):
            vid = f"{rec['article_id']}#{ch.chunk_index}"
            items.append((vid, rec, ch))
    return items


def preflight(subset_ids: list[int]) -> None:
    print(f"{'config':<14}{'chunks':>9}{'embed_tokens':>16}{'est_cost':>12}")
    grand = 0.0
    for size, ov in config.SWEEP_CONFIGS:
        items = build_chunks(subset_ids, size, ov)
        toks = sum(ch.n_tokens for _, _, ch in items)
        cost = toks / 1_000_000 * config.EMBED_PRICE_PER_1M
        grand += cost
        print(f"{f'({size},{ov})':<14}{len(items):>9}{toks:>16,}{f'${cost:.4f}':>12}")
    print(f"{'TOTAL':<14}{'':>9}{'':>16}{f'${grand:.4f}':>12}")


def get_index():
    from pinecone import Pinecone, ServerlessSpec

    env = config.require_env("PINECONE_API_KEY", "PINECONE_INDEX")
    pc = Pinecone(api_key=env["PINECONE_API_KEY"])
    name = env["PINECONE_INDEX"]
    existing = {ix["name"] for ix in pc.list_indexes()}
    if name not in existing:
        print(f"creating Pinecone index '{name}' (dim={config.EMBED_DIM}, cosine)...")
        pc.create_index(
            name=name,
            dimension=config.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(name)


def ingest_config(index, subset_ids, chunk_size, overlap_ratio) -> None:
    from ingest import embed_client

    ns = config.namespace_for(chunk_size, overlap_ratio)
    items = build_chunks(subset_ids, chunk_size, overlap_ratio)
    expected = len(items)

    stats = index.describe_index_stats()
    have = stats.get("namespaces", {}).get(ns, {}).get("vector_count", 0)
    if have >= expected:
        print(f"[{ns}] already has {have}/{expected} vectors -> skip (resumable)")
        return

    print(f"[{ns}] embedding {expected} chunks (batch {config.EMBED_BATCH})...")
    texts = [ch.text for _, _, ch in items]
    vectors = embed_client.embed_texts(texts)

    payload = []
    for (vid, rec, ch), vec in zip(items, vectors):
        payload.append({
            "id": vid,
            "values": vec,
            "metadata": {
                "article_id": str(rec["article_id"]),
                "title": rec["title"],
                "chunk_text": ch.text,
                "chunk_index": ch.chunk_index,
            },
        })

    for i in range(0, len(payload), config.UPSERT_BATCH):
        index.upsert(vectors=payload[i : i + config.UPSERT_BATCH], namespace=ns)
    print(f"[{ns}] upserted {len(payload)} vectors")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually embed & upsert (spends money)")
    args = ap.parse_args()

    subset = json.loads(config.SUBSET_PATH.read_text(encoding="utf-8"))["subset_ids"]
    print(f"subset: {len(subset)} articles\n")
    preflight(subset)

    if not args.yes:
        print("\nDRY RUN. No API calls made. Re-run with --yes to embed & upsert.")
        sys.exit(0)

    print("\n--yes given: embedding for real.\n")
    index = get_index()
    for size, ov in config.SWEEP_CONFIGS:
        ingest_config(index, subset, size, ov)
    print("\ndone. namespaces populated:", [config.namespace_for(s, o) for s, o in config.SWEEP_CONFIGS])


if __name__ == "__main__":
    main()
