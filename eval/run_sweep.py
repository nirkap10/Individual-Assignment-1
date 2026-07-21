"""Phase 2 retrieval evaluation (free: no LLM, no re-embedding).

For each gold question:
  - embed the question ONCE (reused across all namespaces),
  - query each sweep namespace for top_k=30 chunks,
  - dedup to distinct articles (highest-scoring chunk per article_id, then
    collapse identical titles),
  - score hit-rate @ k in {3,5,8,15,30} and reciprocal rank against gold.

Writes eval/sweep_results.md. You choose the winning config from the table.

Run:  python -m eval.run_sweep
"""
import json
from pathlib import Path

from ingest import config
from ingest import embed_client


def dedup_to_articles(matches) -> list[tuple[str, float, str]]:
    """matches: Pinecone results (score-ordered). Return distinct articles as
    (article_id, score, title), keeping the best chunk per article_id and then
    collapsing identical (normalized) titles."""
    best: dict[str, tuple[float, str]] = {}
    order: list[str] = []
    for m in matches:
        md = m["metadata"]
        aid = str(md["article_id"])
        if aid not in best:
            best[aid] = (m["score"], md.get("title", ""))
            order.append(aid)
    ranked = [(aid, best[aid][0], best[aid][1]) for aid in order]

    seen_titles: set[str] = set()
    collapsed = []
    for aid, score, title in ranked:
        key = title.strip().lower()
        if key and key in seen_titles:
            continue
        seen_titles.add(key)
        collapsed.append((aid, score, title))
    return collapsed


def score_single(ranked_ids: list[int], acceptable: set[int]):
    rank = None
    for pos, aid in enumerate(ranked_ids, start=1):
        if aid in acceptable:
            rank = pos
            break
    hits = {k: int(rank is not None and rank <= k) for k in config.HIT_K_VALUES}
    rr = 1.0 / rank if rank else 0.0
    return hits, rr, rank


def score_multi(ranked_ids: list[int], acceptable: set[int], min_distinct: int):
    seen: list[int] = []
    rank_m = None
    for pos, aid in enumerate(ranked_ids, start=1):
        if aid in acceptable and aid not in seen:
            seen.append(aid)
            if len(seen) >= min_distinct and rank_m is None:
                rank_m = pos
    hits = {}
    for k in config.HIT_K_VALUES:
        distinct_in_k = len({a for a in ranked_ids[:k] if a in acceptable})
        hits[k] = int(distinct_in_k >= min_distinct)
    rr = 1.0 / rank_m if rank_m else 0.0
    return hits, rr, rank_m


def main() -> None:
    gold = json.loads(config.GOLD_PATH.read_text(encoding="utf-8"))
    questions = gold["questions"]

    from pinecone import Pinecone
    env = config.require_env("PINECONE_API_KEY", "PINECONE_INDEX")
    index = Pinecone(api_key=env["PINECONE_API_KEY"]).Index(env["PINECONE_INDEX"])

    # embed each question once, reuse across namespaces
    q_vectors = {q["id"]: embed_client.embed_one(q["question"]) for q in questions}

    results = {}   # namespace -> per-question rows
    for size, ov in config.SWEEP_CONFIGS:
        ns = config.namespace_for(size, ov)
        rows = []
        for q in questions:
            res = index.query(
                vector=q_vectors[q["id"]],
                top_k=config.SWEEP_TOP_K,
                namespace=ns,
                include_metadata=True,
            )
            matches = res.get("matches", [])
            articles = dedup_to_articles(matches)
            ranked_ids = [int(a[0]) for a in articles]
            acceptable = set(q["acceptable_article_ids"])
            if q["type"] == "multi":
                hits, rr, rank = score_multi(ranked_ids, acceptable, q["min_distinct"])
            else:
                hits, rr, rank = score_single(ranked_ids, acceptable)
            rows.append({
                "id": q["id"], "type": q["type"], "hits": hits, "rr": rr,
                "rank": rank, "n_distinct": len(ranked_ids),
            })
        results[ns] = rows

    write_report(results)


def write_report(results: dict) -> None:
    lines = ["# Phase 2 — Retrieval sweep results", ""]
    lines.append(f"Subset: {config.SUBSET_SIZE} articles (seed {config.SUBSET_SEED}). "
                 f"top_k={config.SWEEP_TOP_K} chunks fetched, deduplicated to distinct articles.")
    lines.append("Metrics computed over 10 gold questions. Hit-rate @ k = correct "
                 "article present within the first k distinct retrieved articles.")
    lines.append("")

    header = "| config | " + " | ".join(f"hit@{k}" for k in config.HIT_K_VALUES) + \
             " | MRR | avg distinct art. |"
    sep = "|" + "---|" * (len(config.HIT_K_VALUES) + 3)
    lines += ["## Summary", "", header, sep]
    for ns, rows in results.items():
        n = len(rows)
        agg = {k: sum(r["hits"][k] for r in rows) / n for k in config.HIT_K_VALUES}
        mrr = sum(r["rr"] for r in rows) / n
        avg_dist = sum(r["n_distinct"] for r in rows) / n
        cells = " | ".join(f"{agg[k]:.2f}" for k in config.HIT_K_VALUES)
        lines.append(f"| {ns} | {cells} | {mrr:.3f} | {avg_dist:.1f} |")
    lines.append("")

    # per-type MRR
    lines += ["## MRR by question type", "", "| config | fact | multi | summary | recommendation |",
              "|---|---|---|---|---|"]
    for ns, rows in results.items():
        by = {}
        for t in ["fact", "multi", "summary", "recommendation"]:
            tr = [r["rr"] for r in rows if r["type"] == t]
            by[t] = sum(tr) / len(tr) if tr else 0.0
        lines.append(f"| {ns} | {by['fact']:.3f} | {by['multi']:.3f} | "
                     f"{by['summary']:.3f} | {by['recommendation']:.3f} |")
    lines.append("")

    # per-question rank per config
    lines += ["## Per-question rank of correct article (— = not found in top-k)", "",
              "| id | type | " + " | ".join(results.keys()) + " |",
              "|---|---|" + "---|" * len(results)]
    ids = [r["id"] for r in next(iter(results.values()))]
    types = {r["id"]: r["type"] for r in next(iter(results.values()))}
    for qid in ids:
        cells = []
        for ns in results:
            r = next(x for x in results[ns] if x["id"] == qid)
            cells.append(str(r["rank"]) if r["rank"] else "—")
        lines.append(f"| {qid} | {types[qid]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("_You choose the winning config based on this table._")

    out = Path(config.EVAL_DIR) / "sweep_results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
