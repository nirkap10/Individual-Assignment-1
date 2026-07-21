"""Local contract check for the Vercel API. Run before deploying.

Loads .env here (the deployed function never does -- Vercel injects env vars,
and python-dotenv is deliberately not a deployed dependency), then exercises
both endpoints against the gold set and asserts the response shape matches the
assignment contract exactly, including the odd `Augmented_prompt`/`System`/
`User` casing that a grading script may check literally.

Costs one embedding + one chat call per question asked (default 2).

Run: python -m eval.check_api            # 2 questions
     python -m eval.check_api --all      # every gold question
"""
from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from ingest import config

load_dotenv(config.REPO_ROOT / ".env")

# Article text is full Unicode; the Windows console is cp1252 and would raise
# UnicodeEncodeError on characters like the non-breaking hyphen.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CTX_KEYS = {"article_id", "title", "chunk", "score"}


def check_stats(client) -> list[str]:
    errs = []
    r = client.get("/api/stats")
    if r.status_code != 200:
        return [f"/api/stats returned HTTP {r.status_code}"]
    body = r.json()
    if set(body) != {"chunk_size", "overlap_ratio", "top_k"}:
        errs.append(f"/api/stats keys wrong: {sorted(body)}")
    if body.get("chunk_size") != config.FINAL_CHUNK_SIZE:
        errs.append(f"chunk_size {body.get('chunk_size')} != ingested {config.FINAL_CHUNK_SIZE}")
    if abs(body.get("overlap_ratio", -1) - config.FINAL_OVERLAP) > 1e-9:
        errs.append(f"overlap_ratio {body.get('overlap_ratio')} != ingested {config.FINAL_OVERLAP}")
    print(f"GET /api/stats -> {body}")
    return errs


def check_prompt(client, q: dict) -> list[str]:
    errs = []
    r = client.post("/api/prompt", json={"question": q["question"]})
    if r.status_code != 200:
        return [f"[{q['id']}] HTTP {r.status_code}: {r.text[:200]}"]
    d = r.json()

    if set(d) != {"response", "context", "Augmented_prompt"}:
        errs.append(f"[{q['id']}] top-level keys wrong: {sorted(d)}")
    if set(d.get("Augmented_prompt", {})) != {"System", "User"}:
        errs.append(f"[{q['id']}] Augmented_prompt keys wrong: {sorted(d.get('Augmented_prompt', {}))}")

    ctx = d.get("context", [])
    for c in ctx:
        if set(c) != CTX_KEYS:
            errs.append(f"[{q['id']}] context item keys wrong: {sorted(c)}")
            break
    if not all(isinstance(c.get("article_id"), str) for c in ctx):
        errs.append(f"[{q['id']}] article_id must be a string")

    ids = [c["article_id"] for c in ctx]
    if len(ids) != len(set(ids)):
        errs.append(f"[{q['id']}] context contains duplicate article_ids -- dedup broken")

    gold = {str(a) for a in (q.get("acceptable_article_ids") or [q.get("article_id")]) if a is not None}
    hit = gold & set(ids)
    rank = next((i for i, a in enumerate(ids, 1) if a in gold), None)

    print(f"\n[{q['id']}] {q['type']}: {q['question'][:70]}")
    print(f"  context: {len(ctx)} distinct articles | gold hit: {'yes rank ' + str(rank) if hit else 'NO'}")
    print(f"  answer : {d['response'][:200].replace(chr(10), ' ')}")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="test every gold question")
    args = ap.parse_args()

    from fastapi.testclient import TestClient
    from api.index import app

    client = TestClient(app)
    g = json.loads(config.GOLD_PATH.read_text(encoding="utf-8"))
    questions = g["questions"] if isinstance(g, dict) else g
    if not args.all:
        questions = questions[:2]

    errs = check_stats(client)
    for q in questions:
        errs += check_prompt(client, q)

    print("\n" + "=" * 60)
    if errs:
        print(f"FAILED ({len(errs)} contract violations):")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print(f"PASS - contract satisfied on {len(questions)} question(s).")


if __name__ == "__main__":
    main()
