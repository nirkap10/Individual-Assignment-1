"""Build the reproducible ~1,000-article sweep subset.

The subset ALWAYS contains every gold-set target id, plus a fixed random
sample of distractors (seed config.SUBSET_SEED). Writes eval/subset_ids.json.
Local only; no API calls, no cost.

Run:  python -m ingest.build_subset
"""
from __future__ import annotations

import json

import numpy as np

from ingest import config
from ingest import data


def gold_ids() -> list[int]:
    gold = json.loads(config.GOLD_PATH.read_text(encoding="utf-8"))
    ids = {i for q in gold["questions"] for i in q["acceptable_article_ids"]}
    return sorted(ids)


def build() -> dict:
    df = data.load_df()
    all_ids = df["article_id"].tolist()
    g = gold_ids()

    rng = np.random.default_rng(config.SUBSET_SEED)
    pool = [i for i in all_ids if i not in set(g)]
    n_fill = config.SUBSET_SIZE - len(g)
    fill = rng.choice(pool, size=n_fill, replace=False).tolist()
    subset = sorted(set(g) | set(int(x) for x in fill))

    assert set(g).issubset(subset), "subset must contain all gold ids"

    return {
        "size": len(subset),
        "seed": config.SUBSET_SEED,
        "n_gold_ids": len(g),
        "gold_ids": g,
        "subset_ids": subset,
    }


def main() -> None:
    payload = build()
    config.SUBSET_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"subset size      : {payload['size']}")
    print(f"gold ids included: {payload['n_gold_ids']} (all present)")
    print(f"seed             : {payload['seed']}")
    print(f"written          : {config.SUBSET_PATH}")


if __name__ == "__main__":
    main()
