# Phase 2 — Retrieval sweep results

Subset: 1000 articles (seed 42). top_k=30 chunks fetched, deduplicated to distinct articles.
Metrics computed over 10 gold questions. Hit-rate @ k = correct article present within the first k distinct retrieved articles.

## Summary

| config | hit@3 | hit@5 | hit@8 | hit@15 | hit@30 | MRR | avg distinct art. |
|---|---|---|---|---|---|---|---|
| sweep_256_10 | 0.90 | 0.90 | 0.90 | 1.00 | 1.00 | 0.840 | 13.2 |
| sweep_512_15 | 0.80 | 0.80 | 0.90 | 1.00 | 1.00 | 0.757 | 15.8 |
| sweep_1024_20 | 0.80 | 0.80 | 0.80 | 1.00 | 1.00 | 0.750 | 20.3 |

## MRR by question type

| config | fact | multi | summary | recommendation |
|---|---|---|---|---|
| sweep_256_10 | 1.000 | 0.202 | 1.000 | 1.000 |
| sweep_512_15 | 1.000 | 0.200 | 0.722 | 1.000 |
| sweep_1024_20 | 1.000 | 0.205 | 0.697 | 1.000 |

## Per-question rank of correct article (— = not found in top-k)

| id | type | sweep_256_10 | sweep_512_15 | sweep_1024_20 |
|---|---|---|---|---|
| g01 | fact | 1 | 1 | 1 |
| g02 | multi | 14 | 15 | 13 |
| g03 | summary | 1 | 6 | 11 |
| g04 | recommendation | 1 | 1 | 1 |
| g05 | fact | 1 | 1 | 1 |
| g06 | fact | 1 | 1 | 1 |
| g07 | summary | 1 | 1 | 1 |
| g08 | summary | 1 | 1 | 1 |
| g09 | multi | 3 | 3 | 3 |
| g10 | recommendation | 1 | 1 | 1 |

_You choose the winning config based on this table._