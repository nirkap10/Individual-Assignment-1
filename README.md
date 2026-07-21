# Medium Article RAG Assistant

A Retrieval-Augmented Generation system over ~7,600 English Medium articles,
deployed to Vercel with two public HTTP endpoints. University assignment.

**Live URL:** https://individual-assignment-1-nirkap10s-projects.vercel.app

```bash
curl https://individual-assignment-1-nirkap10s-projects.vercel.app/api/stats

curl -X POST https://individual-assignment-1-nirkap10s-projects.vercel.app/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"question": "List exactly 3 articles about education. Return only the titles."}'
```

## Chosen hyperparameters

| parameter | value | can it change without re-embedding? |
|---|---|---|
| `chunk_size` | 256 tokens | no — baked into the vectors |
| `overlap_ratio` | 0.1 | no — baked into the vectors |
| `top_k` | 15 | yes — free to tune at query time |

The full corpus (7,682 articles → 48,585 chunks) is embedded once into the
Pinecone namespace `medium` with `text-embedding-3-small` (1536 dims, cosine).

## Why these values

### chunk_size and overlap: chosen by measurement, not by guessing

All three configs allowed by the assignment were embedded over an identical
1,000-article subset (seed 42, chosen to provably contain every gold-set target)
and evaluated against 10 hand-built gold questions. Retrieval evaluation needs
no LLM calls, so this comparison was free apart from the embeddings.

| config | hit@3 | hit@5 | hit@8 | hit@15 | MRR | avg distinct articles |
|---|---|---|---|---|---|---|
| **256 / 0.1** | **0.90** | **0.90** | **0.90** | 1.00 | **0.840** | 13.2 |
| 512 / 0.15 | 0.80 | 0.80 | 0.90 | 1.00 | 0.757 | 15.8 |
| 1024 / 0.2 | 0.80 | 0.80 | 0.80 | 1.00 | 0.750 | 20.3 |

`hit@k` asks whether the correct article appears in the first *k* results at
all. `MRR` (mean reciprocal rank) scores *how high* it ranked — `1/rank`,
averaged over the questions — so it rewards putting the answer at position 1.

Every config reaches perfect recall by k=15, so the real differentiator is rank
position, and **256 wins on every metric**. The mechanism is visible in the
per-question data: for `g03` the correct article ranks **1st** at 256 but drops
to **6th** at 512 and **11th** at 1024. Smaller chunks keep each vector
topically tight, so a focused question matches cleanly; larger chunks blend
several ideas into one vector and dilute the match.

Smaller chunks also cost slightly less to embed (less duplicated overlap text)
and roughly 4× less per query at generation time, since each retrieved chunk is
a quarter the size. The trade-off is more vectors to store — 6,467 vs 1,966 on
the subset — which is negligible at this scale.

Full results, including per-question ranks: [`eval/sweep_results.md`](eval/sweep_results.md).

### top_k = 15: raised deliberately, and it matters

`top_k` was initially set to 8 and the multi-result question failed: asked to
list 3 articles about education, the system could only see one and correctly
answered *"I don't know based on the provided Medium articles data."*

The sweep had put that question's third distinct article around rank 14. Raising
`top_k` to 15 fixes it — the same question now returns exactly 3 distinct
education articles with justification. All 10 gold questions retrieve their
target at `top_k=15`.

This is the payoff from evaluating `top_k` separately: it is a query-time
parameter, so it was tuned without re-embedding anything.

## Retrieval: over-fetch, then collapse to distinct articles

Pinecone is queried for 30 chunks, which are then grouped by `article_id`
keeping the highest-scoring chunk per article. Without this step, "list 3
articles about education" can return three chunks of the *same* article and fail
the requirement outright.

On the full corpus 30 chunks yield ~22 distinct articles, comfortably more than
`top_k=15`, so the returned list is never starved.

## API

### `POST /api/prompt`

```json
{ "question": "Your natural language question here" }
```

Returns `response`, `context` (one entry per distinct article, with
`article_id`, `title`, `chunk`, `score`), and `Augmented_prompt` with the exact
`System` and `User` strings sent to the chat model.

### `GET /api/stats`

```json
{ "chunk_size": 256, "overlap_ratio": 0.1, "top_k": 15 }
```

## Layout

```
ingest/     chunking, embedding, Pinecone upsert (local only, never deployed)
eval/       gold.json, sweep + API check scripts, results table
api/        index.py — FastAPI app for Vercel
requirements.txt      light deps for the deployed function only
requirements-dev.txt  local ingestion/eval deps (pandas, tiktoken, ...)
```

The deployed function deliberately imports nothing from `ingest/` — those
modules pull in pandas and tiktoken, which have no place in a serverless
function that only embeds one question and queries Pinecone.

## Local setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt
cp .env.example .env      # fill in your keys (gitignored)
```

The dataset (`medium-english-50mb.csv`) is not committed; download it from the
assignment link and place it in the repo root.

## Workflow

```bash
python -m ingest.build_subset        # build the reproducible sweep subset (free)
python -m ingest.sweep_ingest        # dry run: prints chunk counts + cost, no spend
python -m ingest.sweep_ingest --yes  # embed the subset (~$0.09)
python -m eval.run_sweep             # retrieval eval -> eval/sweep_results.md (free)

python -m ingest.full_ingest         # dry run: full-corpus cost estimate, no spend
python -m ingest.full_ingest --yes   # embed all 7,682 articles (~$0.23), resumable
python -m eval.check_api             # API contract check against the gold set
```

Every script that spends money dry-runs by default and requires `--yes`.
`full_ingest` checkpoints after each batch is embedded *and* upserted, so an
interrupted run resumes without paying twice.

## Cost

| step | spend |
|---|---|
| Phase 2 sweep (3 configs × 1,000 articles) | $0.127 |
| Phase 3 full-corpus ingest | $0.231 |
| **Total** | **~$0.36 of the $5 budget** |
