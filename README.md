# Medium Article RAG Assistant

A Retrieval-Augmented Generation system over ~7,600 English Medium articles,
deployed to Vercel with two public HTTP endpoints. University assignment.

> **Status:** work in progress. Phase 1 (gold set) and Phase 2 (parameter-sweep
> tooling) are complete; embedding/eval runs are pending API credentials.
> Chosen hyperparameters and the sweep results table will be added here once the
> winning config is selected.

## Layout

```
ingest/     chunking, embedding, Pinecone upsert (local only, never deployed)
eval/       gold.json, sweep script, results table
api/        FastAPI app for Vercel (Phase 4)
```

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
python -m ingest.build_subset      # build the reproducible sweep subset (free)
python -m ingest.sweep_ingest      # dry run: prints chunk counts + cost, no spend
python -m ingest.sweep_ingest --yes  # embeds the subset (costs ~$0.09)
python -m eval.run_sweep           # retrieval eval -> eval/sweep_results.md (free)
```
