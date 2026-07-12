"""Shared configuration for local ingestion & evaluation scripts.

These scripts run locally only and are NEVER deployed to Vercel.
Loads secrets from a gitignored .env at the repo root.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

# --- data ---
CSV_PATH = REPO_ROOT / "medium-english-50mb.csv"
CSV_ENCODING = "utf-8"  # file is valid UTF-8; must be read explicitly
EVAL_DIR = REPO_ROOT / "eval"
SUBSET_PATH = EVAL_DIR / "subset_ids.json"
GOLD_PATH = EVAL_DIR / "gold.json"

# --- models (course-provided OpenAI-compatible proxy) ---
EMBED_MODEL = "ZYRANGG-text-embedding-3-small"
EMBED_DIM = 1536
CHAT_MODEL = "ZYRANGG-gpt-5-mini"

# --- embedding batching (assignment requires batching; never one-at-a-time) ---
EMBED_BATCH = 128          # chunks per embedding API call (100-200 allowed)
UPSERT_BATCH = 100         # vectors per Pinecone upsert request

# --- pricing (for pre-flight cost estimates only) ---
EMBED_PRICE_PER_1M = 0.02  # USD per 1M tokens, text-embedding-3-small

# --- Phase 2 sweep configs: (chunk_size_tokens, overlap_ratio) ---
SWEEP_CONFIGS = [(256, 0.1), (512, 0.15), (1024, 0.2)]

# --- reproducible subset ---
SUBSET_SIZE = 1000
SUBSET_SEED = 42

# --- retrieval ---
SWEEP_TOP_K = 30           # fetch once; hit-rate at smaller k by truncation
HIT_K_VALUES = [3, 5, 8, 15, 30]


def namespace_for(chunk_size: int, overlap_ratio: float) -> str:
    """e.g. (512, 0.15) -> 'sweep_512_15'."""
    return f"sweep_{chunk_size}_{int(round(overlap_ratio * 100))}"


def require_env(*names: str) -> dict[str, str]:
    """Fetch required env vars or raise a clear error listing what's missing."""
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + f".\nCreate {REPO_ROOT / '.env'} (see .env.example)."
        )
    return {n: os.environ[n] for n in names}
