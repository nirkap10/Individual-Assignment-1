"""FastAPI app for Vercel. Exposes POST /api/prompt and GET /api/stats.

Does not import from ingest/ -- those modules need pandas and tiktoken, which
don't belong in the deployed function.
"""
import os

from fastapi import FastAPI
from pydantic import BaseModel

# Must match what was ingested (ingest/config.py). Changing chunk size or
# overlap here would be a lie -- they're baked into the vectors.
CHUNK_SIZE = 256
OVERLAP_RATIO = 0.1
# 8 was too low: the "list 3 articles about education" question only found one
# of them, so the model refused. Its third article sits at rank 15.
TOP_K = 15

# Fetch extra chunks so that collapsing them to distinct articles still leaves
# at least TOP_K. 30 chunks gives ~22 articles on the full corpus.
OVERFETCH = 30

NAMESPACE = "medium"
EMBED_MODEL = "NBUECSE-text-embedding-3-small"
CHAT_MODEL = "NBUECSE-gpt-5-mini"

# Required by the assignment, verbatim.
REQUIRED_SYSTEM_PROMPT = (
    "You are a Medium-article assistant that answers questions strictly and "
    "only based on the Medium articles dataset context provided to you "
    "(metadata and article passages). You must not use any external knowledge, "
    "the open internet, or information that is not explicitly contained in the "
    "retrieved context. If the answer cannot be determined from the provided "
    "context, respond: \"I don't know based on the provided Medium articles "
    "data.\" Always explain your answer using the given context, quoting or "
    "paraphrasing the relevant article passage or metadata when helpful."
)

# The spec allows appending response-style clarifications as long as the
# constraints above are kept. Without this the model sometimes emitted the
# refusal sentence and then answered anyway, which reads as a wrong answer.
STYLE_CLARIFICATION = (
    " Response style: use the refusal sentence above only when the context "
    "genuinely does not answer the question, and in that case say nothing "
    "else. If the context does answer it, answer directly and do not include "
    "the refusal sentence. When asked for a specific number of articles, "
    "return exactly that many distinct titles."
)

SYSTEM_PROMPT = REQUIRED_SYSTEM_PROMPT + STYLE_CLARIFICATION

app = FastAPI(title="Medium Article RAG Assistant")

_openai = None
_index = None


def openai_client():
    global _openai
    if _openai is None:
        from openai import OpenAI

        _openai = OpenAI(
            api_key=os.environ["ZYRANGG_API_KEY"],
            base_url=os.environ["ZYRANGG_BASE_URL"],
        )
    return _openai


def pinecone_index():
    global _index
    if _index is None:
        from pinecone import Pinecone

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _index = pc.Index(os.environ["PINECONE_INDEX"])
    return _index


class PromptRequest(BaseModel):
    question: str


def dedup_to_articles(matches) -> list[dict]:
    """Collapse chunks to distinct articles, keeping the best chunk of each.

    Matches come back sorted by score, so the first chunk seen for an
    article_id is its highest-scoring one.
    """
    best: dict[str, dict] = {}
    for m in matches:
        meta = m.get("metadata") or {}
        aid = str(meta.get("article_id", ""))
        if aid and aid not in best:
            best[aid] = {
                "article_id": aid,
                "title": str(meta.get("title", "")),
                "chunk": str(meta.get("chunk_text", "")),
                "score": float(m.get("score", 0.0)),
            }
    return list(best.values())


def build_user_prompt(question: str, context: list[dict]) -> str:
    if not context:
        return (
            "No Medium article context was retrieved for this question.\n\n"
            f"Question: {question}"
        )
    blocks = []
    for i, c in enumerate(context, start=1):
        blocks.append(
            f"[{i}] article_id: {c['article_id']}\n"
            f"title: {c['title']}\n"
            f"passage: {c['chunk']}"
        )
    joined = "\n\n".join(blocks)
    return (
        "Use only the Medium article context below to answer the question.\n"
        "When you rely on a passage, refer to its title.\n\n"
        f"=== CONTEXT ===\n{joined}\n=== END CONTEXT ===\n\n"
        f"Question: {question}"
    )


@app.post("/api/prompt")
def prompt(req: PromptRequest):
    question = (req.question or "").strip()
    if not question:
        return {
            "response": "I don't know based on the provided Medium articles data.",
            "context": [],
            "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": ""},
        }

    embedding = openai_client().embeddings.create(
        model=EMBED_MODEL, input=[question]
    ).data[0].embedding

    result = pinecone_index().query(
        vector=embedding,
        top_k=OVERFETCH,
        namespace=NAMESPACE,
        include_metadata=True,
    )
    matches = result.get("matches", []) if isinstance(result, dict) else result.matches
    context = dedup_to_articles(matches)[:TOP_K]

    user_prompt = build_user_prompt(question, context)
    completion = openai_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = completion.choices[0].message.content or ""

    # Capital A in Augmented_prompt, and System/User, are what the spec asks for.
    return {
        "response": answer,
        "context": context,
        "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": user_prompt},
    }


@app.get("/api/stats")
def stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K,
    }
