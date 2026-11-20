
import json
import time
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent

EVIDENCE_FILE = (
    ROOT / "data" / "evaluation" /
    "rag_evidence_80_current_index" /
    "retrieved_evidence_top10.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "evaluation" /
    "rag_generation_smoke_test_q061"
)

OUTPUT_FILE = OUTPUT_DIR / "q061_qwen3_8b.json"

MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def main():

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}"
        )

    evidence = pd.read_csv(EVIDENCE_FILE)

    q061 = evidence[
        evidence["question_id"].astype(str) == "Q061"
    ].copy()

    if q061.empty:
        raise ValueError("Q061 evidence was not found.")

    q061["rank"] = pd.to_numeric(q061["rank"])
    q061 = q061.sort_values("rank")

    q061 = q061.drop_duplicates(
        subset=["retrieved_text"]
    ).head(5)

    question = q061.iloc[0]["question"]

    context_parts = []

    for _, row in q061.iterrows():

        context_parts.append(
            f"[Source: {row['article_id']}, "
            f"p. {row['page']}]\n"
            f"Title: {row['title']}\n"
            f"Passage: {row['retrieved_text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a research question-answering assistant
specialising in crop disease detection.

Answer using only the supplied research passages.

Instructions:
1. Do not use outside knowledge.
2. Do not estimate or invent missing figures.
3. A general statement about deployment does not
   establish an exact financial cost.
4. If the passages do not explicitly provide the
   requested information, state:
   "The requested information cannot be determined
   from the supplied research evidence."
5. Cite sources only when they support your answer.

RESEARCH QUESTION:
{question}

RETRIEVED RESEARCH EVIDENCE:
{context}

FINAL ANSWER:
""".strip()

    print("=" * 65)
    print("RAG ABSTENTION TEST")
    print("=" * 65)

    print("\nModel:", MODEL)
    print("Question ID: Q061")
    print("Distinct evidence passages:", len(q061))
    print("\nGenerating answer...\n")

    start_time = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "0",
            "options": {
                "temperature": 0.1,
                "num_predict": 400
            }
        },
        timeout=600
    )

    response.raise_for_status()

    result = response.json()

    elapsed = time.perf_counter() - start_time

    answer = result.get("response", "").strip()

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty answer. "
            f"Response fields: {list(result.keys())}"
        )

    output = {
        "experiment": "RAG abstention smoke test",
        "question_id": "Q061",
        "model": MODEL,
        "question": question,
        "answer": answer,
        "retrieved_passages": len(q061),
        "source_passages": [
            {
                "article_id": str(row["article_id"]),
                "page": int(row["page"]),
                "chunk_id": str(row["chunk_id"])
            }
            for _, row in q061.iterrows()
        ],
        "latency_seconds": round(elapsed, 3),
        "total_duration_ns": result.get("total_duration"),
        "prompt_tokens": result.get("prompt_eval_count"),
        "generated_tokens": result.get("eval_count")
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("=" * 65)
    print("ABSTENTION TEST COMPLETED")
    print("=" * 65)

    print("\nGenerated answer:\n")
    print(answer)

    print("\nLatency:", output["latency_seconds"], "seconds")
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
