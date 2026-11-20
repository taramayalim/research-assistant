
import json
import time
from pathlib import Path

import pandas as pd
import requests


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

EVIDENCE_FILE = (
    ROOT
    / "data"
    / "evaluation"
    / "rag_evidence_80_current_index"
    / "retrieved_evidence_top10.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "evaluation"
    / "rag_generation_smoke_test_q001"
)

OUTPUT_FILE = OUTPUT_DIR / "q001_qwen3_8b.json"

MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

TOP_PASSAGES = 5
TIMEOUT_SECONDS = 600


# --------------------------------------------------
# 2. LOAD AND PREPARE EVIDENCE
# --------------------------------------------------

def load_evidence():

    if not EVIDENCE_FILE.exists():
        raise FileNotFoundError(
            f"Evidence file not found: {EVIDENCE_FILE}"
        )

    evidence = pd.read_csv(EVIDENCE_FILE)

    q001 = evidence[
        evidence["question_id"].astype(str) == "Q001"
    ].copy()

    if q001.empty:
        raise ValueError("Q001 evidence was not found.")

    q001["rank"] = pd.to_numeric(
        q001["rank"],
        errors="raise"
    )

    q001 = q001.sort_values("rank")

    # Remove duplicate passages while preserving rank order.
    q001 = q001.drop_duplicates(
        subset=["retrieved_text"]
    ).head(TOP_PASSAGES)

    if q001.empty:
        raise ValueError("No usable evidence passages found.")

    return q001


# --------------------------------------------------
# 3. BUILD GROUNDED PROMPT
# --------------------------------------------------

def build_prompt(q001):

    question = str(q001.iloc[0]["question"])

    context_parts = []

    for _, row in q001.iterrows():

        context_parts.append(
            f"[Source: {row['article_id']}, p. {row['page']}]\n"
            f"Title: {row['title']}\n"
            f"Passage: {row['retrieved_text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a research question-answering assistant
specialising in crop disease detection.

Answer the research question using only the
provided research evidence.

Instructions:
1. Use only information explicitly supported by the passages.
2. Do not invent facts, figures, or citations.
3. Cite the supporting source and page number.
4. Use this citation format: [A006, p. 9].
5. If the evidence is insufficient, clearly state that
   the answer cannot be determined from the supplied evidence.
6. Give a concise and direct answer.

RESEARCH QUESTION:
{question}

RETRIEVED RESEARCH EVIDENCE:
{context}

FINAL ANSWER:
""".strip()

    return question, prompt


# --------------------------------------------------
# 4. GENERATE ANSWER USING OLLAMA
# --------------------------------------------------

def generate_answer(prompt):

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": "0",
        "options": {
            "temperature": 0.1,
            "num_predict": 400
        }
    }

    start_time = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=TIMEOUT_SECONDS
    )

    response.raise_for_status()

    result = response.json()

    elapsed = time.perf_counter() - start_time

    answer = result.get("response", "").strip()

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty final answer.\n"
            f"Model: {MODEL}\n"
            f"Done: {result.get('done')}\n"
            f"Done reason: {result.get('done_reason')}\n"
            f"Generated tokens: {result.get('eval_count')}\n"
            f"Thinking field present: "
            f"{bool(result.get('thinking'))}\n"
            "Inspect the Ollama response before retrying."
        )

    return answer, result, elapsed


# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------

def save_result(
    question,
    q001,
    answer,
    result,
    elapsed
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "experiment": "RAG generation smoke test",
        "question_id": "Q001",
        "model": MODEL,
        "question": question,
        "answer": answer,
        "retrieved_passages": len(q001),
        "source_passages": [
            {
                "article_id": str(row["article_id"]),
                "page": int(row["page"]),
                "chunk_id": str(row["chunk_id"])
            }
            for _, row in q001.iterrows()
        ],
        "latency_seconds": round(elapsed, 3),
        "total_duration_ns": result.get("total_duration"),
        "prompt_tokens": result.get("prompt_eval_count"),
        "generated_tokens": result.get("eval_count"),
        "thinking_saved": False
    }

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

    return output


# --------------------------------------------------
# 6. MAIN EXECUTION
# --------------------------------------------------

def main():

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            "Output already exists. "
            f"Inspect it before running again: {OUTPUT_FILE}"
        )

    print("=" * 65)
    print("RAG ANSWER GENERATION TEST")
    print("=" * 65)

    q001 = load_evidence()

    question, prompt = build_prompt(q001)

    print("\nModel:", MODEL)
    print("Question ID: Q001")
    print("Distinct evidence passages:", len(q001))
    print("\nGenerating answer...\n")

    answer, result, elapsed = generate_answer(prompt)

    output = save_result(
        question,
        q001,
        answer,
        result,
        elapsed
    )

    print("=" * 65)
    print("GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("\nGenerated answer:\n")
    print(answer)

    print("\nLatency:", output["latency_seconds"], "seconds")
    print("Prompt tokens:", output["prompt_tokens"])
    print("Generated tokens:", output["generated_tokens"])

    print("\nSaved result:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
