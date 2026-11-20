
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests


# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

EVIDENCE_FILE = (
    ROOT / "data" / "evaluation" /
    "rag_evidence_80_current_index" /
    "retrieved_evidence_top10.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "evaluation" /
    "rag_generation_80"
)

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = {
    "qwen3:8b": "qwen3_8b",
    "gemma3:4b": "gemma3_4b",
    "llama3.2:3b": "llama3_2_3b",
    "phi3:mini": "phi3_mini"
}

TOP_PASSAGES = 5
TIMEOUT_SECONDS = 900
NUM_PREDICT = 400
TEMPERATURE = 0.1


# ==================================================
# 2. LOAD EVIDENCE
# ==================================================

def load_evidence():

    if not EVIDENCE_FILE.exists():
        raise FileNotFoundError(EVIDENCE_FILE)

    evidence = pd.read_csv(EVIDENCE_FILE)

    required_columns = {
        "question_id",
        "question",
        "rank",
        "chunk_id",
        "article_id",
        "title",
        "page",
        "retrieved_text"
    }

    missing = required_columns - set(evidence.columns)

    if missing:
        raise ValueError(
            f"Missing evidence columns: {sorted(missing)}"
        )

    if evidence["question_id"].nunique() != 80:
        raise ValueError("Expected exactly 80 questions.")

    return evidence


# ==================================================
# 3. PREPARE QUESTION CONTEXT
# ==================================================

def prepare_question(question_rows):

    question_rows = question_rows.copy()

    question_rows["rank"] = pd.to_numeric(
        question_rows["rank"],
        errors="raise"
    )

    question_rows = question_rows.sort_values("rank")

    # Remove duplicate text while preserving ranking.
    question_rows = question_rows.drop_duplicates(
        subset=["retrieved_text"]
    ).head(TOP_PASSAGES)

    if question_rows.empty:
        raise ValueError("No evidence passages available.")

    question = str(question_rows.iloc[0]["question"])

    context_parts = []
    sources = []

    for _, row in question_rows.iterrows():

        article_id = str(row["article_id"])
        page = int(row["page"])

        context_parts.append(
            f"[Source: {article_id}, p. {page}]\n"
            f"Title: {row['title']}\n"
            f"Passage: {row['retrieved_text']}"
        )

        sources.append({
            "article_id": article_id,
            "page": page,
            "chunk_id": str(row["chunk_id"])
        })

    return question, "\n\n".join(context_parts), sources


# ==================================================
# 4. BUILD COMMON PROMPT
# ==================================================

def build_prompt(question, context):

    return f"""
You are a research question-answering assistant
specialising in crop disease detection.

Answer the research question using only the
retrieved research evidence provided below.

Instructions:
1. Use only information explicitly supported by the evidence.
2. Do not use outside knowledge.
3. Do not invent facts, figures, or citations.
4. Cite supporting sources using this format:
   [A006, p. 9].
5. If the evidence does not contain enough information
   to answer the question, clearly state that the
   requested information cannot be determined from
   the supplied research evidence.
6. For comparison questions, use evidence from the
   relevant papers where available.
7. Give a concise and direct answer.

RESEARCH QUESTION:
{question}

RETRIEVED RESEARCH EVIDENCE:
{context}

FINAL ANSWER:
""".strip()


# ==================================================
# 5. GENERATE ANSWER
# ==================================================

def generate_answer(model, prompt):

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT
        }
    }

    start = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=TIMEOUT_SECONDS
    )

    response.raise_for_status()

    result = response.json()

    elapsed = time.perf_counter() - start

    answer = result.get("response", "").strip()

    if not answer:
        raise RuntimeError(
            f"Empty response from {model}. "
            f"Done reason: {result.get('done_reason')}; "
            f"Generated tokens: {result.get('eval_count')}"
        )

    return answer, result, elapsed


# ==================================================
# 6. SAVE EACH RESULT
# ==================================================

def append_result(output_file, record):

    with open(
        output_file,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )


# ==================================================
# 7. PROCESS ONE MODEL
# ==================================================

def run_model(model):

    if model not in MODELS:
        raise ValueError(f"Unsupported model: {model}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_name = MODELS[model]

    output_file = (
        OUTPUT_DIR / f"{model_name}_answers.jsonl"
    )

    if not EVIDENCE_FILE.exists():
        raise FileNotFoundError(EVIDENCE_FILE)

    evidence = load_evidence()

    completed_ids = set()

    if output_file.exists():

        with open(
            output_file,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                if line.strip():
                    record = json.loads(line)
                    completed_ids.add(record["question_id"])

    questions = (
        evidence[["question_id", "question"]]
        .drop_duplicates()
        .sort_values("question_id")
    )

    print("=" * 70)
    print("RAG ANSWER GENERATION")
    print("=" * 70)

    print("Model:", model)
    print("Total questions:", len(questions))
    print("Already completed:", len(completed_ids))
    print("Remaining:", len(questions) - len(completed_ids))
    print("Output:", output_file)

    if len(completed_ids) > 80:
        raise ValueError("Unexpected number of saved results.")

    if len(completed_ids) == 80:
        print("\nAll 80 questions are already completed.")
        return

    # Check Ollama availability.
    try:
        tags_response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=15
        )
        tags_response.raise_for_status()

    except requests.RequestException as error:
        raise RuntimeError(
            "Ollama is unavailable. Check that Ollama is running."
        ) from error

    available_models = {
        item["name"]
        for item in tags_response.json().get("models", [])
    }

    if model not in available_models:
        raise RuntimeError(
            f"{model} is not installed. "
            f"Available models: {sorted(available_models)}"
        )

    run_start = time.perf_counter()

    for number, row in enumerate(
        questions.to_dict("records"),
        start=1
    ):

        question_id = str(row["question_id"])

        if question_id in completed_ids:
            continue

        question_rows = evidence[
            evidence["question_id"].astype(str) == question_id
        ]

        question, context, sources = prepare_question(
            question_rows
        )

        prompt = build_prompt(question, context)

        print(
            f"\n[{number}/80] {question_id}: "
            f"{question[:100]}",
            flush=True
        )

        answer, result, elapsed = generate_answer(
            model,
            prompt
        )

        record = {
            "question_id": question_id,
            "model": model,
            "question": question,
            "answer": answer,
            "source_passages": sources,
            "retrieved_passage_count": len(sources),
            "latency_seconds": round(elapsed, 3),
            "total_duration_ns": result.get("total_duration"),
            "load_duration_ns": result.get("load_duration"),
            "prompt_tokens": result.get("prompt_eval_count"),
            "generated_tokens": result.get("eval_count"),
            "generation_timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        }

        append_result(output_file, record)

        completed_ids.add(question_id)

        print("Answer:", answer)
        print("Latency:", round(elapsed, 2), "seconds")
        print("Saved:", len(completed_ids), "/ 80")

    total_elapsed = time.perf_counter() - run_start

    print("\n" + "=" * 70)
    print("MODEL GENERATION COMPLETED")
    print("=" * 70)

    print("Model:", model)
    print("Completed questions:", len(completed_ids))
    print("Total elapsed time:", round(total_elapsed, 2), "seconds")
    print("Results:", output_file)


# ==================================================
# 8. MAIN
# ==================================================

def main():

    parser = argparse.ArgumentParser(
        description="Generate RAG answers using local Ollama models."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODELS.keys()),
        help="Select one installed Ollama model."
    )

    args = parser.parse_args()

    run_model(args.model)


if __name__ == "__main__":
    main()
