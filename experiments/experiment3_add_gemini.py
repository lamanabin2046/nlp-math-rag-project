"""
experiments/experiment3_add_gemini.py
───────────────────────────────────────
Adds Gemini 2.5 Flash to Experiment 3 (Agentic Workflows).
Runs 3 workflows × 33 hard/proof questions = 99 new responses.
Does NOT re-run GPT-4o or Claude.

Run:
    python experiments/experiment3_add_gemini.py
"""

import json
import time
import csv
import sys
import re
import numpy as np
from pathlib import Path
from datetime import datetime

import nltk
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GEMINI_API_KEY,
    TESTSET_DIR, CHROMA_DIR, CHROMA_COLLECTION,
    BERT_MODEL, CLIP_DIM, DESC_DIM,
)

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

RESULTS_DIR        = Path(__file__).parent.parent / "results"
TOP_K              = 5
GEMINI_MODEL       = "gemini-2.5-flash-lite"
SELF_RAG_THRESHOLD = 3.5
SELF_RAG_MAX_ITER  = 3
CRAG_THRESHOLD     = 0.6
WORKFLOWS          = ["static_rag", "self_rag", "crag"]


# ══════════════════════════════════════════════════════════════════════════
# 1. RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════

def load_retriever():
    print(f"Loading BERT: {BERT_MODEL}")
    bert = SentenceTransformer(BERT_MODEL)
    print("Connecting to ChromaDB …")
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(CHROMA_COLLECTION)
    print(f"  {collection.count()} documents indexed\n")
    return bert, collection


def retrieve(question, bert, collection, top_k=TOP_K):
    text_emb = bert.encode([question], normalize_embeddings=True)[0]
    pad      = np.zeros(CLIP_DIM + DESC_DIM, dtype=np.float32)
    query    = np.concatenate([text_emb, pad]).tolist()
    results  = collection.query(
        query_embeddings=[query],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    contexts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        contexts.append({
            "text":          doc,
            "chapter_num":   meta.get("chapter_num"),
            "chapter_title": meta.get("chapter_title"),
            "page":          meta.get("page"),
        })
    return contexts


def format_context(contexts):
    lines = ["Relevant worked examples from the Nepal Grade 10 Mathematics textbook:\n"]
    for i, ctx in enumerate(contexts, 1):
        lines.append(
            f"Example {i} (Unit {ctx['chapter_num']} — "
            f"{ctx['chapter_title']}, p.{ctx['page']}):"
        )
        lines.append(ctx["text"])
        lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# 2. PROMPTS
# ══════════════════════════════════════════════════════════════════════════

SOLVE_PROMPT = """You are a mathematics tutor for Grade 10 students in Nepal.
Use the worked examples below as reference to solve the problem.
Show clear step-by-step working aligned with the Nepal curriculum.

{context}

Now solve this problem step by step:
Problem: {question}

Solution:"""

SELF_EVAL_PROMPT = """You are an expert mathematics educator reviewing a student's solution.

Original question: {question}

Generated solution:
{solution}

Score this solution from 1 to 5:
- 5: All steps correct, clear working, correct final answer
- 4: Minor errors, mostly correct
- 3: Correct approach but significant gaps
- 2: Partially correct, multiple errors
- 1: Incorrect approach

Respond ONLY with:
SCORE: <integer 1-5>
REASON: <one sentence>
MISSING: <what is missing or 'nothing' if score is 5>"""

CRAG_GRADE_PROMPT = """You are grading whether a worked example is relevant to a student's question.

Student question: {question}

Worked example:
{document}

Rate relevance from 0.0 to 1.0:
  1.0 = perfectly relevant
  0.7 = mostly relevant
  0.4 = somewhat relevant
  0.1 = barely relevant
  0.0 = completely irrelevant

Respond ONLY with:
RELEVANCE: <float 0.0 to 1.0>
REASON: <one brief sentence>"""

QUERY_TRANSFORM_PROMPT = """The original question retrieved poor-quality context.
Rephrase the question to better match worked examples in a Grade 10 Nepal mathematics textbook.
Focus on the specific mathematical operation or procedure required.

Original question: {question}

Rephrased question (for textbook search only, do not solve):"""


# ══════════════════════════════════════════════════════════════════════════
# 3. GEMINI CALL
# ══════════════════════════════════════════════════════════════════════════

def call_gemini(prompt: str) -> tuple[str, float]:
    client  = genai.Client(api_key=GEMINI_API_KEY)
    start   = time.time()
    resp    = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
        )
    )
    latency = time.time() - start
    return resp.text.strip(), latency


# ══════════════════════════════════════════════════════════════════════════
# 4. WORKFLOW IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════

def run_static_rag(question, bert, collection):
    t0       = time.time()
    contexts = retrieve(question, bert, collection)
    prompt   = SOLVE_PROMPT.format(
        context=format_context(contexts), question=question
    )
    response, _ = call_gemini(prompt)
    return {
        "response":       response,
        "latency_sec":    round(time.time() - t0, 3),
        "iterations":     1,
        "n_retrieves":    1,
        "final_score":    None,
        "mean_relevance": None,
        "contexts":       contexts,
    }


def parse_score(text: str) -> float:
    m = re.search(r"SCORE:\s*([1-5])", text)
    return float(m.group(1)) if m else 3.0


def run_self_rag(question, bert, collection):
    t0         = time.time()
    contexts   = retrieve(question, bert, collection)
    iterations = 0
    n_retrieves = 1
    response    = ""
    final_score = None

    for i in range(SELF_RAG_MAX_ITER):
        iterations += 1
        prompt      = SOLVE_PROMPT.format(
            context=format_context(contexts), question=question
        )
        response, _ = call_gemini(prompt)

        # Self-evaluate
        eval_prompt  = SELF_EVAL_PROMPT.format(
            question=question, solution=response
        )
        eval_text, _ = call_gemini(eval_prompt)
        score        = parse_score(eval_text)
        final_score  = score

        if score >= SELF_RAG_THRESHOLD:
            break

        # Re-retrieve with original question
        if i < SELF_RAG_MAX_ITER - 1:
            contexts     = retrieve(question, bert, collection)
            n_retrieves += 1

    return {
        "response":       response,
        "latency_sec":    round(time.time() - t0, 3),
        "iterations":     iterations,
        "n_retrieves":    n_retrieves,
        "final_score":    final_score,
        "mean_relevance": None,
        "contexts":       contexts,
    }


def parse_relevance(text: str) -> float:
    m = re.search(r"RELEVANCE:\s*([0-9.]+)", text)
    try:
        return float(m.group(1)) if m else 0.5
    except ValueError:
        return 0.5


def run_crag(question, bert, collection):
    t0          = time.time()
    contexts    = retrieve(question, bert, collection)
    n_retrieves = 1

    # Grade each retrieved document
    scores = []
    for ctx in contexts:
        grade_prompt  = CRAG_GRADE_PROMPT.format(
            question=question, document=ctx["text"]
        )
        grade_text, _ = call_gemini(grade_prompt)
        scores.append(parse_relevance(grade_text))

    mean_relevance = float(np.mean(scores)) if scores else 0.0

    # Corrective re-retrieve if quality is low
    if mean_relevance < CRAG_THRESHOLD:
        transform_prompt = QUERY_TRANSFORM_PROMPT.format(question=question)
        new_query, _     = call_gemini(transform_prompt)
        contexts         = retrieve(new_query.strip(), bert, collection)
        n_retrieves     += 1

    prompt      = SOLVE_PROMPT.format(
        context=format_context(contexts), question=question
    )
    response, _ = call_gemini(prompt)

    return {
        "response":       response,
        "latency_sec":    round(time.time() - t0, 3),
        "iterations":     1,
        "n_retrieves":    n_retrieves,
        "final_score":    None,
        "mean_relevance": round(mean_relevance, 4),
        "contexts":       contexts,
    }


WORKFLOW_RUNNERS = {
    "static_rag": run_static_rag,
    "self_rag":   run_self_rag,
    "crag":       run_crag,
}


# ══════════════════════════════════════════════════════════════════════════
# 5. METRICS
# ══════════════════════════════════════════════════════════════════════════

_bert_scorer = None

def compute_bertscore(reference, hypothesis):
    global _bert_scorer
    from bert_score import BERTScorer
    if _bert_scorer is None:
        _bert_scorer = BERTScorer(
            model_type="distilbert-base-uncased",
            lang="en"
        )
    P, R, F = _bert_scorer.score([hypothesis], [reference])
    return F[0].item()


def compute_bleu(reference, hypothesis):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    return sentence_bleu([ref_tokens], hyp_tokens,
                         smoothing_function=SmoothingFunction().method4)


def compute_rouge_l(reference, hypothesis):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, hypothesis)["rougeL"].fmeasure


def compute_step_correctness(response):
    score = 0.0
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    if len(lines) >= 3:
        score += 0.2
    step_pattern = re.compile(
        r"(step\s*\d+|\d+[\.\)]|\(i+\)|∴|therefore|hence)", re.I)
    steps_found = len(step_pattern.findall(response))
    if steps_found >= 2:
        score += 0.3
    elif steps_found >= 1:
        score += 0.15
    if re.search(r"[\d]+.*=|=.*[\d]+", response):
        score += 0.2
    if re.search(r"(therefore|hence|answer|∴|final|result)", response, re.I):
        score += 0.2
    if len(response.split()) >= 30:
        score += 0.1
    return round(min(score, 1.0), 3)


def compute_metrics(response, question):
    return {
        "bleu":             round(compute_bleu(question, response), 4),
        "rouge_l":          round(compute_rouge_l(question, response), 4),
        "bertscore":        round(compute_bertscore(question, response), 4),
        "step_correctness": compute_step_correctness(response),
    }


# ══════════════════════════════════════════════════════════════════════════
# 6. SUMMARY
# ══════════════════════════════════════════════════════════════════════════

def print_summary(results):
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["model"], r["workflow"])].append(r)

    print("\n" + "=" * 75)
    print(f"{'Model':<20} {'Workflow':<12} {'BLEU':>7} {'ROUGE-L':>8} "
          f"{'BERT':>7} {'Steps':>7} {'Latency':>8} {'Iters':>6}")
    print("-" * 75)
    for (model, workflow), rows in sorted(grouped.items()):
        print(
            f"{model:<20} {workflow:<12} "
            f"{np.mean([r['bleu'] for r in rows]):>7.4f} "
            f"{np.mean([r['rouge_l'] for r in rows]):>8.4f} "
            f"{np.mean([r['bertscore'] for r in rows]):>7.4f} "
            f"{np.mean([r['step_correctness'] for r in rows]):>7.4f} "
            f"{np.mean([r['latency_sec'] for r in rows]):>7.2f}s "
            f"{np.mean([r.get('iterations',1) for r in rows]):>5.1f}"
        )
    print("=" * 75)


def save_summary_csv(results):
    fields = [
        "test_id", "chapter_type", "difficulty", "question_type",
        "model", "workflow", "bleu", "rouge_l", "bertscore",
        "step_correctness", "latency_sec", "iterations",
        "n_retrieves", "self_eval_score", "mean_relevance",
    ]
    path = RESULTS_DIR / "experiment3_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fields})
    print(f"Summary CSV saved → {path}")


# ══════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    # ── Load existing results ─────────────────────────────────────────────
    results_path = RESULTS_DIR / "experiment3_results.json"
    if not results_path.exists():
        print("ERROR: experiment3_results.json not found.")
        return

    with open(results_path, encoding="utf-8") as f:
        existing_results = json.load(f)

    # Remove any previous partial Gemini results
    clean_results = [r for r in existing_results
                     if r.get("model") != GEMINI_MODEL]
    removed = len(existing_results) - len(clean_results)
    if removed > 0:
        print(f"Removed {removed} previous Gemini results — rerunning cleanly")

    print(f"Loaded {len(clean_results)} existing results "
          f"(GPT-4o + Claude)\n")

    # ── Load questions ────────────────────────────────────────────────────
    with open(TESTSET_DIR / "test_set.json", encoding="utf-8") as f:
        all_questions = json.load(f)

    hard_questions = [
        q for q in all_questions
        if q.get("difficulty") == "hard" or q.get("question_type") == "proof"
    ]
    if not hard_questions:
        hard_questions = all_questions[:20]

    print(f"Hard + proof questions: {len(hard_questions)}")
    print(f"Running 3 workflows × {len(hard_questions)} questions = "
          f"{3 * len(hard_questions)} new responses\n")
    print("=" * 65)

    # ── Load retriever ────────────────────────────────────────────────────
    bert, collection = load_retriever()

    # ── Run 3 workflows for Gemini ────────────────────────────────────────
    new_results = []

    for workflow_name in WORKFLOWS:
        runner = WORKFLOW_RUNNERS[workflow_name]
        label  = f"GEMINI / {workflow_name.replace('_', ' ').title()}"
        print(f"\n▶  {label}")
        print("-" * 40)

        for q in tqdm(hard_questions, desc=label[:35]):
            result_info = {}
            error       = None

            for attempt in range(3):
                try:
                    result_info = runner(q["text"], bert, collection)
                    break
                except Exception as e:
                    error = str(e)
                    print(f"  Error (attempt {attempt+1}): {e}")
                    time.sleep(2 ** attempt)

            if not result_info.get("response"):
                result_info = {
                    "response":       f"ERROR: {error}",
                    "latency_sec":    0.0,
                    "iterations":     0,
                    "n_retrieves":    0,
                    "final_score":    None,
                    "mean_relevance": None,
                }

            response = result_info["response"]
            metrics  = compute_metrics(response, q["text"])

            record = {
                "test_id":          q["test_id"],
                "question":         q["text"],
                "chapter_type":     q["chapter_type"],
                "difficulty":       q["difficulty"],
                "question_type":    q["question_type"],
                "model":            GEMINI_MODEL,
                "workflow":         workflow_name,
                "response":         response,
                "latency_sec":      result_info.get("latency_sec", 0.0),
                "iterations":       result_info.get("iterations", 1),
                "n_retrieves":      result_info.get("n_retrieves", 1),
                "self_eval_score":  result_info.get("final_score"),
                "mean_relevance":   result_info.get("mean_relevance"),
                "bleu":             metrics["bleu"],
                "rouge_l":          metrics["rouge_l"],
                "bertscore":        metrics["bertscore"],
                "step_correctness": metrics["step_correctness"],
                "timestamp":        datetime.now().isoformat(),
            }

            new_results.append(record)

            # Save after every response
            all_results = clean_results + new_results
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

        time.sleep(1)

    # ── Final save ────────────────────────────────────────────────────────
    all_results = clean_results + new_results
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nMerged results saved → {results_path}")
    print(f"Total responses: {len(all_results)}")

    print_summary(all_results)
    save_summary_csv(all_results)


if __name__ == "__main__":
    main()
