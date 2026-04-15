"""
experiments/experiment1_add_gemini.py
───────────────────────────────────────
Adds Gemini 2.5 Flash (Baseline + RAG) to existing Experiment 1 results.
Does NOT re-run GPT-4o, Claude, or Qwen.

Run:
    python experiments/experiment1_add_gemini.py
"""

import json
import time
import csv
import re
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

import nltk
from google import genai
from google.genai import types
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GEMINI_API_KEY,
    TESTSET_DIR, CHROMA_DIR, CHROMA_COLLECTION,
    BERT_MODEL, CLIP_DIM, DESC_DIM,
)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

RESULTS_DIR   = Path(__file__).parent.parent / "results"
N_QUESTIONS   = 20
TOP_K         = 5
GEMINI_MODEL  = "gemini-2.5-flash"


# ══════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════

def load_retriever():
    print("Loading BERT …")
    bert = SentenceTransformer(BERT_MODEL)
    print("Connecting to ChromaDB …")
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(CHROMA_COLLECTION)
    print(f"  {collection.count()} documents\n")
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
            "chapter_type":  meta.get("chapter_type"),
            "page":          meta.get("page"),
        })
    return contexts


def format_context(contexts):
    lines = ["Here are relevant worked examples from the Nepal Grade 10 "
             "Mathematics textbook:\n"]
    for i, ctx in enumerate(contexts, 1):
        lines.append(
            f"Example {i} (Unit {ctx['chapter_num']} — "
            f"{ctx['chapter_title']}, p.{ctx['page']}):"
        )
        lines.append(ctx["text"])
        lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════

BASELINE_PROMPT = """You are a mathematics tutor for Grade 10 students in Nepal.
Solve the following problem with clear step-by-step working.
Show every step and give the final answer clearly.

Problem: {question}

Solution:"""

RAG_PROMPT = """You are a mathematics tutor for Grade 10 students in Nepal.
Use the worked examples below as reference to solve the problem.
Show clear step-by-step working aligned with the Nepal curriculum.

{context}

Now solve this problem step by step:
Problem: {question}

Solution:"""


def build_prompt(question, use_rag, contexts=None):
    if use_rag and contexts:
        return RAG_PROMPT.format(
            context=format_context(contexts),
            question=question,
        )
    return BASELINE_PROMPT.format(question=question)


# ══════════════════════════════════════════════════════════════════════════
# GEMINI CALL
# ══════════════════════════════════════════════════════════════════════════

def call_gemini(prompt: str) -> tuple[str, float]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    start  = time.time()
    resp   = client.models.generate_content(
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
# METRICS
# ══════════════════════════════════════════════════════════════════════════

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


def compute_bertscore(reference, hypothesis):
    from bert_score import score as bscore
    P, R, F = bscore(
        [hypothesis], [reference],
        lang="en", verbose=False,
        model_type="distilbert-base-uncased",
    )
    return F[0].item()


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
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════

def print_summary(results):
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["model"], r["setting"])].append(r)

    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'Setting':<10} {'BLEU':>7} {'ROUGE-L':>8} "
          f"{'BERT':>7} {'Steps':>7} {'Latency':>8}")
    print("-" * 70)
    for (model, setting), rows in sorted(grouped.items()):
        print(f"{model:<20} {setting:<10} "
              f"{np.mean([r['bleu'] for r in rows]):>7.4f} "
              f"{np.mean([r['rouge_l'] for r in rows]):>8.4f} "
              f"{np.mean([r['bertscore'] for r in rows]):>7.4f} "
              f"{np.mean([r['step_correctness'] for r in rows]):>7.4f} "
              f"{np.mean([r['latency_sec'] for r in rows]):>7.2f}s")
    print("=" * 70)


def save_summary_csv(results):
    csv_path = RESULTS_DIR / "experiment1_summary.csv"
    fields   = [
        "test_id", "chapter_type", "difficulty", "question_type",
        "model", "setting", "bleu", "rouge_l", "bertscore",
        "step_correctness", "latency_sec", "n_contexts",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})
    print(f"Summary CSV saved → {csv_path}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    # ── Load existing results ─────────────────────────────────────────────
    existing_path = RESULTS_DIR / "experiment1_results.json"
    if not existing_path.exists():
        print("ERROR: experiment1_results.json not found.")
        return

    with open(existing_path, encoding="utf-8") as f:
        existing_results = json.load(f)

    # Remove ALL old gemini results (any model name containing 'gemini')
    clean_results = [r for r in existing_results if "gemini" not in r["model"]]
    removed = len(existing_results) - len(clean_results)
    if removed > 0:
        print(f"Removed {removed} old Gemini results")

    print(f"Loaded {len(clean_results)} existing non-Gemini results")

    # Check if gemini-2.5-flash already fully done
    already_done = [r for r in existing_results if r["model"] == GEMINI_MODEL]
    if len(already_done) >= 40:
        print(f"Gemini already complete ({len(already_done)} results)")
        print_summary(existing_results)
        return

    # ── Load questions ────────────────────────────────────────────────────
    with open(TESTSET_DIR / "test_set.json", encoding="utf-8") as f:
        questions = json.load(f)[:N_QUESTIONS]

    # ── Load retriever ────────────────────────────────────────────────────
    bert, collection = load_retriever()

    # ── Run Gemini Baseline + RAG ─────────────────────────────────────────
    new_results = []
    configs = [
        (False, "Baseline"),
        (True,  "RAG"),
    ]

    for use_rag, setting in configs:
        print(f"\n▶  {GEMINI_MODEL.upper()}  —  {setting}")
        print("-" * 40)

        for q in tqdm(questions, desc=f"gemini {setting}"):
            question_text = q["text"]
            contexts      = retrieve(question_text, bert, collection) \
                            if use_rag else None
            prompt        = build_prompt(question_text, use_rag, contexts)

            response = ""
            latency  = 0.0
            error    = None

            for attempt in range(3):
                try:
                    response, latency = call_gemini(prompt)
                    break
                except Exception as e:
                    error = str(e)
                    print(f"  Error (attempt {attempt+1}): {e}")
                    time.sleep(2 ** attempt)

            if not response:
                response = f"ERROR: {error}"
                latency  = 0.0

            metrics = compute_metrics(response, question_text)

            new_results.append({
                "test_id":          q["test_id"],
                "question":         question_text,
                "chapter_type":     q["chapter_type"],
                "difficulty":       q["difficulty"],
                "question_type":    q["question_type"],
                "model":            GEMINI_MODEL,
                "setting":          setting,
                "use_rag":          use_rag,
                "response":         response,
                "latency_sec":      round(latency, 3),
                "bleu":             metrics["bleu"],
                "rouge_l":          metrics["rouge_l"],
                "bertscore":        metrics["bertscore"],
                "step_correctness": metrics["step_correctness"],
                "n_contexts":       len(contexts) if contexts else 0,
                "timestamp":        datetime.now().isoformat(),
            })

            # Save after every response
            all_results = clean_results + new_results
            with open(existing_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

        time.sleep(1)

    # ── Final save ────────────────────────────────────────────────────────
    all_results = clean_results + new_results
    with open(existing_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nMerged results saved → {existing_path}")
    print(f"Total responses: {len(all_results)}")

    print_summary(all_results)
    save_summary_csv(all_results)


if __name__ == "__main__":
    main()