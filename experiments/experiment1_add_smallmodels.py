"""
experiments/experiment1_add_smallmodels.py
───────────────────────────────────────────
Adds small open-source models (Baseline + RAG) to existing Experiment 1
results, so they can be fairly compared against GPT-4o, Claude-Opus, and
Qwen2-Math.

Models added (all run locally via Ollama):
  - tinyllama       (1.1B)  — smallest possible, establishes lower bound
  - phi3:mini       (3.8B)  — Microsoft small model, strong reasoning
  - mistral:7b      (7B)    — standard small model benchmark

Does NOT re-run GPT-4o, Claude, Qwen, or Gemini.

Before running:
    1. Make sure Ollama is running:   ollama serve
    2. Pull the models:
         ollama pull tinyllama
         ollama pull phi3:mini
         ollama pull mistral:7b

Run:
    python experiments/experiment1_add_smallmodels.py
"""

import json
import time
import csv
import re
import sys
import urllib.request
import numpy as np
from pathlib import Path
from datetime import datetime

import nltk
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TESTSET_DIR, CHROMA_DIR, CHROMA_COLLECTION,
    BERT_MODEL, CLIP_DIM, DESC_DIM,
)

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RESULTS_DIR  = Path(__file__).parent.parent / "results"
N_QUESTIONS  = 20
TOP_K        = 5
OLLAMA_URL   = "http://localhost:11434/api/generate"

# Small models to add — name used in results JSON : ollama model tag
SMALL_MODELS = {
    "tinyllama":  "tinyllama",
    "phi3-mini":  "phi3:mini",
    "mistral-7b": "mistral:7b",
}


# ==============================================================================
# RETRIEVAL  (identical to experiment1_baseline_vs_rag.py)
# ==============================================================================

def load_retriever():
    print("Loading BERT ...")
    bert = SentenceTransformer(BERT_MODEL)
    print("Connecting to ChromaDB ...")
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
            f"Example {i} (Unit {ctx['chapter_num']} - "
            f"{ctx['chapter_title']}, p.{ctx['page']}):"
        )
        lines.append(ctx["text"])
        lines.append("")
    return "\n".join(lines)


# ==============================================================================
# PROMPTS  (identical to experiment1_baseline_vs_rag.py)
# ==============================================================================

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


# ==============================================================================
# OLLAMA CALL
# ==============================================================================

def call_ollama(prompt, ollama_model_tag):
    """
    Call any model running in Ollama.
    ollama_model_tag is the tag as known by Ollama, e.g. 'phi3:mini'.
    Returns (response_text, latency_seconds).
    """
    start   = time.time()
    payload = json.dumps({
        "model":  ollama_model_tag,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1024,   # max tokens to generate
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    latency = time.time() - start
    return result["response"].strip(), latency


# ==============================================================================
# METRICS  (identical to experiment1_baseline_vs_rag.py)
# ==============================================================================

def compute_bleu(reference, hypothesis):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    return sentence_bleu(
        [ref_tokens], hyp_tokens,
        smoothing_function=SmoothingFunction().method4
    )


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
        r"(step\s*\d+|\d+[\.\)]|\(i+\)|therefore|hence)", re.I)
    steps_found = len(step_pattern.findall(response))
    if steps_found >= 2:
        score += 0.3
    elif steps_found >= 1:
        score += 0.15
    if re.search(r"[\d]+.*=|=.*[\d]+", response):
        score += 0.2
    if re.search(r"(therefore|hence|answer|final|result)", response, re.I):
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


# ==============================================================================
# SUMMARY HELPERS
# ==============================================================================

def print_summary(results):
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["model"], r["setting"])].append(r)

    print("\n" + "=" * 75)
    print(f"{'Model':<20} {'Setting':<10} {'BLEU':>7} {'ROUGE-L':>8} "
          f"{'BERT':>7} {'Steps':>7} {'Latency':>8}")
    print("-" * 75)
    for (model, setting), rows in sorted(grouped.items()):
        print(
            f"{model:<20} {setting:<10} "
            f"{np.mean([r['bleu']             for r in rows]):>7.4f} "
            f"{np.mean([r['rouge_l']          for r in rows]):>8.4f} "
            f"{np.mean([r['bertscore']        for r in rows]):>7.4f} "
            f"{np.mean([r['step_correctness'] for r in rows]):>7.4f} "
            f"{np.mean([r['latency_sec']      for r in rows]):>7.2f}s"
        )
    print("=" * 75)


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
    print(f"Summary CSV saved -> {csv_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def check_ollama():
    """Make sure Ollama is reachable before starting."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        urllib.request.urlopen(req, timeout=5)
        print("Ollama is running\n")
        return True
    except Exception:
        print("ERROR: Ollama is not running.")
        print("Open a new terminal and run:  ollama serve")
        return False


def check_models_pulled():
    """
    List models available in Ollama and warn about any that are missing.
    Returns True only if all required models are present.
    """
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data   = json.loads(resp.read().decode("utf-8"))
            pulled = {m["name"].split(":")[0] for m in data.get("models", [])}

        all_present = True
        for friendly_name, tag in SMALL_MODELS.items():
            base = tag.split(":")[0]
            if base not in pulled:
                print(f"  WARNING: '{tag}' not found in Ollama.")
                print(f"           Run:  ollama pull {tag}")
                all_present = False
            else:
                print(f"  OK: {tag}")
        return all_present
    except Exception as e:
        print(f"Could not verify pulled models: {e}")
        return False


def main():
    # ── Preflight checks ──────────────────────────────────────────────────
    if not check_ollama():
        return

    print("Checking required Ollama models ...")
    if not check_models_pulled():
        print("\nSome models are missing. Pull them first (commands shown above).")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != "y":
            return

    # ── Load existing results ─────────────────────────────────────────────
    existing_path = RESULTS_DIR / "experiment1_results.json"
    if not existing_path.exists():
        print("ERROR: experiment1_results.json not found.")
        print("       Run experiment1_baseline_vs_rag.py first.")
        return

    with open(existing_path, encoding="utf-8") as f:
        existing_results = json.load(f)

    print(f"\nLoaded {len(existing_results)} existing results.")

    # ── Load questions ────────────────────────────────────────────────────
    with open(TESTSET_DIR / "test_set.json", encoding="utf-8") as f:
        questions = json.load(f)[:N_QUESTIONS]

    # ── Load retriever ────────────────────────────────────────────────────
    bert, collection = load_retriever()

    # ── Run each small model ──────────────────────────────────────────────
    new_results = []

    for friendly_name, ollama_tag in SMALL_MODELS.items():

        # Skip if already fully done (40 results = 20 questions x 2 settings)
        already = [r for r in existing_results if r["model"] == friendly_name]
        if len(already) >= 40:
            print(f"\n{friendly_name} already complete ({len(already)} results), skipping.")
            continue

        # Remove any partial previous run for this model
        existing_results = [r for r in existing_results
                            if r["model"] != friendly_name]

        for use_rag, setting in [(False, "Baseline"), (True, "RAG")]:
            print(f"\n>>> {friendly_name.upper()} ({ollama_tag})  --  {setting}")
            print("-" * 50)

            for q in tqdm(questions, desc=f"{friendly_name} {setting}"):
                question_text = q["text"]
                contexts      = retrieve(question_text, bert, collection) \
                                if use_rag else None
                prompt        = build_prompt(question_text, use_rag, contexts)

                response = ""
                latency  = 0.0
                error    = None

                for attempt in range(3):
                    try:
                        response, latency = call_ollama(prompt, ollama_tag)
                        break
                    except Exception as e:
                        error = str(e)
                        print(f"  Error (attempt {attempt + 1}): {e}")
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
                    "model":            friendly_name,
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

                # Save after every response so progress is never lost
                all_so_far = existing_results + new_results
                with open(existing_path, "w", encoding="utf-8") as f:
                    json.dump(all_so_far, f, ensure_ascii=False, indent=2)

        time.sleep(1)

    # ── Final save ────────────────────────────────────────────────────────
    all_results = existing_results + new_results
    with open(existing_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nMerged results saved -> {existing_path}")
    print(f"Total responses: {len(all_results)}")

    print_summary(all_results)
    save_summary_csv(all_results)


if __name__ == "__main__":
    main()