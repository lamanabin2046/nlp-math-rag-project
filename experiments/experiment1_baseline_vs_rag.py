"""
experiments/experiment1_baseline_vs_rag.py
───────────────────────────────────────────
Experiment 1 — Baseline LLM vs RAG

For each of 20 test questions, runs 6 configurations:
  - GPT-4o        Baseline
  - GPT-4o        RAG
  - Claude Opus   Baseline
  - Claude Opus   RAG
  - Qwen2-Math    Baseline  (via Ollama local)
  - Qwen2-Math    RAG       (via Ollama local)

Metrics per response:
  - BLEU
  - ROUGE-L
  - BERTScore
  - Step correctness (heuristic)
  - Response latency (seconds)

Output: results/experiment1_results.json
        results/experiment1_summary.csv

Run:
    python experiments/experiment1_baseline_vs_rag.py
"""

import json
import time
import csv
import re
import sys
import urllib.request
from pathlib import Path
from datetime import datetime

import nltk
import numpy as np
from tqdm import tqdm

import openai
import anthropic

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY,
    TESTSET_DIR, PROCESSED_DIR, CHROMA_DIR, CHROMA_COLLECTION,
    BERT_MODEL, CLIP_DIM, DESC_DIM,
)

# ── Download NLTK data ────────────────────────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Results directory ─────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────
N_QUESTIONS  = 20
TOP_K        = 5
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2-math"


# ══════════════════════════════════════════════════════════════════════════
# 1. RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════

def load_retriever():
    print("Loading BERT for query embedding …")
    bert = SentenceTransformer(BERT_MODEL)

    print("Connecting to ChromaDB …")
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(CHROMA_COLLECTION)
    print(f"  Collection: {collection.count()} documents\n")

    return bert, collection


def retrieve(question: str, bert, collection, top_k: int = TOP_K) -> list[dict]:
    text_emb = bert.encode([question], normalize_embeddings=True)[0]
    pad      = np.zeros(CLIP_DIM + DESC_DIM, dtype=np.float32)
    query    = np.concatenate([text_emb, pad]).tolist()

    results = collection.query(
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


def format_context(contexts: list[dict]) -> str:
    lines = ["Here are relevant worked examples from the Nepal Grade 10 Mathematics textbook:\n"]
    for i, ctx in enumerate(contexts, 1):
        lines.append(f"Example {i} (Unit {ctx['chapter_num']} — {ctx['chapter_title']}, p.{ctx['page']}):")
        lines.append(ctx["text"])
        lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# 2. PROMPTS
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


def build_prompt(question: str, use_rag: bool, contexts: list[dict] = None) -> str:
    if use_rag and contexts:
        return RAG_PROMPT.format(
            context=format_context(contexts),
            question=question,
        )
    return BASELINE_PROMPT.format(question=question)


# ══════════════════════════════════════════════════════════════════════════
# 3. MODEL CALLS
# ══════════════════════════════════════════════════════════════════════════

def call_gpt4o(prompt: str) -> tuple[str, float]:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    start  = time.time()
    resp   = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.2,
    )
    latency = time.time() - start
    return resp.choices[0].message.content.strip(), latency


def call_claude(prompt: str) -> tuple[str, float]:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    start  = time.time()
    resp   = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start
    return resp.content[0].text.strip(), latency


def call_qwen(prompt: str) -> tuple[str, float]:
    """Call Qwen2-Math via Ollama local API."""
    start   = time.time()
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
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


MODEL_CALLERS = {
    "gpt-4o":      call_gpt4o,
    "claude-opus": call_claude,
    "qwen2-math":  call_qwen,
}


# ══════════════════════════════════════════════════════════════════════════
# 4. METRICS
# ══════════════════════════════════════════════════════════════════════════

def compute_bleu(reference: str, hypothesis: str) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    smoothie   = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


# Load BERTScore model once globally to avoid reloading per response
_bertscore_loaded = False

def compute_bertscore(reference: str, hypothesis: str) -> float:
    from bert_score import score as bscore
    P, R, F = bscore(
        [hypothesis], [reference],
        lang="en", verbose=False,
        model_type="distilbert-base-uncased",
    )
    return F[0].item()


def compute_step_correctness(response: str) -> float:
    score = 0.0
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    if len(lines) >= 3:
        score += 0.2
    step_pattern = re.compile(r"(step\s*\d+|\d+[\.\)]|\(i+\)|∴|therefore|hence)", re.I)
    steps_found  = len(step_pattern.findall(response))
    if steps_found >= 2:
        score += 0.3
    elif steps_found >= 1:
        score += 0.15
    math_pattern = re.compile(r"[\d]+.*=|=.*[\d]+")
    if math_pattern.search(response):
        score += 0.2
    answer_pattern = re.compile(r"(therefore|hence|answer|∴|final|result)", re.I)
    if answer_pattern.search(response):
        score += 0.2
    if len(response.split()) >= 30:
        score += 0.1
    return round(min(score, 1.0), 3)


def compute_metrics(response: str, question: str) -> dict:
    bleu    = compute_bleu(question, response)
    rouge_l = compute_rouge_l(question, response)
    bert_f  = compute_bertscore(question, response)
    steps   = compute_step_correctness(response)
    return {
        "bleu":             round(bleu, 4),
        "rouge_l":          round(rouge_l, 4),
        "bertscore":        round(bert_f, 4),
        "step_correctness": steps,
    }


# ══════════════════════════════════════════════════════════════════════════
# 5. MAIN EXPERIMENT LOOP
# ══════════════════════════════════════════════════════════════════════════

def run_experiment():
    # ── Check Ollama is running ───────────────────────────────────────────
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        urllib.request.urlopen(req, timeout=5)
        print("✓ Ollama is running\n")
    except Exception:
        print("ERROR: Ollama is not running.")
        print("Start it by opening a new terminal and running: ollama serve")
        return

    # ── Load test questions ───────────────────────────────────────────────
    test_path = TESTSET_DIR / "test_set.json"
    with open(test_path, encoding="utf-8") as f:
        all_questions = json.load(f)

    questions = all_questions[:N_QUESTIONS]
    print(f"Loaded {len(questions)} questions for Experiment 1\n")

    # ── Load retriever ────────────────────────────────────────────────────
    bert, collection = load_retriever()

    # ── Define configurations ─────────────────────────────────────────────
    configs = [
        ("gpt-4o",      False, "Baseline"),
        ("gpt-4o",      True,  "RAG"),
        ("claude-opus", False, "Baseline"),
        ("claude-opus", True,  "RAG"),
        ("qwen2-math",  False, "Baseline"),
        ("qwen2-math",  True,  "RAG"),
    ]

    all_results = []
    total = len(configs) * len(questions)
    print(f"Running {len(configs)} configs × {len(questions)} questions = {total} total\n")
    print("=" * 60)

    for model_name, use_rag, setting in configs:
        print(f"\n▶  {model_name.upper()}  —  {setting}")
        print("-" * 40)

        caller = MODEL_CALLERS[model_name]

        for q in tqdm(questions, desc=f"{model_name} {setting}"):
            question_text = q["text"]

            contexts = None
            if use_rag:
                contexts = retrieve(question_text, bert, collection)

            prompt   = build_prompt(question_text, use_rag, contexts)
            response = ""
            latency  = 0.0
            error    = None

            for attempt in range(3):
                try:
                    response, latency = caller(prompt)
                    break
                except Exception as e:
                    error = str(e)
                    print(f"  Error (attempt {attempt+1}): {e}")
                    time.sleep(2 ** attempt)

            if not response:
                response = f"ERROR: {error}"
                latency  = 0.0

            metrics = compute_metrics(response, question_text)

            result = {
                "test_id":          q["test_id"],
                "question":         question_text,
                "chapter_type":     q["chapter_type"],
                "difficulty":       q["difficulty"],
                "question_type":    q["question_type"],
                "model":            model_name,
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
            }

            all_results.append(result)

            out_path = RESULTS_DIR / "experiment1_results.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

        time.sleep(1)

    out_path = RESULTS_DIR / "experiment1_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n\nFull results saved → {out_path}")

    print_summary(all_results)
    save_summary_csv(all_results)


def print_summary(results: list[dict]):
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        key = (r["model"], r["setting"])
        grouped[key].append(r)

    print("\n" + "=" * 70)
    print(f"{'Model':<15} {'Setting':<10} {'BLEU':>7} {'ROUGE-L':>8} {'BERT':>7} {'Steps':>7} {'Latency':>8}")
    print("-" * 70)

    for (model, setting), rows in sorted(grouped.items()):
        bleu    = np.mean([r["bleu"]             for r in rows])
        rouge   = np.mean([r["rouge_l"]          for r in rows])
        bert    = np.mean([r["bertscore"]         for r in rows])
        steps   = np.mean([r["step_correctness"] for r in rows])
        latency = np.mean([r["latency_sec"]       for r in rows])
        print(f"{model:<15} {setting:<10} {bleu:>7.4f} {rouge:>8.4f} {bert:>7.4f} {steps:>7.4f} {latency:>7.2f}s")

    print("=" * 70)


def save_summary_csv(results: list[dict]):
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


if __name__ == "__main__":
    run_experiment()