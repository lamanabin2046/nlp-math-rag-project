"""
experiments/experiment5_embedding_comparison.py
────────────────────────────────────────────────
Experiment 5 — Embedding Model Comparison (Extended)

Addresses TA/Professor comment #4:
  "You used CLIP embedding but it was trained on images of dogs and cats.
   Find alternatives more appropriate for your problem."

Compares FOUR retrieval collections on the same 20 test questions:

  Collection A (ORIGINAL):
    BERT text (768d) + CLIP image (512d) + BERT desc (768d) = 2048d
    Problem: CLIP trained on photo-caption pairs, not math diagrams.

  Collection B (BGE-M3):
    BGE-M3 (1024d) — retrieval-optimised, multilingual, math-aware.

  Collection C (E5-Large):
    intfloat/e5-large-v2 (1024d) — strong retrieval model, competitor to BGE-M3.

  Collection D (MiniLM):
    all-MiniLM-L6-v2 (384d) — lightweight fast baseline.

All collections use GPT-4o as the generation model (RAG setting only).

Output:
    results/experiment5_results.json
    results/experiment5_summary.csv

Run:
    python experiments/experiment5_embedding_comparison.py
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
import openai
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OPENAI_API_KEY,
    TESTSET_DIR, CHROMA_DIR, CHROMA_COLLECTION,
    BERT_MODEL, CLIP_DIM, DESC_DIM,
)

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Constants ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_QUESTIONS = 20
TOP_K       = 5
BERT_DIM    = 768

# Collection names
ORIGINAL_COLLECTION = CHROMA_COLLECTION
BGE_COLLECTION      = CHROMA_COLLECTION + "_bge"
E5_COLLECTION       = CHROMA_COLLECTION + "_e5"
MINI_COLLECTION     = CHROMA_COLLECTION + "_minilm"

# Model names
BGE_MODEL_NAME  = "BAAI/bge-m3"
E5_MODEL_NAME   = "intfloat/e5-large-v2"
MINI_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ==============================================================================
# SECTION 1 — BUILD COLLECTIONS
# ==============================================================================

def build_collection(chroma_client, model, collection_name, model_label):
    """
    Re-index all documents from the original ChromaDB collection
    using the given embedding model into a new named collection.
    Skips if already built.
    """
    original = chroma_client.get_collection(ORIGINAL_COLLECTION)
    n_docs   = original.count()

    # Check if collection already exists and is complete
    try:
        col = chroma_client.get_collection(collection_name)
        if col.count() >= n_docs:
            print(f"  [{model_label}] Collection already exists "
                  f"({col.count()} docs). Skipping build.")
            return col
        else:
            print(f"  [{model_label}] Incomplete collection. Rebuilding.")
            chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    print(f"  [{model_label}] Fetching {n_docs} documents from original ...")
    result    = original.get(include=["documents", "metadatas"])
    documents = result["documents"]
    metadatas = result["metadatas"]
    ids       = result["ids"]

    col = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    BATCH = 32
    print(f"  [{model_label}] Embedding {n_docs} documents ...")

    for start in tqdm(range(0, n_docs, BATCH),
                      desc=f"  {model_label} batches"):
        batch_docs  = documents[start : start + BATCH]
        batch_metas = metadatas[start : start + BATCH]
        batch_ids   = ids[start : start + BATCH]

        embeddings = model.encode(
            batch_docs,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        col.add(
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embeddings,
            ids=batch_ids,
        )

    print(f"  [{model_label}] Built: {col.count()} documents.\n")
    return col


# ==============================================================================
# SECTION 2 — RETRIEVAL
# ==============================================================================

def retrieve_original(question, bert, original_collection, top_k=TOP_K):
    """Original BERT+CLIP+DESC retrieval — zero-pads CLIP and DESC dims."""
    text_emb = bert.encode([question], normalize_embeddings=True)[0]
    pad      = np.zeros(CLIP_DIM + DESC_DIM, dtype=np.float32)
    query    = np.concatenate([text_emb, pad]).tolist()

    results = original_collection.query(
        query_embeddings=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return _parse_results(results)


def retrieve_dense(question, model, collection, top_k=TOP_K):
    """Generic dense retrieval for BGE-M3, E5-Large, MiniLM."""
    query = model.encode(
        [question], normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return _parse_results(results)


def _parse_results(results):
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":          doc,
            "similarity":    round(1 - (dist / 2), 4),
            "chapter_num":   meta.get("chapter_num"),
            "chapter_title": meta.get("chapter_title"),
            "chapter_type":  meta.get("chapter_type"),
            "page":          meta.get("page"),
        })
    return chunks


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
# SECTION 3 — PROMPT + GPT-4o
# ==============================================================================

RAG_PROMPT = """You are a mathematics tutor for Grade 10 students in Nepal.
Use the worked examples below as reference to solve the problem.
Show clear step-by-step working aligned with the Nepal curriculum.

{context}

Now solve this problem step by step:
Problem: {question}

Solution:"""


def call_gpt4o(question, contexts):
    prompt  = RAG_PROMPT.format(
        context=format_context(contexts),
        question=question,
    )
    client  = openai.OpenAI(api_key=OPENAI_API_KEY)
    start   = time.time()
    resp    = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.2,
    )
    latency = time.time() - start
    return resp.choices[0].message.content.strip(), latency


# ==============================================================================
# SECTION 4 — METRICS
# ==============================================================================

def compute_bleu(reference, hypothesis):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    return sentence_bleu(
        [ref_tokens], hyp_tokens,
        smoothing_function=SmoothingFunction().method4,
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
# SECTION 5 — SUMMARY + SAVE
# ==============================================================================

def print_summary(results):
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[r["embedding"]].append(r)

    order = [
        "BERT+CLIP+DESC (Original)",
        "MiniLM-L6 (384d)",
        "BGE-M3 (1024d)",
        "E5-Large (1024d)",
    ]

    print("\n" + "=" * 85)
    print(f"{'Embedding':<28} {'BLEU':>7} {'ROUGE-L':>8} {'BERT':>7} "
          f"{'Steps':>7} {'MeanSim':>8} {'Latency':>9}")
    print("-" * 85)
    for embedding in order:
        rows = grouped.get(embedding, [])
        if not rows:
            continue
        print(
            f"{embedding:<28} "
            f"{np.mean([r['bleu']             for r in rows]):>7.4f} "
            f"{np.mean([r['rouge_l']          for r in rows]):>8.4f} "
            f"{np.mean([r['bertscore']        for r in rows]):>7.4f} "
            f"{np.mean([r['step_correctness'] for r in rows]):>7.4f} "
            f"{np.mean([r['mean_sim']         for r in rows]):>8.4f} "
            f"{np.mean([r['latency_sec']      for r in rows]):>8.2f}s"
        )
    print("=" * 85)

    # Chapter type breakdown
    chapter_types = sorted(set(r["chapter_type"] for r in results))
    embeddings    = [e for e in order if e in grouped]

    print("\nSTEP CORRECTNESS BY CHAPTER TYPE")
    print("-" * 85)
    header = f"  {'Chapter':<20}"
    for emb in embeddings:
        short = emb[:20]
        header += f" {short:>20}"
    print(header)
    for ctype in chapter_types:
        row = f"  {ctype:<20}"
        for emb in embeddings:
            subset = [r for r in results
                      if r["chapter_type"] == ctype and r["embedding"] == emb]
            val = np.mean([r["step_correctness"] for r in subset]) if subset else 0
            row += f" {val:>20.4f}"
        print(row)

    print("\nMEAN RETRIEVAL SIMILARITY BY CHAPTER TYPE")
    print("-" * 85)
    print(header)
    for ctype in chapter_types:
        row = f"  {ctype:<20}"
        for emb in embeddings:
            subset = [r for r in results
                      if r["chapter_type"] == ctype and r["embedding"] == emb]
            val = np.mean([r["mean_sim"] for r in subset]) if subset else 0
            row += f" {val:>20.4f}"
        print(row)

    # Key comparison
    print("\n" + "=" * 85)
    print("KEY COMPARISON vs ORIGINAL")
    print("=" * 85)
    print(f"  {'Embedding':<28} {'MeanSim':>10} {'SimDelta':>10} "
          f"{'Steps':>8} {'StepDelta':>10}")
    print("  " + "-" * 70)
    orig_sim  = np.mean([r["mean_sim"]         for r in grouped.get("BERT+CLIP+DESC (Original)", [])])
    orig_step = np.mean([r["step_correctness"] for r in grouped.get("BERT+CLIP+DESC (Original)", [])])
    for emb in order:
        rows = grouped.get(emb, [])
        if not rows:
            continue
        sim  = np.mean([r["mean_sim"]         for r in rows])
        step = np.mean([r["step_correctness"] for r in rows])
        print(f"  {emb:<28} {sim:>10.4f} {sim-orig_sim:>+10.4f} "
              f"{step:>8.4f} {step-orig_step:>+10.4f}")
    print("=" * 85)


def save_results(results):
    # JSON
    json_path = RESULTS_DIR / "experiment5_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved -> {json_path}")

    # CSV
    csv_path = RESULTS_DIR / "experiment5_summary.csv"
    fields   = [
        "test_id", "chapter_type", "difficulty", "question_type",
        "embedding", "bleu", "rouge_l", "bertscore",
        "step_correctness", "mean_sim", "latency_sec", "n_contexts",
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

def main():
    print("=" * 70)
    print("EXPERIMENT 5 — Embedding Model Comparison (Extended)")
    print("Comparing 4 embedding strategies for math retrieval")
    print("=" * 70)
    print()

    # Load existing results
    results_path = RESULTS_DIR / "experiment5_results.json"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"Found {len(all_results)} existing results.")
    else:
        all_results = []

    already_done = set((r["test_id"], r["embedding"]) for r in all_results)

    # ── Load embedding models ──────────────────────────────────────────────
    print("\nLoading embedding models ...")
    print(f"  1. Original BERT ({BERT_MODEL}) ...")
    bert_model = SentenceTransformer(BERT_MODEL)

    print(f"  2. BGE-M3 ({BGE_MODEL_NAME}) ...")
    bge_model  = SentenceTransformer(BGE_MODEL_NAME)

    print(f"  3. E5-Large ({E5_MODEL_NAME}) ...")
    e5_model   = SentenceTransformer(E5_MODEL_NAME)

    print(f"  4. MiniLM ({MINI_MODEL_NAME}) ...")
    mini_model = SentenceTransformer(MINI_MODEL_NAME)
    print()

    # ── Connect to ChromaDB ────────────────────────────────────────────────
    chroma_client       = chromadb.PersistentClient(path=str(CHROMA_DIR))
    original_collection = chroma_client.get_collection(ORIGINAL_COLLECTION)
    print(f"Original collection: {original_collection.count()} documents\n")

    # ── Build new collections ──────────────────────────────────────────────
    print("Building embedding collections (skips if already done) ...")
    bge_collection  = build_collection(chroma_client, bge_model,
                                       BGE_COLLECTION,  "BGE-M3")
    e5_collection   = build_collection(chroma_client, e5_model,
                                       E5_COLLECTION,   "E5-Large")
    mini_collection = build_collection(chroma_client, mini_model,
                                       MINI_COLLECTION, "MiniLM")

    # ── Load test questions ────────────────────────────────────────────────
    with open(TESTSET_DIR / "test_set.json", encoding="utf-8") as f:
        questions = json.load(f)[:N_QUESTIONS]
    print(f"Running on {len(questions)} questions\n")

    # ── Define 4 conditions ────────────────────────────────────────────────
    conditions = [
        {
            "label":      "BERT+CLIP+DESC (Original)",
            "retriever":  retrieve_original,
            "model":      bert_model,
            "collection": original_collection,
        },
        {
            "label":      "MiniLM-L6 (384d)",
            "retriever":  retrieve_dense,
            "model":      mini_model,
            "collection": mini_collection,
        },
        {
            "label":      "BGE-M3 (1024d)",
            "retriever":  retrieve_dense,
            "model":      bge_model,
            "collection": bge_collection,
        },
        {
            "label":      "E5-Large (1024d)",
            "retriever":  retrieve_dense,
            "model":      e5_model,
            "collection": e5_collection,
        },
    ]

    # ── Run experiment ─────────────────────────────────────────────────────
    for cond in conditions:
        label = cond["label"]
        print(f"\n>>> Condition: {label}")
        print("-" * 50)

        for q in tqdm(questions, desc=label):
            key = (q["test_id"], label)
            if key in already_done:
                continue

            question_text = q["text"]

            # Retrieve
            contexts = cond["retriever"](
                question_text, cond["model"], cond["collection"]
            )
            mean_sim = round(
                float(np.mean([c["similarity"] for c in contexts])), 4
            ) if contexts else 0.0

            # Call GPT-4o
            response = ""
            latency  = 0.0
            error    = None

            for attempt in range(3):
                try:
                    response, latency = call_gpt4o(question_text, contexts)
                    break
                except Exception as e:
                    error = str(e)
                    print(f"  Error (attempt {attempt+1}): {e}")
                    time.sleep(2 ** attempt)

            if not response:
                response = f"ERROR: {error}"

            metrics = compute_metrics(response, question_text)

            record = {
                "test_id":          q["test_id"],
                "question":         question_text,
                "chapter_type":     q["chapter_type"],
                "difficulty":       q["difficulty"],
                "question_type":    q["question_type"],
                "embedding":        label,
                "response":         response,
                "latency_sec":      round(latency, 3),
                "bleu":             metrics["bleu"],
                "rouge_l":          metrics["rouge_l"],
                "bertscore":        metrics["bertscore"],
                "step_correctness": metrics["step_correctness"],
                "mean_sim":         mean_sim,
                "n_contexts":       len(contexts),
                "timestamp":        datetime.now().isoformat(),
            }
            all_results.append(record)
            already_done.add(key)

            # Save after every response
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

        time.sleep(1)

    # ── Final output ───────────────────────────────────────────────────────
    save_results(all_results)
    print(f"\nTotal records: {len(all_results)}")
    print_summary(all_results)


if __name__ == "__main__":
    main()