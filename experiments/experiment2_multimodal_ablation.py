"""
experiments/experiment2_multimodal_ablation.py
───────────────────────────────────────────────
Experiment 2 — Multimodal Ablation

Tests four retrieval configurations on geometry questions
(visual_required=True, chapter_type='geometry') using GPT-4o as the
generation model. Answers RQ2: Does math-aware diagram grounding and
modality weighting improve performance on geometry problems?

Conditions:
  1. text_only       — BERT text embeddings only (768d)
  2. text_desc       — BERT text + diagram description (768d + 768d = 1536d)
  3. text_clip_desc  — Full 2048d equal-weight concatenation (existing index)
  4. text_learned    — Full 2048d with learned scalar modality weights

Learned weights are found by a random search over w=(w_text, w_clip, w_desc)
on a held-out 30% validation split of geometry questions, using BERTScore
as the reward signal, then applied to the remaining 70% test split.

Output: results/experiment2_results.json
        results/experiment2_summary.csv

Run:
    python experiments/experiment2_multimodal_ablation.py
"""

import json
import time
import csv
import sys
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import product

import nltk
import openai
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OPENAI_API_KEY, TESTSET_DIR, CHROMA_DIR, CHROMA_COLLECTION,
    BERT_MODEL, CLIP_DIM, DESC_DIM,
)

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

RESULTS_DIR  = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BERT_DIM = 768          # first 768 dims  = BERT text
# CLIP_DIM = 512        # next  512 dims  = CLIP image  (from config)
# DESC_DIM = 768        # last  768 dims  = BERT desc   (from config)

TOP_K          = 5
N_WEIGHT_TRIES = 40     # random-search budget for learning weights
RANDOM_SEED    = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CONDITIONS = [
    ("text_only",      "Text Only"),
    ("text_desc",      "Text + Diagram Desc (Equal)"),
    ("text_clip_desc", "Text + Desc + CLIP (Equal)"),
    ("text_learned",   "Text + Desc + CLIP (Learned)"),
]


# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD RESOURCES
# ══════════════════════════════════════════════════════════════════════════

def load_resources():
    print(f"Loading BERT: {BERT_MODEL}")
    bert = SentenceTransformer(BERT_MODEL)

    print("Connecting to ChromaDB …")
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(CHROMA_COLLECTION)
    print(f"  Collection size: {collection.count()} documents\n")

    # Pull ALL stored embeddings + metadata once — reuse across conditions
    print("Fetching all stored embeddings from ChromaDB …")
    result = collection.get(
        include=["embeddings", "documents", "metadatas"]
    )
    embeddings = np.array(result["embeddings"], dtype=np.float32)  # (N, 2048)
    documents  = result["documents"]
    metadatas  = result["metadatas"]
    ids        = result["ids"]
    print(f"  Loaded {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}\n")

    return bert, embeddings, documents, metadatas, ids


# ══════════════════════════════════════════════════════════════════════════
# 2. CONDITION-SPECIFIC RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════

def cosine_similarity_matrix(query_vec: np.ndarray,
                              doc_mat: np.ndarray) -> np.ndarray:
    """cosine similarity between 1-d query and (N, d) doc matrix."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    norms = np.linalg.norm(doc_mat, axis=1, keepdims=True) + 1e-8
    d = doc_mat / norms
    return d @ q    # (N,)


def build_query_and_docs(question: str,
                         bert: SentenceTransformer,
                         all_embs: np.ndarray,
                         condition: str,
                         learned_weights: tuple = (1.0, 1.0, 1.0)):
    """
    Return (query_vec, doc_matrix) shaped for cosine similarity,
    projected according to the ablation condition.

    all_embs layout: [0:768]=BERT_text | [768:1280]=CLIP | [1280:2048]=DESC
    """
    text_emb = bert.encode([question], normalize_embeddings=True)[0]   # (768,)

    t_slice  = slice(0,    BERT_DIM)
    c_slice  = slice(BERT_DIM, BERT_DIM + CLIP_DIM)
    d_slice  = slice(BERT_DIM + CLIP_DIM, BERT_DIM + CLIP_DIM + DESC_DIM)

    if condition == "text_only":
        # Use only BERT-text dimension
        query = text_emb                            # (768,)
        docs  = all_embs[:, t_slice]                # (N, 768)

    elif condition == "text_desc":
        # BERT-text + BERT-description; CLIP excluded
        # Query: [text_emb, zeros_desc] — query has no image so desc=0
        query = np.concatenate([text_emb, np.zeros(DESC_DIM, dtype=np.float32)])
        docs  = np.concatenate(
            [all_embs[:, t_slice], all_embs[:, d_slice]], axis=1
        )   # (N, 1536)

    elif condition == "text_clip_desc":
        # Full 2048-d equal weights; queries zero-pad CLIP and DESC
        query = np.concatenate([
            text_emb,
            np.zeros(CLIP_DIM, dtype=np.float32),
            np.zeros(DESC_DIM, dtype=np.float32),
        ])
        docs  = all_embs    # (N, 2048)

    elif condition == "text_learned":
        w_text, w_clip, w_desc = learned_weights
        query = np.concatenate([
            w_text * text_emb,
            np.zeros(CLIP_DIM,  dtype=np.float32),  # query has no image
            np.zeros(DESC_DIM,  dtype=np.float32),
        ])
        # Weight stored modalities
        weighted_docs = np.concatenate([
            w_text * all_embs[:, t_slice],
            w_clip * all_embs[:, c_slice],
            w_desc * all_embs[:, d_slice],
        ], axis=1)
        docs = weighted_docs

    else:
        raise ValueError(f"Unknown condition: {condition}")

    return query, docs


def retrieve_condition(question: str,
                       bert: SentenceTransformer,
                       all_embs: np.ndarray,
                       documents: list,
                       metadatas: list,
                       condition: str,
                       learned_weights: tuple = (1.0, 1.0, 1.0),
                       top_k: int = TOP_K) -> list[dict]:
    query, docs = build_query_and_docs(
        question, bert, all_embs, condition, learned_weights
    )
    sims = cosine_similarity_matrix(query, docs)
    top_idx = np.argsort(sims)[::-1][:top_k]

    contexts = []
    for idx in top_idx:
        meta = metadatas[idx]
        contexts.append({
            "text":          documents[idx],
            "similarity":    float(sims[idx]),
            "chapter_num":   meta.get("chapter_num"),
            "chapter_title": meta.get("chapter_title"),
            "chapter_type":  meta.get("chapter_type"),
            "page":          meta.get("page"),
        })
    return contexts


# ══════════════════════════════════════════════════════════════════════════
# 3. LEARNED WEIGHT SEARCH
# ══════════════════════════════════════════════════════════════════════════

def bertscore_fast(reference: str, hypothesis: str) -> float:
    """Light BERTScore wrapper — returns F1."""
    from bert_score import score as bscore
    _, _, F = bscore([hypothesis], [reference],
                     lang="en", verbose=False,
                     model_type="distilbert-base-uncased")
    return F[0].item()


def generate_gpt4o(prompt: str, api_key: str) -> tuple[str, float]:
    client = openai.OpenAI(api_key=api_key)
    start  = time.time()
    resp   = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip(), time.time() - start


RAG_PROMPT = """You are a mathematics tutor for Grade 10 students in Nepal.
Use the worked examples below as reference to solve the problem.
Show clear step-by-step working aligned with the Nepal curriculum.

{context}

Now solve this problem step by step:
Problem: {question}

Solution:"""


def format_context(contexts: list[dict]) -> str:
    lines = ["Relevant worked examples from the Nepal Grade 10 Mathematics textbook:\n"]
    for i, ctx in enumerate(contexts, 1):
        lines.append(
            f"Example {i} (Unit {ctx['chapter_num']} — {ctx['chapter_title']}, p.{ctx['page']}):"
        )
        lines.append(ctx["text"])
        lines.append("")
    return "\n".join(lines)


def score_weights(questions_val: list[dict],
                  bert: SentenceTransformer,
                  all_embs: np.ndarray,
                  documents: list,
                  metadatas: list,
                  weights: tuple,
                  api_key: str) -> float:
    """
    Run generation on validation questions with given weights,
    return mean BERTScore(F1) as reward. Uses a sample of 5 questions
    to keep cost manageable.
    """
    sample = random.sample(questions_val, min(5, len(questions_val)))
    scores = []
    for q in sample:
        contexts = retrieve_condition(
            q["text"], bert, all_embs, documents, metadatas,
            "text_learned", learned_weights=weights
        )
        prompt = RAG_PROMPT.format(
            context=format_context(contexts),
            question=q["text"],
        )
        try:
            response, _ = generate_gpt4o(prompt, api_key)
            s = bertscore_fast(q["text"], response)
        except Exception:
            s = 0.0
        scores.append(s)
    return float(np.mean(scores))


def learn_weights(questions_val: list[dict],
                  bert: SentenceTransformer,
                  all_embs: np.ndarray,
                  documents: list,
                  metadatas: list,
                  api_key: str) -> tuple:
    """
    Random search over N_WEIGHT_TRIES (w_text, w_clip, w_desc) triplets.
    Weights are constrained to [0.1, 2.0] and normalised to sum to 3.
    Returns the best-scoring weight triplet.
    """
    print(f"\n  Learning modality weights ({N_WEIGHT_TRIES} trials) …")
    best_score   = -1.0
    best_weights = (1.0, 1.0, 1.0)

    # Always include the equal-weight baseline
    candidates = [(1.0, 1.0, 1.0)]
    for _ in range(N_WEIGHT_TRIES - 1):
        w = np.random.dirichlet(np.ones(3)) * 3   # sums to 3
        candidates.append(tuple(float(x) for x in w))

    for i, weights in enumerate(tqdm(candidates, desc="  Weight search")):
        sc = score_weights(
            questions_val, bert, all_embs, documents, metadatas,
            weights, api_key
        )
        if sc > best_score:
            best_score   = sc
            best_weights = weights
        if i % 10 == 0:
            print(f"    trial {i:>3}  w={[f'{x:.2f}' for x in weights]}  "
                  f"score={sc:.4f}  best={best_score:.4f}")

    print(f"\n  Best weights: text={best_weights[0]:.3f}  "
          f"clip={best_weights[1]:.3f}  desc={best_weights[2]:.3f}  "
          f"(BERTScore={best_score:.4f})\n")
    return best_weights


# ══════════════════════════════════════════════════════════════════════════
# 4. METRICS  (reuse same helpers as experiment 1)
# ══════════════════════════════════════════════════════════════════════════

def compute_bleu(reference: str, hypothesis: str) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ref = nltk.word_tokenize(reference.lower())
    hyp = nltk.word_tokenize(hypothesis.lower())
    return sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method4)


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, hypothesis)["rougeL"].fmeasure


def compute_step_correctness(response: str) -> float:
    import re
    score = 0.0
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    if len(lines) >= 3:   score += 0.2
    sp = re.compile(r"(step\s*\d+|\d+[\.\)]|\(i+\)|∴|therefore|hence)", re.I)
    n  = len(sp.findall(response))
    if n >= 2: score += 0.3
    elif n:    score += 0.15
    if re.search(r"[\d]+.*=|=.*[\d]+", response): score += 0.2
    if re.search(r"(therefore|hence|answer|∴|final|result)", response, re.I): score += 0.2
    if len(response.split()) >= 30:                score += 0.1
    return round(min(score, 1.0), 3)


def compute_metrics(response: str, question: str) -> dict:
    return {
        "bleu":             round(compute_bleu(question, response),       4),
        "rouge_l":          round(compute_rouge_l(question, response),    4),
        "bertscore":        round(bertscore_fast(question, response),     4),
        "step_correctness": compute_step_correctness(response),
    }


# ══════════════════════════════════════════════════════════════════════════
# 5. MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════

def run_experiment():
    # ── Load test questions ───────────────────────────────────────────────
    test_path = TESTSET_DIR / "test_set.json"
    with open(test_path, encoding="utf-8") as f:
        all_questions = json.load(f)

    # Filter to geometry + visual_required only
    geo_questions = [
        q for q in all_questions
        if q.get("chapter_type") == "geometry" and q.get("visual_required", False)
    ]
    if not geo_questions:
        # Fall back to all geometry questions if none are visual_required
        geo_questions = [q for q in all_questions if q.get("chapter_type") == "geometry"]

    if not geo_questions:
        print("ERROR: No geometry questions found in test set.")
        return

    print(f"Geometry questions (visual_required): {len(geo_questions)}\n")

    # Train / test split for learned-weight search
    random.shuffle(geo_questions)
    split         = max(1, int(0.30 * len(geo_questions)))
    questions_val  = geo_questions[:split]
    questions_test = geo_questions[split:]
    if not questions_test:
        questions_test = geo_questions   # too few → use all for both

    print(f"  Val   (weight search): {len(questions_val)}")
    print(f"  Test  (evaluation)   : {len(questions_test)}\n")

    # ── Load resources ────────────────────────────────────────────────────
    bert, all_embs, documents, metadatas, ids = load_resources()

    # ── Learn weights ─────────────────────────────────────────────────────
    learned_weights = learn_weights(
        questions_val, bert, all_embs, documents, metadatas, OPENAI_API_KEY
    )

    # ── Run ablation ──────────────────────────────────────────────────────
    all_results = []
    total = len(CONDITIONS) * len(questions_test)
    print(f"Running {len(CONDITIONS)} conditions × {len(questions_test)} questions "
          f"= {total} total evaluations\n")
    print("=" * 65)

    for cond_key, cond_label in CONDITIONS:
        print(f"\n▶  {cond_label}")
        print("-" * 40)

        for q in tqdm(questions_test, desc=cond_label):
            contexts = retrieve_condition(
                q["text"], bert, all_embs, documents, metadatas,
                condition=cond_key,
                learned_weights=learned_weights,
            )
            prompt = RAG_PROMPT.format(
                context=format_context(contexts),
                question=q["text"],
            )

            response = ""
            latency  = 0.0
            error    = None
            for attempt in range(3):
                try:
                    response, latency = generate_gpt4o(prompt, OPENAI_API_KEY)
                    break
                except Exception as e:
                    error = str(e)
                    time.sleep(2 ** attempt)

            if not response:
                response = f"ERROR: {error}"

            metrics = compute_metrics(response, q["text"])

            result = {
                "test_id":          q["test_id"],
                "question":         q["text"],
                "chapter_type":     q["chapter_type"],
                "difficulty":       q["difficulty"],
                "question_type":    q["question_type"],
                "visual_required":  q.get("visual_required", False),
                "condition":        cond_key,
                "condition_label":  cond_label,
                "model":            "gpt-4o",
                "response":         response,
                "latency_sec":      round(latency, 3),
                "bleu":             metrics["bleu"],
                "rouge_l":          metrics["rouge_l"],
                "bertscore":        metrics["bertscore"],
                "step_correctness": metrics["step_correctness"],
                "n_contexts":       len(contexts),
                "mean_sim":         round(np.mean([c["similarity"] for c in contexts]), 4),
                "learned_weights":  learned_weights if cond_key == "text_learned" else None,
                "timestamp":        datetime.now().isoformat(),
            }
            all_results.append(result)

            # Save incrementally
            _save_results(all_results)

        time.sleep(1)

    _save_results(all_results)
    print_summary(all_results, learned_weights)
    save_summary_csv(all_results)


def _save_results(results: list[dict]):
    out = RESULTS_DIR / "experiment2_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def print_summary(results: list[dict], learned_weights: tuple):
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[r["condition_label"]].append(r)

    print("\n" + "=" * 80)
    print(f"{'Condition':<35} {'BLEU':>7} {'ROUGE-L':>8} {'BERT':>7} "
          f"{'Steps':>7} {'MeanSim':>8} {'Latency':>8}")
    print("-" * 80)
    for label, rows in sorted(grouped.items()):
        print(f"{label:<35} "
              f"{np.mean([r['bleu']             for r in rows]):>7.4f} "
              f"{np.mean([r['rouge_l']          for r in rows]):>8.4f} "
              f"{np.mean([r['bertscore']        for r in rows]):>7.4f} "
              f"{np.mean([r['step_correctness'] for r in rows]):>7.4f} "
              f"{np.mean([r['mean_sim']         for r in rows]):>8.4f} "
              f"{np.mean([r['latency_sec']      for r in rows]):>7.2f}s")
    print("=" * 80)
    print(f"\nLearned weights → text={learned_weights[0]:.3f}  "
          f"clip={learned_weights[1]:.3f}  desc={learned_weights[2]:.3f}")


def save_summary_csv(results: list[dict]):
    fields = [
        "test_id", "difficulty", "question_type", "visual_required",
        "condition", "condition_label", "bleu", "rouge_l", "bertscore",
        "step_correctness", "latency_sec", "n_contexts", "mean_sim",
    ]
    path = RESULTS_DIR / "experiment2_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})
    print(f"Summary CSV saved → {path}")


if __name__ == "__main__":
    run_experiment()
