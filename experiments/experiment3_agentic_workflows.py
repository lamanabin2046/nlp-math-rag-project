"""
experiments/experiment3_agentic_workflows.py
─────────────────────────────────────────────
Experiment 3 — Agentic Workflow Comparison

Compares three retrieval-generation strategies on hard and proof-based
questions where iterative refinement is most valuable. Answers RQ3.

Workflows:
  - Static RAG  — single retrieve → generate pass (baseline)
  - Self-RAG    — generate → self-evaluate → re-retrieve if score < threshold
  - CRAG        — retrieve → grade documents → corrective re-retrieve if weak

Models evaluated: GPT-4o, Claude Opus

Output: results/experiment3_results.json
        results/experiment3_summary.csv

Dependencies:
    pip install langgraph langchain langchain-openai langchain-anthropic

Run:
    python experiments/experiment3_agentic_workflows.py
"""

import json
import time
import csv
import sys
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Annotated, List, Optional
import operator

import nltk
import openai
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY,
    TESTSET_DIR, CHROMA_DIR, CHROMA_COLLECTION,
    BERT_MODEL, CLIP_DIM, DESC_DIM,
)

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TOP_K              = 5
BERT_DIM           = 768
SELF_RAG_THRESHOLD = 3.5   # re-retrieve if self-score < this (out of 5)
SELF_RAG_MAX_ITER  = 3
CRAG_THRESHOLD     = 0.6   # re-retrieve if mean relevance < this (0–1)

WORKFLOWS = ["static_rag", "self_rag", "crag"]
MODELS    = ["gpt-4o", "claude-opus"]


# ══════════════════════════════════════════════════════════════════════════
# 1. SHARED RETRIEVAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def load_retriever():
    print(f"Loading BERT: {BERT_MODEL}")
    bert = SentenceTransformer(BERT_MODEL)
    print("Connecting to ChromaDB …")
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(CHROMA_COLLECTION)
    print(f"  {collection.count()} documents indexed\n")
    return bert, collection


def retrieve(question: str, bert, collection, top_k: int = TOP_K) -> list[dict]:
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


def format_context(contexts: list[dict]) -> str:
    lines = ["Relevant worked examples from the Nepal Grade 10 Mathematics textbook:\n"]
    for i, ctx in enumerate(contexts, 1):
        lines.append(
            f"Example {i} (Unit {ctx['chapter_num']} — {ctx['chapter_title']}, p.{ctx['page']}):"
        )
        lines.append(ctx["text"])
        lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# 2. MODEL CALLS
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

Score this solution from 1 to 5 on the following criteria:
- 5: All steps correct, clear working, correct final answer
- 4: Minor errors, mostly correct working and answer
- 3: Correct approach but significant gaps or errors
- 2: Partially correct approach, multiple errors
- 1: Incorrect approach or fundamentally wrong

Respond ONLY with:
SCORE: <integer 1-5>
REASON: <one sentence>
MISSING: <what is missing or wrong, or 'nothing' if score is 5>"""

CRAG_GRADE_PROMPT = """You are grading whether a worked example is relevant to a student's question.

Student question: {question}

Worked example:
{document}

Is this worked example directly relevant and helpful for solving the question?
Rate relevance from 0.0 to 1.0, where:
  1.0 = perfectly relevant, same concept and approach
  0.7 = mostly relevant, same chapter/topic
  0.4 = somewhat relevant, related topic
  0.1 = barely relevant
  0.0 = completely irrelevant

Respond ONLY with:
RELEVANCE: <float 0.0 to 1.0>
REASON: <one brief sentence>"""

QUERY_TRANSFORM_PROMPT = """The original question retrieved poor-quality context.
Rephrase the question to better match worked examples in a Grade 10 Nepal mathematics textbook.
Focus on the specific mathematical operation, theorem, or procedure required.

Original question: {question}

Rephrased question (for textbook search only, do not solve):"""


def call_llm(prompt: str, model: str) -> tuple[str, float]:
    """Unified LLM caller. Returns (response_text, latency_sec)."""
    start = time.time()
    if model == "gpt-4o":
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        resp   = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()

    elif model == "claude-opus":
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp   = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
    else:
        raise ValueError(f"Unknown model: {model}")

    return text, time.time() - start


# ══════════════════════════════════════════════════════════════════════════
# 3. WORKFLOW IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════

# ── 3a. Static RAG ────────────────────────────────────────────────────────

def run_static_rag(question: str, model: str,
                   bert, collection) -> dict:
    t0       = time.time()
    contexts = retrieve(question, bert, collection)
    prompt   = SOLVE_PROMPT.format(
        context=format_context(contexts), question=question
    )
    response, latency = call_llm(prompt, model)
    return {
        "response":    response,
        "latency_sec": round(time.time() - t0, 3),
        "iterations":  1,
        "n_retrieves": 1,
        "final_score": None,
        "contexts":    contexts,
    }


# ── 3b. Self-RAG (LangGraph) ──────────────────────────────────────────────

class SelfRAGState(TypedDict):
    question:    str
    model:       str
    contexts:    list
    response:    str
    score:       float
    reason:      str
    missing:     str
    iteration:   int
    latency_acc: float


def _selfrag_retrieve(state: SelfRAGState, bert, collection) -> SelfRAGState:
    """Node: retrieve context, optionally appending missing info to query."""
    query = state["question"]
    if state["iteration"] > 0 and state.get("missing") and \
       state["missing"].lower() != "nothing":
        query = f"{query} [Additional focus: {state['missing']}]"
    t0       = time.time()
    contexts = retrieve(query, bert, collection)
    return {**state,
            "contexts":    contexts,
            "latency_acc": state["latency_acc"] + (time.time() - t0)}


def _selfrag_generate(state: SelfRAGState) -> SelfRAGState:
    """Node: generate solution from current context."""
    prompt = SOLVE_PROMPT.format(
        context=format_context(state["contexts"]),
        question=state["question"],
    )
    t0          = time.time()
    response, _ = call_llm(prompt, state["model"])
    return {**state,
            "response":    response,
            "latency_acc": state["latency_acc"] + (time.time() - t0)}


def _selfrag_evaluate(state: SelfRAGState) -> SelfRAGState:
    """Node: self-evaluate the generated solution."""
    prompt = SELF_EVAL_PROMPT.format(
        question=state["question"],
        solution=state["response"],
    )
    t0            = time.time()
    raw, _        = call_llm(prompt, state["model"])
    latency_delta = time.time() - t0

    score   = SELF_RAG_THRESHOLD
    reason  = ""
    missing = "nothing"

    score_m   = re.search(r"SCORE:\s*([1-5])",  raw)
    reason_m  = re.search(r"REASON:\s*(.+)",    raw)
    missing_m = re.search(r"MISSING:\s*(.+)",   raw)

    if score_m:   score   = float(score_m.group(1))
    if reason_m:  reason  = reason_m.group(1).strip()
    if missing_m: missing = missing_m.group(1).strip()

    return {**state,
            "score":       score,
            "reason":      reason,
            "missing":     missing,
            "iteration":   state["iteration"] + 1,
            "latency_acc": state["latency_acc"] + latency_delta}


def _selfrag_should_continue(state: SelfRAGState) -> str:
    """Edge: decide whether to re-retrieve or stop."""
    if state["score"] >= SELF_RAG_THRESHOLD or \
       state["iteration"] >= SELF_RAG_MAX_ITER:
        return "done"
    return "retrieve"


def run_self_rag(question: str, model: str, bert, collection) -> dict:
    try:
        from langgraph.graph import StateGraph, END
        return _run_self_rag_langgraph(question, model, bert, collection)
    except ImportError:
        return _run_self_rag_manual(question, model, bert, collection)


def _run_self_rag_langgraph(question, model, bert, collection) -> dict:
    from langgraph.graph import StateGraph, END

    def retrieve_node(state):
        return _selfrag_retrieve(state, bert, collection)

    graph = StateGraph(SelfRAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", _selfrag_generate)
    graph.add_node("evaluate", _selfrag_evaluate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        _selfrag_should_continue,
        {"retrieve": "retrieve", "done": END},
    )
    app = graph.compile()

    init_state: SelfRAGState = {
        "question":    question,
        "model":       model,
        "contexts":    [],
        "response":    "",
        "score":       0.0,
        "reason":      "",
        "missing":     "",
        "iteration":   0,
        "latency_acc": 0.0,
    }
    final = app.invoke(init_state)
    return {
        "response":    final["response"],
        "latency_sec": round(final["latency_acc"], 3),
        "iterations":  final["iteration"],
        "n_retrieves": final["iteration"],
        "final_score": final["score"],
        "contexts":    final["contexts"],
    }


def _run_self_rag_manual(question, model, bert, collection) -> dict:
    """Fallback: manual loop (no LangGraph dependency)."""
    state: SelfRAGState = {
        "question":    question,
        "model":       model,
        "contexts":    [],
        "response":    "",
        "score":       0.0,
        "reason":      "",
        "missing":     "",
        "iteration":   0,
        "latency_acc": 0.0,
    }
    while True:
        state = _selfrag_retrieve(state, bert, collection)
        state = _selfrag_generate(state)
        state = _selfrag_evaluate(state)
        if _selfrag_should_continue(state) == "done":
            break
    return {
        "response":    state["response"],
        "latency_sec": round(state["latency_acc"], 3),
        "iterations":  state["iteration"],
        "n_retrieves": state["iteration"],
        "final_score": state["score"],
        "contexts":    state["contexts"],
    }


# ── 3c. CRAG (Corrective RAG, LangGraph) ─────────────────────────────────

class CRAGState(TypedDict):
    question:         str
    model:            str
    contexts:         list
    relevance_scores: list
    mean_relevance:   float
    transformed_q:    str
    response:         str
    latency_acc:      float
    n_retrieves:      int


def _crag_retrieve(state: CRAGState, bert, collection) -> CRAGState:
    query    = state.get("transformed_q") or state["question"]
    t0       = time.time()
    contexts = retrieve(query, bert, collection)
    return {**state,
            "contexts":    contexts,
            "n_retrieves": state["n_retrieves"] + 1,
            "latency_acc": state["latency_acc"] + (time.time() - t0)}


def _crag_grade_docs(state: CRAGState) -> CRAGState:
    """Grade each retrieved document for relevance."""
    scores = []
    t0     = time.time()
    for ctx in state["contexts"]:
        prompt = CRAG_GRADE_PROMPT.format(
            question=state["question"], document=ctx["text"]
        )
        raw, _ = call_llm(prompt, state["model"])
        m = re.search(r"RELEVANCE:\s*([0-9.]+)", raw)
        scores.append(float(m.group(1)) if m else 0.5)

    mean_rel = float(np.mean(scores)) if scores else 0.0
    return {**state,
            "relevance_scores": scores,
            "mean_relevance":   mean_rel,
            "latency_acc":      state["latency_acc"] + (time.time() - t0)}


def _crag_transform_query(state: CRAGState) -> CRAGState:
    """Rephrase the question to improve retrieval."""
    t0       = time.time()
    prompt   = QUERY_TRANSFORM_PROMPT.format(question=state["question"])
    new_q, _ = call_llm(prompt, state["model"])
    return {**state,
            "transformed_q": new_q.strip(),
            "latency_acc":   state["latency_acc"] + (time.time() - t0)}


def _crag_filter_and_generate(state: CRAGState) -> CRAGState:
    """Filter to relevant docs only, then generate."""
    threshold = CRAG_THRESHOLD * 0.5
    filtered  = [
        ctx for ctx, sc in zip(state["contexts"], state["relevance_scores"])
        if sc >= threshold
    ] or state["contexts"][:2]

    prompt      = SOLVE_PROMPT.format(
        context=format_context(filtered), question=state["question"]
    )
    t0          = time.time()
    response, _ = call_llm(prompt, state["model"])
    return {**state,
            "response":    response,
            "latency_acc": state["latency_acc"] + (time.time() - t0)}


def _crag_needs_correction(state: CRAGState) -> str:
    """Edge: correct retrieval or generate directly."""
    if state["mean_relevance"] < CRAG_THRESHOLD and state["n_retrieves"] < 2:
        return "transform"
    return "generate"


def run_crag(question: str, model: str, bert, collection) -> dict:
    try:
        from langgraph.graph import StateGraph, END
        return _run_crag_langgraph(question, model, bert, collection)
    except ImportError:
        return _run_crag_manual(question, model, bert, collection)


def _run_crag_langgraph(question, model, bert, collection) -> dict:
    from langgraph.graph import StateGraph, END

    def retrieve_node(state):
        return _crag_retrieve(state, bert, collection)

    graph = StateGraph(CRAGState)
    graph.add_node("retrieve",   retrieve_node)
    graph.add_node("grade_docs", _crag_grade_docs)
    graph.add_node("transform",  _crag_transform_query)
    graph.add_node("generate",   _crag_filter_and_generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_docs")
    graph.add_conditional_edges(
        "grade_docs",
        _crag_needs_correction,
        {"transform": "transform", "generate": "generate"},
    )
    graph.add_edge("transform", "retrieve")
    graph.add_edge("generate",  END)
    app = graph.compile()

    init: CRAGState = {
        "question":         question,
        "model":            model,
        "contexts":         [],
        "relevance_scores": [],
        "mean_relevance":   0.0,
        "transformed_q":    "",
        "response":         "",
        "latency_acc":      0.0,
        "n_retrieves":      0,
    }
    final = app.invoke(init)
    return {
        "response":       final["response"],
        "latency_sec":    round(final["latency_acc"], 3),
        "iterations":     1,
        "n_retrieves":    final["n_retrieves"],
        "mean_relevance": final["mean_relevance"],
        "contexts":       final["contexts"],
    }


def _run_crag_manual(question, model, bert, collection) -> dict:
    """Fallback manual CRAG loop."""
    state: CRAGState = {
        "question":         question,
        "model":            model,
        "contexts":         [],
        "relevance_scores": [],
        "mean_relevance":   0.0,
        "transformed_q":    "",
        "response":         "",
        "latency_acc":      0.0,
        "n_retrieves":      0,
    }
    state = _crag_retrieve(state, bert, collection)
    state = _crag_grade_docs(state)
    if _crag_needs_correction(state) == "transform":
        state = _crag_transform_query(state)
        state = _crag_retrieve(state, bert, collection)
        state = _crag_grade_docs(state)
    state = _crag_filter_and_generate(state)
    return {
        "response":       state["response"],
        "latency_sec":    round(state["latency_acc"], 3),
        "iterations":     1,
        "n_retrieves":    state["n_retrieves"],
        "mean_relevance": state["mean_relevance"],
        "contexts":       state["contexts"],
    }


WORKFLOW_RUNNERS = {
    "static_rag": run_static_rag,
    "self_rag":   run_self_rag,
    "crag":       run_crag,
}


# ══════════════════════════════════════════════════════════════════════════
# 4. METRICS
# ══════════════════════════════════════════════════════════════════════════

def compute_bleu(ref: str, hyp: str) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    return sentence_bleu(
        [nltk.word_tokenize(ref.lower())],
        nltk.word_tokenize(hyp.lower()),
        smoothing_function=SmoothingFunction().method4,
    )


def compute_rouge_l(ref: str, hyp: str) -> float:
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(
        ["rougeL"], use_stemmer=True
    ).score(ref, hyp)["rougeL"].fmeasure


def compute_bertscore(ref: str, hyp: str) -> float:
    from bert_score import score as bscore
    _, _, F = bscore([hyp], [ref], lang="en", verbose=False,
                     model_type="distilbert-base-uncased")
    return F[0].item()


def compute_step_correctness(response: str) -> float:
    score = 0.0
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    if len(lines) >= 3: score += 0.2
    n = len(re.findall(r"(step\s*\d+|\d+[\.\)]|\(i+\)|∴|therefore|hence)", response, re.I))
    if n >= 2: score += 0.3
    elif n:    score += 0.15
    if re.search(r"[\d]+.*=|=.*[\d]+", response):                            score += 0.2
    if re.search(r"(therefore|hence|answer|∴|final|result)", response, re.I): score += 0.2
    if len(response.split()) >= 30:                                           score += 0.1
    return round(min(score, 1.0), 3)


def compute_metrics(response: str, question: str) -> dict:
    return {
        "bleu":             round(compute_bleu(question, response),      4),
        "rouge_l":          round(compute_rouge_l(question, response),   4),
        "bertscore":        round(compute_bertscore(question, response), 4),
        "step_correctness": compute_step_correctness(response),
    }


# ══════════════════════════════════════════════════════════════════════════
# 5. SAVE / SUMMARY HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _save_results(results: list[dict]):
    out = RESULTS_DIR / "experiment3_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def print_summary(results: list[dict]):
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["model"], r["workflow"])].append(r)

    print("\n" + "=" * 85)
    print(f"{'Model':<14} {'Workflow':<14} {'BLEU':>7} {'ROUGE-L':>8} "
          f"{'BERT':>7} {'Steps':>7} {'Iters':>6} {'Retrieves':>9} {'Latency':>8}")
    print("-" * 85)
    for (model, wf), rows in sorted(grouped.items()):
        iters = [r["iterations"]  for r in rows if r["iterations"]  is not None]
        retr  = [r["n_retrieves"] for r in rows if r["n_retrieves"] is not None]
        print(
            f"{model:<14} {wf:<14} "
            f"{np.mean([r['bleu']             for r in rows]):>7.4f} "
            f"{np.mean([r['rouge_l']          for r in rows]):>8.4f} "
            f"{np.mean([r['bertscore']        for r in rows]):>7.4f} "
            f"{np.mean([r['step_correctness'] for r in rows]):>7.4f} "
            f"{np.mean(iters) if iters else 0:>6.1f} "
            f"{np.mean(retr)  if retr  else 0:>9.1f} "
            f"{np.mean([r['latency_sec']      for r in rows]):>7.2f}s"
        )
    print("=" * 85)

    print("\nBreakdown by difficulty:")
    print(f"{'Model':<14} {'Workflow':<14} {'Difficulty':<12} {'Steps':>7} {'BERT':>7}")
    print("-" * 60)
    grouped2 = defaultdict(list)
    for r in results:
        grouped2[(r["model"], r["workflow"], r["difficulty"])].append(r)
    for (model, wf, diff), rows in sorted(grouped2.items()):
        print(f"{model:<14} {wf:<14} {diff:<12} "
              f"{np.mean([r['step_correctness'] for r in rows]):>7.4f} "
              f"{np.mean([r['bertscore']        for r in rows]):>7.4f}")


def save_summary_csv(results: list[dict]):
    fields = [
        "test_id", "chapter_type", "difficulty", "question_type",
        "model", "workflow", "bleu", "rouge_l", "bertscore",
        "step_correctness", "latency_sec", "iterations", "n_retrieves",
        "self_eval_score", "mean_relevance",
    ]
    path = RESULTS_DIR / "experiment3_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fields})
    print(f"Summary CSV saved → {path}")


# ══════════════════════════════════════════════════════════════════════════
# 6. MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════

def run_experiment():
    test_path = TESTSET_DIR / "test_set.json"
    with open(test_path, encoding="utf-8") as f:
        all_questions = json.load(f)

    hard_questions = [
        q for q in all_questions
        if q.get("difficulty") == "hard" or q.get("question_type") == "proof"
    ]
    if not hard_questions:
        print("WARNING: No hard/proof questions found; using all questions.")
        hard_questions = all_questions[:20]

    print(f"Hard + proof-based questions: {len(hard_questions)}\n")
    print(f"  difficulty breakdown: "
          f"hard={sum(1 for q in hard_questions if q['difficulty']=='hard')}, "
          f"proof={sum(1 for q in hard_questions if q['question_type']=='proof')}")

    bert, collection = load_retriever()

    all_results = []
    total = len(MODELS) * len(WORKFLOWS) * len(hard_questions)
    print(f"\nRunning {len(MODELS)} models × {len(WORKFLOWS)} workflows × "
          f"{len(hard_questions)} questions = {total} total\n")
    print("=" * 65)

    for model_name in MODELS:
        for workflow_name in WORKFLOWS:
            label  = f"{model_name.upper()} / {workflow_name.replace('_', ' ').title()}"
            runner = WORKFLOW_RUNNERS[workflow_name]
            print(f"\n▶  {label}")
            print("-" * 40)

            for q in tqdm(hard_questions, desc=label[:35]):
                result_info = {}
                error       = None

                for attempt in range(3):
                    try:
                        result_info = runner(q["text"], model_name, bert, collection)
                        break
                    except Exception as e:
                        error = str(e)
                        print(f"  Error (attempt {attempt+1}): {e}")
                        time.sleep(2 ** attempt)

                if not result_info.get("response"):
                    result_info = {
                        "response":    f"ERROR: {error}",
                        "latency_sec": 0.0,
                        "iterations":  0,
                        "n_retrieves": 0,
                    }

                response = result_info["response"]
                metrics  = compute_metrics(response, q["text"])

                record = {
                    "test_id":          q["test_id"],
                    "question":         q["text"],
                    "chapter_type":     q["chapter_type"],
                    "difficulty":       q["difficulty"],
                    "question_type":    q["question_type"],
                    "model":            model_name,
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
                all_results.append(record)
                _save_results(all_results)

            time.sleep(1)

    _save_results(all_results)
    print_summary(all_results)
    save_summary_csv(all_results)


# ══════════════════════════════════════════════════════════════════════════
# 7. REPAIR UTILITIES  (deduplicate + rerun failed)
# ══════════════════════════════════════════════════════════════════════════

def deduplicate_results():
    """Remove duplicate (test_id, model, workflow) entries, keeping the first."""
    results_path = RESULTS_DIR / "experiment3_results.json"
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    seen  = set()
    clean = []
    for r in results:
        key = (r["test_id"], r["model"], r["workflow"])
        if key not in seen:
            seen.add(key)
            clean.append(r)

    removed = len(results) - len(clean)
    print(f"Deduplicate: removed {removed} duplicate(s). {len(clean)} records remaining.")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)


def rerun_failed(model: str = "gpt-4o", workflow: str = "crag"):
    """Re-run only the questions that previously returned an ERROR response."""
    results_path = RESULTS_DIR / "experiment3_results.json"
    with open(results_path, encoding="utf-8") as f:
        all_results = json.load(f)

    # Collect test_ids where the response is an error
    failed_ids = {
        r["test_id"] for r in all_results
        if r["model"] == model
        and r["workflow"] == workflow
        and str(r["response"]).startswith("ERROR")
    }

    if not failed_ids:
        print(f"No failed entries found for {model} / {workflow}. Nothing to re-run.")
        return

    print(f"Found {len(failed_ids)} failed entries for {model} / {workflow}")

    with open(TESTSET_DIR / "test_set.json", encoding="utf-8") as f:
        all_questions = json.load(f)

    questions_to_retry = [
        q for q in all_questions
        if (q.get("difficulty") == "hard" or q.get("question_type") == "proof")
        and q["test_id"] in failed_ids
    ]
    print(f"Matched {len(questions_to_retry)} question(s) to retry\n")

    bert, collection = load_retriever()
    runner = WORKFLOW_RUNNERS[workflow]

    for q in tqdm(questions_to_retry, desc=f"Re-running {model}/{workflow}"):
        # Remove the old failed entry before replacing it
        all_results = [
            r for r in all_results
            if not (r["test_id"] == q["test_id"]
                    and r["model"] == model
                    and r["workflow"] == workflow)
        ]

        result_info = {}
        error       = None
        for attempt in range(3):
            try:
                result_info = runner(q["text"], model, bert, collection)
                break
            except Exception as e:
                error = str(e)
                print(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)

        if not result_info.get("response"):
            print(f"  Skipping {q['test_id']} — still failing after 3 attempts: {error}")
            continue

        response = result_info["response"]
        metrics  = compute_metrics(response, q["text"])

        record = {
            "test_id":          q["test_id"],
            "question":         q["text"],
            "chapter_type":     q["chapter_type"],
            "difficulty":       q["difficulty"],
            "question_type":    q["question_type"],
            "model":            model,
            "workflow":         workflow,
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
        all_results.append(record)
        _save_results(all_results)   # save after every question

    print("\nAll done! Final summary:")
    print_summary(all_results)
    save_summary_csv(all_results)


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # To run the full experiment from scratch, swap the two lines below:
    #   run_experiment()
    deduplicate_results()
    rerun_failed(model="gpt-4o", workflow="crag")
