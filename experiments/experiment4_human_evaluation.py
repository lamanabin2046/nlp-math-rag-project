"""
experiments/experiment4_human_evaluation.py
─────────────────────────────────────────────
Experiment 4 — Human Evaluation

Two components:

A) RESPONSE PREPARATION
   - Selects representative questions from the test set (2-3 per chapter type)
   - Generates solutions using the best-performing configuration identified
     from Experiments 1-3 (configurable via BEST_MODEL / BEST_WORKFLOW)
   - Also generates baseline (no-RAG) responses for comparison
   - Exports a human_eval_packet.json with all responses for evaluators

B) ANALYSIS (run after human evaluators fill in scores)
   - Loads completed human_eval_scores.json
   - Computes inter-rater reliability (Cohen's kappa: teacher rubric vs
     automated step-correctness)
   - Summarises student Likert ratings (clarity, alignment, usefulness)
   - Compares teacher rubric scores across RAG vs Baseline
   - Saves results/experiment4_analysis.json and experiment4_summary.csv

Usage:
   # Step 1 — generate response packet for evaluators:
   python experiments/experiment4_human_evaluation.py --prepare

   # Step 2 — after human scores are collected, run analysis:
   python experiments/experiment4_human_evaluation.py --analyze

Human evaluators fill in results/human_eval_scores.json:
  [
    {
      "item_id": 0,
      "teacher_rubric": 2,       # 0=incorrect 1=partial 2=correct 3=fully detailed
      "student_clarity": 4,      # Likert 1-5
      "student_alignment": 3,    # Likert 1-5
      "student_usefulness": 4,   # Likert 1-5
      "evaluator_id": "T01"      # teacher ID or student ID
    }, ...
  ]

NOTE: Claude Opus API credits were exhausted during this experiment.
      GPT-4o is used for all generation in Experiment 4.
      The primary comparison (RAG vs Baseline) remains model-independent.
"""

import argparse
import json
import csv
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Best config from Experiments 1-3 ─────────────────────────────────────
# Claude Opus + Static RAG won on quality, but Anthropic API credits ran out.
# GPT-4o is used here — the RAG vs Baseline comparison is model-independent.
BEST_MODEL    = "gpt-4o"
BEST_WORKFLOW = "static_rag"

N_PER_CHAPTER = 3
CHAPTER_TYPES = ["geometry", "algebra", "word_problems"]


# ══════════════════════════════════════════════════════════════════════════
# 1. GENERATION HELPERS
# ══════════════════════════════════════════════════════════════════════════

RAG_PROMPT = """You are a mathematics tutor for Grade 10 students in Nepal.
Use the worked examples below as reference to solve the problem.
Show clear step-by-step working aligned with the Nepal curriculum.
Write your solution as a Grade 10 teacher would — numbered steps, clear
intermediate results, final answer clearly stated.

{context}

Now solve this problem step by step:
Problem: {question}

Solution:"""

BASELINE_PROMPT = """You are a mathematics tutor for Grade 10 students in Nepal.
Solve the following problem with clear step-by-step working.
Write your solution as a Grade 10 teacher would — numbered steps, clear
intermediate results, final answer clearly stated.

Problem: {question}

Solution:"""

AUTO_GRADE_PROMPT = """You are an expert Grade 10 mathematics educator in Nepal.
Grade the following solution using this 4-point rubric:

  0 — Incorrect: wrong approach or fundamental error
  1 — Partially correct: right approach, significant errors in working
  2 — Correct: right approach and answer, but steps could be clearer
  3 — Fully detailed: correct, all steps shown, matches exam requirements

Question: {question}

Solution:
{solution}

Respond ONLY with:
RUBRIC: <0, 1, 2, or 3>
FEEDBACK: <one sentence>"""


def retrieve(question: str, bert, collection, top_k: int = 5) -> list[dict]:
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


def call_llm(prompt: str, model: str = "gpt-4o") -> tuple[str, float]:
    """Unified LLM caller. Only GPT-4o used in Experiment 4."""
    start  = time.time()
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp   = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()
    return text, time.time() - start


def auto_grade(question: str, solution: str) -> dict:
    """GPT-4o grades the solution — returns rubric score + feedback."""
    import re
    try:
        prompt   = AUTO_GRADE_PROMPT.format(question=question, solution=solution)
        raw, _   = call_llm(prompt)
        rubric   = None
        feedback = ""
        rm = re.search(r"RUBRIC:\s*([0-3])", raw)
        fm = re.search(r"FEEDBACK:\s*(.+)",  raw)
        if rm: rubric   = int(rm.group(1))
        if fm: feedback = fm.group(1).strip()
        return {"auto_rubric": rubric, "auto_feedback": feedback}
    except Exception as e:
        print(f"    Auto-grade failed: {e}")
        return {"auto_rubric": None, "auto_feedback": ""}


def compute_step_correctness(response: str) -> float:
    import re
    score = 0.0
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    if len(lines) >= 3: score += 0.2
    n = len(re.findall(r"(step\s*\d+|\d+[\.\)]|\(i+\)|∴|therefore|hence)", response, re.I))
    if n >= 2: score += 0.3
    elif n:    score += 0.15
    if re.search(r"[\d]+.*=|=.*[\d]+", response):                             score += 0.2
    if re.search(r"(therefore|hence|answer|∴|final|result)", response, re.I): score += 0.2
    if len(response.split()) >= 30:                                            score += 0.1
    return round(min(score, 1.0), 3)


# ══════════════════════════════════════════════════════════════════════════
# 2. PREPARE EVALUATION PACKET
# ══════════════════════════════════════════════════════════════════════════

def prepare_eval_packet():
    print("=" * 60)
    print("EXPERIMENT 4 — Preparing Human Evaluation Packet")
    print(f"  Model:    {BEST_MODEL}")
    print(f"  Workflow: {BEST_WORKFLOW}")
    print("=" * 60)

    # Load questions
    with open(TESTSET_DIR / "test_set.json", encoding="utf-8") as f:
        all_questions = json.load(f)

    # Stratified sample: N_PER_CHAPTER per chapter type, balanced difficulty
    selected = []
    for ctype in CHAPTER_TYPES:
        pool = [q for q in all_questions if q["chapter_type"] == ctype]
        if not pool:
            print(f"  WARNING: no questions for chapter_type={ctype}")
            continue

        chosen = []
        for diff in ["easy", "medium", "hard"]:
            diff_pool = [q for q in pool if q["difficulty"] == diff]
            if diff_pool:
                import random
                chosen.append(random.choice(diff_pool))
            if len(chosen) >= N_PER_CHAPTER:
                break

        if len(chosen) < N_PER_CHAPTER:
            import random
            remaining = [q for q in pool if q not in chosen]
            chosen += random.sample(remaining, min(N_PER_CHAPTER - len(chosen), len(remaining)))

        selected.extend(chosen[:N_PER_CHAPTER])

    print(f"\nSelected {len(selected)} questions across {len(CHAPTER_TYPES)} chapter types\n")

    # Load retriever
    print(f"Loading BERT: {BERT_MODEL}")
    bert          = SentenceTransformer(BERT_MODEL)
    client_chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection    = client_chroma.get_collection(CHROMA_COLLECTION)
    print(f"  {collection.count()} documents indexed\n")

    packet_items = []
    item_id      = 0

    for q in tqdm(selected, desc="Generating responses"):
        question = q["text"]

        # ── RAG response ──────────────────────────────────────────────────
        contexts   = retrieve(question, bert, collection)
        rag_prompt = RAG_PROMPT.format(
            context=format_context(contexts), question=question
        )
        rag_resp = ""
        for attempt in range(3):
            try:
                rag_resp, _ = call_llm(rag_prompt)
                if not rag_resp:
                    raise ValueError("Empty response returned")
                break
            except Exception as e:
                print(f"  RAG attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)

        # ── Baseline response (no RAG) ────────────────────────────────────
        base_prompt = BASELINE_PROMPT.format(question=question)
        base_resp   = ""
        for attempt in range(3):
            try:
                base_resp, _ = call_llm(base_prompt)
                if not base_resp:
                    raise ValueError("Empty response returned")
                break
            except Exception as e:
                print(f"  Baseline attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)

        # ── Auto-grade both ───────────────────────────────────────────────
        rag_auto   = auto_grade(question, rag_resp)
        base_auto  = auto_grade(question, base_resp)
        rag_steps  = compute_step_correctness(rag_resp)
        base_steps = compute_step_correctness(base_resp)

        for setting, response, auto, steps in [
            ("RAG",      rag_resp,  rag_auto,  rag_steps),
            ("Baseline", base_resp, base_auto, base_steps),
        ]:
            packet_items.append({
                "item_id":           item_id,
                "test_id":           q["test_id"],
                "question":          question,
                "chapter_type":      q["chapter_type"],
                "difficulty":        q["difficulty"],
                "question_type":     q["question_type"],
                "model":             BEST_MODEL,
                "workflow":          BEST_WORKFLOW if setting == "RAG" else "none",
                "setting":           setting,
                "response":          response,
                "auto_rubric":       auto["auto_rubric"],
                "auto_feedback":     auto["auto_feedback"],
                "step_correctness":  steps,
                "teacher_rubric":    None,
                "student_clarity":   None,
                "student_alignment": None,
                "student_usefulness":None,
                "evaluator_id":      None,
            })
            item_id += 1

        # Save incrementally after each question
        _save_packet(packet_items)

    print(f"\n✓ Evaluation packet saved → {RESULTS_DIR / 'human_eval_packet.json'}")
    print(f"  {len(packet_items)} items  ({len(selected)} questions × 2 settings)")

    _save_evaluator_guide(packet_items)
    _save_scores_template(packet_items)

    print("\nNext step: distribute human_eval_guide.txt to evaluators,")
    print("  collect scores in human_eval_scores.json, then run --analyze\n")


def _save_packet(items: list[dict]):
    path = RESULTS_DIR / "human_eval_packet.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _save_scores_template(items: list[dict]):
    template = [
        {
            "item_id":            item["item_id"],
            "teacher_rubric":     None,
            "student_clarity":    None,
            "student_alignment":  None,
            "student_usefulness": None,
            "evaluator_id":       None,
        }
        for item in items
    ]
    path = RESULTS_DIR / "human_eval_scores_TEMPLATE.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print(f"  Score template     → {path}")


def _save_evaluator_guide(items: list[dict]):
    lines = [
        "HUMAN EVALUATION GUIDE — Nepal Grade 10 Mathematics AI Tutor",
        "=" * 65,
        "",
        "TEACHER RUBRIC (0-3):",
        "  0 = Incorrect        — wrong approach or fundamental error",
        "  1 = Partially correct — right approach, significant errors",
        "  2 = Correct           — right answer, steps could be clearer",
        "  3 = Fully detailed    — correct, all steps, exam-ready",
        "",
        "STUDENT RATINGS (1-5 Likert):",
        "  Clarity    — Is the explanation easy to understand?",
        "  Alignment  — Does it match how you were taught in school?",
        "  Usefulness — Would this help you prepare for the exam?",
        "",
        "=" * 65,
        "",
    ]
    for item in items:
        resp_text = item.get("response") or "[NO RESPONSE GENERATED]"
        lines += [
            f"ITEM {item['item_id']:>3}  |  {item['chapter_type'].upper()}  |  "
            f"{item['difficulty'].upper()}  |  {item['setting'].upper()}",
            "-" * 65,
            f"Question: {item['question']}",
            "",
            "Solution:",
            resp_text,
            "",
            f"  [Auto-rubric: {item.get('auto_rubric')}]  {item.get('auto_feedback', '')}",
            "",
            "  Teacher rubric (fill in):     ___",
            "  Student clarity (fill in):    ___",
            "  Student alignment (fill in):  ___",
            "  Student usefulness (fill in): ___",
            "",
            "=" * 65,
            "",
        ]

    path = RESULTS_DIR / "human_eval_guide.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Evaluator guide    → {path}")


# ══════════════════════════════════════════════════════════════════════════
# 3. ANALYZE COLLECTED SCORES
# ══════════════════════════════════════════════════════════════════════════

def analyze_scores():
    print("=" * 60)
    print("EXPERIMENT 4 — Analyzing Human Evaluation Scores")
    print("=" * 60)

    packet_path = RESULTS_DIR / "human_eval_packet.json"
    scores_path = RESULTS_DIR / "human_eval_scores.json"

    for p in [packet_path, scores_path]:
        if not p.exists():
            print(f"ERROR: {p} not found.")
            if p == scores_path:
                print("  Run --prepare first, collect scores, save as human_eval_scores.json")
            return

    with open(packet_path, encoding="utf-8") as f:
        packet = {item["item_id"]: item for item in json.load(f)}
    with open(scores_path, encoding="utf-8") as f:
        scores = json.load(f)

    # merged = []
    # for sc in scores:
    #     item = packet.get(sc["item_id"], {}).copy()
    #     item.update({k: v for k, v in sc.items() if v is not None})
    #     merged.append(item)
    
    
    from collections import defaultdict
    item_scores = defaultdict(lambda: {
        "teacher_rubric": [], "student_clarity": [],
        "student_alignment": [], "student_usefulness": []
    })

    for sc in scores:
        iid = sc["item_id"]
        for field in ["teacher_rubric", "student_clarity",
                      "student_alignment", "student_usefulness"]:
            if sc.get(field) is not None:
                item_scores[iid][field].append(sc[field])

    merged = []
    for iid, score_lists in item_scores.items():
        item = packet.get(iid, {"item_id": iid}).copy()
        item["teacher_rubric"]    = round(np.mean(score_lists["teacher_rubric"]), 3) if score_lists["teacher_rubric"] else None
        item["student_clarity"]   = round(np.mean(score_lists["student_clarity"]), 3) if score_lists["student_clarity"] else None
        item["student_alignment"] = round(np.mean(score_lists["student_alignment"]), 3) if score_lists["student_alignment"] else None
        item["student_usefulness"]= round(np.mean(score_lists["student_usefulness"]), 3) if score_lists["student_usefulness"] else None
        item["evaluator_id"]      = f"{len(score_lists['teacher_rubric'])}T_{len(score_lists['student_clarity'])}S"
        merged.append(item)

    print(f"Loaded {len(merged)} unique items (averaged across evaluators)\n")

    print(f"Loaded {len(merged)} scored items\n")

    results = {
        "timestamp":        datetime.now().isoformat(),
        "n_items":          len(merged),
        "n_evaluators":     len({m.get("evaluator_id") for m in merged
                                 if m.get("evaluator_id")}),
        "rubric_comparison": _rubric_comparison(merged),
        "likert_summary":    _likert_summary(merged),
        "cohens_kappa":      _cohens_kappa(merged),
        "rag_vs_baseline":   _rag_vs_baseline(merged),
        "per_chapter_type":  _per_chapter_type(merged),
    }

    out_path = RESULTS_DIR / "experiment4_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Analysis saved → {out_path}")

    _save_analysis_csv(merged)
    _print_summary(results)


def _rubric_comparison(merged: list[dict]) -> dict:
    by_setting = defaultdict(list)
    for m in merged:
        if m.get("setting"):
            by_setting[m["setting"]].append(m)
    out = {}
    for setting, items in by_setting.items():
        tr = [i["teacher_rubric"]    for i in items if i.get("teacher_rubric")    is not None]
        ar = [i["auto_rubric"]       for i in items if i.get("auto_rubric")       is not None]
        sc = [i["step_correctness"]  for i in items if i.get("step_correctness")  is not None]
        out[setting] = {
            "mean_teacher_rubric":   round(np.mean(tr), 3) if tr else None,
            "mean_auto_rubric":      round(np.mean(ar), 3) if ar else None,
            "mean_step_correctness": round(np.mean(sc), 3) if sc else None,
            "n":                     len(tr),
        }
    return out


def _likert_summary(merged: list[dict]) -> dict:
    dims = ["student_clarity", "student_alignment", "student_usefulness"]
    out  = {"by_setting": {}, "by_chapter": {}}

    by_setting = defaultdict(list)
    by_chapter = defaultdict(list)
    for m in merged:
        if m.get("setting"):      by_setting[m["setting"]].append(m)
        if m.get("chapter_type"): by_chapter[m["chapter_type"]].append(m)

    for groups, target in [(by_setting, out["by_setting"]),
                           (by_chapter, out["by_chapter"])]:
        for grp, items in groups.items():
            target[grp] = {}
            for dim in dims:
                vals = [i[dim] for i in items if i.get(dim) is not None]
                target[grp][dim] = round(np.mean(vals), 3) if vals else None

    return out


def _cohens_kappa(merged: list[dict]) -> dict:
    tr_vals, ar_vals, sc_vals = [], [], []
    for m in merged:
        if m.get("teacher_rubric") is not None and m.get("auto_rubric") is not None:
            tr_vals.append(m["teacher_rubric"])
            ar_vals.append(m["auto_rubric"])
        if m.get("teacher_rubric") is not None and m.get("step_correctness") is not None:
            sc_vals.append((m["teacher_rubric"], m["step_correctness"]))

    kappa = None
    if tr_vals:
        try:
            from sklearn.metrics import cohen_kappa_score
            kappa = round(float(cohen_kappa_score(tr_vals, ar_vals)), 4)
        except ImportError:
            kappa = round(_manual_kappa(tr_vals, ar_vals), 4)

    sc_corr = None
    if sc_vals:
        tr = [x[0] for x in sc_vals]
        sc = [x[1] for x in sc_vals]
        sc_corr = round(float(np.corrcoef(tr, sc)[0, 1]), 4)

    return {
        "teacher_vs_auto_rubric_kappa":     kappa,
        "teacher_vs_step_correctness_corr": sc_corr,
        "n_pairs":                          len(tr_vals),
    }


def _manual_kappa(y1: list, y2: list) -> float:
    classes = list(set(y1 + y2))
    n       = len(y1)
    p_obs   = sum(a == b for a, b in zip(y1, y2)) / n
    p_exp   = sum((y1.count(c) / n) * (y2.count(c) / n) for c in classes)
    return (p_obs - p_exp) / (1 - p_exp) if (1 - p_exp) > 0 else 0.0


def _rag_vs_baseline(merged: list[dict]) -> dict:
    by_question = defaultdict(dict)
    for m in merged:
        if m.get("test_id") is not None and m.get("setting"):
            by_question[m["test_id"]][m["setting"]] = m

    gains = {"teacher_rubric": [], "auto_rubric": [], "step_correctness": []}
    for qid, settings in by_question.items():
        rag  = settings.get("RAG", {})
        base = settings.get("Baseline", {})
        for metric in gains:
            rv = rag.get(metric)
            bv = base.get(metric)
            if rv is not None and bv is not None:
                try:
                    gains[metric].append(float(rv) - float(bv))
                except (TypeError, ValueError):
                    pass

    return {
        metric: {
            "mean_gain": round(np.mean(vals), 4) if vals else None,
            "positive":  sum(1 for v in vals if v > 0),
            "negative":  sum(1 for v in vals if v < 0),
            "neutral":   sum(1 for v in vals if v == 0),
            "n_pairs":   len(vals),
        }
        for metric, vals in gains.items()
    }


def _per_chapter_type(merged: list[dict]) -> dict:
    by_chapter = defaultdict(list)
    for m in merged:
        if m.get("chapter_type"):
            by_chapter[m["chapter_type"]].append(m)
    out = {}
    for ctype, items in by_chapter.items():
        tr = [i["teacher_rubric"]   for i in items if i.get("teacher_rubric")   is not None]
        cl = [i["student_clarity"]  for i in items if i.get("student_clarity")  is not None]
        out[ctype] = {
            "mean_teacher_rubric":   round(np.mean(tr), 3) if tr else None,
            "mean_student_clarity":  round(np.mean(cl), 3) if cl else None,
            "n":                     len(items),
        }
    return out


def _save_analysis_csv(merged: list[dict]):
    fields = [
        "item_id", "test_id", "chapter_type", "difficulty", "question_type",
        "setting", "teacher_rubric", "auto_rubric", "step_correctness",
        "student_clarity", "student_alignment", "student_usefulness",
        "evaluator_id",
    ]
    path = RESULTS_DIR / "experiment4_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k) for k in fields})
    print(f"Summary CSV saved → {path}")


def _print_summary(results: dict):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    rc = results.get("rubric_comparison", {})
    for setting, vals in rc.items():
        print(f"\n{setting}:")
        print(f"  Teacher rubric (mean):   {vals.get('mean_teacher_rubric')}")
        print(f"  Auto rubric (mean):      {vals.get('mean_auto_rubric')}")
        print(f"  Step correctness (mean): {vals.get('mean_step_correctness')}")

    kappa = results.get("cohens_kappa", {})
    print(f"\nCohen's κ (teacher vs auto rubric):  {kappa.get('teacher_vs_auto_rubric_kappa')}")
    print(f"Pearson r (teacher vs step-correct): {kappa.get('teacher_vs_step_correctness_corr')}")

    rvb = results.get("rag_vs_baseline", {})
    print("\nRAG vs Baseline gain:")
    for metric, vals in rvb.items():
        print(f"  {metric:<25}: mean_gain={vals.get('mean_gain')}  "
              f"+={vals.get('positive')}  -={vals.get('negative')}")

    ls = results.get("likert_summary", {}).get("by_setting", {})
    print("\nStudent Likert scores (1-5):")
    for setting, dims in ls.items():
        print(f"  {setting}: clarity={dims.get('student_clarity')}  "
              f"alignment={dims.get('student_alignment')}  "
              f"usefulness={dims.get('student_usefulness')}")


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 4 — Human Evaluation")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true",
                       help="Generate response packet for human evaluators")
    group.add_argument("--analyze", action="store_true",
                       help="Analyze collected human evaluation scores")
    args = parser.parse_args()

    if args.prepare:
        prepare_eval_packet()
    elif args.analyze:
        analyze_scores()
