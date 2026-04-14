"""
pipeline/stage6_create_testset.py
───────────────────────────────────
Creates the 100-question stratified test set from exercise questions:

  Difficulty   : 33 easy | 34 medium | 33 hard
  Type         : 40 computational | 35 conceptual | 25 proof-based
  Chapter type : geometry | algebra | word_problems

Difficulty is heuristically assigned based on question length and
keyword signals (e.g. "prove", "find", "simplify").
visual_required is inherited from the image linking stage.

Output: data/test_set/test_set.json

Run:
    python pipeline/stage6_create_testset.py
"""

import json
import random
import re
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR, TESTSET_DIR,
    TEST_SET_SIZE, TEST_EASY, TEST_MEDIUM, TEST_HARD,
)

random.seed(42)

# ── Heuristic classifiers ──────────────────────────────────────────────────
PROOF_KEYWORDS  = re.compile(r"\b(prove|show that|verify|establish|demonstrate)\b", re.I)
COMPUTE_KEYWORDS = re.compile(r"\b(find|calculate|evaluate|solve|simplify|factorise|expand)\b", re.I)

def classify_type(text: str) -> str:
    if PROOF_KEYWORDS.search(text):
        return "proof"
    if COMPUTE_KEYWORDS.search(text):
        return "computational"
    return "conceptual"

def classify_difficulty(text: str) -> str:
    words = len(text.split())
    has_proof   = bool(PROOF_KEYWORDS.search(text))
    has_multi   = bool(re.search(r"\band\b.*\band\b", text, re.I))  # multiple conditions
    has_complex = bool(re.search(r"\b(hence|deduce|therefore|using|given that)\b", text, re.I))

    if has_proof or (has_multi and has_complex):
        return "hard"
    if words > 25 or has_complex:
        return "medium"
    return "easy"


def main():
    content_path = PROCESSED_DIR / "content.json"
    if not content_path.exists():
        print("ERROR: content.json not found. Run previous stages first.")
        return

    with open(content_path, encoding="utf-8") as f:
        content = json.load(f)

    # ── Flatten all questions ─────────────────────────────────────────────
    all_questions = []
    qid = 0
    for exercise in content["exercises"]:
        for q in exercise["questions"]:
            text  = q["text"].strip()
            if len(text) < 10:
                continue
            all_questions.append({
                "qid":           qid,
                "text":          text,
                "page":          q["page"],
                "chapter_num":   exercise["chapter_num"],
                "chapter_title": exercise["chapter_title"],
                "chapter_type":  exercise["chapter_type"],
                "exercise_num":  exercise["exercise_num"],
                "difficulty":    classify_difficulty(text),
                "question_type": classify_type(text),
                "visual_required": q.get("visual_required", False),
                "ground_truth":  None,   # to be filled manually or via LLM
            })
            qid += 1

    print(f"Total questions available: {len(all_questions)}")

    # ── Stratify by difficulty ────────────────────────────────────────────
    by_diff = defaultdict(list)
    for q in all_questions:
        by_diff[q["difficulty"]].append(q)

    print(f"  easy   : {len(by_diff['easy'])}")
    print(f"  medium : {len(by_diff['medium'])}")
    print(f"  hard   : {len(by_diff['hard'])}")

    def sample_safe(pool, n):
        return random.sample(pool, min(n, len(pool)))

    selected = (
        sample_safe(by_diff["easy"],   TEST_EASY)   +
        sample_safe(by_diff["medium"], TEST_MEDIUM)  +
        sample_safe(by_diff["hard"],   TEST_HARD)
    )

    # If we got fewer than 100 due to small pool, top up from any difficulty
    if len(selected) < TEST_SET_SIZE:
        selected_ids = {q["qid"] for q in selected}
        remaining    = [q for q in all_questions if q["qid"] not in selected_ids]
        selected    += sample_safe(remaining, TEST_SET_SIZE - len(selected))

    # Shuffle final set
    random.shuffle(selected)

    # ── Re-assign sequential IDs ──────────────────────────────────────────
    for i, q in enumerate(selected):
        q["test_id"] = i

    # ── Summary ───────────────────────────────────────────────────────────
    def count(field, val):
        return sum(1 for q in selected if q[field] == val)

    print(f"\nTest set ({len(selected)} questions):")
    print(f"  easy          : {count('difficulty','easy')}")
    print(f"  medium        : {count('difficulty','medium')}")
    print(f"  hard          : {count('difficulty','hard')}")
    print(f"  computational : {count('question_type','computational')}")
    print(f"  conceptual    : {count('question_type','conceptual')}")
    print(f"  proof         : {count('question_type','proof')}")
    print(f"  geometry      : {count('chapter_type','geometry')}")
    print(f"  algebra       : {count('chapter_type','algebra')}")
    print(f"  word_problems : {count('chapter_type','word_problems')}")
    print(f"  visual_req    : {sum(1 for q in selected if q['visual_required'])}")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = TESTSET_DIR / "test_set.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
