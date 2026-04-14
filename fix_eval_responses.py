"""
fix_eval_responses.py
─────────────────────
Regenerates empty responses in human_eval_packet.json.
Run from your project root:
    python fix_eval_responses.py
"""

import json
import time
import sys
from pathlib import Path

import openai
import anthropic

sys.path.insert(0, str(Path(__file__).parent))
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY

RESULTS_DIR = Path(__file__).parent / "results"
PACKET_PATH = RESULTS_DIR / "human_eval_packet.json"

RAG_PROMPT = """You are a mathematics tutor for Grade 10 students in Nepal.
Use the worked examples below as reference to solve the problem.
Show clear step-by-step working aligned with the Nepal curriculum.
Write your solution as a Grade 10 teacher would — numbered steps, clear
intermediate results, final answer clearly stated.

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
Solution: {solution}

Respond ONLY with:
RUBRIC: <0, 1, 2, or 3>
FEEDBACK: <one sentence>"""


def call_llm(prompt: str, model: str) -> str:
    """Call LLM and return response text. Raises exception on failure."""
    if model == "gpt-4o":
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        resp   = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    elif model == "claude-opus":
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp   = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    else:
        raise ValueError(f"Unknown model: {model}")


def auto_grade(question: str, solution: str) -> dict:
    import re
    try:
        raw = call_llm(
            AUTO_GRADE_PROMPT.format(question=question, solution=solution),
            model="gpt-4o"
        )
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


def main():
    print("=" * 60)
    print("Fix: Regenerating empty responses in human_eval_packet.json")
    print("=" * 60)

    with open(PACKET_PATH, encoding="utf-8") as f:
        packet = json.load(f)

    empty_items = [item for item in packet if not item.get("response")]
    print(f"\nFound {len(empty_items)} empty responses out of {len(packet)} total\n")

    if not empty_items:
        print("Nothing to fix — all responses already filled.")
        return

    # First test API connectivity
    print("Testing API connections...")
    try:
        test = call_llm("Say hello in one word.", "claude-opus")
        print(f"  Claude API: OK  (response: {test[:30]})")
    except Exception as e:
        print(f"  Claude API: FAILED — {e}")
        print("  Check your ANTHROPIC_API_KEY in config.py")

    try:
        test = call_llm("Say hello in one word.", "gpt-4o")
        print(f"  GPT-4o API: OK  (response: {test[:30]})")
    except Exception as e:
        print(f"  GPT-4o API: FAILED — {e}")
        print("  Check your OPENAI_API_KEY in config.py")

    print()

    # Regenerate each empty response
    for item in packet:
        if item.get("response"):
            print(f"  Item {item['item_id']:>2} — already has response, skipping")
            continue

        model    = item["model"]
        setting  = item["setting"]
        question = item["question"]

        print(f"  Item {item['item_id']:>2} | {setting:<9} | {item['chapter_type']:<14} | {item['difficulty']}")

        prompt = RAG_PROMPT.format(question=question) if setting == "RAG" \
                 else BASELINE_PROMPT.format(question=question)

        response = ""
        for attempt in range(3):
            try:
                response = call_llm(prompt, model)
                print(f"    Response generated ({len(response)} chars)")
                break
            except Exception as e:
                print(f"    Attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

        if not response:
            print(f"    SKIPPING — could not generate response after 3 attempts")
            continue

        # Auto-grade
        grade = auto_grade(question, response)
        steps = compute_step_correctness(response)

        # Update item
        item["response"]         = response
        item["auto_rubric"]      = grade["auto_rubric"]
        item["auto_feedback"]    = grade["auto_feedback"]
        item["step_correctness"] = steps

        # Save after every item
        with open(PACKET_PATH, "w", encoding="utf-8") as f:
            json.dump(packet, f, ensure_ascii=False, indent=2)
        print(f"    Saved. Auto-rubric={grade['auto_rubric']}, Steps={steps}")

    # Regenerate guide with filled responses
    _save_updated_guide(packet)

    print("\n" + "=" * 60)
    print("Done! Check results/human_eval_packet.json")
    print("      and results/human_eval_guide.txt")
    print("=" * 60)


def _save_updated_guide(items: list):
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
        resp_display = item.get("response") or "[NO RESPONSE GENERATED]"
        lines += [
            f"ITEM {item['item_id']:>3}  |  {item['chapter_type'].upper()}  |  "
            f"{item['difficulty'].upper()}  |  {item['setting'].upper()}",
            "-" * 65,
            f"Question: {item['question']}",
            "",
            "Solution:",
            resp_display,
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

    guide_path = PACKET_PATH.parent / "human_eval_guide.txt"
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nUpdated guide saved → {guide_path}")


if __name__ == "__main__":
    main()
