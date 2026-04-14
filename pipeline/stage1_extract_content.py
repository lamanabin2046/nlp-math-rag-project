"""
pipeline/stage1_extract_content.py
"""

import re
import json
import fitz
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TEXTBOOK_PDF, PROCESSED_DIR, CHAPTER_TYPES

# ── Regex patterns ─────────────────────────────────────────────────────────
RE_CHAPTER  = re.compile(r"^Unit:\s*(\d+)\s*$")
RE_UNIT_NUM = re.compile(r"^(\d+)\.0\s*$")
RE_EXAMPLE  = re.compile(r"^Example\s+(\d+)[:\.]?\s*(.*)", re.IGNORECASE)
RE_EXERCISE = re.compile(r"^Exercise\s+([\d.]+)\s*$", re.IGNORECASE)
RE_SOLUTION = re.compile(r"^Solution[:\.]?\s*", re.IGNORECASE)
RE_QUESTION = re.compile(r"^(\d+|[ivxlIVXL]+|[a-hA-H])[\.\)]\s*(.*)")


def get_chapter_type(chapter_num: int) -> str:
    for ctype, chapters in CHAPTER_TYPES.items():
        if chapter_num in chapters:
            return ctype
    return "other"


def save_exercise(exercises, current_exercise):
    """Save exercise only if it has at least one question."""
    if current_exercise and current_exercise["questions"]:
        exercises.append(current_exercise)


def extract_content(pdf_path: Path) -> dict:
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    print(f"Opened PDF: {pdf_path.name}  ({total_pages} pages)")

    chapters  = []
    examples  = []
    exercises = []

    current_chapter_num   = None
    current_chapter_title = ""
    current_chapter_text  = []
    current_chapter_start = 1

    current_example  = None
    in_solution      = False
    current_exercise = None
    pending_question = None   # question number with no text yet

    for page_num in tqdm(range(total_pages), desc="Extracting text"):
        page   = doc[page_num]
        blocks = page.get_text("blocks")
        lines  = []
        for b in blocks:
            if b[6] == 0:
                lines.append((b[4].strip(), page_num + 1))

        for raw_text, pno in lines:
            for line in raw_text.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # ── Unit heading ───────────────────────────────────────────
                m = RE_CHAPTER.match(line)
                if m:
                    save_exercise(exercises, current_exercise)
                    if current_chapter_num is not None:
                        chapters.append({
                            "chapter_num":   current_chapter_num,
                            "chapter_title": current_chapter_title,
                            "chapter_type":  get_chapter_type(current_chapter_num),
                            "start_page":    current_chapter_start,
                            "end_page":      pno - 1,
                            "text":          " ".join(current_chapter_text),
                        })
                    current_chapter_num   = int(m.group(1))
                    current_chapter_title = ""
                    current_chapter_text  = []
                    current_chapter_start = pno
                    current_example       = None
                    current_exercise      = None
                    pending_question      = None
                    in_solution           = False
                    continue

                # ── Capture unit title ─────────────────────────────────────
                if current_chapter_num is not None and current_chapter_title == "":
                    if not RE_UNIT_NUM.match(line):
                        current_chapter_title = line
                    continue

                # ── Example heading ────────────────────────────────────────
                m = RE_EXAMPLE.match(line)
                if m:
                    save_exercise(exercises, current_exercise)
                    current_exercise = None
                    pending_question = None
                    if current_example:
                        examples.append(current_example)
                    current_example = {
                        "chapter_num":     current_chapter_num,
                        "chapter_title":   current_chapter_title,
                        "chapter_type":    get_chapter_type(current_chapter_num) if current_chapter_num else "other",
                        "example_index":   int(m.group(1)),
                        "page":            pno,
                        "problem":         m.group(2).strip(),
                        "solution":        "",
                        "visual_required": False,
                        "images":          [],
                    }
                    in_solution = False
                    continue

                # ── Solution marker ────────────────────────────────────────
                if RE_SOLUTION.match(line) and current_example:
                    in_solution = True
                    rest = RE_SOLUTION.sub("", line).strip()
                    if rest:
                        current_example["solution"] += rest + " "
                    continue

                # ── Exercise heading ───────────────────────────────────────
                m = RE_EXERCISE.match(line)
                if m:
                    save_exercise(exercises, current_exercise)
                    if current_example:
                        examples.append(current_example)
                        current_example = None
                    current_exercise = {
                        "chapter_num":   current_chapter_num,
                        "chapter_title": current_chapter_title,
                        "chapter_type":  get_chapter_type(current_chapter_num) if current_chapter_num else "other",
                        "exercise_num":  m.group(1),
                        "page":          pno,
                        "questions":     [],
                    }
                    pending_question = None
                    in_solution      = False
                    continue

                # ── Question inside exercise ───────────────────────────────
                if current_exercise:
                    m = RE_QUESTION.match(line)
                    if m:
                        text_on_same_line = m.group(2).strip()
                        if text_on_same_line:
                            # e.g. "1. Find the value..."
                            current_exercise["questions"].append({
                                "text":            line,
                                "page":            pno,
                                "chapter_type":    current_exercise["chapter_type"],
                                "visual_required": False,
                            })
                            pending_question = None
                        else:
                            # e.g. "1." alone — text comes on next line
                            pending_question = line
                        continue

                    if pending_question:
                        # This line is the text for the pending question number
                        current_exercise["questions"].append({
                            "text":            pending_question + " " + line,
                            "page":            pno,
                            "chapter_type":    current_exercise["chapter_type"],
                            "visual_required": False,
                        })
                        pending_question = None
                        continue

                    # Continuation of last question
                    if current_exercise["questions"]:
                        current_exercise["questions"][-1]["text"] += " " + line
                    continue

                # ── Accumulate chapter / example text ──────────────────────
                if current_example:
                    if in_solution:
                        current_example["solution"] += line + " "
                    else:
                        current_example["problem"] += " " + line
                elif current_chapter_num is not None:
                    current_chapter_text.append(line)

    # ── Flush remaining ────────────────────────────────────────────────────
    save_exercise(exercises, current_exercise)
    if current_example:
        examples.append(current_example)
    if current_chapter_num is not None:
        chapters.append({
            "chapter_num":   current_chapter_num,
            "chapter_title": current_chapter_title,
            "chapter_type":  get_chapter_type(current_chapter_num),
            "start_page":    current_chapter_start,
            "end_page":      total_pages,
            "text":          " ".join(current_chapter_text),
        })

    doc.close()

    content = {
        "chapters":  chapters,
        "examples":  examples,
        "exercises": exercises,
    }

    print(f"\n  Chapters  : {len(chapters)}")
    print(f"  Examples  : {len(examples)}")
    exercise_q_count = sum(len(e["questions"]) for e in exercises)
    print(f"  Exercises : {len(exercises)}  ({exercise_q_count} questions total)")

    return content


def main():
    if not TEXTBOOK_PDF.exists():
        print(f"ERROR: PDF not found at {TEXTBOOK_PDF}")
        return

    content = extract_content(TEXTBOOK_PDF)

    out_path = PROCESSED_DIR / "content.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()