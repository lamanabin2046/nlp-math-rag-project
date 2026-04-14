"""
pipeline/stage3_describe_images.py
────────────────────────────────────
For each extracted image, calls GPT-4o Vision with a structured
math-aware prompt to produce:
  - diagram_type  (e.g. "triangle", "circle", "graph", "table")
  - labels        (e.g. ["AB = 5cm", "angle ACB = 60°"])
  - latex_elements (e.g. ["\\triangle ABC", "\\angle = 60^\\circ"])
  - description   (plain English summary)

Output: data/processed/images_metadata.json  (updated in-place)

Run:
    python pipeline/stage3_describe_images.py
"""

import json
import base64
import time
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, OPENAI_API_KEY, VISION_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

MATH_AWARE_PROMPT = """You are analyzing a diagram from a Nepal Grade 10 Mathematics textbook.
Provide a structured description with EXACTLY these fields:

diagram_type: one of [triangle, circle, quadrilateral, polygon, graph, number_line, 
               table, construction, solid_3d, coordinate_axes, other]

labels: list every measurement, label, or annotation visible in the diagram.
        Use exact notation e.g. ["AB = 5 cm", "angle ACB = 60 degrees", "x = 3"].
        If none visible, return [].

latex_elements: express the key geometric/algebraic relationships in LaTeX.
        e.g. ["\\\\triangle ABC", "\\\\angle ACB = 60^\\\\circ", "AB = 5\\\\text{ cm}"]
        If none, return [].

description: one paragraph in plain English describing what the diagram shows,
        what the key geometric or algebraic relationships are, and what
        mathematical concept it illustrates.

visual_required: true if a student CANNOT solve the associated problem without
        seeing this diagram; false if the diagram is purely decorative or supplementary.

Respond ONLY with valid JSON, no markdown, no extra text."""


def encode_image(image_path: str) -> tuple[str, str]:
    """Convert image to PNG and return (base64_data, media_type).
    Forces PNG conversion to handle unsupported formats like jb2/jpx."""
    try:
        img    = Image.open(image_path).convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
        return data, "image/png"
    except Exception as e:
        raise ValueError(f"Could not open image {image_path}: {e}")


def describe_image(image_path: str, retries: int = 3) -> dict:
    """Call GPT-4o Vision and return the structured description dict."""
    try:
        data, mime = encode_image(image_path)
    except ValueError as e:
        print(f"  Skipping — {e}")
        return {"diagram_type": "other", "labels": [], "latex_elements": [],
                "description": "Image could not be opened.", "visual_required": False}

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=VISION_MODEL,
                max_tokens=800,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": MATH_AWARE_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url":    f"data:{mime};base64,{data}",
                            "detail": "high"
                        }},
                    ],
                }],
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt+1}): {e}")
            if attempt == retries - 1:
                return {"diagram_type": "other", "labels": [], "latex_elements": [],
                        "description": "Parse error.", "visual_required": False}
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            if attempt == retries - 1:
                return {"diagram_type": "other", "labels": [], "latex_elements": [],
                        "description": "API error.", "visual_required": False}
            time.sleep(2 ** attempt)

    return {"diagram_type": "other", "labels": [], "latex_elements": [],
            "description": "Failed after retries.", "visual_required": False}


def build_rich_description(parsed: dict) -> str:
    """Combine all structured fields into a single searchable text string."""
    parts = []
    if parsed.get("diagram_type"):
        parts.append(f"Diagram type: {parsed['diagram_type']}.")
    if parsed.get("labels"):
        parts.append("Labels: " + ", ".join(parsed["labels"]) + ".")
    if parsed.get("latex_elements"):
        parts.append("Notation: " + " ".join(parsed["latex_elements"]) + ".")
    if parsed.get("description"):
        parts.append(parsed["description"])
    return " ".join(parts)


def main():
    meta_path = PROCESSED_DIR / "images_metadata.json"
    if not meta_path.exists():
        print("ERROR: Run stage 2 first (images_metadata.json not found)")
        return

    with open(meta_path, encoding="utf-8") as f:
        records = json.load(f)

    pending = [r for r in records if r.get("description") is None]
    print(f"{len(pending)} images need describing  ({len(records) - len(pending)} already done)")

    for record in tqdm(pending, desc="Describing images"):
        parsed = describe_image(record["path"])

        record["description"]     = build_rich_description(parsed)
        record["diagram_type"]    = parsed.get("diagram_type", "other")
        record["labels"]          = parsed.get("labels", [])
        record["latex_elements"]  = parsed.get("latex_elements", [])
        record["visual_required"] = parsed.get("visual_required", False)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        time.sleep(0.3)

    print(f"\nDescriptions complete. Updated → {meta_path}")
    described = sum(1 for r in records if r.get("description"))
    print(f"  {described} / {len(records)} images described")


if __name__ == "__main__":
    main()