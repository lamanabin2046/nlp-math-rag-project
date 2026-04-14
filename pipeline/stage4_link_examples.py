"""
pipeline/stage4_link_examples.py
──────────────────────────────────
Links each extracted diagram to its most relevant worked example(s)
using two signals:

  1. Page proximity  — diagrams on the same page or within LINK_PAGE_WINDOW
                       pages of an example are candidates.
  2. Semantic similarity — cosine similarity between the image description
                           embedding and the example text embedding.

Updates both images_metadata.json and content.json with cross-references.

Output: data/processed/images_metadata.json  (updated)
        data/processed/content.json          (examples get "images" field)

Run:
    python pipeline/stage4_link_examples.py
"""

import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, BERT_MODEL, LINK_PAGE_WINDOW, LINK_TOP_K


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def main():
    content_path = PROCESSED_DIR / "content.json"
    meta_path    = PROCESSED_DIR / "images_metadata.json"

    for p in [content_path, meta_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run previous stages first.")
            return

    with open(content_path, encoding="utf-8") as f:
        content = json.load(f)
    with open(meta_path, encoding="utf-8") as f:
        images = json.load(f)

    examples = content["examples"]
    print(f"Loaded {len(examples)} examples  |  {len(images)} images")

    # ── Encode example texts ───────────────────────────────────────────────
    print(f"\nLoading BERT model: {BERT_MODEL}")
    model = SentenceTransformer(BERT_MODEL)

    example_texts = [
        f"{e['problem']} {e['solution']}" for e in examples
    ]
    print("Encoding examples …")
    example_embs = model.encode(example_texts, show_progress_bar=True,
                                 normalize_embeddings=True)

    # ── Build page → example index map ────────────────────────────────────
    page_to_examples: dict[int, list[int]] = {}
    for idx, ex in enumerate(examples):
        pg = ex["page"]
        for offset in range(-LINK_PAGE_WINDOW, LINK_PAGE_WINDOW + 1):
            page_to_examples.setdefault(pg + offset, []).append(idx)

    # ── Link each image ────────────────────────────────────────────────────
    # Reset existing links
    for ex in examples:
        ex["images"] = []
    for img in images:
        img["linked_examples"] = []

    described_images = [img for img in images if img.get("description")]
    print(f"\nLinking {len(described_images)} described images …")

    desc_texts = [img["description"] for img in described_images]
    desc_embs  = model.encode(desc_texts, show_progress_bar=True,
                               normalize_embeddings=True)

    for img, desc_emb in tqdm(zip(described_images, desc_embs),
                               total=len(described_images), desc="Linking"):
        page      = img["page"]
        candidates = page_to_examples.get(page, [])

        if not candidates:
            continue

        # Score candidates by semantic similarity
        scored = []
        for idx in candidates:
            sim = cosine_sim(desc_emb, example_embs[idx])
            scored.append((sim, idx))

        scored.sort(reverse=True)
        top = scored[:LINK_TOP_K]

        # Accept links with similarity > 0.3
        linked = []
        for sim, idx in top:
            if sim > 0.30:
                linked.append({
                    "example_index": examples[idx]["example_index"],
                    "chapter_num":   examples[idx]["chapter_num"],
                    "page":          examples[idx]["page"],
                    "similarity":    round(sim, 4),
                })
                examples[idx]["images"].append({
                    "image_id":  img["image_id"],
                    "filename":  img["filename"],
                    "similarity": round(sim, 4),
                })
                # Mark example as visually required if image is
                if img.get("visual_required"):
                    examples[idx]["visual_required"] = True

        img["linked_examples"] = linked

    # ── Report ─────────────────────────────────────────────────────────────
    linked_imgs = sum(1 for img in images if img["linked_examples"])
    linked_exs  = sum(1 for ex in examples if ex.get("images"))
    print(f"\n  Images linked to at least one example : {linked_imgs}")
    print(f"  Examples linked to at least one image  : {linked_exs}")

    # ── Save ───────────────────────────────────────────────────────────────
    content["examples"] = examples
    with open(content_path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(images, f, ensure_ascii=False, indent=2)

    print(f"\nSaved → {content_path}")
    print(f"Saved → {meta_path}")


if __name__ == "__main__":
    main()
