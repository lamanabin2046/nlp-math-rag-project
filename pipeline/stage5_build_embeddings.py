"""
pipeline/stage5_build_embeddings.py
─────────────────────────────────────
Builds the 2048-d multimodal embedding for each worked example:

    BERT(text)   → 768d
    CLIP(image)  → 512d   (zero-padded if no associated image)
    BERT(desc)   → 768d   (zero-padded if no image description)
    ─────────────────────
    concat       → 2048d  → indexed in ChromaDB

Also stores equal-weight and a flag for learned-weight ablation (Exp 2).

Output: data/chroma_db/  (ChromaDB persistent store)
        data/processed/embedding_index.json  (id → metadata map)

Run:
    python pipeline/stage5_build_embeddings.py
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
import chromadb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR, CHROMA_DIR, CHROMA_COLLECTION,
    BERT_MODEL, CLIP_MODEL,
    BERT_DIM, CLIP_DIM, DESC_DIM, TOTAL_DIM,
)


def load_models():
    print(f"Loading BERT  : {BERT_MODEL}")
    bert = SentenceTransformer(BERT_MODEL)

    print(f"Loading CLIP  : {CLIP_MODEL}")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model     = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    return bert, clip_model, clip_processor, device


def embed_text_bert(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_image_clip(model, processor, image_path: str, device: str) -> np.ndarray:
    try:
        image  = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        vec = feats.detach().cpu().numpy().squeeze()
        # L2 normalise
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec.astype(np.float32)
    except Exception as e:
        print(f"  CLIP error on {image_path}: {e}")
        return np.zeros(CLIP_DIM, dtype=np.float32)


def build_embedding(
    text_emb: np.ndarray,     # 768d
    image_emb: np.ndarray,    # 512d  (or zeros)
    desc_emb: np.ndarray,     # 768d  (or zeros)
) -> np.ndarray:
    """Concatenate into 2048-d vector."""
    return np.concatenate([text_emb, image_emb, desc_emb]).astype(np.float32)


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
        images_list = json.load(f)

    # Build image_id → record lookup
    image_by_id = {img["image_id"]: img for img in images_list}

    examples = content["examples"]
    print(f"Building embeddings for {len(examples)} examples …\n")

    bert, clip_model, clip_processor, device = load_models()

    # ── ChromaDB setup ─────────────────────────────────────────────────────
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection to rebuild cleanly
    try:
        chroma_client.delete_collection(CHROMA_COLLECTION)
        print(f"Deleted existing ChromaDB collection: {CHROMA_COLLECTION}")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    index_records = []

    BATCH = 32
    for start in tqdm(range(0, len(examples), BATCH), desc="Embedding batches"):
        batch = examples[start: start + BATCH]

        # ── Text embedding (BERT) ──────────────────────────────────────────
        texts    = [f"{ex['problem']} {ex['solution']}" for ex in batch]
        text_embs = embed_text_bert(bert, texts)   # (B, 768)

        ids        = []
        embeddings = []
        metadatas  = []
        documents  = []

        for i, ex in enumerate(batch):
            global_idx = start + i
            doc_id     = f"example_{global_idx:05d}"

            text_emb = text_embs[i]

            # ── CLIP embedding (best linked image, or zeros) ───────────────
            image_emb = np.zeros(CLIP_DIM, dtype=np.float32)
            best_img_path = None
            if ex.get("images"):
                # Pick the highest-similarity linked image
                best = max(ex["images"], key=lambda x: x.get("similarity", 0))
                img_rec = image_by_id.get(best["image_id"])
                if img_rec and Path(img_rec["path"]).exists():
                    image_emb     = embed_image_clip(clip_model, clip_processor,
                                                     img_rec["path"], device)
                    best_img_path = img_rec["path"]

            # ── Description embedding (BERT over GPT-4V text) ─────────────
            desc_emb = np.zeros(DESC_DIM, dtype=np.float32)
            desc_text = ""
            if ex.get("images"):
                best_img_rec = image_by_id.get(ex["images"][0]["image_id"])
                if best_img_rec and best_img_rec.get("description"):
                    desc_text = best_img_rec["description"]
                    desc_emb  = embed_text_bert(bert, [desc_text])[0]

            # ── Concatenate → 2048d ────────────────────────────────────────
            combined = build_embedding(text_emb, image_emb, desc_emb)

            ids.append(doc_id)
            embeddings.append(combined.tolist())
            metadatas.append({
                "chapter_num":    ex.get("chapter_num", 0),
                "chapter_title":  ex.get("chapter_title", ""),
                "chapter_type":   ex.get("chapter_type", "other"),
                "example_index":  ex.get("example_index", 0),
                "page":           ex.get("page", 0),
                "visual_required": ex.get("visual_required", False),
                "has_image":      bool(ex.get("images")),
                "image_path":     best_img_path or "",
            })
            documents.append(texts[i])

            index_records.append({
                "doc_id":         doc_id,
                "chapter_num":    ex.get("chapter_num"),
                "chapter_type":   ex.get("chapter_type"),
                "example_index":  ex.get("example_index"),
                "page":           ex.get("page"),
                "visual_required": ex.get("visual_required", False),
                "has_image":      bool(ex.get("images")),
                "text_dim":       BERT_DIM,
                "image_dim":      CLIP_DIM,
                "desc_dim":       DESC_DIM,
                "total_dim":      TOTAL_DIM,
            })

        # ── Add batch to ChromaDB ─────────────────────────────────────────
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    print(f"\nIndexed {collection.count()} documents into ChromaDB")
    print(f"  Collection : {CHROMA_COLLECTION}")
    print(f"  Path       : {CHROMA_DIR}")

    # ── Save index map ─────────────────────────────────────────────────────
    index_path = PROCESSED_DIR / "embedding_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_records, f, ensure_ascii=False, indent=2)
    print(f"  Index map  : {index_path}")


if __name__ == "__main__":
    main()
