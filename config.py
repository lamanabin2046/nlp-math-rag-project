"""
config.py — central configuration for the data pipeline.
Copy .env.example to .env and fill in your API keys.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
IMAGES_DIR    = DATA_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"
TESTSET_DIR   = DATA_DIR / "test_set"
CHROMA_DIR    = DATA_DIR / "chroma_db"

for d in [RAW_DIR, IMAGES_DIR, PROCESSED_DIR, TESTSET_DIR, CHROMA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Textbook PDF ───────────────────────────────────────────────────────────
TEXTBOOK_PDF = RAW_DIR / "class10_math.pdf"

# ── API Keys ───────────────────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HF_TOKEN          = os.getenv("HF_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Models ─────────────────────────────────────────────────────────────────
VISION_MODEL = "gpt-4o"
BERT_MODEL   = "sentence-transformers/all-mpnet-base-v2"
CLIP_MODEL   = "openai/clip-vit-base-patch32"

# ── Embedding dimensions ───────────────────────────────────────────────────
BERT_DIM  = 768
CLIP_DIM  = 512
DESC_DIM  = 768
TOTAL_DIM = BERT_DIM + CLIP_DIM + DESC_DIM   # 2048

# ── ChromaDB ───────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "nepal_math_g10"

# ── Pipeline parameters ────────────────────────────────────────────────────
TOP_K_RETRIEVAL  = 5
IMAGE_MIN_WIDTH  = 50
IMAGE_MIN_HEIGHT = 50
LINK_PAGE_WINDOW = 2
LINK_TOP_K       = 3

# ── Test set ───────────────────────────────────────────────────────────────
TEST_SET_SIZE = 100
TEST_EASY     = 33
TEST_MEDIUM   = 34
TEST_HARD     = 33

# ── Chapter type mapping ───────────────────────────────────────────────────
CHAPTER_TYPES = {
    "geometry":      [5, 6, 7, 13, 14, 15, 16],
    "algebra":       [1, 8, 9, 10, 11, 12],
    "word_problems": [2, 3, 4],
}