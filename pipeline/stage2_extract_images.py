"""
pipeline/stage2_extract_images.py
───────────────────────────────────
Extracts all embedded images from the textbook PDF.
Saves each image to data/images/ and records metadata.

Output: data/processed/images_metadata.json

Run:
    python pipeline/stage2_extract_images.py
"""

import json
import fitz
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TEXTBOOK_PDF, IMAGES_DIR, PROCESSED_DIR, IMAGE_MIN_WIDTH, IMAGE_MIN_HEIGHT


def extract_images(pdf_path: Path) -> list[dict]:
    doc   = fitz.open(str(pdf_path))
    total = len(doc)
    print(f"Scanning {total} pages for images …")

    records = []
    img_id  = 0

    for page_num in tqdm(range(total)):
        page      = doc[page_num]
        img_list  = page.get_images(full=True)

        for img_index, img_info in enumerate(img_list):
            xref       = img_info[0]
            base_image = doc.extract_image(xref)
            img_bytes  = base_image["image"]
            ext        = base_image["ext"]        # png, jpeg, etc.

            # Filter out tiny decorative images
            try:
                pil = Image.open(BytesIO(img_bytes))
                w, h = pil.size
            except Exception:
                continue

            if w < IMAGE_MIN_WIDTH or h < IMAGE_MIN_HEIGHT:
                continue

            # Get bounding box on the page
            rects = page.get_image_rects(xref)
            bbox  = list(rects[0]) if rects else [0, 0, 0, 0]

            # Save image file
            filename  = f"img_{img_id:04d}_p{page_num+1}.{ext}"
            save_path = IMAGES_DIR / filename
            with open(save_path, "wb") as f:
                f.write(img_bytes)

            records.append({
                "image_id":     img_id,
                "filename":     filename,
                "path":         str(save_path),
                "page":         page_num + 1,
                "bbox":         bbox,           # [x0, y0, x1, y1] in PDF points
                "width_px":     w,
                "height_px":    h,
                "format":       ext,
                "description":  None,           # filled in stage 3
                "linked_examples": [],          # filled in stage 4
            })
            img_id += 1

    doc.close()
    print(f"\nExtracted {len(records)} images → {IMAGES_DIR}")
    return records


def main():
    if not TEXTBOOK_PDF.exists():
        print(f"ERROR: PDF not found at {TEXTBOOK_PDF}")
        return

    records  = extract_images(TEXTBOOK_PDF)
    out_path = PROCESSED_DIR / "images_metadata.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
