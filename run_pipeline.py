"""
run_pipeline.py
────────────────
Runs all 6 pipeline stages in order.
Safe to re-run — each stage skips work that is already done.

Usage:
    python run_pipeline.py              # run all stages
    python run_pipeline.py --from 3    # resume from stage 3
    python run_pipeline.py --only 5    # run only stage 5
"""

import sys
import argparse
import importlib
from pathlib import Path


STAGES = [
    (1, "pipeline.stage1_extract_content",   "Extract text content (chapters, examples, exercises)"),
    (2, "pipeline.stage2_extract_images",    "Extract images from PDF"),
    (3, "pipeline.stage3_describe_images",   "Math-aware GPT-4V diagram descriptions  [$$ costs API calls]"),
    (4, "pipeline.stage4_link_examples",     "Link images to worked examples"),
    (5, "pipeline.stage5_build_embeddings",  "Build 2048-d embeddings + index in ChromaDB"),
    (6, "pipeline.stage6_create_testset",    "Create stratified 100-question test set"),
]


def run_stage(num: int, module: str, label: str):
    print(f"\n{'='*60}")
    print(f"STAGE {num}: {label}")
    print(f"{'='*60}")
    mod = importlib.import_module(module)
    mod.main()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from",  type=int, default=1,    dest="from_stage",
                        help="Start from this stage number (default: 1)")
    parser.add_argument("--only",  type=int, default=None, dest="only_stage",
                        help="Run only this stage number")
    args = parser.parse_args()

    if args.only_stage:
        stages_to_run = [s for s in STAGES if s[0] == args.only_stage]
    else:
        stages_to_run = [s for s in STAGES if s[0] >= args.from_stage]

    if not stages_to_run:
        print("No stages to run.")
        return

    print("Pipeline stages to run:")
    for num, _, label in stages_to_run:
        print(f"  Stage {num}: {label}")

    for num, module, label in stages_to_run:
        run_stage(num, module, label)

    print(f"\n{'='*60}")
    print("Pipeline complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
