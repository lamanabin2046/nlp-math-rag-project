"""
error_analysis.py
Run:
    python error_analysis.py > results/error_analysis_output.txt
"""


import json
import numpy as np
from collections import defaultdict
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('results/experiment1_results.json', encoding='utf-8') as f:
    data = json.load(f)

def avg(lst):
    return round(np.mean(lst), 4) if lst else 0

FAIL_THRESHOLD = 0.5
TOP3 = ['gpt-4o', 'claude-opus', 'qwen2-math']
CHAPTER_TYPES  = ['geometry', 'algebra', 'word_problems', 'other']
QUESTION_TYPES = ['computational', 'proof', 'conceptual']
DIFFICULTIES   = ['easy', 'medium', 'hard']
ALL_MODELS     = sorted(set(r['model'] for r in data))


# ==============================================================
# SECTION 1 - PERFORMANCE BREAKDOWN
# ==============================================================

print("=" * 60)
print("SECTION 1A - BY CHAPTER TYPE (all models, all settings)")
print("=" * 60)
by_chapter = defaultdict(list)
for r in data:
    by_chapter[r['chapter_type']].append(r['step_correctness'])
for k, v in sorted(by_chapter.items(), key=lambda x: -avg(x[1])):
    print(f"  {k:20} step={avg(v):.4f}  n={len(v)}")

print()
print("=" * 60)
print("SECTION 1B - BY QUESTION TYPE (all models, all settings)")
print("=" * 60)
by_qtype = defaultdict(list)
for r in data:
    by_qtype[r['question_type']].append(r['step_correctness'])
for k, v in sorted(by_qtype.items(), key=lambda x: -avg(x[1])):
    print(f"  {k:20} step={avg(v):.4f}  n={len(v)}")


# ==============================================================
# SECTION 2 - RAG vs BASELINE DELTA
# ==============================================================

print()
print("=" * 60)
print("SECTION 2A - RAG vs BASELINE BY CHAPTER TYPE (top 3 models)")
print("=" * 60)
for ctype in CHAPTER_TYPES:
    base = [r for r in data if r['chapter_type'] == ctype
            and r['setting'] == 'Baseline' and r['model'] in TOP3]
    rag  = [r for r in data if r['chapter_type'] == ctype
            and r['setting'] == 'RAG'      and r['model'] in TOP3]
    b  = avg([r['step_correctness'] for r in base])
    ra = avg([r['step_correctness'] for r in rag])
    direction = "^ RAG helps" if ra > b else ("v RAG hurts" if ra < b else "= no change")
    print(f"  {ctype:20} Baseline={b:.4f}  RAG={ra:.4f}  Delta={ra-b:+.4f}  {direction}")

print()
print("=" * 60)
print("SECTION 2B - RAG vs BASELINE BY QUESTION TYPE (top 3 models)")
print("=" * 60)
for qtype in QUESTION_TYPES:
    base = [r for r in data if r['question_type'] == qtype
            and r['setting'] == 'Baseline' and r['model'] in TOP3]
    rag  = [r for r in data if r['question_type'] == qtype
            and r['setting'] == 'RAG'      and r['model'] in TOP3]
    b  = avg([r['step_correctness'] for r in base])
    ra = avg([r['step_correctness'] for r in rag])
    direction = "^ RAG helps" if ra > b else ("v RAG hurts" if ra < b else "= no change")
    print(f"  {qtype:20} Baseline={b:.4f}  RAG={ra:.4f}  Delta={ra-b:+.4f}  {direction}")


# ==============================================================
# SECTION 3 - FAILURE RATE ANALYSIS
# ==============================================================

print()
print("=" * 60)
print(f"SECTION 3A - OVERALL FAILURE RATE (step_correctness < {FAIL_THRESHOLD})")
print("=" * 60)
failures_all = [r for r in data if r['step_correctness'] < FAIL_THRESHOLD]
print(f"  Total records : {len(data)}")
print(f"  Failures      : {len(failures_all)}  ({len(failures_all)/len(data):.1%})")

print()
print("FAILURE RATE BY CHAPTER TYPE")
print("-" * 40)
for ctype in CHAPTER_TYPES:
    subset = [r for r in data if r['chapter_type'] == ctype]
    fail   = [r for r in subset if r['step_correctness'] < FAIL_THRESHOLD]
    rate   = len(fail) / len(subset) if subset else 0
    bar    = "#" * int(rate * 20)
    print(f"  {ctype:20} {rate:5.1%}  ({len(fail):>2}/{len(subset):<2})  {bar}")

print()
print("FAILURE RATE BY QUESTION TYPE")
print("-" * 40)
for qtype in QUESTION_TYPES:
    subset = [r for r in data if r['question_type'] == qtype]
    fail   = [r for r in subset if r['step_correctness'] < FAIL_THRESHOLD]
    rate   = len(fail) / len(subset) if subset else 0
    bar    = "#" * int(rate * 20)
    print(f"  {qtype:20} {rate:5.1%}  ({len(fail):>2}/{len(subset):<2})  {bar}")

print()
print("FAILURE RATE BY MODEL")
print("-" * 40)
for model in ALL_MODELS:
    subset = [r for r in data if r['model'] == model]
    fail   = [r for r in subset if r['step_correctness'] < FAIL_THRESHOLD]
    rate   = len(fail) / len(subset) if subset else 0
    bar    = "#" * int(rate * 20)
    print(f"  {model:25} {rate:5.1%}  ({len(fail):>2}/{len(subset):<2})  {bar}")

print()
print("FAILURE RATE BY SETTING (RAG vs Baseline)")
print("-" * 40)
for setting in ['Baseline', 'RAG']:
    subset = [r for r in data if r['setting'] == setting]
    fail   = [r for r in subset if r['step_correctness'] < FAIL_THRESHOLD]
    rate   = len(fail) / len(subset) if subset else 0
    print(f"  {setting:10} {rate:5.1%}  ({len(fail)}/{len(subset)})")


# ==============================================================
# SECTION 4 - DIFFICULTY INTERACTION
# ==============================================================

print()
print("=" * 60)
print("SECTION 4A - PERFORMANCE BY DIFFICULTY (all models)")
print("=" * 60)
for diff in DIFFICULTIES:
    subset = [r for r in data if r['difficulty'] == diff]
    fail   = [r for r in subset if r['step_correctness'] < FAIL_THRESHOLD]
    print(f"  {diff:10} avg={avg([r['step_correctness'] for r in subset]):.4f}  "
          f"fail_rate={len(fail)/len(subset):.1%}  n={len(subset)}")

print()
print("SECTION 4B - DIFFICULTY x SETTING (RAG vs Baseline, top 3 models)")
print("=" * 60)
print(f"  {'Difficulty':<10} {'Setting':<10} {'Avg Step':>9}  {'Fail Rate':>10}  n")
print("  " + "-" * 48)
for diff in DIFFICULTIES:
    for setting in ['Baseline', 'RAG']:
        subset = [r for r in data if r['difficulty'] == diff
                  and r['setting'] == setting and r['model'] in TOP3]
        fail   = [r for r in subset if r['step_correctness'] < FAIL_THRESHOLD]
        if subset:
            print(f"  {diff:<10} {setting:<10} "
                  f"{avg([r['step_correctness'] for r in subset]):>9.4f}  "
                  f"{len(fail)/len(subset):>9.1%}  {len(subset)}")

print()
print("SECTION 4C - DIFFICULTY x CHAPTER TYPE (RAG only, top 3 models)")
print("=" * 60)
print(f"  {'Chapter':<20} {'Easy':>8} {'Medium':>8} {'Hard':>8}")
print("  " + "-" * 48)
for ctype in CHAPTER_TYPES:
    row = f"  {ctype:<20}"
    for diff in DIFFICULTIES:
        subset = [r for r in data if r['chapter_type'] == ctype
                  and r['difficulty'] == diff
                  and r['setting'] == 'RAG'
                  and r['model'] in TOP3]
        row += f" {avg([r['step_correctness'] for r in subset]):>8.4f}" if subset else f"{'N/A':>9}"
    print(row)


# ==============================================================
# SECTION 5 - CASES WHERE RAG MADE THINGS WORSE
# ==============================================================

print()
print("=" * 60)
print("SECTION 5 - CASES WHERE RAG HURT vs BASELINE (per question+model pair)")
print("=" * 60)

pairs = {}
for r in data:
    key = (r['test_id'], r['model'])
    pairs.setdefault(key, {})[r['setting']] = r

rag_hurt    = []
rag_helped  = []
rag_neutral = []

for key, settings in pairs.items():
    if 'RAG' not in settings or 'Baseline' not in settings:
        continue
    rag_score  = settings['RAG']['step_correctness']
    base_score = settings['Baseline']['step_correctness']
    delta      = rag_score - base_score
    if delta < -0.05:
        rag_hurt.append((key, settings, delta))
    elif delta > 0.05:
        rag_helped.append((key, settings, delta))
    else:
        rag_neutral.append((key, settings, delta))

total_pairs = len(pairs)
print(f"  Total question-model pairs  : {total_pairs}")
print(f"  RAG helped  (delta > +0.05) : {len(rag_helped)}  ({len(rag_helped)/total_pairs:.1%})")
print(f"  RAG neutral (delta <= 0.05) : {len(rag_neutral)}  ({len(rag_neutral)/total_pairs:.1%})")
print(f"  RAG hurt    (delta < -0.05) : {len(rag_hurt)}  ({len(rag_hurt)/total_pairs:.1%})")

if rag_hurt:
    print()
    print("TOP RAG-HURT CASES (largest regressions):")
    print("-" * 60)
    for (test_id, model), settings, delta in sorted(rag_hurt, key=lambda x: x[2])[:5]:
        r = settings['RAG']
        print(f"\n  test_id={test_id}  model={model}")
        print(f"  Chapter: {r['chapter_type']}  |  Type: {r['question_type']}  |  Difficulty: {r['difficulty']}")
        print(f"  Baseline={settings['Baseline']['step_correctness']:.3f}  RAG={r['step_correctness']:.3f}  Delta={delta:+.3f}")
        print(f"  Question : {r['question'][:150]}")

print()
print("RAG REGRESSION BY CHAPTER TYPE")
print("-" * 40)
for ctype in CHAPTER_TYPES:
    hurt     = [(k, s, d) for k, s, d in rag_hurt if s['RAG']['chapter_type'] == ctype]
    total_ct = [k for k, s in pairs.items() if 'RAG' in s and s['RAG']['chapter_type'] == ctype]
    rate     = len(hurt) / len(total_ct) if total_ct else 0
    print(f"  {ctype:20} regressions={len(hurt)}  rate={rate:.1%}")


# ==============================================================
# SECTION 6 - SAMPLE FAILURE CASES WITH RESPONSE TEXT
# ==============================================================

print()
print("=" * 60)
print("SECTION 6 - SAMPLE FAILURE CASES (step_correctness < 0.5, RAG setting)")
print("=" * 60)

rag_failures = sorted(
    [r for r in data if r['setting'] == 'RAG' and r['step_correctness'] < FAIL_THRESHOLD],
    key=lambda x: x['step_correctness']
)

shown = defaultdict(int)
MAX_PER_TYPE = 2
printed = 0

for r in rag_failures:
    ctype = r['chapter_type']
    if shown[ctype] >= MAX_PER_TYPE:
        continue
    shown[ctype] += 1
    printed += 1

    print(f"\n  -- Failure Case {printed} --")
    print(f"  Chapter type  : {r['chapter_type']}")
    print(f"  Question type : {r['question_type']}")
    print(f"  Difficulty    : {r['difficulty']}")
    print(f"  Model         : {r['model']}")
    print(f"  Step score    : {r['step_correctness']}")
    print(f"  Question      : {r['question'][:200]}")
    print(f"  Response      :")
    for line in r['response'][:600].split('\n'):
        print(f"    {line}")


# ==============================================================
# SECTION 7 - PER MODEL BREAKDOWN (RAG only)
# ==============================================================

print()
print("=" * 60)
print("SECTION 7A - BY CHAPTER TYPE PER MODEL (RAG only)")
print("=" * 60)
for model in ALL_MODELS:
    recs = [r for r in data if r['model'] == model and r['setting'] == 'RAG']
    by_ch = defaultdict(list)
    for r in recs:
        by_ch[r['chapter_type']].append(r['step_correctness'])
    print(f"\n  {model}:")
    for ch, vals in sorted(by_ch.items(), key=lambda x: -avg(x[1])):
        fail_rate = sum(1 for s in vals if s < FAIL_THRESHOLD) / len(vals)
        print(f"    {ch:20} step={avg(vals):.4f}  fail_rate={fail_rate:.1%}  n={len(vals)}")

print()
print("=" * 60)
print("SECTION 7B - BY QUESTION TYPE PER MODEL (RAG only)")
print("=" * 60)
for model in ALL_MODELS:
    recs = [r for r in data if r['model'] == model and r['setting'] == 'RAG']
    by_qt = defaultdict(list)
    for r in recs:
        by_qt[r['question_type']].append(r['step_correctness'])
    print(f"\n  {model}:")
    for qt, vals in sorted(by_qt.items(), key=lambda x: -avg(x[1])):
        fail_rate = sum(1 for s in vals if s < FAIL_THRESHOLD) / len(vals)
        print(f"    {qt:20} step={avg(vals):.4f}  fail_rate={fail_rate:.1%}  n={len(vals)}")

print()
print("=" * 60)
print("SECTION 7C - BEST AND WORST MODEL PER CATEGORY (RAG only)")
print("=" * 60)
print("\n  By chapter type:")
for ctype in CHAPTER_TYPES:
    model_scores = {}
    for model in ALL_MODELS:
        vals = [r['step_correctness'] for r in data
                if r['model'] == model and r['chapter_type'] == ctype and r['setting'] == 'RAG']
        if vals:
            model_scores[model] = avg(vals)
    if model_scores:
        best  = max(model_scores, key=model_scores.get)
        worst = min(model_scores, key=model_scores.get)
        print(f"    {ctype:20}  best={best} ({model_scores[best]:.4f})  worst={worst} ({model_scores[worst]:.4f})")

print("\n  By question type:")
for qtype in QUESTION_TYPES:
    model_scores = {}
    for model in ALL_MODELS:
        vals = [r['step_correctness'] for r in data
                if r['model'] == model and r['question_type'] == qtype and r['setting'] == 'RAG']
        if vals:
            model_scores[model] = avg(vals)
    if model_scores:
        best  = max(model_scores, key=model_scores.get)
        worst = min(model_scores, key=model_scores.get)
        print(f"    {qtype:20}  best={best} ({model_scores[best]:.4f})  worst={worst} ({model_scores[worst]:.4f})")


# ==============================================================
# SUMMARY
# ==============================================================

print()
print("=" * 60)
print("SUMMARY - KEY FINDINGS")
print("=" * 60)

all_rag      = [r for r in data if r['setting'] == 'RAG']
all_baseline = [r for r in data if r['setting'] == 'Baseline']
overall_fail = len([r for r in data if r['step_correctness'] < FAIL_THRESHOLD])

print(f"\n  Overall failure rate       : {overall_fail/len(data):.1%}")
print(f"  Avg step (RAG)             : {avg([r['step_correctness'] for r in all_rag]):.4f}")
print(f"  Avg step (Baseline)        : {avg([r['step_correctness'] for r in all_baseline]):.4f}")

hardest_chap  = min(CHAPTER_TYPES,
    key=lambda c: avg([r['step_correctness'] for r in data if r['chapter_type'] == c]))
hardest_qtype = min(QUESTION_TYPES,
    key=lambda q: avg([r['step_correctness'] for r in data if r['question_type'] == q]))

print(f"  Hardest chapter type       : {hardest_chap}")
print(f"  Hardest question type      : {hardest_qtype}")
print(f"  RAG hurt  : {len(rag_hurt)}/{total_pairs} pairs ({len(rag_hurt)/total_pairs:.1%})")
print(f"  RAG helped: {len(rag_helped)}/{total_pairs} pairs ({len(rag_helped)/total_pairs:.1%})")
