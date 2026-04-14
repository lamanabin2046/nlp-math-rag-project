# RAG-Based AI Mathematics Tutoring System for Nepal Grade 10 Students

A research project investigating the effectiveness of Retrieval-Augmented Generation (RAG) for delivering AI-powered step-by-step mathematics solutions aligned with the Nepal national curriculum (Grade 10 / SEE preparation).

---

## Research Overview

This project compares AI models, retrieval strategies, and agentic workflows for solving Nepal Grade 10 mathematics problems. A human evaluation study validates automated metrics against real teacher and student judgments.

**Research Questions:**
- RQ1: Which LLM produces the best step-by-step solutions for Nepal Grade 10 mathematics?
- RQ2: Which retrieval strategy produces the most relevant textbook context?
- RQ3: Do agentic workflows improve quality over static single-pass RAG?
- RQ4: Do teachers and students perceive RAG solutions as better than baseline?

---

## Project Structure

```
project/
├── config.py                          # API keys, paths, constants
├── data/
│   └── test_set.json                  # 120+ Nepal Grade 10 math questions
├── experiments/
│   ├── experiment1_model_comparison.py
│   ├── experiment2_multimodal_ablation.py
│   ├── experiment3_agentic_workflows.py
│   └── experiment4_human_evaluation.py
├── results/
│   ├── experiment1_results.json
│   ├── experiment1_summary.csv
│   ├── experiment2_results.json
│   ├── experiment2_summary.csv
│   ├── experiment3_results.json
│   ├── experiment3_summary.csv
│   ├── human_eval_packet.json
│   └── human_eval_scores.json
└── eval-app/                          # MERN human evaluation web app
    ├── backend/
    │   ├── server.js
    │   ├── models/Score.js
    │   └── routes/
    └── frontend/
        └── src/
```

---

## Experiments

### Experiment 1 — Model Comparison

Compares three LLMs on step-by-step solution quality using Static RAG.

**Models:** Claude Opus, GPT-4o, Qwen2-Math  
**Questions:** 120 total (40 per model)  
**Metrics:** BLEU, ROUGE-L, BERTScore, Step Correctness, Latency

| Model | Step Correctness | BERTScore | Latency |
|-------|-----------------|-----------|---------|
| Claude Opus | **0.955** | 0.697 | 10.84s |
| GPT-4o | 0.866 | **0.701** | 5.20s |
| Qwen2-Math | 0.905 | 0.689 | 17.19s |

**Finding:** Claude Opus achieves the highest step correctness. GPT-4o offers the best speed-quality trade-off.

---

### Experiment 2 — Mutimodal Ablation

Compares four retrieval conditions varying embedding modalities.

**Model:** GPT-4o (fixed)  
**Questions:** 124 total (~31 per condition)

| Condition | Step Correctness | BERTScore |
|-----------|-----------------|-----------|
| Text Only | **0.890** | 0.718 |
| Text + Diagram Desc (Equal) | 0.871 | 0.720 |
| Text + Desc + CLIP (Equal) | 0.871 | 0.719 |
| Text + Desc + CLIP (Learned) | 0.810 | **0.726** |

**Finding:** Text-only retrieval achieves the highest step correctness. Adding visual embeddings introduces noise for this primarily text-based curriculum.

---

### Experiment 3 — Agentic Workflow Comparison

Compares Static RAG, Self-RAG, and CRAG on hard and proof-based questions.

**Models:** Claude Opus, GPT-4o  
**Questions:** 27 hard/proof questions  
**Workflows:**
- **Static RAG** — retrieve once, generate once
- **Self-RAG** — generate, self-score, re-retrieve if score < 3.5 (up to 3 iterations)
- **CRAG** — retrieve, grade documents, re-retrieve if relevance < 0.6

| Model | Workflow | Step Correctness | Latency |
|-------|----------|-----------------|---------|
| Claude Opus | **Static RAG** | **0.958** | 12.71s |
| Claude Opus | Self-RAG | 0.927 | 26.78s |
| Claude Opus | CRAG | 0.939 | 40.60s |
| GPT-4o | Static RAG | 0.891 | 5.57s |
| GPT-4o | Self-RAG | 0.861 | 11.20s |
| GPT-4o | CRAG | 0.849 | 19.64s |

**Finding:** Static RAG with Claude Opus wins on quality. Complex agentic workflows add up to 3× latency without improving step correctness.

---

### Experiment 4 — Human Evaluation

Validates automated metrics against human judgment from Nepal Grade 10 teachers and students.

**Setup:**
- 9 representative questions (3 per chapter: geometry, algebra, word problems)
- Each question has 2 responses: RAG (Solution A) and Baseline (Solution B)
- 18 evaluation items total

**Evaluators:**

| Role | Instrument | Scale |
|------|-----------|-------|
| Teachers | Correctness rubric | 0–3 |
| Students | Clarity, Alignment, Usefulness | 1–5 Likert |

**Analysis:** Cohen's kappa (teacher vs auto rubric), RAG vs Baseline gain, student Likert averages.

**Status:** Data collection in progress via web app deployed on AWS EC2.

---

## Best Configuration

Based on Experiments 1–3:

```
Model:     Claude Opus (claude-opus-4-5)
Retrieval: Text-only BERT embeddings, top-5 passages
Workflow:  Static RAG (single retrieve → generate)
```

Expected step correctness: **0.958**  
Expected latency: **~12–13 seconds per question**

---

## Setup

### Requirements

```bash
pip install openai anthropic chromadb sentence-transformers
pip install nltk rouge-score bert-score numpy tqdm
pip install langgraph langchain langchain-openai langchain-anthropic
```

### Configuration

Create `config.py`:
```python
OPENAI_API_KEY    = "your-openai-key"
ANTHROPIC_API_KEY = "your-anthropic-key"
TESTSET_DIR       = Path("data/")
CHROMA_DIR        = Path("chroma_db/")
CHROMA_COLLECTION = "nepal_math"
BERT_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_DIM          = 512
DESC_DIM          = 384
```

### Running Experiments

```bash
# Experiment 1 — Model Comparison
python experiments/experiment1_model_comparison.py

# Experiment 2 — Retrieval Strategy
python experiments/experiment2_retrieval_comparison.py

# Experiment 3 — Agentic Workflows
python experiments/experiment3_agentic_workflows.py

# Experiment 4 — Prepare human eval packet
python experiments/experiment4_human_evaluation.py --prepare

# Experiment 4 — Analyze scores (after collection)
python experiments/experiment4_human_evaluation.py --analyze
```

---

## Human Evaluation Web App

A MERN stack application for collecting teacher and student evaluations.

### Tech Stack
- **Frontend:** React + Vite + Tailwind CSS + KaTeX
- **Backend:** Node.js + Express
- **Database:** MongoDB Atlas
- **Deployment:** AWS EC2 + Nginx + PM2

### Pages

| URL | Purpose |
|-----|---------|
| `/` | Role selection (Teacher / Student) |
| `/teacher` | Grade solutions 0–3 rubric |
| `/student` | Rate solutions 1–5 stars |
| `/admin` | Live dashboard + download scores |
| `/thankyou` | Submission confirmation |

### Local Setup

```bash
# Backend
cd eval-app/backend
npm install
# Create .env with MONGO_URI and PORT=5000
npm run dev

# Frontend
cd eval-app/frontend
npm install
npm run dev
```

### Export Scores for Analysis

```
GET /api/scores/export  →  downloads human_eval_scores.json
```

Then run:
```bash
python experiments/experiment4_human_evaluation.py --analyze
```

---

## Evaluation Metrics

| Metric | Description | Primary? |
|--------|-------------|---------|
| Step Correctness | Heuristic: numbered steps, equations, conclusion markers (0–1) | ✅ Yes |
| BERTScore | Semantic similarity using distilBERT | ✅ Yes |
| BLEU | N-gram overlap with question text | ⚠️ Proxy only |
| ROUGE-L | Longest common subsequence | ⚠️ Proxy only |

> Note: BLEU and ROUGE-L are computed against the question text (no gold-standard reference solutions available). They serve as surface-level proxies only. Step Correctness and BERTScore are the primary evaluation signals.

---

## Results Summary

| Research Question | Answer | Best Config |
|-------------------|--------|-------------|
| RQ1 — Best model? | Claude Opus (steps: 0.955) | Claude Opus |
| RQ2 — Best retrieval? | Text-only (steps: 0.890) | Text Only |
| RQ3 — Agentic workflows help? | No — Static RAG wins | Static RAG |
| RQ4 — Human validation? | In progress | TBD |

---

## Technologies Used

- **LLMs:** Claude Opus (Anthropic), GPT-4o (OpenAI), Qwen2-Math
- **Embeddings:** sentence-transformers, CLIP
- **Vector DB:** ChromaDB
- **Agentic Workflows:** LangGraph
- **Evaluation:** NLTK, rouge-score, bert-score
- **Web App:** React, Node.js, MongoDB Atlas, AWS EC2, Nginx, PM2

---

## License

This project is for academic research purposes only.
