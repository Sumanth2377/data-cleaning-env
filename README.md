---
title: Data Cleaning OpenEnv
emoji: 🧹
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
license: mit
tags:
 - openenv
 - data-cleaning
 - reinforcement-learning
 - tabular-data
 - real-world
---

<div align="center">

# 🧹 Data Cleaning OpenEnv

**A real-world AI environment for training agents to clean and preprocess tabular data**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-4CAF50?style=for-the-badge)](https://openenv.dev)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*Built for the Meta × PyTorch Hackathon 2024*

</div>

---

## 🌍 Why This Environment?

Data cleaning consumes **60–80% of a data scientist's time** — yet no high-quality RL benchmark exists for it. This environment fills that gap.

An agent learns to act as a data engineer: it receives a dirty dataset, observes column-level quality statistics and detected issues, and chooses data-cleaning actions to maximise a composite quality score. **Reward is shaped at every step** — not just at the episode end — giving rich training signal throughout the trajectory.

> **Real-world impact**: Agents trained here could be deployed as autonomous data preparation pipelines, dramatically reducing one of the most labour-intensive bottlenecks in modern ML.

---

## 🗺️ Environment Overview

### Observation Space

Each observation is a rich, typed `DataCleaningObservation` with:

| Field | Type | Description |
|---|---|---|
| `task_name` | `str` | Active task identifier |
| `task_description` | `str` | Natural-language objective |
| `dataset_id` | `str` | Unique episode identifier |
| `step_count` | `int` | Steps taken so far |
| `columns` | `List[ColumnInfo]` | Per-column dtype, null %, unique count, sample values, detected issues |
| `stats` | `DatasetStats` | Aggregate quality metrics |
| `issues` | `List[IssueDetail]` | Severity-ranked list of detected data problems |
| `actions_history` | `List[str]` | All actions taken this episode |
| `target_schema` | `dict \| None` | Expected column formats (medium task) |
| `auxiliary_datasets` | `dict \| None` | Secondary tables preview (hard task) |
| `current_score` | `float [0, 1]` | Running quality score |
| `max_steps` | `int` | Episode step budget |

### Action Space

12 typed cleaning actions:

| Action | Key Parameters | Effect |
|---|---|---|
| `fill_missing` | `column`, `strategy` (mean/median/mode/constant/drop/forward_fill/backward_fill/unknown) | Fill NaN values |
| `drop_duplicates` | `subset`, `keep` | Remove duplicate rows |
| `cast_column` | `column`, `dtype` (int/float/string/datetime/bool/category) | Type conversion |
| `normalize_format` | `column`, `format_type` (phone/email/date/text_case/strip_currency/zip_code) | Standardise format |
| `apply_regex` | `column`, `pattern`, `replacement` | Regex substitution |
| `standardize_text` | `column`, `operations` (strip/lower/upper/title/remove_extra_spaces) | Text normalisation |
| `drop_column` | `column` | Remove column entirely |
| `drop_rows_by_condition` | `column`, `operator`, `value` | Conditional row removal |
| `clip_outliers` | `column`, `method` (iqr/zscore/drop), `threshold` | Outlier treatment |
| `fix_referential_integrity` | `child_column`, `parent_table`, `parent_column`, `action` (drop/flag) | FK violation repair |
| `merge_tables` | `right_table`, `left_on`, `right_on`, `how`, `columns` | Table join |
| `rename_column` | `old_name`, `new_name` | Column rename |

### Reward Function

Reward is **shaped at every step** — agents receive signal proportional to the improvement they cause:

```
R(t) = score_delta(t) + step_cost + destructive_penalty

where:
 score_delta   = new_score − prev_score  (can be negative for bad actions)
 step_cost    = −0.005          (efficiency incentive)
 destructive_penalty = −0.10         (if >30% rows dropped in one step)

score = weighted_sum(completeness, consistency, validity, downstream_quality)
```

**Reward range**: `[−1.0, 1.0]` per step 
**Episode score**: `[0.0, 1.0]` (final observation's `current_score`)

---

## Tasks

### Task 1: `csv-doctor` (Easy) — max 15 steps

**Domain**: Customer/employee records 
**Dataset**: 200 rows, 7 columns 

**Injected Issues**: 
- 15% missing age values 
- 12% missing email addresses 
- 10% missing salary values 
- Salary stored as currency string (`"$45,200.00"`) instead of float 
- Age stored as `float64` instead of integer 
- ~8% duplicate rows 
- Name column has mixed casing (ALL CAPS, lowercase, TitleCase) 
- Department column has leading/trailing whitespace 

**Grader** (deterministic):
```
score = 0.30 × completeness   (null reduction in age/salary/email)
   + 0.30 × type_correctness (salary numeric, age integer)
   + 0.20 × deduplication   (fraction unique rows)
   + 0.20 × format_consistency (name title-case, dept stripped)
```

**Baseline score**: ~0.78 (rule-based agent, 8 steps)

---

### Task 2: `schema-enforcer` (Medium) — max 20 steps

**Domain**: Contact directory 
**Dataset**: 300 rows, 7 columns 

**Injected Issues**: 
- Phone numbers in 5 different formats (`(123) 456-7890`, `123-456-7890`, `123.456.7890`, `1234567890`, `+1 (123) 456-7890`) 
- Dates in 4 different formats (`YYYY-MM-DD`, `MM/DD/YYYY`, `DD-MM-YYYY`, `Mon DD, YYYY`) 
- Email addresses with mixed case and stray whitespace 
- Zip codes occasionally include `+4` suffix (e.g. `94105-1234`) 
- Country codes in inconsistent case (`US`, `us`, `Us`, `USA`) 
- First/last name casing inconsistent 

**Target schema** (provided in observation):
```json
{
 "phone":   "(XXX) XXX-XXXX",
 "email":   "lowercase, stripped",
 "birth_date": "YYYY-MM-DD",
 "zip_code":  "5-digit only",
 "country":  "UPPER_CASE 2-3 chars",
 "first_name": "Title Case",
 "last_name": "Title Case"
}
```

**Grader**: Column-level compliance score (regex matching) weighted average 
**Baseline score**: ~0.72 (rule-based agent, 7 steps)

---

### Task 3: `pipeline-debugger` (Hard) — max 30 steps

**Domain**: E-commerce orders + customers 
**Dataset**: Two tables — 300+ orders rows, 100 customers rows 

**Injected Issues**: 
- ~10% of orders reference non-existent customer IDs (FK violations) 
- ~5% of `price` values are 10–50× the normal range (outliers) 
- ~5% of `quantity` values are 20–50× the normal range (outliers) 
- ~8% of orders are implicit duplicates (same data, different `order_id`) 
- `segment` column missing from orders (requires merge with customers table) 

**Grader**:
```
score = 0.30 × referential_integrity (fraction of valid FK references)
   + 0.20 × deduplication     (uniqueness of order content)
   + 0.20 × outlier_removal    (IQR compression vs. original)
   + 0.30 × downstream_ml     (R² of revenue ~ price+qty+product+segment)
```

The downstream ML component trains a `LinearRegression` on the cleaned dataset — the only way to maximise this is to fix all upstream quality issues correctly. 

**Baseline score**: ~0.61 (rule-based agent, 5 steps)

---

## Quick Start

### Local (Python)

```bash
# 1. Clone and install
git clone https://huggingface.co/spaces/suman-kar/data-cleaning-env
cd data-cleaning-env
pip install -r requirements.txt

# 2. Run the Local Validation Server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# 3. Test it
curl -X POST http://localhost:7860/reset \
 -H "Content-Type: application/json" \
 -d '{"task_name": "csv-doctor", "seed": 42}'
```

### Docker

```bash
# Build
docker build -t data-cleaning-env .

# Run
docker run -p 7860:7860 \
 -e HF_TOKEN=hf_your_token \
 -e API_BASE_URL=https://router.huggingface.co/v1 \
 -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
 data-cleaning-env

# Health check
curl http://localhost:7860/health
```

### Run the Baseline Inference Script

```bash
# Set environment variables
export HF_TOKEN=hf_your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# Run all 3 tasks
python inference.py

# Run a single task
TASK_NAME=csv-doctor python inference.py
```

---

## 🔌 API Reference

### `POST /reset`
Start a new episode.

```json
Request:
{
 "task_name": "csv-doctor",  // "csv-doctor" | "schema-enforcer" | "pipeline-debugger"
 "seed": 42          // integer, for reproducibility
}

Response: DataCleaningObservation + info
```

### `POST /step`
Apply a cleaning action.

```json
Request:
{
 "action": {
  "action_type": "fill_missing",
  "parameters": {
   "column": "age",
   "strategy": "median"
  }
 }
}

Response: { observation, reward, done, info }
```

### `GET /state`
Read current episode state (non-mutating).

### `GET /health`
Liveness probe — returns `{"status": "ok"}`.

### `GET /info`
Environment metadata, available actions, task list.

### `GET /docs`
Interactive Swagger UI (auto-generated by FastAPI).

---

## Baseline Scores

All scores are reproducible with `seed=42` using the rule-based fallback agent in `inference.py`.

| Task | Initial Score | Final Score (Rule-Based) | Steps | Δ Improvement |
|---|---|---|---|---|
| `csv-doctor` | 0.689 | **0.999** | 8/15 | +45% |
| `schema-enforcer` | 0.529 | **0.949** | 7/20 | +79% |
| `pipeline-debugger` | 0.673 | **1.000** | 5/30 | +49% |

> LLM agents (e.g. Qwen2.5-72B) typically score slightly lower than the rule-based agent since they must discover the right sequence of actions from natural language.

---

## 🏗️ Project Structure

```
.
├── environment/
│  ├── __init__.py
│  ├── env.py       ← Main DataCleaningEnv class (reset/step/state)
│  ├── models.py      ← Pydantic models (Observation, Action, Reward)
│  ├── actions.py     ← 12 action handlers
│  ├── datasets/
│  │  └── generator.py  ← Reproducible synthetic dirty-data generators
│  └── graders/
│    └── graders.py   ← Deterministic graders per task
├── server/
│  └── main.py       ← FastAPI server (HF Spaces)
├── inference.py      ← Baseline inference script
├── openenv.yaml      ← OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔬 Design Decisions

**Why synthetic data?** 
Synthetic generation (seeded `numpy` + `random`) ensures deterministic grading — every judge running `seed=42` gets identical datasets and scores. No external data downloads; no licensing issues.

**Why shaped rewards?** 
Sparse rewards (only at episode end) are known to be difficult for RL training. Our step-level `score_delta` signal lets agents attribute credit to individual actions and learn faster.

**Why a downstream ML grader?** 
Simply fixing obvious issues (nulls, types) is necessary but not sufficient. The hard task's R² grader ensures the agent must get the *right* data distribution, not just pass surface-level checks.

**Why a destructive penalty?** 
Dropping all rows trivially resolves many issues (no nulls in an empty dataset). The −0.10 penalty when >30% of rows are removed in a single action prevents this exploitation while still allowing necessary row removal.

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

Built with ❤️ for the **Meta × PyTorch Hackathon 2024**

[🤗 HuggingFace Space](https://huggingface.co/spaces/suman-kar/data-cleaning-env) · [ API Docs](https://suman-kar.hf.space/docs) · [ Issues](https://huggingface.co/spaces/suman-kar/data-cleaning-env/discussions)

</div>
