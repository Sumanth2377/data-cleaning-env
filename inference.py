"""
inference.py — Baseline inference script for the Data Cleaning OpenEnv environment.

Mandatory configuration (set as environment variables):
  API_BASE_URL   The LLM API endpoint (default: HuggingFace Inference Router)
  MODEL_NAME     The model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       Your HuggingFace API key

Optional:
  TASK_NAME      Task to run (default: runs all three tasks sequentially)
  MAX_STEPS      Maximum steps per episode (overrides task default)

STDOUT FORMAT (mandatory — do not alter field names or order):
  [START] task=<task_name> env=data-cleaning-env model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Import environment directly (no Docker client needed for local/HF execution)
# ---------------------------------------------------------------------------
from environment.env import DataCleaningEnv, TASK_CONFIG
from environment.models import DataCleaningAction, DataCleaningObservation

# ============================================================================
# Configuration
# ============================================================================
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK: str = "data-cleaning-env"

# Which tasks to run (comma-separated list, or 'all')
_TASK_ENV = os.getenv("TASK_NAME", "all")
TASKS_TO_RUN: List[str] = (
    ["csv-doctor", "schema-enforcer", "pipeline-debugger"]
    if _TASK_ENV == "all"
    else [t.strip() for t in _TASK_ENV.split(",")]
)

SEED: int = int(os.getenv("SEED", "42"))
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "0"))   # 0 → use task default
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))
SUCCESS_THRESHOLD: float = 0.65  # score in [0, 1] that counts as "success"

# ============================================================================
# Logging helpers (mandatory format — do not change)
# ============================================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Sanitise action string — no newlines allowed on a single [STEP] line
    action_clean = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ============================================================================
# Prompt builders
# ============================================================================

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert data scientist operating an AI data cleaning environment.
Your goal is to fix data quality issues in the provided dataset by issuing
data-cleaning actions one at a time.

AVAILABLE ACTIONS (issue one per turn as valid JSON):
  fill_missing       — fill null values in a column
  drop_duplicates    — remove duplicate rows
  cast_column        — change a column's data type
  normalize_format   — standardise phone/email/date/zip_code/text_case/strip_currency
  apply_regex        — regex substitution on a column
  drop_column        — remove a column entirely
  drop_rows_by_condition — drop rows matching a condition
  clip_outliers      — clip statistical outliers (iqr/zscore)
  standardize_text   — apply strip/lower/upper/title/remove_extra_spaces
  fix_referential_integrity — fix foreign-key violations
  merge_tables       — merge an auxiliary table

RESPONSE FORMAT (respond with ONLY this JSON — no markdown, no explanation):
{
  "action_type": "<action_name>",
  "parameters": { ... }
}

Examples:
{"action_type": "fill_missing", "parameters": {"column": "age", "strategy": "median"}}
{"action_type": "drop_duplicates", "parameters": {}}
{"action_type": "normalize_format", "parameters": {"column": "phone", "format_type": "phone"}}
{"action_type": "cast_column", "parameters": {"column": "salary", "dtype": "float"}}
{"action_type": "clip_outliers", "parameters": {"column": "price", "method": "iqr", "threshold": 1.5}}

PRIORITIES:
1. Fix the highest-severity issues first.
2. For 'csv-doctor': fix salary currency strings → cast salary, fix age dtype, fill nulls, drop dupes, title case names & strip dept whitespace.
3. For 'schema-enforcer': use normalize_format for phone/email/date/zip_code, then fix country case and name casing.
4. For 'pipeline-debugger': fix FK violations → drop dupes → clip outliers → merge_tables to add segment.
5. Never drop more than 30% of rows in a single action (incurs penalty).
6. Stop issuing redundant actions once an issue is fixed.
""").strip()


def _obs_summary(obs: DataCleaningObservation) -> str:
    """Build a concise observation string for the LLM."""
    issues_text = "\n".join(
        f"  - [{i.severity.upper()}] {i.issue_type}: {i.description}"
        + (f" (column: {i.column})" if i.column else "")
        for i in obs.issues[:10]
    ) or "  None detected"

    columns_text = "\n".join(
        f"  {c.name} ({c.dtype}): {c.null_count} nulls, {c.unique_count} unique"
        + (f" | issues: {'; '.join(c.detected_issues)}" if c.detected_issues else "")
        for c in obs.columns[:12]
    )

    history_text = (
        "\n".join(f"  {h}" for h in obs.actions_history[-5:])
        if obs.actions_history
        else "  None yet"
    )

    schema_text = ""
    if obs.target_schema:
        schema_text = "\nTARGET SCHEMA:\n" + json.dumps(obs.target_schema, indent=2)[:800]

    aux_text = ""
    if obs.auxiliary_datasets:
        aux_text = "\nAUXILIARY TABLES (preview):\n" + json.dumps(
            obs.auxiliary_datasets, default=str
        )[:400]

    return textwrap.dedent(f"""
    TASK: {obs.task_name}
    OBJECTIVE: {obs.task_description[:300]}
    
    DATASET STATS:
      Rows: {obs.stats.total_rows} | Cols: {obs.stats.total_cols}
      Missing cells: {obs.stats.missing_cells} ({obs.stats.missing_pct:.1f}%)
      Duplicate rows: {obs.stats.duplicate_rows}
      Dtype issues: {obs.stats.dtype_issues}
    
    CURRENT SCORE: {obs.current_score:.3f} (step {obs.step_count}/{obs.max_steps})
    
    COLUMNS:
    {columns_text}
    
    DETECTED ISSUES:
    {issues_text}{schema_text}{aux_text}
    
    RECENT ACTIONS:
    {history_text}
    
    Issue the next cleaning action as JSON:
    """).strip()


def _parse_action(text: str) -> Optional[DataCleaningAction]:
    """Extract a DataCleaningAction from the model's response."""
    text = text.strip()

    # Try to extract JSON block
    try:
        # Sometimes the model wraps it in ```json ... ```
        if "```" in text:
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if match:
                text = match.group(1).strip()

        data = json.loads(text)
        return DataCleaningAction(
            action_type=data.get("action_type", ""),
            parameters=data.get("parameters", {}),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _get_llm_action(
    client: OpenAI,
    obs: DataCleaningObservation,
    step: int,
) -> tuple[Optional[DataCleaningAction], str]:
    """Call the LLM and return (action, raw_text)."""
    user_prompt = _obs_summary(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        raw = ""

    action = _parse_action(raw)
    return action, raw


# ============================================================================
# Fallback rule-based agent (for when LLM fails or is unavailable)
# ============================================================================

def _rule_based_action(obs: DataCleaningObservation, step: int) -> DataCleaningAction:
    """
    Deterministic rule-based agent used as fallback.

    Follows a fixed priority queue per task, ensuring reproducible baseline scores.
    """
    task = obs.task_name
    issues = obs.issues
    columns = {c.name: c for c in obs.columns}

    if task == "csv-doctor":
        # Priority order
        if obs.stats.duplicate_rows > 0:
            return DataCleaningAction(action_type="drop_duplicates", parameters={})

        # Fix salary currency string
        if "salary" in columns and any("currency" in i for i in columns["salary"].detected_issues):
            return DataCleaningAction(
                action_type="normalize_format",
                parameters={"column": "salary", "format_type": "strip_currency"},
            )

        # Fill missing age
        if "age" in columns and columns["age"].null_count > 0:
            return DataCleaningAction(
                action_type="fill_missing",
                parameters={"column": "age", "strategy": "median"},
            )

        # Cast age to int
        if "age" in columns and columns["age"].dtype in ("float64", "float32"):
            return DataCleaningAction(
                action_type="cast_column",
                parameters={"column": "age", "dtype": "int"},
            )

        # Fill missing salary
        if "salary" in columns and columns["salary"].null_count > 0:
            return DataCleaningAction(
                action_type="fill_missing",
                parameters={"column": "salary", "strategy": "median"},
            )

        # Fill missing email
        if "email" in columns and columns["email"].null_count > 0:
            return DataCleaningAction(
                action_type="fill_missing",
                parameters={"column": "email", "strategy": "constant", "fill_value": "unknown@example.com"},
            )

        # Title-case names
        if "name" in columns:
            return DataCleaningAction(
                action_type="standardize_text",
                parameters={"column": "name", "operations": ["title"]},
            )

        # Strip department whitespace
        if "department" in columns:
            return DataCleaningAction(
                action_type="standardize_text",
                parameters={"column": "department", "operations": ["strip"]},
            )

    elif task == "schema-enforcer":
        order = [
            ("phone",      "normalize_format", {"column": "phone",      "format_type": "phone"}),
            ("birth_date", "normalize_format", {"column": "birth_date", "format_type": "date"}),
            ("email",      "normalize_format", {"column": "email",      "format_type": "email"}),
            ("zip_code",   "normalize_format", {"column": "zip_code",   "format_type": "zip_code"}),
            ("country",    "normalize_format", {"column": "country",    "format_type": "text_case", "output_format": "upper"}),
            ("first_name", "standardize_text", {"column": "first_name", "operations": ["title"]}),
            ("last_name",  "standardize_text", {"column": "last_name",  "operations": ["title"]}),
        ]
        idx = min(step - 1, len(order) - 1)
        col, act, params = order[idx]
        return DataCleaningAction(action_type=act, parameters=params)

    elif task == "pipeline-debugger":
        order = [
            ("fix_referential_integrity", {"child_column": "customer_id", "parent_table": "customers", "parent_column": "customer_id", "action": "drop"}),
            ("drop_duplicates",           {"subset": ["customer_id", "product", "price", "quantity", "order_date"]}),
            ("clip_outliers",             {"column": "price",    "method": "iqr", "threshold": 1.5}),
            ("clip_outliers",             {"column": "quantity", "method": "iqr", "threshold": 1.5}),
            ("merge_tables",              {"right_table": "customers", "left_on": "customer_id", "right_on": "customer_id", "how": "left", "columns": ["segment"]}),
        ]
        idx = min(step - 1, len(order) - 1)
        act, params = order[idx]
        return DataCleaningAction(action_type=act, parameters=params)

    # Fallback no-op: drop duplicates (safe)
    return DataCleaningAction(action_type="drop_duplicates", parameters={})


# ============================================================================
# Single-task episode runner
# ============================================================================

def run_task(
    env: DataCleaningEnv,
    client: Optional[OpenAI],
    task_name: str,
) -> Dict[str, Any]:
    """Run one full episode and return result metrics."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg: Optional[str] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    max_steps = MAX_STEPS if MAX_STEPS > 0 else TASK_CONFIG[task_name]["max_steps"]

    try:
        reset_result = env.reset(task_name=task_name, seed=SEED)
        obs = reset_result.observation

        for step in range(1, max_steps + 1):
            if obs.step_count > 0 and step > obs.max_steps:
                break

            # Try LLM agent first; fall back to rule-based on failure
            action: Optional[DataCleaningAction] = None
            action_str = ""

            if client is not None:
                action, raw_text = _get_llm_action(client, obs, step)
                action_str = raw_text[:150] if raw_text else "parse_error"

            if action is None:
                action = _rule_based_action(obs, step)
                action_str = f"{action.action_type}({action.parameters})"

            step_result = env.step(action)
            obs = step_result.observation
            reward = step_result.reward
            done = step_result.done
            err = step_result.info.get("action_message") if not step_result.info.get("action_success", True) else None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=err)

            if done:
                break

        state = env.state()
        score = float(state.current_score)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        error_msg = str(exc)
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score = float(env.state().current_score) if steps_taken > 0 else 0.0

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task_name,
        "score": round(score, 4),
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
        "error": error_msg,
    }


# ============================================================================
# Main entry point
# ============================================================================

def main() -> None:
    # Build OpenAI client (pointing to HF Inference Router or custom endpoint)
    try:
        client: Optional[OpenAI] = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        # Quick connectivity test
        client.models.list()
    except Exception as exc:
        print(f"[DEBUG] LLM client unavailable ({exc}), using rule-based fallback.", flush=True)
        client = None

    env = DataCleaningEnv()
    all_results: List[Dict[str, Any]] = []

    for task in TASKS_TO_RUN:
        result = run_task(env, client, task)
        all_results.append(result)
        print("", flush=True)  # blank line between tasks

    # Summary
    avg_score = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0.0
    n_success = sum(1 for r in all_results if r["success"])
    print(
        f"[SUMMARY] tasks={len(all_results)} success={n_success}/{len(all_results)} "
        f"avg_score={avg_score:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
