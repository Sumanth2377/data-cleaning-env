"""
FastAPI server for the Data Cleaning OpenEnv environment.

Endpoints
---------
GET  /                   — Redirects to interactive API docs
POST /reset              — Start a new episode
POST /step               — Apply a cleaning action
GET  /state              — Inspect current state (read-only)
GET  /health             — Liveness probe
GET  /info               — Environment metadata + available actions
GET  /tasks              — List all tasks with metadata
POST /grade              — Call grader directly (useful for judges)
GET  /actions            — List all supported actions with parameter docs
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from environment.env import DataCleaningEnv, TASK_CONFIG
from environment.actions import SUPPORTED_ACTIONS
from environment.models import (
    DataCleaningAction,
    ResetRequest,
    ResetResult,
    StateResult,
    StepRequest,
    StepResult,
)

# ============================================================================
# App setup
# ============================================================================

app = FastAPI(
    title="Data Cleaning OpenEnv",
    description=(
        "A real-world AI environment for training agents to clean and preprocess "
        "tabular data. Implements the full OpenEnv spec: typed observations, actions, "
        "rewards, and episode management across 3 difficulty levels.\n\n"
        "**Tasks**: `csv-doctor` (Easy) → `schema-enforcer` (Medium) → `pipeline-debugger` (Hard)\n\n"
        "**Reward**: Shaped at every step — `score_delta + step_cost + destructive_penalty`"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single-session)
_env = DataCleaningEnv()

# ============================================================================
# Action documentation
# ============================================================================

ACTION_DOCS: Dict[str, Dict[str, Any]] = {
    "fill_missing": {
        "description": "Fill NaN values in a column",
        "parameters": {
            "column": "str — target column name",
            "strategy": "str — mean | median | mode | constant | forward_fill | backward_fill | drop | unknown",
            "fill_value": "Any (optional) — used when strategy='constant'",
        },
        "example": {"column": "age", "strategy": "median"},
    },
    "drop_duplicates": {
        "description": "Remove duplicate rows",
        "parameters": {
            "subset": "List[str] (optional) — columns to consider for dedup",
            "keep": "str (optional) — first | last | False",
        },
        "example": {"subset": ["customer_id", "product", "price"]},
    },
    "cast_column": {
        "description": "Change a column's data type",
        "parameters": {
            "column": "str — target column",
            "dtype": "str — int | float | string | datetime | bool | category",
        },
        "example": {"column": "age", "dtype": "int"},
    },
    "normalize_format": {
        "description": "Standardise values to a canonical format",
        "parameters": {
            "column": "str — target column",
            "format_type": "str — phone | email | date | text_case | strip_currency | zip_code",
            "output_format": "str (optional) — for date: strftime format; for text_case: lower|upper|title",
        },
        "examples": [
            {"column": "phone", "format_type": "phone"},
            {"column": "salary", "format_type": "strip_currency"},
            {"column": "birth_date", "format_type": "date", "output_format": "%Y-%m-%d"},
        ],
    },
    "standardize_text": {
        "description": "Apply one or more text normalisation operations",
        "parameters": {
            "column": "str — target column",
            "operations": "List[str] — strip | lower | upper | title | remove_extra_spaces",
        },
        "example": {"column": "name", "operations": ["title"]},
    },
    "clip_outliers": {
        "description": "Clip statistical outliers in a numeric column",
        "parameters": {
            "column": "str — target column",
            "method": "str — iqr | zscore | drop",
            "threshold": "float (optional) — IQR multiplier or Z-score threshold (default 1.5)",
        },
        "example": {"column": "price", "method": "iqr", "threshold": 1.5},
    },
    "fix_referential_integrity": {
        "description": "Remove or flag rows with foreign-key violations",
        "parameters": {
            "child_column": "str — FK column in main table",
            "parent_table": "str — auxiliary table name (e.g. 'customers')",
            "parent_column": "str — PK column in parent table",
            "action": "str — drop | flag",
        },
        "example": {
            "child_column": "customer_id",
            "parent_table": "customers",
            "parent_column": "customer_id",
            "action": "drop",
        },
    },
    "merge_tables": {
        "description": "Merge an auxiliary table into the main dataset",
        "parameters": {
            "right_table": "str — auxiliary table name",
            "left_on": "str — join key in main table",
            "right_on": "str — join key in auxiliary table",
            "how": "str — left | inner | outer | right",
            "columns": "List[str] (optional) — columns to bring in from right table",
        },
        "example": {
            "right_table": "customers",
            "left_on": "customer_id",
            "right_on": "customer_id",
            "how": "left",
            "columns": ["segment"],
        },
    },
    "apply_regex": {
        "description": "Apply a regex substitution across a column",
        "parameters": {
            "column": "str — target column",
            "pattern": "str — regex pattern",
            "replacement": "str — replacement string",
        },
        "example": {"column": "salary", "pattern": r"[\$,]", "replacement": ""},
    },
    "drop_column": {
        "description": "Remove a column entirely",
        "parameters": {"column": "str — column to drop"},
        "example": {"column": "unnamed_col"},
    },
    "drop_rows_by_condition": {
        "description": "Drop rows matching a condition",
        "parameters": {
            "column": "str — target column",
            "operator": "str — == | != | > | < | >= | <= | isnull | notnull | contains",
            "value": "Any — comparison value",
        },
        "example": {"column": "age", "operator": "<", "value": 0},
    },
    "rename_column": {
        "description": "Rename a column",
        "parameters": {
            "old_name": "str — current column name",
            "new_name": "str — desired column name",
        },
        "example": {"old_name": "Salary $", "new_name": "salary"},
    },
}


from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to interactive API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["System"])
async def health() -> Dict[str, str]:
    """Liveness probe — returns 200 OK if the server is running."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/tasks", tags=["System"])
async def list_tasks() -> Dict[str, Any]:
    """List all available tasks with difficulty, max_steps, and reward threshold."""
    return {
        "tasks": [
            {
                "name": name,
                "difficulty": {"csv-doctor": "easy", "schema-enforcer": "medium", "pipeline-debugger": "hard"}[name],
                "max_steps": cfg["max_steps"],
                "reward_threshold": {"csv-doctor": 0.75, "schema-enforcer": 0.70, "pipeline-debugger": 0.60}[name],
                "description": cfg["description"][:200] + "…",
            }
            for name, cfg in TASK_CONFIG.items()
        ]
    }


@app.get("/actions", tags=["System"])
async def list_actions() -> Dict[str, Any]:
    """List all supported actions with parameter documentation and examples."""
    return {"actions": ACTION_DOCS, "total": len(ACTION_DOCS)}


@app.get("/info", tags=["System"])
async def info() -> Dict[str, Any]:
    """Return environment metadata."""
    return {
        "name": "data-cleaning-env",
        "version": "1.0.0",
        "description": "Real-world AI Data Cleaning & Preprocessing OpenEnv environment",
        "tasks": list(TASK_CONFIG.keys()),
        "supported_actions": SUPPORTED_ACTIONS,
        "reward_range": [0.0, 1.0],
        "reward_description": "Shaped per-step: score_delta + step_cost(-0.005) + destructive_penalty(-0.10 if >30% rows dropped)",
        "docs_url": "/docs",
    }


@app.post("/reset", response_model=ResetResult, tags=["Environment"])
async def reset(request: ResetRequest = ResetRequest()) -> ResetResult:
    """
    Start a new episode.

    - **task_name**: `csv-doctor` | `schema-enforcer` | `pipeline-debugger`
    - **seed**: integer for reproducibility (default 42)
    """
    try:
        return _env.reset(task_name=request.task_name, seed=request.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset failed: {exc}") from exc


@app.post("/step", response_model=StepResult, tags=["Environment"])
async def step(request: StepRequest) -> StepResult:
    """
    Apply a data-cleaning action.

    **Example body:**
    ```json
    {
      "action": {
        "action_type": "fill_missing",
        "parameters": {"column": "age", "strategy": "median"}
      }
    }
    ```
    See `GET /actions` for all action types and their parameters.
    """
    try:
        return _env.step(request.action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Step failed: {exc}") from exc


@app.get("/state", response_model=StateResult, tags=["Environment"])
async def state() -> StateResult:
    """Return a lightweight read-only snapshot of the current episode state."""
    try:
        return _env.state()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"State failed: {exc}") from exc


@app.post("/grade", tags=["Environment"])
async def grade() -> Dict[str, Any]:
    """
    Run the grader on the current dataset state and return the full score breakdown.

    This is a **read-only** operation — it does not advance the episode.
    Useful for judges and for inspecting per-dimension quality scores.
    """
    try:
        from environment.graders.graders import grade_easy, grade_medium, grade_hard
        import pandas as pd

        task = _env._task_name  # type: ignore[attr-defined]
        df = _env._df           # type: ignore[attr-defined]

        if task == "csv-doctor":
            score, breakdown = grade_easy(df)
        elif task == "schema-enforcer":
            score, breakdown = grade_medium(df)
        elif task == "pipeline-debugger":
            customers = _env._aux_dfs.get("customers", pd.DataFrame())  # type: ignore[attr-defined]
            orig = _env._original_orders or df                           # type: ignore[attr-defined]
            score, breakdown = grade_hard(df.copy(), customers.copy(), orig.copy())
        else:
            raise HTTPException(status_code=400, detail=f"Unknown task: {task}")

        return {
            "task": task,
            "score": round(float(score), 4),
            "breakdown": {k: round(float(v), 4) for k, v in breakdown.items()},
            "step_count": _env._step_count,  # type: ignore[attr-defined]
            "rows": len(df),
            "columns": list(df.columns),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Grade failed: {exc}") from exc


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    """Entry point for the command-line script."""
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
