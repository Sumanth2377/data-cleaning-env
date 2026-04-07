"""
DataCleaningEnv — the main OpenEnv environment class.

Exposes:
  reset(task_name, seed)  → ResetResult
  step(action)            → StepResult
  state()                 → StateResult

All state is held in-memory; one instance = one episode.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from environment.actions import ActionHandler, SUPPORTED_ACTIONS
from environment.datasets.generator import (
    generate_easy_dataset,
    generate_medium_dataset,
    generate_hard_dataset,
    get_medium_target_schema,
)
from environment.graders.graders import grade_easy, grade_medium, grade_hard
from environment.models import (
    ColumnInfo,
    DataCleaningAction,
    DataCleaningObservation,
    DatasetStats,
    IssueDetail,
    ResetResult,
    StateResult,
    StepResult,
)


# ============================================================================
# Constants
# ============================================================================

TASK_CONFIG = {
    "csv-doctor": {
        "description": (
            "Fix data quality issues in a 200-row customer dataset. "
            "Issues include missing values in age/salary/email, salary stored as a "
            "currency string (e.g. '$45,200.00'), float ages instead of integers, "
            "duplicate rows, and inconsistently capitalised names and departments "
            "with stray whitespace. Use the available cleaning actions to maximise "
            "the data quality score."
        ),
        "max_steps": 15,
    },
    "schema-enforcer": {
        "description": (
            "Conform a 300-row contacts dataset to a strict target schema. "
            "Phone numbers appear in 5 different formats; normalise to (XXX) XXX-XXXX. "
            "Dates appear in 4 different formats; normalise to YYYY-MM-DD. "
            "Emails have mixed casing and stray whitespace; normalise to lowercase/stripped. "
            "Zip codes occasionally include the +4 suffix; keep only the 5-digit prefix. "
            "Country codes should be uppercase 2-3 letter codes. "
            "First and last names should be title case. "
            "The target_schema field in the observation shows the exact expected format per column."
        ),
        "max_steps": 20,
    },
    "pipeline-debugger": {
        "description": (
            "Repair a two-table dataset (orders + customers). "
            "Issues: ~10% of orders reference non-existent customer_ids (FK violations); "
            "~5% of price values are 10-50x outliers; ~5% of quantity values are 20-50x outliers; "
            "~8% of orders are implicit duplicates (same data, different order_id); "
            "orders lack the customer segment column (obtainable via merge). "
            "The grader measures referential integrity, deduplication, outlier removal, "
            "AND downstream ML R² improvement. "
            "The auxiliary_datasets field contains a sample view of the customers table."
        ),
        "max_steps": 30,
    },
}

# Penalty for destructive actions (dropping >30 % of rows in one step)
DESTRUCTION_PENALTY = -0.10
STEP_COST = -0.005


# ============================================================================
# Helper functions
# ============================================================================

def _column_info(df: pd.DataFrame) -> List[ColumnInfo]:
    infos = []
    for col in df.columns:
        series = df[col]
        n_null = int(series.isnull().sum())
        total = len(series)
        null_pct = round(100.0 * n_null / total, 2) if total else 0.0

        sample = series.dropna().head(5).tolist()
        # Ensure JSON-serialisable
        sample = [
            v if not isinstance(v, (np.integer, np.floating)) else v.item()
            for v in sample
        ]

        issues: List[str] = []
        if n_null > 0:
            issues.append(f"missing_values ({n_null} nulls, {null_pct:.1f}%)")

        # Detect currency strings
        if series.dtype == object:
            str_vals = series.dropna().astype(str)
            if str_vals.str.contains(r"[\$,€£]", regex=True).any():
                issues.append("currency_string — should be numeric")

        # Detect float age (should be int)
        if col == "age" and pd.api.types.is_float_dtype(series):
            non_null = series.dropna()
            if (non_null % 1 == 0).all():
                issues.append("stored_as_float — should be int")

        infos.append(
            ColumnInfo(
                name=col,
                dtype=str(series.dtype),
                null_count=n_null,
                null_pct=null_pct,
                unique_count=int(series.nunique()),
                sample_values=sample,
                detected_issues=issues,
            )
        )
    return infos


def _dataset_stats(df: pd.DataFrame) -> DatasetStats:
    total_cells = df.size
    missing = int(df.isnull().sum().sum())
    dupes = int(df.duplicated().sum())

    # Count dtype issues (currency strings)
    dtype_issues = 0
    for col in df.columns:
        if df[col].dtype == object:
            str_vals = df[col].dropna().astype(str)
            if str_vals.str.contains(r"[\$,€£]", regex=True).any():
                dtype_issues += 1
        if col == "age" and pd.api.types.is_float_dtype(df[col]):
            dtype_issues += 1

    # Format violations (basic heuristic: null + type issues)
    format_violations = dtype_issues

    return DatasetStats(
        total_rows=len(df),
        total_cols=len(df.columns),
        missing_cells=missing,
        missing_pct=round(100.0 * missing / total_cells, 2) if total_cells else 0.0,
        duplicate_rows=dupes,
        dtype_issues=dtype_issues,
        format_violations=format_violations,
    )


def _detect_issues(df: pd.DataFrame, task_name: str) -> List[IssueDetail]:
    issues: List[IssueDetail] = []

    # Missing values
    for col in df.columns:
        n = int(df[col].isnull().sum())
        if n > 0:
            issues.append(
                IssueDetail(
                    issue_type="missing_values",
                    column=col,
                    severity="high" if n / len(df) > 0.10 else "medium",
                    description=f"{n} missing values ({100*n/len(df):.1f}%)",
                    affected_rows=n,
                )
            )

    # Duplicates
    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        issues.append(
            IssueDetail(
                issue_type="duplicate_rows",
                severity="medium",
                description=f"{n_dupes} duplicate rows detected",
                affected_rows=n_dupes,
            )
        )

    # Currency strings
    for col in df.columns:
        if df[col].dtype == object:
            str_vals = df[col].dropna().astype(str)
            bad = str_vals.str.contains(r"[\$,€£]", regex=True).sum()
            if bad:
                issues.append(
                    IssueDetail(
                        issue_type="wrong_dtype",
                        column=col,
                        severity="high",
                        description=f"'{col}' contains {bad} currency-formatted strings; expected numeric",
                        affected_rows=int(bad),
                    )
                )

    return issues


# ============================================================================
# Main environment class
# ============================================================================

class DataCleaningEnv:
    """
    OpenEnv-compliant Data Cleaning & Preprocessing Environment.

    Usage
    -----
    env = DataCleaningEnv()
    reset_result = env.reset(task_name="csv-doctor", seed=42)
    step_result  = env.step(DataCleaningAction(action_type="drop_duplicates", parameters={}))
    state_result = env.state()
    """

    SUPPORTED_TASKS = list(TASK_CONFIG.keys())

    def __init__(self) -> None:
        self._task_name: str = "csv-doctor"
        self._seed: int = 42
        self._df: pd.DataFrame = pd.DataFrame()
        self._aux_dfs: Dict[str, pd.DataFrame] = {}
        self._original_orders: Optional[pd.DataFrame] = None
        self._handler: Optional[ActionHandler] = None
        self._step_count: int = 0
        self._done: bool = False
        self._dataset_id: str = ""
        self._actions_history: List[str] = []
        self._current_score: float = 0.0
        self._prev_score: float = 0.0

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> ResetResult:
        """Initialise a new episode."""
        if task_name is not None:
            if task_name not in self.SUPPORTED_TASKS:
                raise ValueError(
                    f"Unknown task '{task_name}'. Supported: {self.SUPPORTED_TASKS}"
                )
            self._task_name = task_name
        if seed is not None:
            self._seed = seed

        self._step_count = 0
        self._done = False
        self._dataset_id = str(uuid.uuid4())[:8]
        self._actions_history = []
        self._current_score = 0.0
        self._prev_score = 0.0
        self._aux_dfs = {}
        self._original_orders = None

        # Load dataset
        if self._task_name == "csv-doctor":
            self._df = generate_easy_dataset(seed=self._seed)
            self._handler = ActionHandler(self._df)

        elif self._task_name == "schema-enforcer":
            self._df = generate_medium_dataset(seed=self._seed)
            self._handler = ActionHandler(self._df)

        elif self._task_name == "pipeline-debugger":
            customers, orders = generate_hard_dataset(seed=self._seed)
            self._df = orders.copy()
            self._original_orders = orders.copy()
            self._aux_dfs = {"customers": customers}
            self._handler = ActionHandler(self._df, self._aux_dfs)

        # Compute initial score
        self._current_score = self._compute_score()
        self._prev_score = self._current_score

        obs = self._build_observation()
        return ResetResult(observation=obs, info={"task": self._task_name, "seed": self._seed})

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: DataCleaningAction) -> StepResult:
        """Apply *action* and return the new observation, reward, done, info."""
        if self._done:
            obs = self._build_observation()
            return StepResult(
                observation=obs,
                reward=0.0,
                done=True,
                info={"error": "Episode already done. Call reset() to start a new episode."},
            )

        self._step_count += 1
        rows_before = len(self._df)

        # Execute action
        new_df, message, success = self._handler.execute(  # type: ignore[union-attr]
            action.action_type, action.parameters
        )

        # Destructive-action penalty
        rows_after = len(new_df)
        destructive_penalty = 0.0
        if rows_before > 0 and (rows_before - rows_after) / rows_before > 0.30:
            destructive_penalty = DESTRUCTION_PENALTY

        # Update main df
        self._df = new_df
        self._handler.df = new_df  # type: ignore[union-attr]

        # Sync aux_dfs for hard task
        if self._task_name == "pipeline-debugger":
            self._aux_dfs = self._handler.aux_dfs  # type: ignore[union-attr]

        # Track action history
        action_str = f"step={self._step_count}: {action.action_type}({action.parameters}) → {message}"
        self._actions_history.append(action_str)

        # Compute reward
        new_score = self._compute_score()
        score_delta = new_score - self._prev_score
        reward = float(
            np.clip(score_delta + STEP_COST + destructive_penalty, -1.0, 1.0)
        )
        self._prev_score = new_score
        self._current_score = new_score

        # Episode termination
        max_steps = TASK_CONFIG[self._task_name]["max_steps"]
        self._done = (
            self._step_count >= max_steps
            or self._current_score >= 0.95
        )

        info: Dict[str, Any] = {
            "action_message": message,
            "action_success": success,
            "score_before": round(self._prev_score, 4),
            "score_after": round(new_score, 4),
            "score_delta": round(score_delta, 4),
            "destructive_penalty": destructive_penalty,
            "step_cost": STEP_COST,
            "rows_before": rows_before,
            "rows_after": rows_after,
        }

        obs = self._build_observation()
        return StepResult(observation=obs, reward=round(reward, 4), done=self._done, info=info)

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------

    def state(self) -> StateResult:
        """Return a lightweight snapshot of the current episode state."""
        return StateResult(
            task_name=self._task_name,
            step_count=self._step_count,
            done=self._done,
            current_score=round(self._current_score, 4),
            stats=_dataset_stats(self._df),
            actions_history=list(self._actions_history),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_score(self) -> float:
        import sys
        try:
            if self._task_name == "csv-doctor":
                score, _ = grade_easy(self._df)
            elif self._task_name == "schema-enforcer":
                score, _ = grade_medium(self._df)
            elif self._task_name == "pipeline-debugger":
                customers = self._aux_dfs.get("customers", pd.DataFrame())
                # Ensure we pass clean copies to avoid pandas/numpy state issues
                score, _ = grade_hard(
                    self._df.copy(),
                    customers.copy() if len(customers) > 0 else pd.DataFrame(),
                    (self._original_orders.copy()
                     if self._original_orders is not None else self._df.copy()),
                )
            else:
                score = 0.0
        except Exception as exc:
            print(f"[DEBUG] _compute_score exception ({self._task_name}): {exc}", file=sys.stderr)
            score = 0.0
        return float(np.clip(float(score), 0.0, 1.0))

    def _build_observation(self) -> DataCleaningObservation:
        config = TASK_CONFIG[self._task_name]

        # Auxiliary dataset preview (hard task — show customers column names + 3 rows)
        aux_preview: Optional[Dict[str, Any]] = None
        if self._task_name == "pipeline-debugger" and "customers" in self._aux_dfs:
            cdf = self._aux_dfs["customers"]
            aux_preview = {
                "customers": {
                    "columns": list(cdf.columns),
                    "rows": cdf.head(3).to_dict(orient="records"),
                    "total_rows": len(cdf),
                }
            }

        # Target schema preview (medium task)
        ts: Optional[Dict[str, Any]] = None
        if self._task_name == "schema-enforcer":
            ts = get_medium_target_schema()

        return DataCleaningObservation(
            task_name=self._task_name,
            task_description=config["description"],
            dataset_id=self._dataset_id,
            step_count=self._step_count,
            columns=_column_info(self._df),
            stats=_dataset_stats(self._df),
            issues=_detect_issues(self._df, self._task_name),
            actions_history=list(self._actions_history),
            target_schema=ts,
            auxiliary_datasets=aux_preview,
            current_score=round(self._current_score, 4),
            max_steps=config["max_steps"],
        )
