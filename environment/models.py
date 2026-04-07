"""
DataCleaningEnv — Pydantic models for the OpenEnv Data Cleaning environment.

All public types used by step() / reset() / state() are defined here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Column-level info
# ---------------------------------------------------------------------------

class ColumnInfo(BaseModel):
    """Statistics and detected issues for a single column."""
    name: str
    dtype: str
    null_count: int
    null_pct: float = Field(ge=0.0, le=100.0)
    unique_count: int
    sample_values: List[Any] = Field(default_factory=list)
    detected_issues: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset-level statistics
# ---------------------------------------------------------------------------

class DatasetStats(BaseModel):
    """Aggregate data-quality statistics for the current dataset."""
    total_rows: int
    total_cols: int
    missing_cells: int
    missing_pct: float = Field(ge=0.0, le=100.0)
    duplicate_rows: int
    dtype_issues: int
    format_violations: int


# ---------------------------------------------------------------------------
# Issue catalogue
# ---------------------------------------------------------------------------

class IssueDetail(BaseModel):
    """A single detected data-quality issue."""
    issue_type: Literal[
        "missing_values",
        "wrong_dtype",
        "duplicate_rows",
        "format_violation",
        "outlier",
        "referential_integrity",
        "schema_mismatch",
    ]
    column: Optional[str] = None
    severity: Literal["low", "medium", "high"]
    description: str
    affected_rows: int


# ---------------------------------------------------------------------------
# Observation returned by reset() and step()
# ---------------------------------------------------------------------------

class DataCleaningObservation(BaseModel):
    """Full observation returned to the agent after every step."""
    task_name: str
    task_description: str
    dataset_id: str
    step_count: int = Field(ge=0)
    columns: List[ColumnInfo]
    stats: DatasetStats
    issues: List[IssueDetail]
    actions_history: List[str] = Field(default_factory=list)
    # Medium task: target schema the agent must conform the dataset to
    target_schema: Optional[Dict[str, Any]] = None
    # Hard task: auxiliary tables (column preview only to keep payload manageable)
    auxiliary_datasets: Optional[Dict[str, Any]] = None
    current_score: float = Field(default=0.0, ge=0.0, le=1.0)
    max_steps: int


# ---------------------------------------------------------------------------
# Action sent by the agent
# ---------------------------------------------------------------------------

class DataCleaningAction(BaseModel):
    """A data-cleaning operation issued by the agent."""
    action_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reward with breakdown
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    completeness: float = 0.0
    consistency: float = 0.0
    validity: float = 0.0
    efficiency: float = 0.0
    downstream_quality: float = 0.0   # Hard task only


class DataCleaningReward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    breakdown: RewardBreakdown
    message: str = ""


# ---------------------------------------------------------------------------
# Step / Reset / State return types
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: DataCleaningObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    observation: DataCleaningObservation
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResult(BaseModel):
    task_name: str
    step_count: int
    done: bool
    current_score: float
    stats: DatasetStats
    actions_history: List[str]


# ---------------------------------------------------------------------------
# HTTP request helpers (used by FastAPI server)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: Optional[str] = "csv-doctor"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    action: DataCleaningAction
