"""
Deterministic graders for all three tasks.

Each grader:
  - receives the agent's current DataFrame (and auxiliary tables where needed)
  - returns a float score in [0.0, 1.0]
  - is fully deterministic (no randomness after dataset generation)
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# Task 1 — CSV Doctor  (Easy)
# ============================================================================

EASY_WEIGHTS = {
    "completeness": 0.30,
    "type_correctness": 0.30,
    "deduplication": 0.20,
    "format_consistency": 0.20,
}


def grade_easy(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Score the CSV-Doctor task.

    Parameters
    ----------
    df : pd.DataFrame
        The agent's current dataset state.

    Returns
    -------
    (overall_score, breakdown_dict)
    """
    breakdown: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1. Completeness — how many cells are non-null?
    # ------------------------------------------------------------------
    key_cols = [c for c in ["age", "salary", "email"] if c in df.columns]
    if key_cols:
        total_cells = len(df) * len(key_cols)
        non_null = sum(df[c].notna().sum() for c in key_cols)
        completeness = non_null / total_cells if total_cells else 0.0
    else:
        completeness = 0.0
    breakdown["completeness"] = round(float(completeness), 4)

    # ------------------------------------------------------------------
    # 2. Type correctness
    #    - salary should be numeric (float/int)
    #    - age should be numeric
    # ------------------------------------------------------------------
    type_scores = []
    if "salary" in df.columns:
        try:
            numeric_salary = pd.to_numeric(df["salary"], errors="coerce")
            frac_numeric = numeric_salary.notna().mean()
            # Also check that no '$' symbols remain
            str_vals = df["salary"].astype(str)
            frac_no_currency = (~str_vals.str.contains(r"[\$,€£]", regex=True)).mean()
            type_scores.append((frac_numeric + frac_no_currency) / 2)
        except Exception:
            type_scores.append(0.0)

    if "age" in df.columns:
        try:
            numeric_age = pd.to_numeric(df["age"], errors="coerce")
            frac_numeric = numeric_age.notna().mean()
            # Bonus: check integers (no decimal part)
            frac_int = (numeric_age.dropna() % 1 == 0).mean() if numeric_age.notna().any() else 0.0
            type_scores.append((frac_numeric + frac_int) / 2)
        except Exception:
            type_scores.append(0.0)

    type_correctness = float(np.mean(type_scores)) if type_scores else 0.0
    breakdown["type_correctness"] = round(type_correctness, 4)

    # ------------------------------------------------------------------
    # 3. Deduplication — measure duplicate row fraction (inverted)
    # ------------------------------------------------------------------
    if len(df) == 0:
        dedup = 0.0
    else:
        dupe_frac = df.duplicated().sum() / len(df)
        dedup = max(0.0, 1.0 - dupe_frac)
    breakdown["deduplication"] = round(float(dedup), 4)

    # ------------------------------------------------------------------
    # 4. Format consistency
    #    - name should be title case
    #    - department should have no leading/trailing whitespace
    # ------------------------------------------------------------------
    fmt_scores = []
    if "name" in df.columns:
        names = df["name"].dropna().astype(str)
        frac_title = (names == names.str.title()).mean() if len(names) else 0.0
        fmt_scores.append(float(frac_title))

    if "department" in df.columns:
        depts = df["department"].dropna().astype(str)
        frac_stripped = (depts == depts.str.strip()).mean() if len(depts) else 0.0
        fmt_scores.append(float(frac_stripped))

    format_consistency = float(np.mean(fmt_scores)) if fmt_scores else 0.0
    breakdown["format_consistency"] = round(format_consistency, 4)

    # ------------------------------------------------------------------
    # Weighted overall score
    # ------------------------------------------------------------------
    overall = (
        EASY_WEIGHTS["completeness"]    * breakdown["completeness"]
        + EASY_WEIGHTS["type_correctness"] * breakdown["type_correctness"]
        + EASY_WEIGHTS["deduplication"]    * breakdown["deduplication"]
        + EASY_WEIGHTS["format_consistency"] * breakdown["format_consistency"]
    )
    overall = float(np.clip(overall, 0.0, 1.0))
    return overall, breakdown


# ============================================================================
# Task 2 — Schema Enforcer  (Medium)
# ============================================================================

SCHEMA_PATTERNS: Dict[str, str] = {
    "phone":      r"^\(\d{3}\) \d{3}-\d{4}$",
    "email":      r"^[a-z0-9_.+\-]+@[a-z0-9\-]+\.[a-z]{2,}$",
    "birth_date": r"^\d{4}-\d{2}-\d{2}$",
    "zip_code":   r"^\d{5}$",
    # Accept 2-3 uppercase letters (US, CA, UK, USA, etc.)
    "country":    r"^[A-Z]{2,3}$",
}

MEDIUM_WEIGHTS: Dict[str, float] = {
    "phone":      0.20,
    "email":      0.20,
    "birth_date": 0.20,
    "zip_code":   0.15,
    "country":    0.10,
    "name_case":  0.15,
}


def grade_medium(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """
    Score the Schema-Enforcer task by measuring conformance to the target schema.
    """
    breakdown: Dict[str, float] = {}

    for col, pattern in SCHEMA_PATTERNS.items():
        if col not in df.columns:
            breakdown[col] = 0.0
            continue
        vals = df[col].dropna().astype(str)
        if len(vals) == 0:
            breakdown[col] = 0.0
            continue
        frac = vals.str.match(pattern, na=False).mean()
        breakdown[col] = round(float(frac), 4)

    # Name case (first_name, last_name should be title case)
    name_scores = []
    for col in ["first_name", "last_name"]:
        if col in df.columns:
            vals = df[col].dropna().astype(str)
            if len(vals):
                name_scores.append(float((vals == vals.str.title()).mean()))
    breakdown["name_case"] = round(float(np.mean(name_scores)) if name_scores else 0.0, 4)

    overall = sum(
        MEDIUM_WEIGHTS.get(k, 0.0) * v for k, v in breakdown.items()
    )
    overall = float(np.clip(overall, 0.0, 1.0))
    return overall, breakdown


# ============================================================================
# Task 3 — Pipeline Debugger  (Hard)
# ============================================================================

HARD_WEIGHTS = {
    "referential_integrity": 0.30,
    "deduplication":         0.20,
    "outlier_removal":       0.20,
    "downstream_ml":         0.30,
}


def _downstream_score(orders: pd.DataFrame, customers: pd.DataFrame) -> float:
    """
    Train a simple LinearRegression on (price × quantity) and return R².

    A perfectly cleaned dataset should yield a higher R² than the raw one.
    We normalise by clamping to [0, 1].
    """
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import LabelEncoder

        df = orders.copy()

        # Ensure required columns exist
        for col in ["price", "quantity"]:
            if col not in df.columns:
                return 0.0

        df["revenue"] = pd.to_numeric(df["price"], errors="coerce") * pd.to_numeric(
            df["quantity"], errors="coerce"
        )
        df = df.dropna(subset=["price", "quantity", "revenue"])
        if len(df) < 10:
            return 0.0

        # Feature: product category (encoded)
        if "product" in df.columns:
            le = LabelEncoder()
            df["product_enc"] = le.fit_transform(df["product"].astype(str))
        else:
            df["product_enc"] = 0

        # Feature: segment (encoded) if present
        if "segment" in df.columns:
            le2 = LabelEncoder()
            df["segment_enc"] = le2.fit_transform(df["segment"].astype(str))
        else:
            df["segment_enc"] = 0

        X = df[["price", "quantity", "product_enc", "segment_enc"]].values
        y = df["revenue"].values

        model = LinearRegression()
        model.fit(X, y)
        r2 = float(model.score(X, y))
        return float(np.clip(r2, 0.0, 1.0))

    except Exception:
        return 0.0


def grade_hard(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    original_orders: pd.DataFrame,
) -> Tuple[float, Dict[str, float]]:
    """
    Score the Pipeline-Debugger task.

    Parameters
    ----------
    orders          : agent's cleaned orders DataFrame
    customers       : agent's cleaned customers DataFrame
    original_orders : raw orders DataFrame (for baseline comparison)
    """
    breakdown: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1. Referential integrity — what fraction of order customer_ids are valid?
    # ------------------------------------------------------------------
    if "customer_id" in orders.columns and "customer_id" in customers.columns:
        valid_ids = set(customers["customer_id"].dropna())
        if len(orders):
            frac_valid = orders["customer_id"].isin(valid_ids).mean()
        else:
            frac_valid = 0.0
    else:
        frac_valid = 0.0
    breakdown["referential_integrity"] = round(float(frac_valid), 4)

    # ------------------------------------------------------------------
    # 2. Deduplication of orders
    # ------------------------------------------------------------------
    if len(orders):
        dupe_frac = orders.duplicated(
            subset=[c for c in ["customer_id", "product", "price", "quantity", "order_date"]
                    if c in orders.columns]
        ).sum() / len(orders)
        dedup = float(np.clip(1.0 - dupe_frac, 0.0, 1.0))
    else:
        dedup = 0.0
    breakdown["deduplication"] = round(dedup, 4)

    # ------------------------------------------------------------------
    # 3. Outlier removal — compare max value reduction of price/quantity to original
    #    Score is based on how much the extreme values were tamed relative to
    #    what a perfect IQR-clip would achieve.
    # ------------------------------------------------------------------
    outlier_scores = []
    for col in ["price", "quantity"]:
        if col not in orders.columns or col not in original_orders.columns:
            continue
        orig_vals = pd.to_numeric(original_orders[col], errors="coerce").dropna()
        clean_vals = pd.to_numeric(orders[col], errors="coerce").dropna()
        if len(orig_vals) == 0 or len(clean_vals) == 0:
            continue

        # Compute ideal clipped range (IQR method, threshold=1.5)
        q1, q3 = float(orig_vals.quantile(0.25)), float(orig_vals.quantile(0.75))
        iqr = q3 - q1
        ideal_max = q3 + 1.5 * iqr
        ideal_min = q1 - 1.5 * iqr
        orig_max = float(orig_vals.max())
        orig_min = float(orig_vals.min())
        clean_max = float(clean_vals.max())
        clean_min = float(clean_vals.min())

        # Score: how close is clean_max to ideal_max?
        orig_excess_high = max(orig_max - ideal_max, 0.0)
        clean_excess_high = max(clean_max - ideal_max, 0.0)
        if orig_excess_high == 0:
            score_high = 1.0
        else:
            score_high = float(np.clip(1.0 - clean_excess_high / orig_excess_high, 0.0, 1.0))

        orig_excess_low = max(ideal_min - orig_min, 0.0)
        clean_excess_low = max(ideal_min - clean_min, 0.0)
        if orig_excess_low == 0:
            score_low = 1.0
        else:
            score_low = float(np.clip(1.0 - clean_excess_low / orig_excess_low, 0.0, 1.0))

        outlier_scores.append((score_high + score_low) / 2.0)
    breakdown["outlier_removal"] = round(float(np.mean(outlier_scores)) if outlier_scores else 0.5, 4)

    # ------------------------------------------------------------------
    # 4. Downstream ML quality — R² of revenue prediction
    # ------------------------------------------------------------------
    breakdown["downstream_ml"] = round(_downstream_score(orders, customers), 4)

    # ------------------------------------------------------------------
    # Weighted overall
    # ------------------------------------------------------------------
    overall = sum(HARD_WEIGHTS[k] * breakdown[k] for k in HARD_WEIGHTS if k in breakdown)
    overall = float(np.clip(overall, 0.0, 1.0))
    return overall, breakdown
