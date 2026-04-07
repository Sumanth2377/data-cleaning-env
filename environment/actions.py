"""
Action handlers for the Data Cleaning environment.

Every public method ``_action_<name>`` corresponds to an ``action_type`` string.
All handlers receive the current DataFrame, apply the requested transformation,
and return (new_df, human_readable_message, success_flag).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ActionError(Exception):
    """Raised when an action cannot be applied due to bad parameters."""


SUPPORTED_ACTIONS: List[str] = [
    "fill_missing",
    "drop_duplicates",
    "cast_column",
    "rename_column",
    "normalize_format",
    "apply_regex",
    "drop_column",
    "drop_rows_by_condition",
    "clip_outliers",
    "standardize_text",
    "fix_referential_integrity",
    "merge_tables",
]


class ActionHandler:
    """
    Stateful handler that applies cleaning actions to a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The main working dataset.
    aux_dfs : dict[str, pd.DataFrame]
        Auxiliary tables (used by hard task).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        aux_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        self.df = df.copy()
        self.aux_dfs: Dict[str, pd.DataFrame] = {
            k: v.copy() for k, v in (aux_dfs or {}).items()
        }

    # ------------------------------------------------------------------
    # Public dispatcher
    # ------------------------------------------------------------------

    def execute(
        self,
        action_type: str,
        parameters: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, str, bool]:
        """
        Dispatch an action.

        Returns
        -------
        (new_df, message, success)
        """
        if action_type not in SUPPORTED_ACTIONS:
            return (
                self.df,
                f"Unknown action '{action_type}'. Supported: {SUPPORTED_ACTIONS}",
                False,
            )

        method = getattr(self, f"_action_{action_type}", None)
        if method is None:
            return self.df, f"Action '{action_type}' not implemented.", False

        try:
            new_df, msg = method(**parameters)
            self.df = new_df
            return new_df, msg, True
        except ActionError as exc:
            return self.df, f"ActionError: {exc}", False
        except TypeError as exc:
            return self.df, f"Bad parameters for '{action_type}': {exc}", False
        except Exception as exc:  # noqa: BLE001
            return self.df, f"Unexpected error in '{action_type}': {exc}", False

    # ------------------------------------------------------------------
    # Individual action implementations
    # ------------------------------------------------------------------

    def _action_fill_missing(
        self,
        column: str,
        strategy: str = "mean",
        fill_value: Any = None,
    ) -> Tuple[pd.DataFrame, str]:
        """Fill NaN values in *column* using *strategy*."""
        df = self.df.copy()
        if column not in df.columns:
            raise ActionError(f"Column '{column}' not found. Available: {list(df.columns)}")

        before = int(df[column].isnull().sum())

        if strategy == "mean":
            df[column] = df[column].fillna(pd.to_numeric(df[column], errors="coerce").mean())
        elif strategy == "median":
            df[column] = df[column].fillna(pd.to_numeric(df[column], errors="coerce").median())
        elif strategy == "mode":
            mode_vals = df[column].mode()
            if len(mode_vals):
                df[column] = df[column].fillna(mode_vals.iloc[0])
        elif strategy == "forward_fill":
            df[column] = df[column].ffill()
        elif strategy == "backward_fill":
            df[column] = df[column].bfill()
        elif strategy == "constant":
            df[column] = df[column].fillna(fill_value)
        elif strategy == "drop":
            df = df.dropna(subset=[column]).reset_index(drop=True)
        elif strategy == "unknown":
            df[column] = df[column].fillna("unknown")
        else:
            raise ActionError(
                f"Unknown strategy '{strategy}'. Choose from: "
                "mean, median, mode, forward_fill, backward_fill, constant, drop, unknown"
            )

        after = int(df[column].isnull().sum())
        filled = before - after
        return df, f"fill_missing: filled {filled} nulls in '{column}' via '{strategy}'"

    def _action_drop_duplicates(
        self,
        subset: Optional[List[str]] = None,
        keep: str = "first",
    ) -> Tuple[pd.DataFrame, str]:
        """Remove duplicate rows."""
        df = self.df.copy()
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        removed = before - len(df)
        return df, f"drop_duplicates: removed {removed} duplicate rows"

    def _action_cast_column(
        self,
        column: str,
        dtype: str,
    ) -> Tuple[pd.DataFrame, str]:
        """Cast *column* to *dtype*."""
        df = self.df.copy()
        if column not in df.columns:
            raise ActionError(f"Column '{column}' not found.")

        original = str(df[column].dtype)

        dtype_map = {
            "int":      lambda s: pd.to_numeric(s, errors="coerce").astype("Int64"),
            "integer":  lambda s: pd.to_numeric(s, errors="coerce").astype("Int64"),
            "float":    lambda s: pd.to_numeric(s, errors="coerce"),
            "str":      lambda s: s.astype(str),
            "string":   lambda s: s.astype(str),
            "datetime": lambda s: pd.to_datetime(s, infer_datetime_format=True, errors="coerce"),
            "bool":     lambda s: s.astype(bool),
            "category": lambda s: s.astype("category"),
        }

        if dtype not in dtype_map:
            raise ActionError(f"Unsupported dtype '{dtype}'. Choose from: {list(dtype_map)}")

        df[column] = dtype_map[dtype](df[column])
        return df, f"cast_column: '{column}' {original} → {dtype}"

    def _action_rename_column(
        self,
        old_name: str,
        new_name: str,
    ) -> Tuple[pd.DataFrame, str]:
        """Rename a column."""
        df = self.df.copy()
        if old_name not in df.columns:
            raise ActionError(f"Column '{old_name}' not found.")
        df = df.rename(columns={old_name: new_name})
        return df, f"rename_column: '{old_name}' → '{new_name}'"

    def _action_normalize_format(
        self,
        column: str,
        format_type: str,
        output_format: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Normalise values in *column* to a standard format.

        format_type options
        -------------------
        phone          → (XXX) XXX-XXXX
        email          → lowercase, stripped
        date           → output_format (default: %Y-%m-%d)
        text_case      → output_format: lower | upper | title
        strip_currency → remove $ , € £ and cast to float
        zip_code       → 5-digit only
        """
        df = self.df.copy()
        if column not in df.columns:
            raise ActionError(f"Column '{column}' not found.")

        if format_type == "phone":
            def _norm_phone(val: Any) -> Any:
                if pd.isna(val):
                    return val
                digits = re.sub(r"\D", "", str(val))
                if len(digits) == 10:
                    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
                if len(digits) == 11 and digits[0] == "1":
                    return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
                return val  # cannot normalise — leave unchanged

            df[column] = df[column].apply(_norm_phone)
            return df, f"normalize_format: phone numbers in '{column}' → (XXX) XXX-XXXX"

        if format_type == "email":
            df[column] = df[column].astype(str).str.lower().str.strip()
            return df, f"normalize_format: email addresses in '{column}' → lowercase/stripped"

        if format_type == "date":
            target = output_format or "%Y-%m-%d"
            df[column] = (
                pd.to_datetime(df[column], infer_datetime_format=True, errors="coerce")
                .dt.strftime(target)
            )
            return df, f"normalize_format: dates in '{column}' → {target}"

        if format_type == "text_case":
            case = (output_format or "lower").lower()
            if case == "lower":
                df[column] = df[column].astype(str).str.lower()
            elif case == "upper":
                df[column] = df[column].astype(str).str.upper()
            elif case == "title":
                df[column] = df[column].astype(str).str.title()
            else:
                raise ActionError(f"Unknown case '{case}'. Choose: lower, upper, title")
            return df, f"normalize_format: '{column}' → {case} case"

        if format_type == "strip_currency":
            df[column] = (
                df[column].astype(str)
                .str.replace(r"[\$,€£]", "", regex=True)
                .str.strip()
                .pipe(pd.to_numeric, errors="coerce")
            )
            return df, f"normalize_format: stripped currency symbols from '{column}'"

        if format_type == "zip_code":
            df[column] = (
                df[column].astype(str).str.extract(r"(\d{5})")[0]
            )
            return df, f"normalize_format: zip codes in '{column}' → 5-digit"

        raise ActionError(
            f"Unknown format_type '{format_type}'. Choose from: "
            "phone, email, date, text_case, strip_currency, zip_code"
        )

    def _action_apply_regex(
        self,
        column: str,
        pattern: str,
        replacement: str,
    ) -> Tuple[pd.DataFrame, str]:
        """Apply a regex substitution across all values in *column*."""
        df = self.df.copy()
        if column not in df.columns:
            raise ActionError(f"Column '{column}' not found.")
        df[column] = df[column].astype(str).str.replace(pattern, replacement, regex=True)
        return df, f"apply_regex: '{pattern}' → '{replacement}' on '{column}'"

    def _action_drop_column(self, column: str) -> Tuple[pd.DataFrame, str]:
        """Drop *column* entirely."""
        df = self.df.copy()
        if column not in df.columns:
            raise ActionError(f"Column '{column}' not found.")
        df = df.drop(columns=[column])
        return df, f"drop_column: removed '{column}'"

    def _action_drop_rows_by_condition(
        self,
        column: str,
        operator: str,
        value: Any,
    ) -> Tuple[pd.DataFrame, str]:
        """Drop rows matching column [operator] value."""
        df = self.df.copy()
        if column not in df.columns:
            raise ActionError(f"Column '{column}' not found.")

        before = len(df)
        ops = {
            "==":       lambda s: s == value,
            "!=":       lambda s: s != value,
            ">":        lambda s: pd.to_numeric(s, errors="coerce") > float(value),
            "<":        lambda s: pd.to_numeric(s, errors="coerce") < float(value),
            ">=":       lambda s: pd.to_numeric(s, errors="coerce") >= float(value),
            "<=":       lambda s: pd.to_numeric(s, errors="coerce") <= float(value),
            "isnull":   lambda s: s.isnull(),
            "notnull":  lambda s: s.notnull(),
            "contains": lambda s: s.astype(str).str.contains(str(value), na=False),
        }
        if operator not in ops:
            raise ActionError(f"Unknown operator '{operator}'. Choose from: {list(ops)}")

        mask = ops[operator](df[column])
        df = df[~mask].reset_index(drop=True)
        removed = before - len(df)
        return df, f"drop_rows_by_condition: dropped {removed} rows where {column} {operator} {value}"

    def _action_clip_outliers(
        self,
        column: str,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> Tuple[pd.DataFrame, str]:
        """Clip or remove statistical outliers in a numeric column."""
        df = self.df.copy()
        if column not in df.columns:
            raise ActionError(f"Column '{column}' not found.")

        numeric = pd.to_numeric(df[column], errors="coerce")

        if method == "iqr":
            q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
            df[column] = numeric.clip(lower=lo, upper=hi)
            return df, f"clip_outliers: IQR [{lo:.2f}, {hi:.2f}] on '{column}'"

        if method == "zscore":
            mean, std = numeric.mean(), numeric.std()
            lo = mean - threshold * std
            hi = mean + threshold * std
            df[column] = numeric.clip(lower=lo, upper=hi)
            return df, f"clip_outliers: Z-score [{lo:.2f}, {hi:.2f}] on '{column}'"

        if method == "drop":
            q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
            before = len(df)
            df = df[(numeric >= lo) & (numeric <= hi)].reset_index(drop=True)
            return df, f"clip_outliers: dropped {before - len(df)} outliers in '{column}'"

        raise ActionError(f"Unknown method '{method}'. Choose from: iqr, zscore, drop")

    def _action_standardize_text(
        self,
        column: str,
        operations: List[str],
    ) -> Tuple[pd.DataFrame, str]:
        """Apply one or more text-standardisation operations to *column*."""
        df = self.df.copy()
        if column not in df.columns:
            raise ActionError(f"Column '{column}' not found.")

        op_map = {
            "strip":              lambda s: s.str.strip(),
            "lower":              lambda s: s.str.lower(),
            "upper":              lambda s: s.str.upper(),
            "title":              lambda s: s.str.title(),
            "remove_extra_spaces": lambda s: s.str.replace(r"\s+", " ", regex=True).str.strip(),
        }
        for op in operations:
            if op not in op_map:
                raise ActionError(f"Unknown text operation '{op}'. Choose from: {list(op_map)}")
            df[column] = op_map[op](df[column].astype(str))

        return df, f"standardize_text: [{', '.join(operations)}] on '{column}'"

    def _action_fix_referential_integrity(
        self,
        child_column: str,
        parent_table: str,
        parent_column: str,
        action: str = "drop",
    ) -> Tuple[pd.DataFrame, str]:
        """Remove or flag rows with FK violations."""
        if parent_table not in self.aux_dfs:
            raise ActionError(
                f"Table '{parent_table}' not found. Available: {list(self.aux_dfs)}"
            )
        parent_df = self.aux_dfs[parent_table]
        if parent_column not in parent_df.columns:
            raise ActionError(f"Column '{parent_column}' not found in '{parent_table}'.")

        df = self.df.copy()
        valid_ids = set(parent_df[parent_column].dropna())
        invalid_mask = ~df[child_column].isin(valid_ids)
        n_invalid = int(invalid_mask.sum())

        if action == "drop":
            df = df[~invalid_mask].reset_index(drop=True)
            return df, f"fix_referential_integrity: dropped {n_invalid} FK-violating rows"
        if action == "flag":
            df[f"{child_column}_valid"] = (~invalid_mask).astype(int)
            return df, f"fix_referential_integrity: flagged {n_invalid} FK violations"

        raise ActionError(f"Unknown action '{action}'. Choose from: drop, flag")

    def _action_merge_tables(
        self,
        right_table: str,
        left_on: str,
        right_on: str,
        how: str = "left",
        columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, str]:
        """Merge auxiliary *right_table* into the main DataFrame."""
        if right_table not in self.aux_dfs:
            raise ActionError(f"Table '{right_table}' not found.")

        right_df = self.aux_dfs[right_table].copy()
        if columns:
            keep = list({right_on} | set(columns))
            right_df = right_df[[c for c in keep if c in right_df.columns]]

        df = self.df.copy()
        before_cols = set(df.columns)
        df = df.merge(right_df, left_on=left_on, right_on=right_on, how=how, suffixes=("", f"_{right_table}"))
        added = set(df.columns) - before_cols
        return df, f"merge_tables: merged '{right_table}' on {left_on}={right_on}, added columns: {sorted(added)}"
