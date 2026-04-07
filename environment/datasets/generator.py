"""
Synthetic dirty-data generator for all three tasks.

All generators accept a ``seed`` parameter so that every episode is fully
reproducible.  Inject deliberately designed flaws so that graders can measure
exactly how well the agent cleaned the data.
"""
from __future__ import annotations

import random
import re
import string
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# Helpers
# ============================================================================

RNG_SEED = 42


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rand(seed: int) -> random.Random:
    r = random.Random(seed)
    return r


# ============================================================================
# Task 1 — CSV Doctor (Easy)
# ============================================================================

FIRST_NAMES = [
    "alice", "BOB", "Charlie", "DIANA", "eve", "Frank", "Grace", "HENRY",
    "Iris", "JACK", "karen", "LIAM", "Mia", "NOAH", "olivia", "PETE",
    "quinn", "RACHEL", "Sam", "TINA", "uma", "VICTOR", "wendy", "XANDER",
    "yara", "ZACH",
]
LAST_NAMES = [
    "Smith", "JONES", "Williams", "BROWN", "davis", "MILLER", "wilson",
    "MOORE", "taylor", "ANDERSON", "thomas", "JACKSON", "white", "HARRIS",
    "martin", "THOMPSON", "garcia", "MARTINEZ", "robinson", "CLARK",
]
DEPARTMENTS = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"]
CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia"]


def generate_easy_dataset(seed: int = RNG_SEED, n_rows: int = 200) -> pd.DataFrame:
    """
    Generate a 200-row customer/employee CSV with deliberately injected issues:

    Issues injected
    ---------------
    1. ~15 % null values in ``age`` column.
    2. ~12 % null values in ``email`` column.
    3. ~10 % null values in ``salary`` column.
    4. ``salary`` stored as a currency string (e.g. "$45,200.00") instead of float.
    5. ``age`` stored as float (e.g. 34.0) instead of int.
    6. ~8 % exact duplicate rows.
    7. ``name`` column has inconsistent casing (all-caps, all-lower, title).
    8. ``department`` column has trailing/leading whitespace in some values.
    """
    rng = _rng(seed)
    r = _rand(seed)

    n_clean = int(n_rows * 0.92)   # leave room for duplicates

    ids = list(range(1001, 1001 + n_clean))
    names = [
        f"{r.choice(FIRST_NAMES)} {r.choice(LAST_NAMES)}"
        for _ in range(n_clean)
    ]
    ages = rng.integers(22, 65, size=n_clean).astype(float)
    departments = [r.choice(DEPARTMENTS) for _ in range(n_clean)]
    cities = [r.choice(CITIES) for _ in range(n_clean)]
    salaries = rng.integers(35_000, 120_000, size=n_clean)
    emails = [
        f"{name.split()[0].lower()}.{name.split()[1].lower()}{r.randint(1,99)}@example.com"
        for name in names
    ]

    df = pd.DataFrame({
        "id": ids,
        "name": names,
        "age": ages,
        "department": departments,
        "city": cities,
        "salary": [f"${s:,.2f}" for s in salaries],
        "email": emails,
    })

    # ---- Inject issues ----

    # 1. Missing ages (~15 %)
    null_age_idx = rng.choice(n_clean, size=int(n_clean * 0.15), replace=False)
    df.loc[null_age_idx, "age"] = np.nan

    # 2. Missing emails (~12 %)
    null_email_idx = rng.choice(n_clean, size=int(n_clean * 0.12), replace=False)
    df.loc[null_email_idx, "email"] = np.nan

    # 3. Missing salary (~10 %)
    null_salary_idx = rng.choice(n_clean, size=int(n_clean * 0.10), replace=False)
    df.loc[null_salary_idx, "salary"] = np.nan

    # 4. Trailing/leading whitespace in department
    ws_idx = rng.choice(n_clean, size=int(n_clean * 0.12), replace=False)
    df.loc[ws_idx, "department"] = df.loc[ws_idx, "department"].apply(
        lambda v: f"  {v}  " if isinstance(v, str) else v
    )

    # 5. Duplicate rows (~8 %)
    n_dupes = int(n_clean * 0.08)
    dupe_idx = rng.choice(n_clean, size=n_dupes, replace=False)
    dupes = df.iloc[dupe_idx].copy()
    df = pd.concat([df, dupes], ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    return df


def get_easy_ground_truth(seed: int = RNG_SEED) -> pd.DataFrame:
    """Return the fully-cleaned version for grader comparison."""
    df = generate_easy_dataset(seed)

    # Drop duplicates
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "id"])

    # Fix salary: strip currency, convert to float
    df["salary"] = (
        df["salary"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Fix age: convert to int (nullable)
    df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Int64")

    # Fix name casing: title case
    df["name"] = df["name"].str.title()

    # Fix department: strip whitespace
    df["department"] = df["department"].str.strip()

    # Fill missing age with median
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age)

    # Fill missing email with placeholder
    df["email"] = df["email"].fillna("unknown@example.com")

    # Fill missing salary with median
    median_salary = df["salary"].median()
    df["salary"] = df["salary"].fillna(median_salary)

    return df.reset_index(drop=True)


# ============================================================================
# Task 2 — Schema Enforcer (Medium)
# ============================================================================

TARGET_SCHEMA = {
    "first_name":  {"dtype": "string",   "format": "title_case"},
    "last_name":   {"dtype": "string",   "format": "title_case"},
    "phone":       {"dtype": "string",   "format": r"^\(\d{3}\) \d{3}-\d{4}$"},
    "email":       {"dtype": "string",   "format": r"^[a-z0-9_.+-]+@[a-z0-9-]+\.[a-z]{2,}$"},
    "birth_date":  {"dtype": "datetime", "format": "%Y-%m-%d"},
    "zip_code":    {"dtype": "string",   "format": r"^\d{5}$"},
    "country":     {"dtype": "string",   "format": "upper_case"},
}

PHONE_FORMATS = [
    lambda a, b, c: f"({a}) {b}-{c}",        # canonical
    lambda a, b, c: f"{a}-{b}-{c}",           # dashes
    lambda a, b, c: f"{a}.{b}.{c}",           # dots
    lambda a, b, c: f"{a}{b}{c}",             # no separator
    lambda a, b, c: f"+1 ({a}) {b}-{c}",      # +1 prefix
]

DATE_FORMATS = [
    "%Y-%m-%d",    # canonical
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%b %d, %Y",
]

COUNTRIES = ["US", "us", "Us", "USA", "CA", "ca", "Ca", "UK", "uk"]
ZIP_SUFFIXES = ["", "", "", "", "-1234", "-5678"]  # occasionally has +4


def generate_medium_dataset(seed: int = RNG_SEED, n_rows: int = 300) -> pd.DataFrame:
    """
    Generate a 300-row contacts dataset with schema violations.
    """
    rng = _rng(seed)
    r = _rand(seed)

    fnames = [r.choice(FIRST_NAMES).title() for _ in range(n_rows)]
    lnames = [r.choice(LAST_NAMES).title() for _ in range(n_rows)]

    # Phone — randomly pick one of 5 formats per row
    def rand_phone(rr: random.Random) -> str:
        area   = str(rr.randint(200, 999))
        prefix = str(rr.randint(200, 999))
        line   = str(rr.randint(1000, 9999))
        fmt    = rr.choice(PHONE_FORMATS)
        return fmt(area, prefix, line)

    phones = [rand_phone(r) for _ in range(n_rows)]

    # Email — mixed case + leading/trailing whitespace on ~20 %
    emails = [
        f"{fn.lower()}.{ln.lower()}{r.randint(1,99)}@example.com"
        for fn, ln in zip(fnames, lnames)
    ]
    bad_email_idx = rng.choice(n_rows, size=int(n_rows * 0.20), replace=False)
    for i in bad_email_idx:
        emails[i] = emails[i].upper() if r.random() < 0.5 else f"  {emails[i]}  "

    # Birth dates — random format per row
    import datetime
    dates = []
    for _ in range(n_rows):
        y = r.randint(1950, 2003)
        m = r.randint(1, 12)
        d = r.randint(1, 28)
        dt = datetime.date(y, m, d)
        fmt = r.choice(DATE_FORMATS)
        dates.append(dt.strftime(fmt))

    # Zip codes — occasionally include +4 suffix
    zips = [
        f"{r.randint(10000,99999)}{r.choice(ZIP_SUFFIXES)}"
        for _ in range(n_rows)
    ]

    countries = [r.choice(COUNTRIES) for _ in range(n_rows)]

    df = pd.DataFrame({
        "first_name":  fnames,
        "last_name":   lnames,
        "phone":       phones,
        "email":       emails,
        "birth_date":  dates,
        "zip_code":    zips,
        "country":     countries,
    })

    return df


def get_medium_target_schema() -> dict:
    return TARGET_SCHEMA


# ============================================================================
# Task 3 — Pipeline Debugger (Hard)
# ============================================================================

def generate_hard_dataset(
    seed: int = RNG_SEED,
    n_customers: int = 100,
    n_orders: int = 300,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate two-table dataset (customers + orders) with complex issues:

    1. Referential integrity: ~10 % of orders reference non-existent customer_ids.
    2. Statistical outliers: ~5 % of price values are 10-50x the normal value.
    3. Statistical outliers: ~5 % of quantity values are 20-50x the normal value.
    4. Implicit duplicates: ~8 % of order rows are near-duplicates (same data,
       different order_id, tiny timestamp difference).
    5. Orders lack the customer ``segment`` column, obtainable via a merge.
    """
    rng = _rng(seed)
    r = _rand(seed)

    # --- Customers table ---
    cust_ids = list(range(1, n_customers + 1))
    segments = ["Premium", "Standard", "Basic"]
    customers = pd.DataFrame({
        "customer_id": cust_ids,
        "name": [
            f"{r.choice(FIRST_NAMES).title()} {r.choice(LAST_NAMES).title()}"
            for _ in range(n_customers)
        ],
        "segment": [r.choice(segments) for _ in range(n_customers)],
        "country": [r.choice(["US", "CA", "UK", "AU"]) for _ in range(n_customers)],
    })

    # --- Orders table ---
    valid_cust_ids = list(range(1, n_customers + 1))
    order_cust_ids = [r.choice(valid_cust_ids) for _ in range(n_orders)]

    base_prices = rng.uniform(10.0, 500.0, size=n_orders)
    base_qtys   = rng.integers(1, 20, size=n_orders).astype(float)

    # Inject price outliers (~5 %)
    outlier_p_idx = rng.choice(n_orders, size=int(n_orders * 0.05), replace=False)
    base_prices[outlier_p_idx] *= rng.uniform(10, 50, size=len(outlier_p_idx))

    # Inject quantity outliers (~5 %)
    outlier_q_idx = rng.choice(n_orders, size=int(n_orders * 0.05), replace=False)
    base_qtys[outlier_q_idx] *= rng.uniform(20, 50, size=len(outlier_q_idx))

    import datetime
    base_date = datetime.date(2023, 1, 1)
    order_dates = [
        (base_date + datetime.timedelta(days=r.randint(0, 364))).strftime("%Y-%m-%d")
        for _ in range(n_orders)
    ]

    orders = pd.DataFrame({
        "order_id":    list(range(10001, 10001 + n_orders)),
        "customer_id": order_cust_ids,
        "product":     [r.choice(["Laptop", "Phone", "Tablet", "Monitor", "Keyboard"]) for _ in range(n_orders)],
        "price":       np.round(base_prices, 2),
        "quantity":    base_qtys.astype(int),
        "order_date":  order_dates,
    })

    # Inject FK violations (~10 %)
    bad_fk_idx = rng.choice(n_orders, size=int(n_orders * 0.10), replace=False)
    orders.loc[bad_fk_idx, "customer_id"] = rng.integers(
        n_customers + 1, n_customers + 50, size=len(bad_fk_idx)
    )

    # Inject implicit duplicates (~8 %)
    n_dupes = int(n_orders * 0.08)
    dupe_src_idx = rng.choice(n_orders, size=n_dupes, replace=False)
    dupes = orders.iloc[dupe_src_idx].copy()
    # Give them new order ids
    dupes["order_id"] = list(range(20001, 20001 + n_dupes))
    orders = pd.concat([orders, dupes], ignore_index=True)

    return customers, orders


def get_hard_ground_truth(seed: int = RNG_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return cleaned customers + orders for grader comparison."""
    customers, orders = generate_hard_dataset(seed)

    # 1. Fix FK violations — drop orders with invalid customer_id
    valid_ids = set(customers["customer_id"])
    orders = orders[orders["customer_id"].isin(valid_ids)].copy()

    # 2. Drop implicit duplicates (keep first occurrence per order content)
    orders = orders.drop_duplicates(
        subset=["customer_id", "product", "price", "quantity", "order_date"]
    ).copy()

    # 3. Clip price outliers using IQR
    q1, q3 = orders["price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    orders["price"] = orders["price"].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

    # 4. Clip quantity outliers using IQR
    q1q, q3q = orders["quantity"].quantile([0.25, 0.75])
    iqrq = q3q - q1q
    orders["quantity"] = orders["quantity"].clip(
        lower=q1q - 1.5 * iqrq, upper=q3q + 1.5 * iqrq
    )

    # 5. Merge to add segment column
    orders = orders.merge(
        customers[["customer_id", "segment"]],
        on="customer_id",
        how="left",
    )

    return customers, orders
