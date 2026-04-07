"""Final comprehensive validation of all 3 tasks."""
from environment.env import DataCleaningEnv
from environment.models import DataCleaningAction

env = DataCleaningEnv()

tasks = {
    "csv-doctor": [
        ("drop_duplicates", {}),
        ("normalize_format", {"column": "salary", "format_type": "strip_currency"}),
        ("fill_missing", {"column": "age", "strategy": "median"}),
        ("cast_column", {"column": "age", "dtype": "int"}),
        ("fill_missing", {"column": "salary", "strategy": "median"}),
        ("fill_missing", {"column": "email", "strategy": "constant", "fill_value": "unknown@example.com"}),
        ("standardize_text", {"column": "name", "operations": ["title"]}),
        ("standardize_text", {"column": "department", "operations": ["strip"]}),
    ],
    "schema-enforcer": [
        ("normalize_format", {"column": "phone", "format_type": "phone"}),
        ("normalize_format", {"column": "birth_date", "format_type": "date"}),
        ("normalize_format", {"column": "email", "format_type": "email"}),
        ("normalize_format", {"column": "zip_code", "format_type": "zip_code"}),
        ("normalize_format", {"column": "country", "format_type": "text_case", "output_format": "upper"}),
        ("standardize_text", {"column": "first_name", "operations": ["title"]}),
        ("standardize_text", {"column": "last_name", "operations": ["title"]}),
    ],
    "pipeline-debugger": [
        ("fix_referential_integrity", {"child_column": "customer_id", "parent_table": "customers", "parent_column": "customer_id", "action": "drop"}),
        ("drop_duplicates", {"subset": ["customer_id", "product", "price", "quantity", "order_date"]}),
        ("clip_outliers", {"column": "price", "method": "iqr", "threshold": 1.5}),
        ("clip_outliers", {"column": "quantity", "method": "iqr", "threshold": 1.5}),
        ("merge_tables", {"right_table": "customers", "left_on": "customer_id", "right_on": "customer_id", "how": "left", "columns": ["segment"]}),
    ],
}

print("=" * 60)
print("  FINAL SCORE REPORT (seed=42, rule-based agent)")
print("=" * 60)

all_pass = True
for task, actions in tasks.items():
    r = env.reset(task_name=task, seed=42)
    init = r.observation.current_score
    step_rewards = []
    for at, params in actions:
        sr = env.step(DataCleaningAction(action_type=at, parameters=params))
        step_rewards.append(sr.reward)
    final = env.state().current_score
    improved = final > init
    ok = improved and 0.0 <= final <= 1.0
    all_pass = all_pass and ok
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {task}")
    print(f"    initial={init:.4f}  ->  final={final:.4f}  delta={final-init:+.4f}")
    print(f"    rewards: {[round(x,3) for x in step_rewards]}")

print("=" * 60)
print(f"  OVERALL: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
print("=" * 60)
