from environment.env import DataCleaningEnv
from environment.models import DataCleaningAction
env = DataCleaningEnv()
results = []
for task, actions in [
    ("csv-doctor", [
        ("drop_duplicates", {}),
        ("normalize_format", {"column":"salary","format_type":"strip_currency"}),
        ("fill_missing", {"column":"age","strategy":"median"}),
        ("cast_column", {"column":"age","dtype":"int"}),
        ("fill_missing", {"column":"salary","strategy":"median"}),
        ("fill_missing", {"column":"email","strategy":"constant","fill_value":"unknown@example.com"}),
        ("standardize_text", {"column":"name","operations":["title"]}),
        ("standardize_text", {"column":"department","operations":["strip"]}),
    ]),
    ("schema-enforcer", [
        ("normalize_format", {"column":"phone","format_type":"phone"}),
        ("normalize_format", {"column":"birth_date","format_type":"date"}),
        ("normalize_format", {"column":"email","format_type":"email"}),
        ("normalize_format", {"column":"zip_code","format_type":"zip_code"}),
        ("normalize_format", {"column":"country","format_type":"text_case","output_format":"upper"}),
        ("standardize_text", {"column":"first_name","operations":["title"]}),
        ("standardize_text", {"column":"last_name","operations":["title"]}),
    ]),
    ("pipeline-debugger", [
        ("fix_referential_integrity", {"child_column":"customer_id","parent_table":"customers","parent_column":"customer_id","action":"drop"}),
        ("drop_duplicates", {"subset":["customer_id","product","price","quantity","order_date"]}),
        ("clip_outliers", {"column":"price","method":"iqr","threshold":1.5}),
        ("clip_outliers", {"column":"quantity","method":"iqr","threshold":1.5}),
        ("merge_tables", {"right_table":"customers","left_on":"customer_id","right_on":"customer_id","how":"left","columns":["segment"]}),
    ]),
]:
    r = env.reset(task_name=task, seed=42)
    init = r.observation.current_score
    for at, params in actions:
        env.step(DataCleaningAction(action_type=at, parameters=params))
    final = env.state().current_score
    results.append((task, init, final))
    print(f"SCORE|{task}|{init:.4f}|{final:.4f}|{final-init:+.4f}|{'PASS' if final > init else 'FAIL'}")
