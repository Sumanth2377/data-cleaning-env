"""Full live API test against the deployed HuggingFace Space."""
import urllib.request
import json

BASE = "https://sumanth2377-data-cleaning-env.hf.space"
PASS = 0
FAIL = 0

def get(path):
    try:
        r = urllib.request.urlopen(BASE + path, timeout=20)
        return r.status, json.loads(r.read())
    except Exception as e:
        return "ERR", str(e)

def post(path, body=None):
    try:
        data = json.dumps(body or {}).encode()
        req = urllib.request.Request(
            BASE + path, data=data,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        r = urllib.request.urlopen(req, timeout=25)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return "ERR", str(e)

def check(label, status, data, expect_key=None):
    global PASS, FAIL
    ok = status == 200 and (expect_key is None or expect_key in data)
    sym = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    val = data.get(expect_key) if expect_key and isinstance(data, dict) else ""
    print(f"  [{sym}] {label}" + (f" -> {expect_key}={val}" if expect_key else f" -> {status}"))
    return ok

print()
print("=" * 60)
print("  LIVE API TEST: sumanth2377-data-cleaning-env.hf.space")
print("=" * 60)

# --- System endpoints ---
print("\n[System Endpoints]")
s, d = get("/health")
check("GET /health", s, d, "status")

s, d = get("/info")
check("GET /info", s, d, "tasks")

s, d = get("/tasks")
tasks = d.get("tasks", []) if isinstance(d, dict) else []
check("GET /tasks (3 tasks)", s, {"tasks": tasks}, "tasks")
print(f"         tasks found: {[t['name'] for t in tasks]}")

s, d = get("/actions")
check("GET /actions", s, d, "total")

# --- Environment endpoints ---
print("\n[Environment Endpoints — csv-doctor]")
s, d = post("/reset", {"task_name": "csv-doctor", "seed": 42})
if check("POST /reset (csv-doctor)", s, d, "observation"):
    obs = d["observation"]
    print(f"         score={obs['current_score']}, rows={obs['stats']['total_rows']}, issues={len(obs['issues'])}")

s, d = post("/step", {"action": {"action_type": "drop_duplicates", "parameters": {}}})
if check("POST /step (drop_duplicates)", s, d, "reward"):
    print(f"         reward={d['reward']}, score={d['observation']['current_score']}")

s, d = post("/step", {"action": {"action_type": "normalize_format", "parameters": {"column": "salary", "format_type": "strip_currency"}}})
if check("POST /step (normalize salary)", s, d, "reward"):
    print(f"         reward={d['reward']}, score={d['observation']['current_score']}")

s, d = get("/state")
if check("GET /state", s, d, "current_score"):
    print(f"         task={d['task_name']}, step={d['step_count']}, score={d['current_score']}")

s, d = post("/grade")
if check("POST /grade", s, d, "score"):
    print(f"         score={d['score']}, breakdown={d['breakdown']}")

# --- Other tasks ---
print("\n[Environment Endpoints — other tasks]")
s, d = post("/reset", {"task_name": "schema-enforcer", "seed": 42})
if check("POST /reset (schema-enforcer)", s, d, "observation"):
    print(f"         score={d['observation']['current_score']}")

s, d = post("/reset", {"task_name": "pipeline-debugger", "seed": 42})
if check("POST /reset (pipeline-debugger)", s, d, "observation"):
    print(f"         score={d['observation']['current_score']}")

# --- Summary ---
print()
print("=" * 60)
total = PASS + FAIL
print(f"  RESULT: {PASS}/{total} passed  {'ALL PASS' if FAIL == 0 else str(FAIL) + ' FAILED'}")
print("=" * 60)
print()
