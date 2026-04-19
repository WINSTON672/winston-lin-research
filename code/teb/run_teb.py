"""
run_teb.py
----------
Task Efficiency Benchmark (TEB) — Model Query Runner

Queries models on all TEB tasks and records:
  - response text
  - response time (seconds)
  - estimated output tokens
  - input token estimate

Saves raw results to teb_results_{timestamp}.json for scoring by score_teb.py.

Usage:
    HF_TOKEN=your_token python run_teb.py
    HF_TOKEN=your_token python run_teb.py --tasks T2-01 T3-02   # specific tasks only
    HF_TOKEN=your_token python run_teb.py --tier 2               # tier 2 tasks only
"""

import os, sys, re, json, time, argparse
from datetime import datetime
import requests

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    sys.exit("Set HF_TOKEN environment variable first.")

# Models: (display_name, huggingface_id, flops_teraflops)
# FLOPs proxy = parameter count (B) × 1T, matching CES paper convention
MODELS = [
    ("Qwen2.5-7B",    "Qwen/Qwen2.5-7B-Instruct",              7.0),
    ("Llama-3-8B",    "meta-llama/Meta-Llama-3-8B-Instruct",    8.0),
    ("Llama-3.3-70B", "meta-llama/Llama-3.3-70B-Instruct",     70.0),
    ("Qwen2.5-72B",   "Qwen/Qwen2.5-72B-Instruct",             72.0),
]

HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
REQUEST_TIMEOUT = 120  # seconds
RETRY_DELAY = 30       # seconds to wait if model is loading

TASKS_FILE = os.path.join(os.path.dirname(__file__), "tasks.json")


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 tokens per word, or 4 chars per token."""
    return max(1, len(text) // 4)


def build_prompt(task: dict) -> str:
    """Combine task prompt and input into a single user message."""
    prompt = task["prompt"]
    if "input" in task:
        prompt += f"\n\n---\n{task['input']}"
    elif "input_files" in task:
        parts = []
        for fname, content in task["input_files"].items():
            parts.append(f"### {fname}\n```python\n{content}\n```")
        prompt += "\n\n---\n" + "\n\n".join(parts)
    return prompt


def query_model(hf_id: str, user_message: str, retries: int = 3) -> dict:
    """
    Query a HuggingFace Inference API model.
    Returns dict with keys: text, duration_s, error (if any).
    """
    url = HF_CHAT_URL
    payload = {
        "model": hf_id,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": 1024,
        "temperature": 0.0,
        "stream": False,
    }

    for attempt in range(retries):
        try:
            t0 = time.time()
            resp = requests.post(url, headers=HEADERS, json=payload, timeout=REQUEST_TIMEOUT)
            duration = time.time() - t0

            if resp.status_code == 503:
                # Model loading — wait and retry
                wait = RETRY_DELAY * (attempt + 1)
                print(f"      Model loading, waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                return {"text": None, "duration_s": duration,
                        "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}

            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return {
                "text": text,
                "duration_s": round(duration, 3),
                "input_tokens": usage.get("prompt_tokens", estimate_tokens(user_message)),
                "output_tokens": usage.get("completion_tokens", estimate_tokens(text)),
                "error": None,
            }

        except requests.exceptions.Timeout:
            return {"text": None, "duration_s": REQUEST_TIMEOUT,
                    "error": "Request timed out"}
        except Exception as e:
            return {"text": None, "duration_s": 0, "error": str(e)}

    return {"text": None, "duration_s": 0, "error": "Max retries exceeded"}


def run_benchmark(task_filter=None, tier_filter=None):
    with open(TASKS_FILE) as f:
        task_data = json.load(f)

    tasks = task_data["tasks"]

    # Apply filters
    if task_filter:
        tasks = [t for t in tasks if t["id"] in task_filter]
    if tier_filter is not None:
        tasks = [t for t in tasks if t["tier"] == tier_filter]

    if not tasks:
        sys.exit("No tasks match the specified filters.")

    print(f"\nTask Efficiency Benchmark")
    print(f"Tasks: {[t['id'] for t in tasks]}")
    print(f"Models: {[m[0] for m in MODELS]}")
    print(f"Starting run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = {
        "run_timestamp": datetime.now().isoformat(),
        "tasks_run": [t["id"] for t in tasks],
        "models": {},
    }

    for model_name, hf_id, flops in MODELS:
        print(f"[{model_name}]")
        results["models"][model_name] = {
            "hf_id": hf_id,
            "flops_T": flops,
            "responses": {},
        }

        for task in tasks:
            task_id = task["id"]
            user_prompt = build_prompt(task)
            input_tokens_est = estimate_tokens(user_prompt)

            print(f"  {task_id} ({task['title'][:40]})...", end=" ", flush=True)

            response = query_model(hf_id, user_prompt)

            if response["error"]:
                print(f"ERROR: {response['error']}")
            else:
                out_tok = response["output_tokens"]
                dur = response["duration_s"]
                print(f"done ({dur:.1f}s, ~{out_tok} output tokens)")

            results["models"][model_name]["responses"][task_id] = {
                "task_id": task_id,
                "tier": task["tier"],
                "human_baseline_minutes": task["human_baseline_minutes"],
                "input_tokens": response.get("input_tokens", input_tokens_est),
                "output_tokens": response.get("output_tokens", 0),
                "duration_s": response.get("duration_s", 0),
                "speedup_vs_human": round(
                    task["human_baseline_minutes"] * 60 / max(response.get("duration_s", 1), 0.1), 1
                ),
                "response_text": response.get("text"),
                "error": response.get("error"),
            }

            # Brief pause between requests to avoid rate limiting
            time.sleep(1)

        print()

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(os.path.dirname(__file__), f"teb_results_{ts}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {out_file}")
    print(f"\nNext step: python score_teb.py {out_file}")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TEB benchmark")
    parser.add_argument("--tasks", nargs="+", help="Run specific task IDs only (e.g. T2-01 T3-02)")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], help="Run all tasks of a specific tier")
    args = parser.parse_args()

    run_benchmark(task_filter=args.tasks, tier_filter=args.tier)
