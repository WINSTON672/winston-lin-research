"""
pilot_experiment.py
-------------------
Pilot CES Validation Experiment.

Queries 4 models available on HuggingFace free inference:
  - Phi-3-mini-4k-instruct   (3.8T FLOPs)
  - Mistral-7B-Instruct-v0.3 (7T FLOPs)
  - Meta-Llama-3-8B-Instruct (8T FLOPs)
  - Mixtral-8x7B-Instruct-v0.1 (46.7T FLOPs)

Question set (designed for this experiment, not drawn from MMLU/HumanEval):
  - 10 knowledge MCQ questions (A/B/C/D)
  - 10 code tasks (write a Python function, scored by test-case execution)

Scoring → bits_resolved → I(M) → raw_CES → normalized CES
Compare pilot CES ranking to paper CES ranking (Spearman).
"""

import os, sys, re, math, json, textwrap, time, subprocess, tempfile
import requests

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    sys.exit("Set HF_TOKEN environment variable first.")

# ── Models ────────────────────────────────────────────────────────────────────
# FLOPs proxy = model parameter count (billions) × 1T, matching paper convention.
# Llama-3-8B is in paper (CES=0.2819). Llama-3.3-70B ≈ Llama-3-70B (CES=0.6333).
# Qwen2.5-7B/72B add independent data points with comparable sizes.
MODELS = [
    ("Qwen2.5-7B",      "Qwen/Qwen2.5-7B-Instruct",                  7.0),
    ("Llama-3-8B",      "meta-llama/Meta-Llama-3-8B-Instruct",        8.0),
    ("Qwen2.5-72B",     "Qwen/Qwen2.5-72B-Instruct",                 72.0),
    ("Llama-3.3-70B",   "meta-llama/Llama-3.3-70B-Instruct",         70.0),
]

HUMAN = (0.898, 1.000, 1000.0)   # MMLU, HE, FLOPs — same as paper
W_MMLU = 0.47
W_HE   = 0.53

# ── Question set (10 MCQ + 10 code) ──────────────────────────────────────────
#
# MCQ: 4-choice, answer is one of A/B/C/D
# Chosen to span science, logic, math, and language — all solvable above
# 50% by capable models, not sourced from MMLU verbatim.

MCQ = [
    # (question_text, choices_dict, correct_letter)
    (
        "A gas occupies 4 L at 300 K and 1 atm. What volume does it occupy at 600 K and 1 atm?",
        {"A": "2 L", "B": "4 L", "C": "8 L", "D": "16 L"},
        "C"
    ),
    (
        "Which sorting algorithm has average-case time complexity O(n log n) and worst-case O(n^2)?",
        {"A": "Merge sort", "B": "Heap sort", "C": "Quick sort", "D": "Bubble sort"},
        "C"
    ),
    (
        "In classical mechanics, if net force on an object is zero, the object:",
        {"A": "Must be stationary", "B": "Must be accelerating",
         "C": "Maintains constant velocity", "D": "Loses kinetic energy"},
        "C"
    ),
    (
        "The determinant of a 2×2 matrix [[a,b],[c,d]] is:",
        {"A": "ab - cd", "B": "ad + bc", "C": "ac - bd", "D": "ad - bc"},
        "D"
    ),
    (
        "Which data structure supports O(1) average-case lookup, insertion, and deletion?",
        {"A": "Binary search tree", "B": "Hash table", "C": "Linked list", "D": "Stack"},
        "B"
    ),
    (
        "In Python, what does `list(range(3))` evaluate to?",
        {"A": "[1, 2, 3]", "B": "[0, 1, 2, 3]", "C": "[0, 1, 2]", "D": "[1, 2]"},
        "C"
    ),
    (
        "The derivative of e^(3x) with respect to x is:",
        {"A": "e^(3x)", "B": "3e^x", "C": "3e^(3x)", "D": "e^(3x-1)"},
        "C"
    ),
    (
        "Which of the following is NOT a property of a binary heap?",
        {"A": "Complete binary tree", "B": "Parent ≥ children (max-heap)",
         "C": "O(log n) insertion", "D": "O(1) arbitrary element search"},
        "D"
    ),
    (
        "A fair coin is flipped 3 times. What is the probability of getting exactly 2 heads?",
        {"A": "1/4", "B": "3/8", "C": "1/2", "D": "2/3"},
        "B"
    ),
    (
        "In SQL, which clause filters groups after aggregation?",
        {"A": "WHERE", "B": "GROUP BY", "C": "HAVING", "D": "ORDER BY"},
        "C"
    ),
]

# Code tasks: each is (description, function_signature_hint, test_cases)
# test_cases: list of (args_tuple, expected_return)
CODE_TASKS = [
    # Each entry: (description, function_hint, [(args_tuple, expected), ...])
    # args_tuple must match the function's parameter list exactly.
    (
        "Write a Python function `sum_even(n)` that returns the sum of all even numbers from 2 to n (inclusive). Return 0 if n < 2.",
        "def sum_even(n):",
        [((4,), 6), ((6,), 12), ((1,), 0), ((10,), 30), ((2,), 2)]
    ),
    (
        "Write a Python function `is_palindrome(s)` that returns True if string s is a palindrome (ignoring case), False otherwise.",
        "def is_palindrome(s):",
        [(("racecar",), True), (("Hello",), False), (("Madam",), True), (("abc",), False)]
    ),
    (
        "Write a Python function `count_vowels(s)` that returns the number of vowels (a,e,i,o,u, case-insensitive) in string s.",
        "def count_vowels(s):",
        [(("hello",), 2), (("AEIOU",), 5), (("xyz",), 0), (("Python",), 1)]
    ),
    (
        "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number (0-indexed: fibonacci(0)=0, fibonacci(1)=1).",
        "def fibonacci(n):",
        [((0,), 0), ((1,), 1), ((6,), 8), ((10,), 55)]
    ),
    (
        "Write a Python function `flatten(lst)` that takes a list of lists (one level deep) and returns a flat list. E.g. flatten([[1,2],[3],[4,5]])=[1,2,3,4,5].",
        "def flatten(lst):",
        [(([[1,2],[3]],), [1,2,3]),
         (([[4,5],[6,7]],), [4,5,6,7]),
         (([[],[1]],), [1])]
    ),
    (
        "Write a Python function `second_largest(lst)` that returns the second largest unique value in a list. Assume at least 2 distinct values.",
        "def second_largest(lst):",
        [(([3,1,4,1,5,9,2,6],), 6),
         (([10,5,10,3],), 5),
         (([1,2],), 1)]
    ),
    (
        "Write a Python function `run_length_encode(s)` that returns a list of (char, count) tuples for consecutive runs. E.g. 'aaabbc' → [('a',3),('b',2),('c',1)].",
        "def run_length_encode(s):",
        [(("aaabbc",), [('a',3),('b',2),('c',1)]),
         (("abcd",), [('a',1),('b',1),('c',1),('d',1)]),
         (("aaa",), [('a',3)])]
    ),
    (
        "Write a Python function `rotate_list(lst, k)` that rotates the list to the right by k positions. E.g. rotate_list([1,2,3,4,5], 2) = [4,5,1,2,3].",
        "def rotate_list(lst, k):",
        [(([1,2,3,4,5], 2), [4,5,1,2,3]),
         (([1,2,3], 1), [3,1,2]),
         (([1], 5), [1])]
    ),
    (
        "Write a Python function `prime_factors(n)` that returns a sorted list of prime factors of n (with repetition). E.g. prime_factors(12)=[2,2,3].",
        "def prime_factors(n):",
        [((12,), [2,2,3]), ((7,), [7]), ((36,), [2,2,3,3]), ((30,), [2,3,5])]
    ),
    (
        "Write a Python function `max_subarray_sum(lst)` that returns the maximum subarray sum (Kadane's algorithm). Assume at least one positive number exists.",
        "def max_subarray_sum(lst):",
        [(([-2,1,-3,4,-1,2,1,-5,4],), 6),
         (([1],), 1),
         (([5,4,-1,7,8],), 23)]
    ),
]

# ── Scoring helpers ───────────────────────────────────────────────────────────

def bits_resolved(p):
    if p <= 0.5:
        return 0.0
    p = min(p, 0.9999)
    return 1.0 + p * math.log2(p) + (1 - p) * math.log2(1 - p)

def ces_score(p_mmlu, p_he, flops):
    i = W_MMLU * bits_resolved(p_mmlu) + W_HE * bits_resolved(p_he)
    raw = i / math.log2(1 + flops)
    h_i = W_MMLU * bits_resolved(HUMAN[0]) + W_HE * bits_resolved(HUMAN[1])
    h_raw = h_i / math.log2(1 + HUMAN[2])
    return raw / h_raw

def extract_letter(text):
    """Extract A/B/C/D answer from model response."""
    text = text.strip()
    # Try explicit patterns first
    for pattern in [
        r'\b([ABCD])\b[\s.):–—]',
        r'^([ABCD])$',
        r'answer[:\s]+([ABCD])\b',
        r'\*\*([ABCD])\*\*',
        r'([ABCD])\)',
    ]:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    # Last resort: first standalone capital letter
    m = re.search(r'\b([ABCD])\b', text)
    return m.group(1).upper() if m else None

def run_code_test(code_str, test_cases, task_idx):
    """Execute generated code against test cases. Returns (passed, total).
    test_cases: list of (args_tuple, expected) — args_tuple is passed as *args_tuple.
    """
    # Extract function name from first 'def ' line in code
    func_name = None
    for line in code_str.split("\n"):
        m = re.match(r'def (\w+)\s*\(', line.strip())
        if m:
            func_name = m.group(1)
            break
    if not func_name:
        return 0, len(test_cases)

    passed = 0
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(code_str + "\n\n")
        f.write("import sys, json\n")
        f.write("_results = []\n")
        for args, expected in test_cases:
            # args is a tuple; each element is one argument to the function
            f.write(f"try:\n")
            f.write(f"    _r = {func_name}(*{repr(args)})\n")
            f.write(f"    _results.append(_r == {repr(expected)})\n")
            f.write(f"except Exception as _e:\n")
            f.write(f"    _results.append(False)\n")
        f.write("print(json.dumps(_results))\n")
        fname = f.name

    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            results_list = json.loads(result.stdout.strip())
            passed = sum(results_list)
    except Exception:
        pass
    finally:
        try:
            os.unlink(fname)
        except Exception:
            pass
    return passed, len(test_cases)

def extract_code(text, hint):
    """Extract Python function from model response."""
    # Try fenced code block first
    m = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1)
    # Try to find function definition
    func_name = hint.split('(')[0].replace('def ', '').strip()
    m = re.search(rf'(def {func_name}\(.*?)(?=\ndef |\Z)', text, re.DOTALL)
    if m:
        return m.group(1)
    return text  # fallback: use entire response

def call_model(model_id, messages, max_tokens=512, retries=4):
    """Call HuggingFace inference via InferenceClient with retry."""
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=HF_TOKEN)
    for attempt in range(retries):
        try:
            r = client.chat_completion(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.01,   # near-deterministic
            )
            return r.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Rate" in msg:
                wait = 30 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
            elif "503" in msg or "loading" in msg.lower():
                print(f"    Model loading, waiting 20s...", flush=True)
                time.sleep(20)
            else:
                print(f"    Error attempt {attempt+1}: {msg[:100]}", flush=True)
                time.sleep(5)
    return None

# ── Main experiment loop ──────────────────────────────────────────────────────

MCQ_PROMPT = (
    "Answer the following multiple-choice question. "
    "Respond with ONLY the letter of the correct answer (A, B, C, or D). "
    "Do not explain.\n\n"
)

CODE_PROMPT = (
    "Write a Python function as described. "
    "Respond with ONLY a Python code block (```python ... ```) containing the complete function. "
    "No explanation, no example calls.\n\n"
)

results = {}  # model_name → {mcq_correct, he_passed, he_total, ...}

for model_name, model_id, flops in MODELS:
    print(f"\n{'='*60}")
    print(f"Model: {model_name}  ({model_id})")
    print(f"{'='*60}")

    mcq_correct = 0
    he_passed   = 0
    he_total    = 0
    raw_mcq     = []
    raw_code    = []

    # ── MCQ ──────────────────────────────────────────────
    print(f"\n  MCQ ({len(MCQ)} questions):")
    for i, (question, choices, correct) in enumerate(MCQ):
        choices_str = "\n".join(f"  {k}) {v}" for k, v in choices.items())
        prompt = MCQ_PROMPT + question + "\n" + choices_str
        response = call_model(model_id, [{"role": "user", "content": prompt}], max_tokens=10)

        if response is None:
            print(f"    Q{i+1}: [API FAIL]")
            raw_mcq.append(None)
            continue

        predicted = extract_letter(response)
        ok = (predicted == correct)
        mcq_correct += int(ok)
        raw_mcq.append((correct, predicted, ok))
        print(f"    Q{i+1}: expected={correct}  got={predicted}  {'✓' if ok else '✗'}  ({response.strip()[:30]})")
        time.sleep(1)  # rate limit courtesy

    # ── Code ──────────────────────────────────────────────
    print(f"\n  Code ({len(CODE_TASKS)} tasks):")
    for i, (description, hint, test_cases) in enumerate(CODE_TASKS):
        prompt = CODE_PROMPT + description
        response = call_model(model_id, [{"role": "user", "content": prompt}], max_tokens=400)

        if response is None:
            print(f"    T{i+1}: [API FAIL]")
            raw_code.append(None)
            continue

        code = extract_code(response, hint)
        passed, total = run_code_test(code, test_cases, i)
        he_passed += passed
        he_total  += total
        ok = (passed == total)
        raw_code.append((passed, total))
        print(f"    T{i+1}: {passed}/{total} tests  {'✓' if ok else '✗'}  ({hint.split('(')[0]}...)")
        time.sleep(1)

    # ── CES computation ───────────────────────────────────
    valid_mcq = [r for r in raw_mcq if r is not None]
    p_mmlu = mcq_correct / len(valid_mcq) if valid_mcq else 0.0
    p_he   = he_passed   / he_total       if he_total  else 0.0
    pilot_ces = ces_score(p_mmlu, p_he, flops)

    results[model_name] = {
        "flops":   flops,
        "mcq_n":   len(valid_mcq),
        "mcq_ok":  mcq_correct,
        "p_mmlu":  p_mmlu,
        "he_pass": he_passed,
        "he_tot":  he_total,
        "p_he":    p_he,
        "pilot_ces": pilot_ces,
    }

    print(f"\n  → p_MMLU = {mcq_correct}/{len(valid_mcq)} = {p_mmlu:.3f}")
    print(f"  → p_HE   = {he_passed}/{he_total}  = {p_he:.3f}")
    print(f"  → Pilot CES = {pilot_ces:.4f}")

# ── Summary table ─────────────────────────────────────────────────────────────

# Paper CES for models in paper; Qwen2.5 not in paper so estimated from similar-size models.
PAPER_CES = {
    "Qwen2.5-7B":    None,    # not in paper
    "Llama-3-8B":    0.2819,  # from paper Table 1
    "Qwen2.5-72B":   None,    # not in paper
    "Llama-3.3-70B": 0.6333,  # ≈ Llama-3-70B from paper (same parameter count)
}

print(f"\n{'='*75}")
print(f"{'Model':<16} {'p_MCQ':>7} {'p_Code':>7} {'Pilot CES':>10} {'Paper CES':>12} {'Pilot Rank':>12}")
print(f"{'='*75}")

ces_vals   = [(n, results[n]["pilot_ces"]) for n in results]
sorted_pilot = sorted(ces_vals, key=lambda x: -x[1])
rank_pilot = {n: i+1 for i, (n, _) in enumerate(sorted_pilot)}

for name in results:
    r = results[name]
    paper = PAPER_CES[name]
    paper_str = f"{paper:.4f}" if paper is not None else "   N/A  "
    print(f"{name:<16} {r['p_mmlu']:>7.3f} {r['p_he']:>7.3f} {r['pilot_ces']:>10.4f} {paper_str:>12} {rank_pilot[name]:>12}")

# Spearman: compare pilot ranking vs expected capability ordering
# Expected: 70B-class models (Llama-3.3-70B, Qwen2.5-72B) should rank above 8B-class
print(f"\nPilot ranking (1=highest CES): {sorted([f'{rank_pilot[n]}:{n}' for n in rank_pilot])}")

# Compute Spearman for 2 models with known paper CES (Llama-3-8B, Llama-3.3-70B)
known = {n: v for n, v in PAPER_CES.items() if v is not None}
if len(known) >= 2:
    paper_rank_of_known = sorted(known.keys(), key=lambda x: -known[x])
    paper_rank_map = {n: i+1 for i, n in enumerate(paper_rank_of_known)}
    pilot_vals_known = [(n, results[n]["pilot_ces"]) for n in known]
    pilot_rank_known = sorted(pilot_vals_known, key=lambda x: -x[1])
    pilot_rank_map = {n: i+1 for i, (n, _) in enumerate(pilot_rank_known)}
    d2 = sum((pilot_rank_map[n] - paper_rank_map[n])**2 for n in known)
    n_k = len(known)
    rho = 1 - 6*d2 / (n_k*(n_k**2 - 1)) if n_k > 2 else (1.0 if d2 == 0 else -1.0)
    print(f"Spearman ρ (pilot vs paper, {n_k} models with known paper CES): {rho:.4f}  Σd²={d2}")
else:
    rho, d2 = None, None

# Save JSON for paper
with open("pilot_results.json", "w") as f:
    json.dump({
        "models": results,
        "pilot_ranking": [n for n, _ in sorted_pilot],
        "spearman_known_rho": rho,
        "sum_d2": d2,
    }, f, indent=2)
print("\nResults saved to pilot_results.json")
