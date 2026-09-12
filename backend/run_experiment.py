"""
run_experiment.py — NumeriX ACW Research Data Collector
=========================================================
Run this script to automatically test all 3 strategies
across 30 queries and generate your research results.

Usage:
    cd numerix-na-webapp/backend
    pip install requests
    python run_experiment.py

Make sure your FastAPI server is running first:
    uvicorn main:app --reload
"""

import json
import time
import requests
from datetime import datetime

BASE_URL = "http://localhost:8000/api/chatbot/ask"

# ── 30 test queries covering all modules ─────────────────────────────────────
TEST_QUERIES = [
    # Root Finder (6 queries)
    "What is the Bisection method?",
    "How does Newton-Raphson converge?",
    "What is False Position method?",
    "When does Fixed-Point iteration fail?",
    "What is quadratic convergence?",
    "How do I choose the initial bracket for Bisection?",

    # Integration (5 queries)
    "What is Simpson's 1/3 rule?",
    "When should I use Trapezoidal rule?",
    "What is the difference between Simpson 1/3 and 3/8?",
    "How many intervals does Simpson's 1/3 require?",
    "What is numerical integration used for?",

    # ODE (4 queries)
    "Why is RK4 more accurate than Euler's method?",
    "What is Heun's method?",
    "How does Euler's method work?",
    "What is a first order ODE?",

    # Interpolation (4 queries)
    "What is Lagrange interpolation?",
    "When do I use Newton's forward difference?",
    "What is divided difference?",
    "What is the difference between forward and backward difference?",

    # Differentiation (3 queries)
    "What is finite difference?",
    "How do you compute a numerical derivative?",
    "What is forward difference formula?",

    # Linear Systems (3 queries)
    "What is LU decomposition?",
    "What is the Doolittle method?",
    "How does back substitution work?",

    # Error Analysis (3 queries)
    "What is truncation error?",
    "What is round-off error?",
    "What is absolute vs relative error?",

    # General (2 queries)
    "What is NumeriX?",
    "What topics does this app cover?",
]

STRATEGIES = ["baseline", "moderate", "aggressive"]
RESULTS = {s: [] for s in STRATEGIES}


def run_query(query: str, strategy: str) -> dict:
    payload = {
        "message": query,
        "history": [],
        "strategy": strategy
    }
    try:
        start = time.time()
        resp = requests.post(BASE_URL, json=payload, timeout=30)
        latency = round((time.time() - start) * 1000, 1)  # ms
        data = resp.json()
        metrics = data.get("acw_metrics", {})
        return {
            "query": query,
            "strategy": strategy,
            "latency_ms": latency,
            "tokens_before": metrics.get("tokens_before", 0),
            "tokens_after": metrics.get("tokens_after", 0),
            "token_reduction_pct": metrics.get("token_reduction_pct", 0),
            "chunks_retrieved": metrics.get("chunks_retrieved", 0),
            "chunks_selected": metrics.get("chunks_selected", 0),
            "reply_length": len(data.get("reply", "")),
            "sources": data.get("sources", []),
            "status": "ok"
        }
    except Exception as e:
        return {
            "query": query,
            "strategy": strategy,
            "status": "error",
            "error": str(e)
        }


def print_summary(results: dict):
    print("\n" + "="*65)
    print("  NumeriX ACW — Research Results Summary")
    print(f"  Run at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("="*65)

    print(f"\n{'Strategy':<12} {'Queries':>8} {'Avg Tok Before':>15} "
          f"{'Avg Tok After':>14} {'Reduction %':>12} {'Avg Latency':>12}")
    print("-"*65)

    for strategy in STRATEGIES:
        records = [r for r in results[strategy] if r.get("status") == "ok"]
        if not records:
            continue
        n = len(records)
        avg_before = sum(r["tokens_before"] for r in records) / n
        avg_after = sum(r["tokens_after"] for r in records) / n
        avg_reduction = sum(r["token_reduction_pct"] for r in records) / n
        avg_latency = sum(r["latency_ms"] for r in records) / n

        print(f"{strategy:<12} {n:>8} {avg_before:>15.1f} "
              f"{avg_after:>14.1f} {avg_reduction:>11.1f}% {avg_latency:>10.0f}ms")

    print("\n" + "="*65)
    print("  Full results saved to: acw_experiment_results.json")
    print("  Use these numbers in Section 5 of your research paper!")
    print("="*65 + "\n")


def main():
    print(f"\nNumeriX ACW Experiment")
    print(f"Testing {len(TEST_QUERIES)} queries x {len(STRATEGIES)} strategies")
    print(f"= {len(TEST_QUERIES) * len(STRATEGIES)} total API calls\n")
    print("Make sure your FastAPI server is running at localhost:8000\n")

    total = len(TEST_QUERIES) * len(STRATEGIES)
    done = 0

    for strategy in STRATEGIES:
        print(f"\n--- Strategy: {strategy.upper()} ---")
        for query in TEST_QUERIES:
            result = run_query(query, strategy)
            RESULTS[strategy].append(result)
            done += 1
            status = "OK" if result.get("status") == "ok" else "ERR"
            reduction = result.get("token_reduction_pct", 0)
            print(f"  [{done:02d}/{total}] [{status}] "
                  f"Reduction: {reduction:5.1f}%  | {query[:45]}")
            time.sleep(0.5)  # be polite to the API

    # Save full results
    with open("acw_experiment_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)

    print_summary(RESULTS)


if __name__ == "__main__":
    main()
