"""
run_ablation.py — Ablation Study for the ACW Scoring Function
=================================================================
Drop this file into: numerix/backend/  (same folder as run_experiment.py)

WHAT THIS TESTS
----------------
Your scoring function is:  combined = alpha * cosine_similarity + (1 - alpha) * density
This script isolates WHY that particular formulation (alpha = 0.7) works, by
sweeping:

    cosine_only     (i.e. the original un-modified retrieval ranking)
    density_only    (i.e. ignore semantic relevance entirely)
    alpha = 0.3
    alpha = 0.5
    alpha = 0.7   <- your paper's chosen value
    alpha = 0.9

For each config it runs your 30 test queries (strategy="moderate", top-2),
capturing:
    - token_reduction_pct       (efficiency side of the trade-off)
    - answer quality scores     (correctness / relevance / completeness /
                                  faithfulness, via the same LLM-judge used
                                  in evaluate_answer_quality.py)

This gives you a single table you can drop straight into an "Ablation
Study" section: efficiency AND quality, side by side, per configuration.

REQUIRES
--------
    - acw.py must have the `alpha` / `scoring_mode` parameters (already
      added if you're using the updated acw.py that ships with this file)
    - chatbot.py's ChatInput/ask() must forward `alpha` and `scoring_mode`
      to adaptive_context_wrapper (already added — see the diff notes)
    - Server running:  uvicorn main:app --reload

USAGE
-----
    cd numerix/backend
    python run_ablation.py
"""

import json
import time
from typing import Dict, List

import requests

from run_experiment import TEST_QUERIES
from evaluate_answer_quality import judge_answer  # reuse the exact same judge

ASK_URL = "http://localhost:8000/api/chatbot/ask"
SLEEP_BETWEEN_CALLS = 1.0

# ─── The configurations being ablated ───────────────────────────────────────
CONFIGS = [
    {"label": "cosine_only",  "scoring_mode": "cosine_only",  "alpha": None},
    {"label": "density_only", "scoring_mode": "density_only", "alpha": None},
    {"label": "alpha_0.3",    "scoring_mode": "hybrid",       "alpha": 0.3},
    {"label": "alpha_0.5",    "scoring_mode": "hybrid",       "alpha": 0.5},
    {"label": "alpha_0.7",    "scoring_mode": "hybrid",       "alpha": 0.7},  # your paper's value
    {"label": "alpha_0.9",    "scoring_mode": "hybrid",       "alpha": 0.9},
]


def ask_chatbot(query: str, config: Dict) -> Dict:
    payload = {
        "message": query,
        "history": [],
        "strategy": "moderate",           # top-2, token-budgeted — where scoring matters most
        "alpha": config["alpha"],
        "scoring_mode": config["scoring_mode"],
    }
    try:
        resp = requests.post(ASK_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"reply": "", "sources": [], "acw_metrics": {}, "error": str(e)}


def main():
    all_results: List[Dict] = []
    total = len(CONFIGS) * len(TEST_QUERIES)
    done = 0

    for config in CONFIGS:
        print(f"\n--- Config: {config['label']} ---")
        for query in TEST_QUERIES:
            done += 1
            chat_resp = ask_chatbot(query, config)
            answer = chat_resp.get("reply", "")
            metrics = chat_resp.get("acw_metrics", {})

            if not answer:
                print(f"  [{done:03d}/{total}] SKIP (no answer) — {query[:40]}")
                continue

            scores = judge_answer(query, answer)
            record = {
                "config": config["label"],
                "query": query,
                "token_reduction_pct": metrics.get("token_reduction_pct"),
                "chunks_selected": metrics.get("chunks_selected"),
                **scores,
            }
            all_results.append(record)
            print(
                f"  [{done:03d}/{total}] reduction={metrics.get('token_reduction_pct')}%  "
                f"correctness={scores.get('correctness')}  "
                f"faithfulness={scores.get('faithfulness')}  | {query[:35]}"
            )
            time.sleep(SLEEP_BETWEEN_CALLS)

    with open("ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary table: this answers "why alpha=0.7" ─────────────────────
    print("\n" + "=" * 100)
    print("  ACW Scoring Function — Ablation Study")
    print("=" * 100)
    header = (
        f"{'Config':<14}{'Avg Reduction %':>17}{'Correctness':>14}"
        f"{'Relevance':>12}{'Completeness':>14}{'Faithfulness':>14}"
    )
    print(header)
    print("-" * 100)
    for config in CONFIGS:
        label = config["label"]
        rows = [r for r in all_results if r["config"] == label and r.get("correctness") is not None]
        if not rows:
            continue
        n = len(rows)
        avg = lambda k: sum(r[k] for r in rows) / n
        print(
            f"{label:<14}{avg('token_reduction_pct'):>16.1f}%{avg('correctness'):>14.2f}"
            f"{avg('relevance'):>12.2f}{avg('completeness'):>14.2f}{avg('faithfulness'):>14.2f}"
        )
    print("=" * 100)
    print(
        "Interpretation guide:\n"
        "  - cosine_only  → shows the ceiling on quality when relevance is the only signal\n"
        "                   (but keeps redundant/verbose chunks -> less reduction)\n"
        "  - density_only → shows what happens when semantic relevance is ignored\n"
        "                   (expect correctness/relevance to drop the most here)\n"
        "  - alpha sweep  → shows where quality starts degrading as density is weighted\n"
        "                   more heavily; the paper's alpha=0.7 should sit at or near\n"
        "                   the best reduction-vs-quality trade-off, not necessarily the\n"
        "                   single highest quality score (that's expected to be cosine_only\n"
        "                   or a high alpha, at the cost of reduction).\n"
    )
    print("Saved: ablation_results.json\n")


if __name__ == "__main__":
    main()
