"""
evaluate_answer_quality.py — Answer Quality / Accuracy Evaluation for NumeriX ACW
===================================================================================
Drop this file into: numerix/backend/  (same folder as run_experiment.py)

WHY THIS EXISTS
----------------
run_experiment.py proves ACW saves tokens. It does NOT prove the saved
tokens don't hurt answer quality. This script closes that gap using an
LLM-as-judge (a well-established approach for RAG evaluation, e.g. RAGAS).

For every query, for every strategy (baseline / moderate / aggressive), it:
  1. Calls your running FastAPI server's /api/chatbot/ask endpoint
  2. Sends the question + the model's answer to a judge LLM (Gemini),
     along with the FULL NumeriX knowledge base as ground truth
  3. Asks the judge to score 1-5 on: correctness, relevance,
     completeness, faithfulness (the exact table from your paper outline)
  4. Averages the scores per strategy and prints/saves a comparison table

This directly answers: "Does saving tokens (moderate/aggressive) hurt
answer quality compared to baseline?"

SETUP
-----
    cd numerix/backend
    pip install requests google-generativeai python-dotenv
    # GOOGLE_API_KEY must already be set (it's used by chatbot.py too)
    # Start your server first:  uvicorn main:app --reload
    python evaluate_answer_quality.py

OPTIONAL — HUMAN EVALUATION
-----------------------------
Even the paper's own note says "a human evaluation ... would substantially
strengthen the paper." This script also writes answer_quality_for_human_review.csv
so you (or classmates) can independently score a sample by hand and report
inter-rater agreement alongside the automated scores — reviewers like seeing
both.

OUTPUT
------
  answer_quality_results.json           — full per-query judge scores
  answer_quality_for_human_review.csv   — same data, spreadsheet-friendly,
                                            with blank columns for a human rater
Printed to console: the Metric x Strategy summary table for your paper.
"""

import csv
import json
import os
import time
from typing import Dict, List

import requests

try:
    import google.generativeai as genai
except ImportError:
    raise SystemExit("Run: pip install google-generativeai")

from run_experiment import TEST_QUERIES  # reuse the same 30 queries you already validated

# ─── Config ───────────────────────────────────────────────────────────────
ASK_URL = "http://localhost:8000/api/chatbot/ask"
STRATEGIES = ["baseline", "moderate", "aggressive"]

# Judge model: use a *different/stronger* model than the chatbot where possible
# to reduce self-grading bias. gemini-2.5-pro is a good default if you have
# quota; fall back to gemini-2.5-flash if you're rate-limited.
JUDGE_MODEL = os.environ.get("ACW_JUDGE_MODEL", "gemini-2.5-pro")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise SystemExit("GOOGLE_API_KEY is not set in your environment/.env")
genai.configure(api_key=GOOGLE_API_KEY)
_judge = genai.GenerativeModel(JUDGE_MODEL)

SLEEP_BETWEEN_CALLS = 1.0  # be polite to both your API and the judge API

# ─── Ground truth: reuse the SAME knowledge base your chatbot retrieves from ──
# Importing straight from chatbot.py guarantees the judge's "ground truth"
# never drifts out of sync with what the chatbot can actually say.
from routers.chatbot import KNOWLEDGE_BASE  # noqa: E402

REFERENCE_TEXT = "\n\n".join(
    f"[{c['title']}]\n{c['content']}" for c in KNOWLEDGE_BASE
)

JUDGE_PROMPT_TEMPLATE = """You are an impartial evaluator grading a tutoring chatbot's answer.

You are given:
1. QUESTION — a student's question
2. REFERENCE MATERIAL — the complete, authoritative knowledge base the chatbot
   is built on (the correct answer must be consistent with this, but the
   chatbot only ever sees a filtered SUBSET of it — that is what we are testing)
3. ANSWER — what the chatbot actually replied

Score the ANSWER from 1 (very poor) to 5 (excellent) on each of these,
independently:

- correctness: Are the facts in the answer accurate and consistent with the reference material?
- relevance: Does the answer actually address what the question asked?
- completeness: Does the answer cover the key points a good answer should include?
- faithfulness: Does the answer avoid inventing details not supported by the reference material (no hallucination)?

Respond with ONLY a JSON object, no markdown fences, no commentary:
{{"correctness": <1-5>, "relevance": <1-5>, "completeness": <1-5>, "faithfulness": <1-5>, "notes": "<one short sentence>"}}

QUESTION:
{question}

REFERENCE MATERIAL:
{reference}

ANSWER:
{answer}
"""


def judge_answer(question: str, answer: str, retries: int = 2) -> Dict:
    """Calls the judge LLM and parses its JSON score. Never raises — returns
    a dict with an 'error' key on failure so a bad call doesn't kill a run."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, reference=REFERENCE_TEXT, answer=answer
    )
    for attempt in range(retries + 1):
        try:
            resp = _judge.generate_content(prompt)
            text = resp.text.strip()
            # Strip accidental ```json fences
            if text.startswith("```"):
                text = text.strip("`")
                text = text.split("\n", 1)[1] if "\n" in text else text
                text = text.rsplit("```", 1)[0]
            data = json.loads(text)
            for key in ("correctness", "relevance", "completeness", "faithfulness"):
                data[key] = float(data.get(key, 0))
            return data
        except Exception as e:
            if attempt == retries:
                return {
                    "correctness": None, "relevance": None,
                    "completeness": None, "faithfulness": None,
                    "error": str(e),
                }
            time.sleep(2)


def ask_chatbot(query: str, strategy: str) -> Dict:
    payload = {"message": query, "history": [], "strategy": strategy}
    try:
        resp = requests.post(ASK_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"reply": "", "sources": [], "acw_metrics": {}, "error": str(e)}


def main():
    all_results: List[Dict] = []
    total = len(TEST_QUERIES) * len(STRATEGIES)
    done = 0

    for strategy in STRATEGIES:
        print(f"\n--- Evaluating strategy: {strategy.upper()} ---")
        for query in TEST_QUERIES:
            done += 1
            chat_resp = ask_chatbot(query, strategy)
            answer = chat_resp.get("reply", "")

            if not answer:
                print(f"  [{done:03d}/{total}] SKIP (no answer) — {query[:40]}")
                continue

            scores = judge_answer(query, answer)
            record = {
                "query": query,
                "strategy": strategy,
                "answer": answer,
                "acw_metrics": chat_resp.get("acw_metrics", {}),
                **scores,
            }
            all_results.append(record)
            print(
                f"  [{done:03d}/{total}] correctness={scores.get('correctness')} "
                f"relevance={scores.get('relevance')} "
                f"completeness={scores.get('completeness')} "
                f"faithfulness={scores.get('faithfulness')}  | {query[:40]}"
            )
            time.sleep(SLEEP_BETWEEN_CALLS)

    # ── Save raw results ─────────────────────────────────────────────────
    with open("answer_quality_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── CSV for optional human review (blank human_* columns to fill in) ──
    with open("answer_quality_for_human_review.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query", "strategy", "answer",
            "llm_correctness", "llm_relevance", "llm_completeness", "llm_faithfulness",
            "human_correctness", "human_relevance", "human_completeness", "human_faithfulness",
        ])
        for r in all_results:
            writer.writerow([
                r["query"], r["strategy"], r["answer"],
                r.get("correctness"), r.get("relevance"),
                r.get("completeness"), r.get("faithfulness"),
                "", "", "", "",
            ])

    # ── Summary table (this is your Section 5 table) ────────────────────
    print("\n" + "=" * 78)
    print("  Answer Quality Summary  (1-5 scale, LLM-as-judge)")
    print("=" * 78)
    header = f"{'Strategy':<12}{'Correctness':>14}{'Relevance':>12}{'Completeness':>14}{'Faithfulness':>14}"
    print(header)
    print("-" * 78)
    for strategy in STRATEGIES:
        rows = [r for r in all_results if r["strategy"] == strategy and r.get("correctness") is not None]
        if not rows:
            continue
        n = len(rows)
        avg = lambda k: sum(r[k] for r in rows) / n
        print(
            f"{strategy:<12}{avg('correctness'):>14.2f}{avg('relevance'):>12.2f}"
            f"{avg('completeness'):>14.2f}{avg('faithfulness'):>14.2f}"
        )
    print("=" * 78)
    print("Saved: answer_quality_results.json, answer_quality_for_human_review.csv\n")


if __name__ == "__main__":
    main()
