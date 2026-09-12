"""
cost_analysis.py — Cost Reduction Analysis for NumeriX ACW
==============================================================
Drop this file into: numerix/backend/  (same folder as run_experiment.py)

WHAT THIS DOES
---------------
Reads the MEASURED token counts from acw_experiment_results.json (produced
by run_experiment.py — run that first) and:

  1. Computes measured token reduction % per strategy (this part is real data)
  2. Projects hypothetical $ savings at a few deployment volumes, using a
     pricing constant YOU must fill in / verify (this part is clearly
     labeled as an ESTIMATE, not measured data — do not present it as
     experimental results in your paper; present it as a "practical
     implication" discussion, per your outline)

Formula used (as specified in the assignment):
    Cost Saving (%) ≈ Token Reduction (%)
This holds because commercial LLM APIs bill per input token, and ACW only
reduces INPUT tokens (the retrieved context) — output tokens are unaffected,
so this is an approximation, not an exact accounting identity. Say so in the
paper.

BEFORE RUNNING
--------------
Edit PRICE_PER_1K_INPUT_TOKENS below to match current pricing for whatever
model your paper targets (Gemini 2.5 Flash, GPT-4o-mini, etc.) — check the
provider's pricing page at the time you write the paper, since prices change.
Example (illustrative only, verify before citing):
    https://ai.google.dev/gemini-api/docs/pricing
    https://openai.com/api/pricing

USAGE
-----
    python run_experiment.py        # first, to generate acw_experiment_results.json
    python cost_analysis.py
"""

import json

# ─── EDIT ME: pricing is an estimate, not measured — verify before publishing ──
PRICE_PER_1K_INPUT_TOKENS_USD = 0.000075   # <-- placeholder, e.g. Gemini 2.5 Flash input tier
PRICING_SOURCE_NOTE = (
    "Illustrative only — verify current pricing at your provider's pricing "
    "page before citing a dollar figure in the paper."
)

# Hypothetical deployment volumes for the discussion table
QUERY_VOLUMES_PER_MONTH = [1_000, 10_000, 100_000, 1_000_000]

RESULTS_FILE = "acw_experiment_results.json"


def load_measured_data():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def summarize_strategy(records):
    ok = [r for r in records if r.get("status") == "ok"]
    if not ok:
        return None
    n = len(ok)
    avg_before = sum(r["tokens_before"] for r in ok) / n
    avg_after = sum(r["tokens_after"] for r in ok) / n
    avg_reduction_pct = sum(r["token_reduction_pct"] for r in ok) / n
    return {
        "n": n,
        "avg_tokens_before": avg_before,
        "avg_tokens_after": avg_after,
        "avg_reduction_pct": avg_reduction_pct,
    }


def main():
    data = load_measured_data()

    print("\n" + "=" * 78)
    print("  Cost Reduction Analysis — NumeriX ACW")
    print("=" * 78)
    print(f"  Note: {PRICING_SOURCE_NOTE}")
    print(f"  Assumed price: ${PRICE_PER_1K_INPUT_TOKENS_USD:.6f} per 1K input tokens")
    print("=" * 78)

    summaries = {}
    for strategy, records in data.items():
        summary = summarize_strategy(records)
        if summary:
            summaries[strategy] = summary

    baseline = summaries.get("baseline")
    if not baseline:
        print("No baseline data found — run run_experiment.py first.")
        return

    # ── Measured reduction table ─────────────────────────────────────────
    print(f"\n{'Strategy':<12}{'Avg Tok Before':>16}{'Avg Tok After':>16}{'Reduction %':>14}")
    print("-" * 60)
    for strategy, s in summaries.items():
        print(
            f"{strategy:<12}{s['avg_tokens_before']:>16.1f}"
            f"{s['avg_tokens_after']:>16.1f}{s['avg_reduction_pct']:>13.1f}%"
        )

    # ── Hypothetical monthly cost projection ─────────────────────────────
    print("\n" + "-" * 78)
    print("  HYPOTHETICAL monthly cost projection (context/input tokens only)")
    print("  *** Estimate, not experimental data — label clearly in the paper ***")
    print("-" * 78)
    header = f"{'Queries/mo':>12}" + "".join(f"{s:>16}" for s in summaries)
    print(header)
    for volume in QUERY_VOLUMES_PER_MONTH:
        row = f"{volume:>12,}"
        for strategy, s in summaries.items():
            monthly_tokens = s["avg_tokens_after"] * volume
            monthly_cost = (monthly_tokens / 1000) * PRICE_PER_1K_INPUT_TOKENS_USD
            row += f"{'$' + format(monthly_cost, ',.2f'):>16}"
        print(row)

    # ── Savings vs baseline ──────────────────────────────────────────────
    print("\n" + "-" * 78)
    print("  Estimated $ saved per month vs. BASELINE (no filtering)")
    print("-" * 78)
    header = f"{'Queries/mo':>12}" + "".join(
        f"{s:>16}" for s in summaries if s != "baseline"
    )
    print(header)
    baseline_after = baseline["avg_tokens_after"]
    for volume in QUERY_VOLUMES_PER_MONTH:
        row = f"{volume:>12,}"
        for strategy, s in summaries.items():
            if strategy == "baseline":
                continue
            tokens_saved = (baseline_after - s["avg_tokens_after"]) * volume
            dollars_saved = (tokens_saved / 1000) * PRICE_PER_1K_INPUT_TOKENS_USD
            row += f"{'$' + format(dollars_saved, ',.2f'):>16}"
        print(row)

    print("\n" + "=" * 78)
    print("  Cost Saving (%) ≈ Token Reduction (%):")
    for strategy, s in summaries.items():
        if strategy == "baseline":
            continue
        print(f"    {strategy:<12} → ~{s['avg_reduction_pct']:.1f}% estimated cost saving")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
