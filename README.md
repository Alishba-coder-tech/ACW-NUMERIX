Adaptive Context Wrapper (ACW)

Overview

This project implements an Adaptive Context Wrapper (ACW) for reducing unnecessary context passed to Large Language Models (LLMs) in Retrieval-Augmented Generation (RAG) systems.

ACW dynamically controls the amount of retrieved and conversational context provided to the model, aiming to reduce token usage and latency while maintaining answer quality.

An extended, risk-aware variant of ACW is also included, which adjusts compression aggressiveness per-query based on estimated risk, validates that compression didn't drop critical information before generation, and can retry with additional context if the generated answer's confidence is low.

Strategies

The core scoring function ranks retrieved chunks using a weighted combination of cosine similarity and information density:

combined_score = α · cosine_similarity + (1 − α) · information_density

The system evaluates three context-selection strategies built on this score:





Baseline: Full available context is used (no filtering).



Moderate: Top-2 chunks selected within a token budget.



Aggressive: Top-1 chunk only, for maximum token savings.



Risk-Aware Compression Pipeline

Implemented and available now via the use_risk_aware: true flag on /ask (default false, so plain requests keep using the fixed baseline/moderate/aggressive strategies above). When enabled, it adds three features implementing a Compress → Generate → Evaluate → Recover loop:

1. Risk-Aware Compression Selection — modules/risk_aware_compression.py

Scores each incoming query (0–100) based on query complexity, domain risk, predicted answer type, and presence of critical keywords (constraints, precision language, numerical/temporal terms). The score maps to one of five compression levels (NONE → EXTREME), which is then mapped onto the three ACW strategies. Higher-risk queries (e.g. ones asking for exact tolerances or step-by-step procedures) automatically get less aggressive compression than simple factual queries.

2. Semantic Safety Guard — modules/semantic_safety_guard.py

Before any LLM call, checks whether the chunks ACW selected still preserve enough critical information (numbers, dates, measurements, constraints/negations) relative to the full retrieved set. If the preservation rate falls below a configurable threshold, compression is rejected and the pipeline automatically escalates to fuller context — catching information loss before it reaches the model, not after.

3. Context Recovery — modules/context_recovery.py

Scores each generated answer's confidence (relevance, completeness, coherence, citation use, hedging language). If confidence is low, the pipeline retries generation with progressively less compression, up to a configurable retry budget (max_recovery_attempts), falling back toward full context if needed.

Pipeline flow





Risk scoring — estimates query risk and selects an initial compression level.



Semantic safety guard — validates the selected chunks before generation; escalates if unsafe.



Generation — the answer is produced using the (possibly escalated) selected context.



Confidence evaluation — the generated answer is scored on relevance, completeness, coherence, and confidence signals.



Recovery — if confidence is low, the pipeline retries with progressively less compression, up to the retry budget.

Risk-aware mode is fully implemented and tested; it is opt-in per-request (not enabled by default) so that the core evaluation scripts above continue to exercise the fixed-strategy ACW path unaffected.

Evaluation

Four experiments are included, each independently runnable:







Script



Measures



Output





run_experiment.py



Tokens before/after, token reduction %, latency, chunks retrieved/selected, response length



acw_experiment_results.json





answer_quality.py



Answer correctness, relevance, completeness, and faithfulness per strategy, via LLM-as-judge against the full knowledge base



answer_quality_results.json, answer_quality_for_human_review.csv





run_ablation.py



Token reduction and answer quality across scoring configurations: cosine-only, density-only, and α ∈ {0.3, 0.5, 0.7, 0.9}



ablation_results.json





cost_analysis.py



Measured token reduction converted into an estimated per-query and monthly API cost saving (pricing constant must be set/verified before citing)



printed summary

Each strategy/configuration is evaluated using the same query set and retrieval configuration to ensure a fair comparison.

Running

# start the backend first
uvicorn main:app --reload

# then, from backend/:
python run_experiment.py
python answer_quality.py
python run_ablation.py
python cost_analysis.py



Technology





Python, FastAPI



Google Gemini (embeddings + generation, and as LLM-judge for answer quality)



Pinecone (vector retrieval)



React (frontend)



Retrieval-Augmented Generation (RAG)



Research Objective

The primary objective is to determine whether adaptive context selection can reduce LLM context/token consumption without significantly affecting response quality, and to characterize the token-reduction/answer-quality trade-off across scoring configurations (via the ablation study) and across query risk levels (via the risk-aware pipeline).

Project

ACW is implemented as part of the NumeriX AI-powered numerical analysis chatbot and is intended for academic research and experimental evaluation.

File Structure

numerix/backend/
├── main.py
├── requirements.txt
├── run_experiment.py
├── answer_quality.py
├── run_ablation.py
├── cost_analysis.py
├── acw_experiment_results.json
├── acw_experiment_results_v2.json
├── answer_quality_results.json
├── answer_quality_for_human_review.csv
├── modules/
│   ├── __init__.py
│   ├── risk_aware_compression.py
│   ├── semantic_safety_guard.py
│   └── context_recovery.py
└── routers/
    ├── __init__.py
    ├── acw.py
    ├── acw_log.jsonl
    ├── chatbot.py
    ├── differentiation.py
    ├── errors.py
    ├── integration.py
    ├── interpolation.py
    ├── linear.py
    ├── ode.py
    └── rootfinder.py

