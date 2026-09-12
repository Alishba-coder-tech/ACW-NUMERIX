
# Adaptive Context Wrapper (ACW)

## Overview

This project implements an **Adaptive Context Wrapper (ACW)** for reducing unnecessary context passed to Large Language Models (LLMs) in Retrieval-Augmented Generation (RAG) systems.

ACW dynamically controls the amount of retrieved and conversational context provided to the model, aiming to reduce **token usage and latency while maintaining answer quality**.

## Strategies

The system evaluates three strategies:

* **Baseline:** Full available context is used.
* **Moderate:** Context is reduced while preserving relevant information.
* **Aggressive:** Stronger context reduction is applied to maximize token savings.

## Evaluation Metrics

The experiments measure:

* Tokens before and after context reduction
* Token reduction percentage
* Response latency
* Retrieved and selected chunks
* Response length
* Answer quality

## Experimental Setup

Each strategy is evaluated using the **same query set and retrieval configuration** to ensure a fair comparison.

Results are stored in:

```text
acw_experiment_results.json
```

Run the experiment with:

```bash
python run_experiment.py
```

## Technology

* Python
* FastAPI
* Google Gemini
* Pinecone
* React
* Retrieval-Augmented Generation (RAG)

## Research Objective

The primary objective is to determine whether **adaptive context selection can reduce LLM context/token consumption without significantly affecting response quality**.

## Project

ACW is implemented as part of the **NumeriX AI-powered numerical analysis chatbot** and is intended for academic research and experimental evaluation.

This version is much better for a GitHub repository: **short, technical, and directly focused on the research contribution.**
