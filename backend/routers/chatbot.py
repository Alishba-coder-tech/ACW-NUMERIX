import sys, os 
sys.path.insert(0, r'C:\Users\wajiz.pk\Downloads\numerix_project_with_chatbot\numerix\backend') 
"""
NumeriX AI Assistant — with Adaptive Context Wrapper (ACW)
-----------------------------------------------------------
Changes from original chatbot.py:
  1. Imported acw.py (fixed path)
  2. /ask endpoint now accepts a `strategy` field ("baseline" | "moderate" | "aggressive")
  3. ACW filters chunks before building context
  4. Response includes `acw_metrics` for research data visibility
  5. Everything else unchanged
"""

import os
import sys
from typing import List, Optional

# ── FIX: add backend/ folder to path so acw.py can be found ─────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import google.generativeai as genai
from fastapi import APIRouter, HTTPException
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel
from routers.acw import adaptive_context_wrapper

router = APIRouter()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")

EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768
CHAT_MODEL = "gemini-3.5-flash-lite"
model = genai.GenerativeModel(CHAT_MODEL)
INDEX_NAME = "acw-index"

_pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
_index = None


def _get_index():
    global _index
    if _index is not None:
        return _index
    if _pc is None:
        raise HTTPException(
            status_code=500,
            detail="PINECONE_API_KEY is not configured on the server.",
        )
    existing = [i.name for i in _pc.list_indexes()]
    if INDEX_NAME not in existing:
        _pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    _index = _pc.Index(INDEX_NAME)
    return _index


KNOWLEDGE_BASE = [
    {
        "id": "errors",
        "title": "Error Analyzer (Lab 2)",
        "content": (
            "The Error Analyzer computes Absolute Error, Relative Error, "
            "Round-off Error, and Truncation Error. Absolute error is the "
            "magnitude of the difference between the true value and the "
            "approximate value. Relative error divides that by the true "
            "value, often expressed as a percentage. Round-off error comes "
            "from representing numbers with finite precision. Truncation "
            "error comes from approximating an infinite process (like a "
            "Taylor series) with a finite number of terms. Endpoint: "
            "/api/errors."
        ),
    },
    {
        "id": "rootfinder",
        "title": "Root Finder (Labs 4-5)",
        "content": (
            "The Root Finder module solves f(x) = 0 using Bisection, False "
            "Position (Regula Falsi), Newton-Raphson, and Fixed-Point "
            "iteration. Bisection and False Position need a bracketing "
            "interval [a, b] where f(a) and f(b) have opposite signs. "
            "Newton-Raphson needs f(x), its derivative f'(x), and an "
            "initial guess x0, and converges quadratically near the root. "
            "Fixed-Point iteration rewrites f(x)=0 as x = g(x) and iterates "
            "x_new = g(x). Each method returns an iteration table with the "
            "approximate error at each step. Endpoint: /api/roots."
        ),
    },
    {
        "id": "interpolation",
        "title": "Interpolator (Labs 6-7)",
        "content": (
            "The Interpolator module estimates values between known data "
            "points using Newton's Forward Difference, Newton's Backward "
            "Difference, Newton's Divided Difference, and Lagrange "
            "Interpolation. Forward/backward difference formulas assume "
            "equally spaced x-values, while divided difference and "
            "Lagrange work for unequally spaced points. Endpoint: "
            "/api/interpolation."
        ),
    },
    {
        "id": "differentiation",
        "title": "Differentiator (Lab 8)",
        "content": (
            "The Differentiator module estimates derivatives numerically "
            "using Forward and Backward Finite Difference formulas, for "
            "both first and second order derivatives, given a step size h "
            "and a set of function values. Endpoint: /api/differentiation."
        ),
    },
    {
        "id": "integration",
        "title": "Integrator (Labs 9-10)",
        "content": (
            "The Integrator module approximates definite integrals using "
            "the Trapezoidal Rule, Simpson's 1/3 Rule, Simpson's 3/8 Rule, "
            "and rules for unequally spaced segments. Simpson's 1/3 "
            "requires an even number of intervals; Simpson's 3/8 works in "
            "groups of three intervals. Endpoint: /api/integration."
        ),
    },
    {
        "id": "ode",
        "title": "ODE Solver (Labs 11-12)",
        "content": (
            "The ODE Solver module solves first-order ordinary "
            "differential equations dy/dx = f(x, y) using Euler's Method, "
            "Improved Euler (Heun's Method), and 4th Order Runge-Kutta "
            "(RK4). RK4 is generally the most accurate for a given step "
            "size h because it evaluates the slope at four points per "
            "step. Endpoint: /api/ode."
        ),
    },
    {
        "id": "linear",
        "title": "Linear Systems (Lab 13)",
        "content": (
            "The Linear Systems module solves systems of linear equations "
            "Ax = b using LU Decomposition, with both the Doolittle method "
            "(unit lower-triangular L) and the Crout method (unit "
            "upper-triangular U). The system is factored into L and U, "
            "then solved via forward substitution (Ly = b) followed by "
            "back substitution (Ux = y). Endpoint: /api/linear."
        ),
    },
    {
        "id": "about",
        "title": "About NumeriX",
        "content": (
            "NumeriX is a full-stack interactive numerical methods "
            "calculator built for the BSE-6C Numerical Analysis course at "
            "Bahria University Karachi Campus. It has a React + Tailwind "
            "frontend and a Python FastAPI backend, covering 13 labs "
            "across error analysis, root finding, interpolation, "
            "differentiation, integration, ODE solving, and linear "
            "systems, each with iteration tables and live charts."
        ),
    },
]

_kb_seeded = False


def _ensure_kb_seeded():
    global _kb_seeded
    if _kb_seeded:
        return
    index = _get_index()
    stats = index.describe_index_stats()
    if stats.get("total_vector_count", 0) >= len(KNOWLEDGE_BASE):
        _kb_seeded = True
        return
    vectors = []
    for chunk in KNOWLEDGE_BASE:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=chunk["content"],
            task_type="retrieval_document",
            title=chunk["title"],
            output_dimensionality=EMBED_DIM,
        )
        vectors.append(
            {
                "id": chunk["id"],
                "values": result["embedding"],
                "metadata": {"title": chunk["title"], "content": chunk["content"]},
            }
        )
    index.upsert(vectors=vectors)
    _kb_seeded = True


def _retrieve(query: str, top_k: int = 3) -> List[dict]:
    _ensure_kb_seeded()
    index = _get_index()
    q_embed = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query",
        output_dimensionality=EMBED_DIM,
    )["embedding"]
    results = index.query(vector=q_embed, top_k=top_k, include_metadata=True)
    return [
        {
            "id": match.id,
            "title": match.metadata.get("title", ""),
            "content": match.metadata.get("content", ""),
            "score": match.score,
        }
        for match in results.matches
    ]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatInput(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    strategy: Optional[str] = "moderate"
    # ── Ablation-study controls (optional; defaults reproduce original behavior) ──
    alpha: Optional[float] = None            # override ACW alpha, e.g. 0.3 / 0.5 / 0.9
    scoring_mode: Optional[str] = "hybrid"   # "hybrid" | "cosine_only" | "density_only"


@router.post("/ask")
def ask(data: ChatInput):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not configured.")
    if not PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY is not configured.")

    try:
        # Step 1: Retrieve from Pinecone
        raw_chunks = _retrieve(data.message, top_k=3)

        # Step 2: ACW filters chunks
        strategy = data.strategy or "moderate"
        selected_chunks, acw_metrics = adaptive_context_wrapper(
            query=data.message,
            chunks=raw_chunks,
            strategy=strategy,
            alpha=data.alpha,
            scoring_mode=data.scoring_mode or "hybrid",
        )

        # Step 3: Build context from selected chunks only
        context = "\n\n".join(
            f"[{c['title']}]\n{c['content']}" for c in selected_chunks
        )

        system_prompt = (
            "You are the NumeriX Assistant, a helpful tutor embedded in a "
            "numerical methods web app for a Numerical Analysis course. "
            "Answer the user's question using the reference context below "
            "when relevant. Explain concepts clearly and concisely, use "
            "the method names and endpoints from the context when helpful, "
            "and if a question is unrelated to numerical methods or this "
            "app, politely say so and redirect to what you can help with.\n\n"
            f"Reference context:\n{context}"
        )

        history_payload = []
        for m in data.history or []:
            role = "model" if m.role == "assistant" else "user"
            history_payload.append({"role": role, "parts": [m.content]})

        model = genai.GenerativeModel(
            model_name=CHAT_MODEL,
            system_instruction=system_prompt,
        )
        chat = model.start_chat(history=history_payload)
        response = chat.send_message(data.message)

        return {
            "reply": response.text,
            "sources": [c["title"] for c in selected_chunks],
            "acw_metrics": {
                "strategy": strategy,
                "chunks_retrieved": acw_metrics["chunks_retrieved"],
                "chunks_selected": acw_metrics["chunks_selected"],
                "tokens_before": acw_metrics["tokens_before"],
                "tokens_after": acw_metrics["tokens_after"],
                "token_reduction_pct": acw_metrics["token_reduction_pct"],
                "scoring_mode": acw_metrics.get("scoring_mode"),
                "alpha_used": acw_metrics.get("alpha_used"),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/suggestions")
def suggestions():
    return {
        "suggestions": [
            "What's the difference between Bisection and Newton-Raphson?",
            "When should I use Simpson's 1/3 vs Simpson's 3/8?",
            "Why does RK4 give better accuracy than Euler's method?",
            "What's the difference between Doolittle and Crout?",
        ]
    }
