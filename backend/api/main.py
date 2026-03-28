# =============================================================================
# FILE: backend/api/main.py
#
# PURPOSE:
#   FastAPI application entry point.
#   Creates the app, configures CORS, loads all ML models at startup,
#   and registers the pipeline routes.
#
# THIS IS THE FILE THAT STARTS THE SERVER.
# Run from the crisislens/ root directory:
#   uvicorn backend.api.main:app --reload --port 8000
#
# STARTUP SEQUENCE:
#   1. FastAPI app created
#   2. CORS middleware configured
#   3. Pipeline routes registered
#   4. Server starts accepting connections
#   5. @lifespan startup fires:
#      - bouncer models loaded        → model_status["bouncer"] = "ok"
#      - deduplicator model loaded    → model_status["deduplicator"] = "ok"
#      - detective model loaded       → model_status["detective"] = "ok"
#      - geocoder gazetteer loaded    → model_status["geocoder"] = "ok"
#      - any failure is caught and logged — server continues with partial models
#
# BUG FIXED IN THIS FILE:
#   Bug 10 — Each model loaded in individual try/except.
#             One failed model does not crash the entire server.
#             /api/health exposes per-model status so the UI shows red/green.
#
# EDITOR FIX:
#   We import pipeline as a module object (not its variables directly).
#   lambda: pipeline_module.active_clusters reads the CURRENT value each time.
#   The old way (importing active_clusters directly) captured the empty list
#   at import time and never saw updates — editor always saw 0 clusters.
#
# CORS EXPLAINED:
#   React runs on localhost:5173 (Vite).
#   FastAPI runs on localhost:8000.
#   Without CORS config, the browser blocks React from calling FastAPI.
#   With CORS config: FastAPI says "I allow requests from :5173" → works.
# =============================================================================

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio

# Import pipeline as a module — NOT individual variables from it.
# This is the fix for the editor always seeing empty clusters.
# pipeline_module.active_clusters reads the live value every time it's called.
# Importing `active_clusters` directly would capture the empty [] at startup
# and never reflect updates made by POST /api/process.
from backend.api.routes import pipeline as pipeline_module
from backend.api.routes.pipeline import router, model_status

# Import all pipeline components
from backend.components import bouncer, deduplicator, detective, geocoder
from backend.components import editor


# =============================================================================
# SECTION 1 — LIFESPAN (STARTUP / SHUTDOWN)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at server startup (before first request).
    Loads all ML models into memory with individual error handling.
    Starts the editor background task.
    """
    print("=" * 50)
    print("CrisisLens — Server Starting Up")
    print("=" * 50)
    print("Loading ML models...")

    # ── Load Bouncer (.pkl files) ─────────────────────────────────────
    try:
        bouncer.load_models()
        model_status["bouncer"] = "ok"
    except Exception as e:
        model_status["bouncer"] = f"error: {str(e)}"
        print(f"   ❌ Bouncer failed: {e}")

    # ── Load Deduplicator (MiniLM) ────────────────────────────────────
    try:
        deduplicator.load_model()
        model_status["deduplicator"] = "ok"
    except Exception as e:
        model_status["deduplicator"] = f"error: {str(e)}"
        print(f"   ❌ Deduplicator failed: {e}")

    # ── Load Detective (BERT NER) ─────────────────────────────────────
    try:
        detective.load_model()
        model_status["detective"] = "ok"
    except Exception as e:
        model_status["detective"] = f"error: {str(e)}"
        print(f"   ❌ Detective failed: {e}")

    # ── Load Geocoder (Chennai Gazetteer JSON) ────────────────────────
    try:
        geocoder.load_gazetteer()
        model_status["geocoder"] = "ok"
    except Exception as e:
        model_status["geocoder"] = f"error: {str(e)}"
        print(f"   ❌ Geocoder failed: {e}")

    # ── Startup Summary ───────────────────────────────────────────────
    ready_count = sum(1 for v in model_status.values() if v == "ok")
    print(f"\n   Models ready: {ready_count}/{len(model_status)}")
    for component, status in model_status.items():
        icon = "✅" if status == "ok" else "❌"
        print(f"   {icon} {component}: {status}")

    print("\n   CrisisLens is running.")
    print(f"   API docs: http://localhost:8000/docs")
    print(f"   Health:   http://localhost:8000/api/health")
    print("=" * 50)

    # ── Start Editor Background Task ──────────────────────────────────
    # pipeline_module.active_clusters → reads the LIVE list every call
    # pipeline_module.set_report      → writes to the LIVE report store
    # Both use the module object so they always see current state,
    # not a snapshot captured at import time.
    asyncio.create_task(
    editor.run_editor_loop(
        get_clusters_fn  = pipeline_module.get_active_clusters,
        update_report_fn = pipeline_module.set_report
    )
)
    print("   ✅ Editor background task started (runs every 5 minutes)")

    yield  # Server runs here — handles all requests until shutdown

    # Shutdown
    print("CrisisLens server shutting down.")


# =============================================================================
# SECTION 2 — APP CREATION
# =============================================================================

app = FastAPI(
    title       = "CrisisLens API",
    description = "Real-time crisis intelligence pipeline for disaster response",
    version     = "1.0.0",
    lifespan    = lifespan
)


# =============================================================================
# SECTION 3 — CORS CONFIGURATION
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",     # React local development
        "http://localhost:5173",     # Vite default port
        "https://*.vercel.app",      # Vercel production deployment
    ],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# =============================================================================
# SECTION 4 — REGISTER ROUTES
# =============================================================================

app.include_router(router, prefix="/api")
