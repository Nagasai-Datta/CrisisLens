# =============================================================================
# FILE: backend/api/routes/pipeline.py
#
# PURPOSE:
#   Defines all REST API endpoints.
#   Maintains the in-memory cluster store that React polls.
#   Wires the four pipeline components into one complete flow.
#
# ENDPOINTS:
#   POST /api/process     → runs the full pipeline on submitted tweets
#   GET  /api/clusters    → returns current active cluster list (React polls this)
#   GET  /api/report      → returns latest BART situation report
#   GET  /api/health      → returns per-model status (green/red for UI sidebar)
#
# IN-MEMORY STATE:
#   active_clusters : list of cluster dicts — the current War Room state
#   latest_report   : string — the most recent BART-generated summary
#   model_status    : dict — which models loaded successfully at startup
#
# WHY POLLING INSTEAD OF WEBSOCKETS:
#   React calls GET /api/clusters every 3 seconds.
#   FastAPI returns the full current cluster list.
#   React compares to previous state — if different, re-renders.
#   This is simple, reliable, and works on all deployment platforms
#   including HuggingFace Spaces which does not support WebSockets on free tier.
#
# THE FULL PIPELINE (called by POST /api/process):
#   raw tweets
#       → bouncer.predict()        : filter noise
#       → deduplicator.cluster()   : group into incidents
#       → detective.extract_all()  : add location strings
#       → geocoder.geocode_all()   : add lat/lng coordinates
#       → stored in active_clusters
#       → returned as JSON response
# =============================================================================

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from datetime import datetime

from backend.components import bouncer, deduplicator, detective, geocoder


# =============================================================================
# SECTION 1 — SHARED STATE
#
# These module-level variables are the "database" of CrisisLens.
# Because FastAPI runs as a single process (for our use case), these
# Python objects persist in memory for the entire server lifetime.
#
# active_clusters: the current list of cluster dicts — one per unique incident.
#                  Updated every time /api/process runs.
#                  React polls /api/clusters to get this list.
#
# latest_report:   the most recent BART-generated situation summary.
#                  Updated every 5 minutes by editor.py (Phase 7).
#                  React polls /api/report to get this string.
#
# model_status:    which components loaded successfully at startup.
#                  Set by main.py's lifespan function.
#                  React reads /api/health to show green/red per model.
#                  Defined here (not in main.py) to avoid circular imports.
# =============================================================================

active_clusters: list = []
latest_report:   str  = ""
model_status:    dict = {
    "bouncer"     : "not_loaded",
    "deduplicator": "not_loaded",
    "detective"   : "not_loaded",
    "geocoder"    : "not_loaded",
}


# =============================================================================
# SECTION 2 — REQUEST MODELS
#
# Pydantic models define the expected shape of incoming request bodies.
# FastAPI automatically validates the request body against these models.
# If the request doesn't match, FastAPI returns a 422 error with details.
# =============================================================================

class ProcessRequest(BaseModel):
    """
    Request body for POST /api/process.

    tweets: list of raw tweet strings to run through the pipeline.

    Example request body:
    {
        "tweets": [
            "Family trapped near Velachery bridge, water rising fast",
            "Help needed at Koyambedu bus stand",
            "Heavy flooding on OMR near Sholinganallur"
        ]
    }
    """
    tweets: List[str]


class ReportUpdateRequest(BaseModel):
    """
    Request body for POST /api/report (used by editor.py to update the report).

    report: the BART-generated situation summary string.
    """
    report: str


# =============================================================================
# SECTION 3 — ROUTER
# =============================================================================

router = APIRouter()


# =============================================================================
# SECTION 4 — ENDPOINTS
# =============================================================================

@router.post("/process")
async def process_tweets(request: ProcessRequest):
    """
    POST /api/process

    The main pipeline endpoint. Takes a list of raw tweets, runs them
    through the full 4-stage pipeline, and updates the cluster store.

    This is called when:
    - A user submits tweets via the React UI input
    - A test script sends demo tweets during development
    - The demo automation runs the synthetic Chennai tweet dataset

    PIPELINE FLOW:
        request.tweets (raw strings)
            ↓
        bouncer.predict()
            → removes non-disaster tweets
            → returns filtered list (may be shorter)
            ↓
        deduplicator.cluster()
            → groups similar tweets into incident clusters
            → returns list of cluster dicts
            → manages the 30-minute rolling window internally
            ↓
        detective.extract_all()
            → adds "locations" field to each cluster dict
            → runs BERT NER on representative tweet
            → async — runs in thread pool to avoid blocking event loop
            ↓
        geocoder.geocode_all()
            → adds "lat", "lng", "resolved_location" to each cluster dict
            → pure local dictionary lookup — no network calls
            ↓
        active_clusters updated
            ↓
        response returned to React
    """
    global active_clusters

    tweets = request.tweets

    if not tweets:
        return {"message": "No tweets provided", "clusters": []}

    # ── Stage 1: Bouncer ─────────────────────────────────────────────
    # Filter noise — only disaster tweets proceed
    filtered_tweets = bouncer.predict(tweets)

    if not filtered_tweets:
        # All tweets were noise — update clusters from deduplicator's
        # rolling window (old tweets may still be active)
        # Still run deduplicator with empty input to trigger window maintenance
        clusters = deduplicator.cluster([], timestamps=[])
        active_clusters = clusters
        return {
            "message"     : f"All {len(tweets)} tweets filtered as noise",
            "clusters"    : clusters,
            "filter_stats": {
                "submitted": len(tweets),
                "passed"   : 0,
                "filtered" : len(tweets)
            }
        }

    print(f"\n Processing batch: {len(tweets)} submitted → {len(filtered_tweets)} passed Bouncer")

    # ── Stage 2: Deduplicator ─────────────────────────────────────────
    # Group similar tweets into incident clusters
    # timestamps=None → deduplicator uses current time for all (fine for batch input)
    clusters = deduplicator.cluster(filtered_tweets, timestamps=None) # type: ignore
    print(f"   Clusters formed: {len(clusters)}")

    # ── Stage 3: Detective ────────────────────────────────────────────
    # Extract location strings from representative tweets
    # async call — BERT runs in thread pool (Bug 8 fix in detective.py)
    clusters = await detective.extract_all(clusters)

    # ── Stage 4: Geocoder ─────────────────────────────────────────────
    # Map location strings to lat/lng coordinates
    clusters = geocoder.geocode_all(clusters)

    geocoded_count = sum(1 for c in clusters if c.get("lat") is not None)
    print(f"   Geocoded: {geocoded_count}/{len(clusters)} clusters have coordinates")

    # ── Update global cluster store ───────────────────────────────────
    # React polls /api/clusters every 3 seconds and will pick this up
    active_clusters = clusters

    return {
        "message"     : "Pipeline complete",
        "clusters"    : clusters,
        "filter_stats": {
            "submitted": len(tweets),
            "passed"   : len(filtered_tweets),
            "filtered" : len(tweets) - len(filtered_tweets)
        }
    }


@router.get("/clusters")
async def get_clusters():
    """
    GET /api/clusters

    Returns the current list of active incident clusters.
    React polls this endpoint every 3 seconds.

    Each cluster dict contains:
    - cluster_id            : unique identifier
    - representative_tweet  : the most informative tweet
    - tweet_count           : how many tweets are in this cluster
    - source_tweets         : all tweets (for expand toggle in UI)
    - first_seen            : ISO timestamp
    - last_seen             : ISO timestamp
    - urgency_score         : float 0.0 to 1.0
    - urgency_label         : "CRITICAL", "HIGH", or "MODERATE"
    - locations             : list of location strings from BERT NER
    - lat                   : float or null (null = no dot on map)
    - lng                   : float or null
    - resolved_location     : full location name string or null
    """
    return active_clusters


@router.post("/report")
async def update_report(request: ReportUpdateRequest):
    """
    POST /api/report

    Used by editor.py (BART component) to update the situation report.
    editor.py runs on a 5-minute timer and calls this endpoint
    whenever a new BART summary is generated.

    Not called directly by React — React uses GET /api/report.
    """
    global latest_report
    latest_report = request.report
    return {"message": "Report updated"}


@router.get("/report")
async def get_report():
    """
    GET /api/report

    Returns the latest BART-generated situation report.
    React polls this endpoint every 5 minutes to refresh Zone C.

    Returns empty string if BART hasn't run yet
    (editor.py is the last component built — Phase 7).
    """
    return {
        "report"    : latest_report,
        "generated" : datetime.utcnow().isoformat() if latest_report else None
    }


@router.get("/health")
async def health():
    """
    GET /api/health

    Returns per-model load status.
    React's control sidebar reads this to show green/red per model.

    Response example:
    {
        "bouncer":      "ok",
        "deduplicator": "ok",
        "detective":    "ok",
        "geocoder":     "ok",
        "overall":      "healthy"
    }

    If a model failed to load:
    {
        "bouncer":      "error: FileNotFoundError ...",
        "deduplicator": "ok",
        ...
        "overall":      "degraded"
    }
    """
    all_ok  = all(v == "ok" for v in model_status.values())
    overall = "healthy" if all_ok else "degraded"

    return {
        **model_status,     # spread all individual model statuses
        "overall"        : overall,
        "active_clusters": len(active_clusters),
        "window_size"    : deduplicator.get_window_size(),
        "gazetteer_size" : geocoder.get_coverage(),
    }
@router.delete("/clusters/{cluster_id}")
async def resolve_cluster(cluster_id: str):
    """
    DELETE /api/clusters/{cluster_id}

    Marks an incident cluster as resolved and removes it from the active store.
    Called when a commander clicks "Mark Resolved" on a cluster card.
    The cluster is removed immediately — it will not reappear unless
    new tweets about the same incident arrive and form a new cluster.
    """
    global active_clusters

    before = len(active_clusters)
    active_clusters = [
        c for c in active_clusters
        if c["cluster_id"] != cluster_id
    ]
    after = len(active_clusters)

    if before == after:
        return {"message": f"Cluster {cluster_id} not found", "removed": False}

    print(f"   ✅ Cluster {cluster_id} marked as resolved and removed")
    return {"message": f"Cluster {cluster_id} resolved", "removed": True}


@router.delete("/clusters")
async def clear_all_clusters():
    """
    DELETE /api/clusters

    Clears ALL active clusters — resets the War Room to empty state.
    Useful for starting a new monitoring session or resetting after a demo.
    """
    global active_clusters
    count = len(active_clusters)
    active_clusters = []
    print(f"   🗑️  All {count} clusters cleared")
    return {"message": f"Cleared {count} clusters", "removed": count}


def set_report(report_text: str):
    """Called by editor.py background task to update the situation report."""
    global latest_report
    latest_report = report_text


def get_active_clusters() -> list:
    """
    Returns the current active cluster list.
    Called by editor.py background task every 5 minutes.
    Defined here (not as a lambda in main.py) so it always
    reads the live module-level variable, not a captured snapshot.
    """
    return active_clusters