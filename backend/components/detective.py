# =============================================================================
# FILE: backend/components/detective.py
#
# PURPOSE:
#   Named Entity Recognition (NER) on cluster representative tweets.
#   Extracts location strings so the Geocoder can map them to coordinates.
#
# WHY BERT FOR THIS:
#   Context-aware entity recognition. BERT reads the full sentence
#   bidirectionally and correctly identifies ambiguous entities:
#   "rushing to Apollo hospital" → Apollo = LOCATION (correct)
#   "Apple announced new iPhone" → Apple  = ORGANISATION (correct)
#   Regex or keyword lists cannot distinguish these — BERT can.
#
# WHY RUN ON REPRESENTATIVE TWEET ONLY (not all tweets):
#   The Deduplicator already collapsed 400 near-duplicate tweets into one
#   high-quality representative. Running BERT on all 400 would be wasteful.
#   The representative is specifically chosen to be the most detail-rich.
#
# BUGS FIXED IN THIS FILE:
#   Bug 7 — aggregation_strategy="simple" reconstructs subword tokens
#            Without this: "Koyambedu" → ["Ko", "##yam", "##bed", "##u"]
#            "Ko" is useless to the Geocoder.
#            With this:    "Koyambedu" → "Koyambedu" ✓
#
#   Bug 8 — BERT inference runs in ThreadPoolExecutor
#            Without this: BERT blocks FastAPI's async event loop for 200-500ms
#            No WebSocket updates or REST calls can process during that time.
#            With this:    BERT runs in a background thread, event loop stays free.
#
# INPUT:
#   list of cluster dicts (from deduplicator.cluster())
#   each dict has: representative_tweet, cluster_id, ... etc.
#
# OUTPUT:
#   same list of cluster dicts, each now enriched with:
#   "locations": ["Velachery bridge", "Tambaram"] ← list of location strings
#
# FLOW:
#   cluster dict arrives
#       → extract representative_tweet
#       → run BERT NER in thread pool (Bug 8 fix)
#       → aggregation_strategy joins subwords (Bug 7 fix)
#       → filter only LOC and GPE entity types
#       → deduplicate location strings
#       → add "locations" field to cluster dict
#       → pass enriched dict to Geocoder
# =============================================================================

import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# HuggingFace model for Named Entity Recognition
# dslim/bert-base-NER: fine-tuned BERT on CoNLL-2003 NER dataset
# Recognises: PER (person), ORG (organisation), LOC (location), MISC (misc)
# We only use LOC and GPE tags for our purposes
NER_MODEL = "dslim/bert-base-NER"

# Entity types we care about:
# LOC = physical location (river, bridge, neighbourhood)
# GPE = geo-political entity (city, country, region)
# Both are relevant for geocoding during a disaster
LOCATION_ENTITY_TYPES = {"LOC", "GPE", "B-LOC", "I-LOC", "B-GPE", "I-GPE"}

# Thread pool for running BERT in background threads (Bug 8 fix)
# max_workers=2: allows up to 2 concurrent BERT inference calls
# More workers = more RAM used; 2 is appropriate for a dev machine
_executor = ThreadPoolExecutor(max_workers=2)


# =============================================================================
# SECTION 2 — MODULE-LEVEL MODEL STORAGE
# =============================================================================

_ner_pipeline = None   # HuggingFace NER pipeline
_loaded       = False  # flag to prevent reloading


# =============================================================================
# SECTION 3 — MODEL LOADING
# =============================================================================

def load_model():
    """
    Loads the BERT NER pipeline into memory.
    Called once at FastAPI startup.

    Key parameters explained:

    aggregation_strategy="simple" (Bug 7 fix):
        BERT tokenises words into subword pieces.
        "Koyambedu" → ["Ko", "##yam", "##bed", "##u"]
        Without aggregation, NER returns each subword separately.
        "simple" strategy: merges consecutive subwords that share an entity
        tag into one complete word. "Koyambedu" comes out as one piece.
        This is critical for Chennai place names which are always multi-syllable.

    device selection:
        "mps" on Apple Silicon = runs on Metal GPU (~3-5x faster than CPU)
        "cpu" = fallback
        We use device index 0 for MPS (HuggingFace pipeline format)
    """
    global _ner_pipeline, _loaded

    if _loaded:
        return True

    print("   Loading BERT NER model...")

    # Detect best available device
    # HuggingFace pipeline uses device=0 for MPS (not "mps" string)
    try:
        import torch
        if torch.backends.mps.is_available():
            device = 0   # HuggingFace uses integer 0 for first GPU/MPS device
            print("   Using Apple Silicon MPS GPU ✅")
        else:
            device = -1  # -1 = CPU in HuggingFace convention
            print("   MPS not available — using CPU")
    except Exception:
        device = -1

    # aggregation_strategy="simple" is the Bug 7 fix
    # It must be set HERE at pipeline creation, not later
    _ner_pipeline = pipeline(
        task                 = "ner",
        model                = NER_MODEL,
        aggregation_strategy = "simple",  # Bug 7 fix — reconstruct subwords
        device               = device
    )

    _loaded = True
    print("   ✅ Detective (BERT NER) ready")

    return True


# =============================================================================
# SECTION 4 — SYNCHRONOUS NER INFERENCE
#
# This is the actual BERT call. It's a plain synchronous function because
# BERT's transformers library is not async-compatible.
#
# We NEVER call this directly from an async FastAPI route.
# Instead we use _extract_locations_async() which runs this in a thread.
# =============================================================================

def _run_ner(text: str) -> list:
    """
    Runs BERT NER on a single text string.
    Returns list of entity dicts from HuggingFace pipeline.

    This is synchronous (blocking). Do not call directly from async code.
    Use extract_locations_async() instead.

    Args:
        text: the representative tweet string

    Returns:
        list of dicts like:
        [{"entity_group": "LOC", "word": "Velachery bridge", "score": 0.98},
         {"entity_group": "GPE", "word": "Chennai", "score": 0.95}]
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return []

    # Truncate very long texts to BERT's 512 token limit
    # Tweets are rarely this long but worth being safe
    text = text[:512]

    try:
        return _ner_pipeline(text) # type: ignore
    except Exception as e:
        print(f"     BERT NER inference error: {e}")
        return []


# =============================================================================
# SECTION 5 — ASYNC WRAPPER (Bug 8 fix)
#
# FastAPI's route handlers are async functions.
# If you call a slow synchronous function (like BERT) directly inside async,
# Python's event loop is blocked — nothing else can run for 200-500ms.
# That means no WebSocket pushes, no other API calls, nothing.
#
# The fix: run_in_executor() offloads the synchronous BERT call to a
# background thread from our ThreadPoolExecutor.
# The event loop can process other tasks while BERT runs in the thread.
# When BERT finishes, the result is returned via await.
#
# Think of it like this:
#   Without fix: "Hold on everyone, I need to call BERT. Nobody move."
#   With fix:    "Thread, go call BERT. I'll keep working. Tell me when done."
# =============================================================================

async def _extract_locations_async(text: str) -> list:
    """
    Async wrapper around _run_ner().
    Runs BERT in background thread so the event loop stays free.

    Args:
        text: representative tweet string

    Returns:
        list of location strings extracted from the text
    """
    loop = asyncio.get_event_loop()

    # run_in_executor: runs _run_ner(text) in _executor thread pool
    # The await yields control back to the event loop until BERT finishes
    raw_entities = await loop.run_in_executor(_executor, _run_ner, text)

    # Filter to only location-type entities
    location_strings = []
    for entity in raw_entities:
        # entity_group is the aggregated label (e.g. "LOC", "GPE", "PER")
        # After aggregation_strategy="simple", multi-token entities are merged
        entity_type = entity.get("entity_group", "").upper()
        entity_word = entity.get("word", "").strip()

        if entity_type in LOCATION_ENTITY_TYPES and entity_word:
            # Clean up any residual subword artifacts (## prefixes)
            # aggregation_strategy="simple" handles most cases but
            # we clean defensively
            clean = entity_word.replace("##", "").strip()
            if len(clean) > 1:  # skip single characters like "(" or ")"
                location_strings.append(clean)

    # Deduplicate while preserving order
    # (same location can appear multiple times in one tweet)
    seen = set()
    unique_locations = []
    for loc in location_strings:
        loc_lower = loc.lower()
        if loc_lower not in seen:
            seen.add(loc_lower)
            unique_locations.append(loc)

    return unique_locations


# =============================================================================
# SECTION 6 — PUBLIC API
#
# These are the functions that FastAPI's pipeline route calls.
# =============================================================================

async def extract_locations(cluster: dict) -> dict:
    """
    Enriches a single cluster dict with extracted location strings.

    Takes a cluster dict, reads its representative_tweet,
    runs BERT NER on it, adds a "locations" field.

    Args:
        cluster: cluster dict from deduplicator.cluster()

    Returns:
        the same cluster dict with "locations" field added:
        {"locations": ["Velachery bridge", "Tambaram"]}
        or {"locations": []} if no locations found
    """
    # Auto-load model if needed (for notebook testing)
    if not _loaded:
        load_model()

    representative = cluster.get("representative_tweet", "")

    locations = await _extract_locations_async(representative)

    # Add locations field to the cluster dict
    # If BERT found nothing, empty list — Geocoder handles this gracefully
    cluster["locations"] = locations

    if locations:
        print(f"   🔍 Cluster {cluster['cluster_id']}: found {locations}")
    else:
        print(f"   🔍 Cluster {cluster['cluster_id']}: no locations found")

    return cluster


async def extract_all(clusters: list) -> list:
    """
    Processes all clusters concurrently.
    Runs extract_locations on every cluster in the list.

    Using asyncio.gather() runs all NER calls concurrently —
    while one BERT inference is running in a thread, the next
    cluster's call can be queued. Faster than sequential processing.

    Args:
        clusters: list of cluster dicts from deduplicator.cluster()

    Returns:
        list of cluster dicts, each enriched with "locations" field
    """
    if not clusters:
        return []

    # asyncio.gather runs all coroutines concurrently
    enriched = await asyncio.gather(
        *[extract_locations(cluster) for cluster in clusters]
    )

    return list(enriched)


# =============================================================================
# SECTION 7 — STATUS
# =============================================================================

def is_loaded() -> bool:
    """Returns True if BERT NER model is loaded and ready."""
    return _loaded


