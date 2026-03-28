# =============================================================================
# FILE: backend/components/geocoder.py
#
# PURPOSE:
#   Converts location strings extracted by detective.py into lat/lng coordinates.
#   Uses the pre-built Chennai gazetteer JSON — zero network calls at runtime.
#
# WHY LOCAL LOOKUP AND NOT RUNTIME API:
#   Nominatim has rate limits, requires active internet, and can go down.
#   For a demo or deployment that must run reliably offline, API dependence
#   is unacceptable. The gazetteer JSON is a permanent static asset built
#   once via populate_gazetteer.py — instantaneous lookup, always available.
#
# BUGS FIXED IN THIS FILE:
#   Bug 9 — Case sensitivity: BERT NER output is unpredictable case.
#            Fix: lowercase both gazetteer keys AND BERT output before lookup.
#            Without this, "Velachery Bridge" fails to match "velachery bridge".
#
# INPUT:
#   list of cluster dicts from detective.py
#   each dict has a "locations" field: ["Velachery bridge", "Tambaram"]
#
# OUTPUT:
#   same cluster dicts enriched with:
#   "lat": 12.9716,
#   "lng": 80.2209,
#   "resolved_location": "Velachery bridge, Chennai"
#
# LOOKUP STRATEGY (two levels):
#   Level 1: Exact match after lowercasing → instant O(1) dict lookup
#   Level 2: Fuzzy match via difflib → handles partial names and typos
#   Level 3: null lat/lng → no dot placed on map (honest behaviour)
#
# WHICH LOCATION IS USED FOR COORDINATES:
#   If a cluster has multiple location strings, we use the FIRST one
#   that successfully resolves to coordinates. This is the primary location.
# =============================================================================

import os
import json
import difflib


# =============================================================================
# SECTION 1 — PATH CONFIGURATION
# =============================================================================

ROOT_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GAZETTEER_PATH = os.path.join(ROOT_DIR, "backend", "data", "gazetteers", "chennai_gazetteer.json")


# =============================================================================
# SECTION 2 — MODULE-LEVEL STATE
# =============================================================================

_gazetteer    = {}     # the full gazetteer dict loaded from JSON
_gazetteer_keys = []   # pre-computed lowercase key list for fuzzy matching
_loaded       = False  # flag to prevent reloading


# =============================================================================
# SECTION 3 — GAZETTEER LOADING
# =============================================================================

def load_gazetteer():
    """
    Loads the Chennai gazetteer JSON into memory.
    Called once at FastAPI startup.

    The gazetteer is a dict with lowercase keys:
    {
      "velachery": {"lat": 12.9783, "lng": 80.2209, "full_name": "Velachery, Chennai"},
      "gh":        {"lat": 13.0827, "lng": 80.2707, "full_name": "Govt General Hospital"},
      ...
    }

    Keys are already lowercase from populate_gazetteer.py.
    We pre-compute the key list here for use by difflib fuzzy matching.
    """
    global _gazetteer, _gazetteer_keys, _loaded

    if _loaded:
        return True

    if not os.path.exists(GAZETTEER_PATH):
        raise FileNotFoundError(
            f"Chennai gazetteer not found at {GAZETTEER_PATH}\n"
            f"Run populate_gazetteer.py first."
        )

    print("   Loading Chennai gazetteer...")
    with open(GAZETTEER_PATH, "r", encoding="utf-8") as f:
        _gazetteer = json.load(f)

    # Pre-compute key list for fuzzy matching
    # We do this once at load time rather than on every lookup call
    _gazetteer_keys = list(_gazetteer.keys())

    _loaded = True
    print(f"   ✅ Geocoder ready — {len(_gazetteer):,} locations loaded")

    return True


# =============================================================================
# SECTION 4 — LOOKUP LOGIC
#
# Bug 9 fix is implemented here.
# The lookup has three levels:
#
# Level 1 — Exact lowercase match:
#   "Velachery Bridge" → lowercase → "velachery bridge" → found in dict
#   Most lookups succeed here. O(1) time.
#
# Level 2 — Fuzzy match via difflib:
#   "velachery brige" (typo) → difflib finds "velachery bridge" → success
#   "tambaram bus" (partial) → difflib finds "tambaram bus stand" → success
#   difflib.get_close_matches() is in Python's standard library — no install needed
#   cutoff=0.6 means at least 60% string similarity to accept a match
#
# Level 3 — Not found:
#   Returns None, None, None
#   Caller sets lat=None, lng=None on the cluster dict
#   React/Leaflet ignores clusters with null coordinates (no dot on map)
# =============================================================================

def _lookup(location_string: str) -> tuple:
    """
    Looks up a location string in the gazetteer.
    Returns (lat, lng, full_name) or (None, None, None) if not found.

    Args:
        location_string: raw location string from BERT NER

    Returns:
        tuple of (lat: float|None, lng: float|None, full_name: str|None)
    """
    if not location_string or not isinstance(location_string, str):
        return None, None, None

    # Bug 9 fix: always lowercase before lookup
    # BERT might return "Velachery Bridge", "VELACHERY BRIDGE", or "velachery bridge"
    # All three must match the same gazetteer entry
    key = location_string.lower().strip()

    # Level 1: Exact match
    if key in _gazetteer:
        entry = _gazetteer[key]
        return entry["lat"], entry["lng"], entry.get("full_name", location_string)

    # Level 2: Fuzzy match
    # get_close_matches returns a list of close matches, sorted by similarity
    # n=1: return only the best match
    # cutoff=0.6: require at least 60% similarity to accept
    matches = difflib.get_close_matches(key, _gazetteer_keys, n=1, cutoff=0.6)
    if matches:
        best_match = matches[0]
        entry = _gazetteer[best_match]
        print(f"   🔍 Fuzzy match: '{location_string}' → '{best_match}'")
        return entry["lat"], entry["lng"], entry.get("full_name", best_match)

    # Level 3: Not found
    return None, None, None


# =============================================================================
# SECTION 5 — PUBLIC API
# =============================================================================

def geocode(cluster: dict) -> dict:
    """
    Enriches a single cluster dict with lat/lng coordinates.

    Takes the "locations" list from the cluster dict (added by detective.py).
    Tries each location string in order until one resolves to coordinates.
    Uses the first successful resolution as the cluster's map position.

    If no location resolves, lat and lng are set to None.
    FastAPI's pipeline route handles null coordinates gracefully —
    clusters with null coordinates are still returned to React but
    Leaflet simply doesn't place a dot for them.

    Args:
        cluster: cluster dict with "locations" field

    Returns:
        same cluster dict enriched with:
        - "lat": float or None
        - "lng": float or None
        - "resolved_location": str or None
    """
    # Auto-load if needed (for notebook testing)
    if not _loaded:
        load_gazetteer()

    locations = cluster.get("locations", [])

    # Default to null — will be overwritten if any location resolves
    cluster["lat"]               = None
    cluster["lng"]               = None
    cluster["resolved_location"] = None

    if not locations:
        # Detective found no locations in this tweet
        # Cluster is valid and will still appear in the feed,
        # but no map dot will be placed for it
        print(f"    Cluster {cluster['cluster_id']}: no locations to geocode")
        return cluster

    # Try each extracted location in order — use first that resolves
    for location_string in locations:
        lat, lng, full_name = _lookup(location_string)

        if lat is not None and lng is not None:
            cluster["lat"]               = lat
            cluster["lng"]               = lng
            cluster["resolved_location"] = full_name
            print(f"   📍 Cluster {cluster['cluster_id']}: '{location_string}' → {lat:.4f}, {lng:.4f}")
            return cluster

    # None of the extracted locations resolved
    print(f"    Cluster {cluster['cluster_id']}: locations {locations} — none found in gazetteer")
    return cluster


def geocode_all(clusters: list) -> list:
    """
    Geocodes all clusters in a list.
    Simple sequential processing — no async needed since
    this is pure in-memory dict lookup (microseconds per call).

    Args:
        clusters: list of cluster dicts from detective.extract_all()

    Returns:
        list of fully enriched cluster dicts, each with lat/lng added
    """
    return [geocode(cluster) for cluster in clusters]


# =============================================================================
# SECTION 6 — STATUS
# =============================================================================

def is_loaded() -> bool:
    """Returns True if gazetteer is loaded and ready."""
    return _loaded


def get_coverage() -> int:
    """Returns number of locations in the loaded gazetteer."""
    return len(_gazetteer)