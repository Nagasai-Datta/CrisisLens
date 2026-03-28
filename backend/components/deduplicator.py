# =============================================================================
# FILE: backend/components/deduplicator.py
#
# PURPOSE:
#   Receives a batch of disaster tweets (already filtered by the Bouncer).
#   Groups semantically similar tweets about the same incident into clusters.
#   Returns one cluster dictionary per unique incident.
#
# WHY THIS EXISTS:
#   During a real disaster, hundreds of people tweet about the same event.
#   Without deduplication, the commander sees 400 alerts for one incident.
#   With deduplication, they see one card: "Velachery bridge — 400 reports"
#   This is the component that makes CrisisLens genuinely useful vs noise.
#
# BUGS FIXED IN THIS FILE:
#   Bug 3  — DBSCAN noise points now become singleton clusters (not discarded)
#   Bug 4  — eps=0.15 (cosine DISTANCE), not 0.85 (which would be similarity)
#   Bug 5  — Rolling window re-selects representative after retiring old tweets
#   Bug 6  — Urgency score formula fully defined and implemented
#   Bug 17 — Fewer than 3 tweets skips DBSCAN entirely (prevents edge case crash)
#   Bug 18 — Embeddings explicitly deleted when tweets retire (prevents memory leak)
#
# INPUT:
#   list of tweet strings (from bouncer.predict())
#   list of timestamps (datetime objects — when each tweet arrived)
#
# OUTPUT:
#   list of cluster dictionaries — one per unique incident
#   Each dict has: cluster_id, representative_tweet, tweet_count,
#                  source_tweets, first_seen, last_seen, urgency_score
#
# FLOW:
#   new tweets arrive
#       → embed them (MiniLM → 384-dim vectors)
#       → add to the rolling window store
#       → retire tweets older than 30 minutes
#       → run DBSCAN on all active embeddings
#       → build cluster dicts
#       → return cluster list
# =============================================================================

import os
import uuid
import numpy  as np
from datetime  import datetime, timedelta
from sklearn.cluster import DBSCAN

# sentence-transformers: the library that wraps MiniLM
from sentence_transformers import SentenceTransformer


# =============================================================================
# SECTION 1 — CONFIGURATION CONSTANTS
#
# These are defined at the top so they're easy to find and tune.
# =============================================================================

# Rolling window: how long a tweet stays active in the clustering pool
WINDOW_MINUTES = 30

# DBSCAN epsilon — cosine DISTANCE threshold for grouping tweets.
# eps = 1 - similarity_threshold = 1 - 0.85 = 0.15
# Bug 4 fix: if this were 0.85 (similarity), completely unrelated tweets
# would be clustered together. 0.4 (distance) is the correct value.
DBSCAN_EPS = 0.4

# DBSCAN min_samples — minimum tweets to form a dense cluster.
# Set to 2: at least 2 tweets must be near each other to form a cluster.
# Anything with only 1 neighbour becomes noise → we handle as singleton.
DBSCAN_MIN_SAMPLES = 2

# Urgency keyword list — presence of these words increases urgency score
URGENT_KEYWORDS = {
    "trapped", "trap", "stuck", "dying", "dead", "death",
    "help", "rescue", "sos", "urgent", "critical", "emergency",
    "drowning", "drown", "flood", "fire", "collapse", "collapsed",
    "missing", "injured", "injury", "hospital", "ambulance"
}

# MiniLM model name — distilled transformer for fast sentence embeddings
# Produces 384-dimensional vectors, 5x faster than large transformers
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =============================================================================
# SECTION 2 — MODULE-LEVEL STATE
#
# The rolling window stores tweets + embeddings in memory.
# All module-level variables persist for the lifetime of the server process.
#
# _model         : the loaded MiniLM model (loaded once, used forever)
# _tweet_store   : list of dicts, one per active tweet
#                  each dict: {id, text, embedding, timestamp}
# =============================================================================

_model       = None   # MiniLM sentence transformer
_tweet_store = []     # the rolling 30-minute window of active tweets
_loaded      = False  # flag to prevent reloading


# =============================================================================
# SECTION 3 — MODEL LOADING
# =============================================================================

def load_model():
    """
    Loads MiniLM sentence transformer model into memory.
    Called once at FastAPI startup.

    Apple Silicon note:
        We set device="mps" to use the Mac's Metal GPU.
        This is approximately 3-5x faster than CPU for embedding batches.
        Falls back to CPU automatically if MPS is not available.
    """
    global _model, _loaded

    if _loaded:
        return True

    print("   Loading MiniLM sentence transformer...")

    # Detect best available device
    # mps = Apple Silicon Metal GPU
    # cpu = fallback for Intel Macs or if MPS unavailable
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
            print("   Using Apple Silicon MPS GPU")
        else:
            device = "cpu"
            print("   MPS not available — using CPU")
    except Exception:
        device = "cpu"

    _model  = SentenceTransformer(MINILM_MODEL, device=device)
    _loaded = True
    print("    Deduplicator (MiniLM) ready")

    return True


# =============================================================================
# SECTION 4 — ROLLING WINDOW MANAGEMENT
#
# The rolling window is a list of tweet records kept in memory.
# Each record stores the tweet text, its 384-dim embedding, and timestamp.
# Records older than WINDOW_MINUTES are retired (deleted) on each update.
#
# WHY A ROLLING WINDOW?
#   1. Prevents temporal bleed: a flood at Velachery 3 hours ago should not
#      absorb a new report about a fire at Velachery now. Different incidents
#      happen at the same location over time.
#   2. Bounds memory: without retirement, embeddings accumulate forever.
#      After 6 hours of heavy tweets, memory would be exhausted.
# =============================================================================

def _add_tweets_to_window(tweets: list, timestamps: list) -> list:
    """
    Embeds new tweets and adds them to the rolling window store.

    Args:
        tweets     : list of tweet strings (already Bouncer-filtered)
        timestamps : list of datetime objects, one per tweet

    Returns:
        list of tweet record dicts that were successfully added
    """
    global _tweet_store, _model

    if not tweets:
        return []

    # Ensure model is loaded before encoding
    if _model is None:
        load_model()

    # Verify model loaded successfully
    if _model is None:
        raise RuntimeError("Failed to load MiniLM model")

    # Embed all new tweets in one batch (faster than one at a time)
    # encode() returns a numpy array of shape (n_tweets, 384)
    # main miniLM task
    embeddings = _model.encode(tweets, convert_to_numpy=True, show_progress_bar=False)

    new_records = []
    for text, timestamp, embedding in zip(tweets, timestamps, embeddings):
        record = {
            "id"        : str(uuid.uuid4()),  # unique ID for this tweet record
            "text"      : text,
            "embedding" : embedding,           # 384-dim numpy array
            "timestamp" : timestamp,
        }
        _tweet_store.append(record)
        new_records.append(record)

    return new_records


def _retire_old_tweets():
    """
    Removes tweets older than WINDOW_MINUTES from the store.
    Explicitly deletes their embeddings to free memory (Bug 18 fix).

    WHY EXPLICIT DELETION?
        Python's garbage collector does not guarantee immediate cleanup.
        numpy arrays holding 384 floats each are small individually
        but accumulate significantly over hours of high-volume tweeting.
        Explicitly deleting ensures memory is freed promptly.
    """
    global _tweet_store

    cutoff = datetime.utcnow() - timedelta(minutes=WINDOW_MINUTES)

    surviving = []
    for record in _tweet_store:
        if record["timestamp"] >= cutoff:
            surviving.append(record)
        else:
            # Explicitly delete the embedding array (Bug 18 fix)
            del record["embedding"]

    retired_count = len(_tweet_store) - len(surviving)
    _tweet_store  = surviving

    if retired_count > 0:
        print(f"    Retired {retired_count} tweets outside 30-min window")


# =============================================================================
# SECTION 5 — URGENCY SCORE COMPUTATION (Bug 6 fix)
#
# Formula:
#   urgency_score = 0.4 × keyword_weight
#                 + 0.4 × cluster_density
#                 + 0.2 × recency_weight
#
# keyword_weight  : 0.0 to 1.0 — fraction of tweets containing urgent keywords
# cluster_density : 0.0 to 1.0 — average cosine similarity within cluster
#                   tight cluster = similar reports = confirmed incident
# recency_weight  : 0.0 to 1.0 — how recently the last tweet arrived
#                   1.0 = just now, 0.0 = 30 minutes ago
#
# Thresholds:
#   score >= 0.7 → CRITICAL
#   score >= 0.4 → HIGH
#   score <  0.4 → MODERATE
# =============================================================================

def _compute_urgency(tweet_texts: list, embeddings: np.ndarray, last_seen: datetime) -> tuple:
    """
    Computes urgency score for a cluster.

    Args:
        tweet_texts : list of tweet strings in this cluster
        embeddings  : numpy array of shape (n, 384) — one row per tweet
        last_seen   : datetime of the most recent tweet in the cluster

    Returns:
        (urgency_score: float, urgency_label: str)
    """
    # ── keyword_weight ────────────────────────────────────────────────────
    # What fraction of tweets in this cluster contain at least one urgent keyword?
    keyword_hits = 0
    for text in tweet_texts:
        words = set(text.lower().split())
        if words & URGENT_KEYWORDS:  # set intersection — any overlap?
            keyword_hits += 1
    keyword_weight = keyword_hits / len(tweet_texts) if tweet_texts else 0.0

    # ── cluster_density ───────────────────────────────────────────────────
    # Average pairwise cosine similarity within the cluster.
    # Tight clusters (high similarity) = confirmed incident with many reports.
    # Loose clusters = possibly different events grouped together.
    if len(embeddings) == 1:
        # Singleton cluster — can't compute pairwise similarity
        cluster_density = 0.5  # neutral score for singletons
    else:
        # Normalise embeddings to unit vectors (required for cosine similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors
        norms = np.where(norms == 0, 1, norms)
        normalised = embeddings / norms

        # Cosine similarity matrix: dot product of normalised vectors
        # Shape: (n, n) — each cell is similarity between two tweets
        sim_matrix = np.dot(normalised, normalised.T)

        # Extract upper triangle (avoid counting pairs twice and self-similarity)
        upper_triangle = sim_matrix[np.triu_indices(len(embeddings), k=1)]
        cluster_density = float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else 0.5

    # ── recency_weight ────────────────────────────────────────────────────
    # How recent is the last tweet?
    # 1.0 = arrived right now
    # 0.0 = arrived exactly 30 minutes ago
    now     = datetime.utcnow()
    age_sec = (now - last_seen).total_seconds()
    window_sec    = WINDOW_MINUTES * 60
    recency_weight = max(0.0, 1.0 - (age_sec / window_sec))

    # ── Final score ───────────────────────────────────────────────────────
    urgency_score = (
        0.4 * keyword_weight
      + 0.4 * cluster_density
      + 0.2 * recency_weight
    )

    # Clamp to [0.0, 1.0] — floating point arithmetic can produce tiny overflows
    urgency_score = float(np.clip(urgency_score, 0.0, 1.0))

    # Assign label
    if urgency_score >= 0.7:
        urgency_label = "CRITICAL"
    elif urgency_score >= 0.4:
        urgency_label = "HIGH"
    else:
        urgency_label = "MODERATE"

    return urgency_score, urgency_label


# =============================================================================
# SECTION 6 — REPRESENTATIVE SELECTION (Bug 5 fix)
#
# WHY LONGEST TWEET?
#   Longer tweets statistically carry more location detail, victim counts,
#   and emergency type. "Help" is less useful to BERT NER than
#   "Family of 4 trapped on roof near Velachery bridge, water rising fast".
#
# Bug 5 fix:
#   This function is called AFTER retiring old tweets so the representative
#   is always selected from currently active tweets only.
#   If all tweets in a cluster retire, the cluster is dissolved entirely.
# =============================================================================

def _select_representative(tweet_texts: list) -> str:
    """
    Selects the best representative tweet from a cluster.
    Currently: longest tweet by character count.

    Args:
        tweet_texts: list of tweet strings

    Returns:
        The single tweet string that best represents the cluster.
    """
    return max(tweet_texts, key=len)


# =============================================================================
# SECTION 7 — CLUSTER BUILDING
#
# After DBSCAN assigns labels, we build the cluster dictionary.
# This is the lingua franca of the pipeline — every downstream component
# receives this dict and adds fields to it.
# =============================================================================

def _build_cluster_dict(tweet_records: list, cluster_label: str) -> dict:
    """
    Builds the cluster dictionary from a list of tweet records.

    Args:
        tweet_records  : list of tweet record dicts (from _tweet_store)
        cluster_label  : string like "c_001" or "singleton_abc123"

    Returns:
        cluster dict with all required fields
    """
    texts      = [r["text"]      for r in tweet_records]
    embeddings = np.array([r["embedding"] for r in tweet_records])
    timestamps = [r["timestamp"] for r in tweet_records]

    first_seen = min(timestamps)
    last_seen  = max(timestamps)

    representative = _select_representative(texts)

    urgency_score, urgency_label = _compute_urgency(texts, embeddings, last_seen)

    return {
        "cluster_id"          : cluster_label,
        "representative_tweet": representative,
        "tweet_count"         : len(texts),
        "source_tweets"       : texts,
        "first_seen"          : first_seen.isoformat(),
        "last_seen"           : last_seen.isoformat(),
        "urgency_score"       : urgency_score,
        "urgency_label"       : urgency_label,
        # These fields are added by downstream components:
        # "locations"         : []     ← added by detective.py
        # "lat"               : None   ← added by geocoder.py
        # "lng"               : None   ← added by geocoder.py
        # "resolved_location" : None   ← added by geocoder.py
    }


# =============================================================================
# SECTION 8 — MAIN CLUSTERING FUNCTION
#
# This is the function FastAPI calls on each new batch of tweets.
# Full flow:
#   1. Add new tweets to rolling window (embed them)
#   2. Retire tweets older than 30 minutes
#   3. If fewer than 3 active tweets → skip DBSCAN, make singletons
#   4. Run DBSCAN on all active embeddings
#   5. DBSCAN noise points → singleton clusters (Bug 3 fix)
#   6. Build cluster dict for each group
#   7. Return list of cluster dicts
# =============================================================================

def cluster(tweets: list, timestamps: list = None) -> list: # type: ignore
    """
    Main entry point for the Deduplicator.
    Called by FastAPI's pipeline route on every new tweet batch.

    Args:
        tweets     : list of tweet strings (Bouncer output)
        timestamps : list of datetime objects. If None, uses current time
                     for all tweets (useful for testing).

    Returns:
        list of cluster dicts — one per unique active incident
    """
    global _tweet_store

    # Auto-load model if not yet loaded (for notebook testing)
    if not _loaded:
        load_model()

    # Default timestamps to now if not provided
    if timestamps is None:
        now = datetime.utcnow()
        timestamps = [now] * len(tweets)

    # ── Step 1: Add new tweets to window ─────────────────────────────────
    if tweets:
        _add_tweets_to_window(tweets, timestamps)

    # ── Step 2: Retire old tweets ─────────────────────────────────────────
    _retire_old_tweets()

    # ── Step 3: Check minimum points (Bug 17 fix) ─────────────────────────
    # If fewer than 3 active tweets, DBSCAN behaves unpredictably.
    # Skip it entirely and treat each tweet as its own singleton cluster.
    if len(_tweet_store) < 3:
        clusters = []
        for i, record in enumerate(_tweet_store):
            cluster_dict = _build_cluster_dict(
                tweet_records = [record],
                cluster_label = f"singleton_{record['id'][:8]}"
            )
            clusters.append(cluster_dict)
        print(f"    {len(clusters)} singleton cluster(s) (< 3 tweets, DBSCAN skipped)")
        return clusters

    # ── Step 4: Run DBSCAN ───────────────────────────────────────────────
    # Extract all active embeddings into one matrix
    all_embeddings = np.array([r["embedding"] for r in _tweet_store])

    # DBSCAN with cosine metric
    # eps=0.15  → cosine DISTANCE threshold (Bug 4 fix — not 0.85 similarity)
    # metric='cosine' → uses cosine distance, correct for text embeddings
    # min_samples=2 → at least 2 tweets needed to form a cluster
    #                 any tweet without 2 neighbours → noise → singleton (Bug 3)
    dbscan = DBSCAN(
        eps        = DBSCAN_EPS,
        min_samples= DBSCAN_MIN_SAMPLES,
        metric     = "cosine",
        algorithm  = "brute"    # brute force needed for cosine metric
    )

    labels = dbscan.fit_predict(all_embeddings)
    # labels is a numpy array: one integer per tweet
    # -1       = noise point (no cluster)
    # 0, 1, 2, ... = cluster IDs

    # ── Step 5: Group tweets by cluster label ─────────────────────────────
    clusters = []

    # Group indices by cluster label
    unique_labels = set(labels)

    for label in unique_labels:
        # Get indices of all tweets with this label
        indices = [i for i, l in enumerate(labels) if l == label]
        records = [_tweet_store[i] for i in indices]

        if label == -1:
            # Bug 3 fix: noise points are NOT discarded.
            # Each noise point = a tweet about a unique incident nobody else
            # has reported yet. It becomes its own singleton cluster.
            # This is the most important fix in the entire system.
            for record in records:
                cluster_dict = _build_cluster_dict(
                    tweet_records = [record],
                    cluster_label = f"singleton_{record['id'][:8]}"
                )
                clusters.append(cluster_dict)
        else:
            # Normal cluster — multiple tweets about same incident
            cluster_dict = _build_cluster_dict(
                tweet_records = records,
                cluster_label = f"c_{str(label).zfill(3)}"
            )
            clusters.append(cluster_dict)

    # Sort by urgency score descending so most critical clusters come first
    clusters.sort(key=lambda c: c["urgency_score"], reverse=True)

    n_real      = sum(1 for l in unique_labels if l != -1)
    n_singleton = sum(1 for l in labels if l == -1)
    print(f"    Clusters: {n_real} multi-tweet + {n_singleton} singletons = {len(clusters)} total")

    return clusters


# =============================================================================
# SECTION 9 — STATUS
# =============================================================================

def is_loaded() -> bool:
    """Returns True if MiniLM model is loaded and ready."""
    return _loaded


def get_window_size() -> int:
    """Returns number of tweets currently in the active window."""
    return len(_tweet_store)