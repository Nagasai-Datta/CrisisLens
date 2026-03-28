# =============================================================================
# FILE: backend/components/bouncer.py
#
# PURPOSE:
#   This is the RUNTIME Bouncer component.
#   It loads the trained .pkl files and exposes a predict() function.
#
# THIS FILE IS NOT A SCRIPT — it is a module that gets imported.
# FastAPI imports it at server startup.
# You can also test it in your notebook:
#   from backend.components.bouncer import predict
#   results = predict(["flooding near bridge", "this sale is on fire"])
#
# WHAT IT DOES AT RUNTIME:
#   1. When imported → loads both .pkl files into memory ONCE
#   2. predict() is called → cleans tweets → vectorises → SVM predicts
#   3. Returns only tweets classified as disaster (label = 1)
#   4. Adds confidence scores so low-confidence predictions can be filtered
#
# INPUT:  list of raw tweet strings
# OUTPUT: list of tweet strings that passed the filter (disaster tweets only)
# =============================================================================

import os
import re
import joblib
import numpy as np


# =============================================================================
# SECTION 1 — PATHS
# =============================================================================

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TFIDF_PATH = os.path.join(ROOT_DIR, "backend", "models", "tfidf_vectoriser.pkl")
SVM_PATH   = os.path.join(ROOT_DIR, "backend", "models", "svm_classifier.pkl")


# =============================================================================
# SECTION 2 — MODULE-LEVEL MODEL STORAGE
#
# These variables are set ONCE when the module is first imported.
# They stay in memory for the entire lifetime of the process.
# Every call to predict() uses these same loaded objects — no reloading.
#
# This is the standard pattern for ML model serving.
# DO NOT load models inside the predict() function — that would
# reload from disk on every single tweet batch (catastrophically slow).
# =============================================================================

_tfidf  = None   # will hold the fitted TfidfVectorizer
_svm    = None   # will hold the trained LinearSVC
_loaded = False  # flag so we only load once


# =============================================================================
# SECTION 3 — TEXT CLEANING
# CRITICAL: MUST be identical to prepare_data.py and train_bouncer.py.
# All three must clean text the same way.
# =============================================================================

def clean_text(text: str) -> str:
    """
    Cleans an incoming tweet string from frontend before vectorisation.
    Must match exactly the cleaning in prepare_data.py and train_bouncer.py.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================================================================
# SECTION 4 — MODEL LOADING
# =============================================================================

def load_models():
    """
    Loads both .pkl files into memory.
    Called once at FastAPI server startup.
    Sets module-level _tfidf and _svm variables.

    Returns True if successful, False if files not found.
    FastAPI uses the return value to set model health status.
    """
    global _tfidf, _svm, _loaded

    # Don't reload if already loaded
    if _loaded:
        return True

    # Check files exist before trying to load
    if not os.path.exists(TFIDF_PATH):
        raise FileNotFoundError(
            f"TF-IDF vectoriser not found at {TFIDF_PATH}\n"
            f"Run train_bouncer.py first."
        )
    if not os.path.exists(SVM_PATH):
        raise FileNotFoundError(
            f"SVM classifier not found at {SVM_PATH}\n"
            f"Run train_bouncer.py first."
        )

    print("   Loading Bouncer models...")
    #loading models and makinf flag true
    _tfidf  = joblib.load(TFIDF_PATH)
    _svm    = joblib.load(SVM_PATH)
    _loaded = True
    print(" Bouncer ready")

    return True


# =============================================================================
# SECTION 5 — PREDICT
#
# HOW CONFIDENCE THRESHOLD WORKS (Bug 2 fix):
#
#   SVM's standard predict() returns a hard 0 or 1.
#   SVM's decision_function() returns a distance from the decision boundary.
#       Positive value → leans toward class 1 (disaster)
#       Negative value → leans toward class 0 (not disaster)
#       Large positive → very confident it's disaster
#       Small positive → barely over the line — likely a metaphorical tweet
#
#   Example:
#       "flooding kills 10 in Chennai"  → decision score: +2.4  (very confident)
#       "flooding the market with deals"→ decision score: +0.2  (barely over line)
#
#   With threshold=0.3, the market tweet gets filtered out.
#   With threshold=0.0 (default hard predict), both pass.
#
#   The threshold is tunable — start at 0.3 and adjust based on demo results.
# =============================================================================

# Confidence threshold — tweets with decision score below this are dropped
# Increase this to let fewer tweets through (stricter filter)
# Decrease this to let more tweets through (more permissive)
CONFIDENCE_THRESHOLD = 0.3

def predict(tweets: list) -> list:
    """
    Takes a list of raw tweet strings.
    Returns only those classified as disaster (label = 1)
    with confidence above CONFIDENCE_THRESHOLD.

    This is the function FastAPI calls every time new tweets arrive.

    Args:
        tweets: list of raw tweet strings

    Returns:
        filtered_tweets: list of strings — only the disaster ones

    Example:
        predict([
            "family trapped in flood near velachery bridge",
            "this sale is flooding the market with deals",
            "earthquake rocks the music charts"
        ])
        → ["family trapped in flood near velachery bridge"]
    """
    global _tfidf, _svm, _loaded

    # Auto-load models if not loaded yet
    # This handles the case where bouncer.py is used in a notebook
    # without going through FastAPI's startup sequence
    if not _loaded:
        load_models()

    # Check that models are available before proceeding
    if _tfidf is None or _svm is None:
        raise RuntimeError("Models failed to load. Check that model files exist at the configured paths.")

    # Handle empty input gracefully
    if not tweets:
        return []

    # Step 1 — Clean all tweets
    cleaned = [clean_text(t) for t in tweets]

    # Filter out tweets that became empty after cleaning
    # Keep track of original tweets alongside cleaned versions
    valid_pairs = [
        (original, cleaned_text)
        for original, cleaned_text in zip(tweets, cleaned)
        if len(cleaned_text) > 0
    ]

    if not valid_pairs:
        return []

    original_tweets  = [pair[0] for pair in valid_pairs]
    cleaned_tweets   = [pair[1] for pair in valid_pairs]

    # Step 2 — TF-IDF transform (uses the SAVED vocabulary, not a new one)
    # transform() not fit_transform() — we never refit the vectoriser at runtime
    X = _tfidf.transform(cleaned_tweets)

    # Step 3 — Get confidence scores from SVM (Bug 2 fix)
    # decision_function returns distance from decision boundary
    # Higher value = more confident it's disaster
    confidence_scores = _svm.decision_function(X)

    # Step 4 — Filter by confidence threshold
    # Keep only tweets where confidence score exceeds threshold
    disaster_tweets = [
        original
        for original, score in zip(original_tweets, confidence_scores)
        if score >= CONFIDENCE_THRESHOLD
    ]

    return disaster_tweets


# =============================================================================
# SECTION 6 — STATUS CHECK
# Used by FastAPI's /api/health endpoint
# =============================================================================

def is_loaded() -> bool:
    """Returns True if models are loaded and ready."""
    return _loaded
