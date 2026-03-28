# =============================================================================
# FILE: backend/components/editor.py
#
# PURPOSE:
#   Generates the Commander's Situation Report using BART summarisation.
#   Runs every 5 minutes as a background task inside FastAPI.
#   Sends active cluster summaries to facebook/bart-large-cnn via
#   the HuggingFace Inference API and updates the report store.
#
# BUG 12 FIX — BART INPUT LENGTH OVERFLOW:
#   BART max input is 1,024 tokens (~750 words).
#   We sort clusters by urgency descending and truncate at 900 tokens.
#   Most critical incidents always make it into the report.
#

# THIS FILE IS NOT IMPORTED DIRECTLY — main.py starts it as a background task.
# =============================================================================

import os
import asyncio
import requests
from dotenv import load_dotenv

# Load .env file — reads HF_API_TOKEN into environment
load_dotenv(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "backend", ".env"
))

# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# HuggingFace Inference API endpoint for BART
BART_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"

# Maximum tokens to send to BART
# BART's hard limit is 1,024 — we use 900 to leave room for the summary
MAX_INPUT_TOKENS = 900

# Approximate tokens per word (rough estimate for token counting)
WORDS_PER_TOKEN = 0.75

# How often to generate a new report (seconds)
REPORT_INTERVAL_SECONDS = 10   # 5 minutes


# =============================================================================
# SECTION 2 — REPORT GENERATION
# =============================================================================

def _build_bart_input(clusters: list) -> str:
    """
    Formats the active clusters into a structured text input for BART.
    Sorts by urgency descending (most critical first).
    Truncates at MAX_INPUT_TOKENS to stay within BART's limit (Bug 12 fix).

    Args:
        clusters: list of cluster dicts from the in-memory store

    Returns:
        Formatted string ready to send to BART
    """
    if not clusters:
        return ""

    # Sort by urgency_score descending — most critical incidents first
    # If BART input gets truncated, we lose low-priority clusters not high ones
    urgency_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2}
    sorted_clusters = sorted(
        clusters,
        key=lambda c: (urgency_order.get(c.get("urgency_label", "MODERATE"), 2),
                       -c.get("tweet_count", 0))
    )

    lines       = []
    word_count  = 0
    max_words   = int(MAX_INPUT_TOKENS / WORDS_PER_TOKEN)   # ~1,200 words

    for i, cluster in enumerate(sorted_clusters, 1):
        urgency  = cluster.get("urgency_label", "MODERATE")
        location = cluster.get("resolved_location", "Unknown location")
        tweet    = cluster.get("representative_tweet", "")
        count    = cluster.get("tweet_count", 1)

        line = f"INCIDENT {i} [{urgency}] at {location}: {tweet} ({count} reports)"
        line_words = len(line.split())

        # Stop adding clusters if we'd exceed the token budget (Bug 12 fix)
        if word_count + line_words > max_words:
            lines.append(f"[{len(sorted_clusters) - i + 1} additional incidents not shown]")
            break

        lines.append(line)
        word_count += line_words

    return "\n".join(lines)


def generate_report(clusters: list) -> str: # type: ignore
    """
    Sends the cluster summary to BART and returns the generated report.

    Args:
        clusters: list of active cluster dicts

    Returns:
        BART-generated situation report string
        Empty string if API call fails or clusters is empty
    """
    hf_token = os.getenv("HF_API_TOKEN", "")

    if not hf_token:
        print("   ⚠️  Editor: HF_API_TOKEN not set in backend/.env — skipping")
        return ""

    if not clusters:
        return ""

    # Build the input text
    bart_input = _build_bart_input(clusters)

    if not bart_input:
        return ""

    print(f"   📝 Editor: Generating report for {len(clusters)} clusters...")

    # Call HuggingFace Inference API
    headers  = {"Authorization": f"Bearer {hf_token}"}
    payload  = {
        "inputs": bart_input,
        "parameters": {
            "max_length":    250,   # summary length in tokens
            "min_length":    60,    # minimum summary length
            "do_sample":     False, # deterministic output
        }
    }

    try:
        response = requests.post(
            BART_API_URL,
            headers = headers,
            json    = payload,
            timeout = 30        # 30 second timeout
        )

        if response.status_code == 200:
            result = response.json()
            # API returns a list — first element has the summary
            if isinstance(result, list) and len(result) > 0:
                summary = result[0].get("summary_text", "")
                print(f"   ✅ Editor: Report generated ({len(summary)} chars)")
                return summary

        elif response.status_code == 503:
            # Model is loading on HuggingFace's side — this happens after cold start
            print("   ⚠️  Editor: BART model loading on HuggingFace (503). Will retry next cycle.")
            return ""

        else:
            print(f"   ⚠️  Editor: API error {response.status_code}: {response.text[:200]}")
            return ""

    except requests.Timeout:
        print("   ⚠️  Editor: API request timed out. Will retry next cycle.")
        return ""
    except Exception as e:
        print(f"   ⚠️  Editor: Unexpected error: {e}")
        return ""


# =============================================================================
# SECTION 3 — BACKGROUND TASK
#
# This is the function that runs continuously in the background.
# FastAPI starts it as an asyncio task during server startup.
# It loops forever: wait 5 minutes → generate report → update store → repeat.
#
# WHY asyncio.sleep AND NOT time.sleep:
#   time.sleep() blocks the entire event loop for 5 minutes.
#   asyncio.sleep() yields control back to the event loop during the wait.
#   Other requests can still be processed while the editor is sleeping.
# =============================================================================

async def run_editor_loop(get_clusters_fn, update_report_fn):
    """
    Background loop that generates situation reports every 5 minutes.

    Args:
        get_clusters_fn:  function that returns current active_clusters list
        update_report_fn: function that stores the generated report
    """
    print("   ✅ Editor background task started (5-minute interval)")

    while True:
        # Wait 5 minutes before first report (let clusters accumulate first)
        await asyncio.sleep(REPORT_INTERVAL_SECONDS)

        # Get current clusters from the pipeline store
        clusters = get_clusters_fn()

        if not clusters:
            print("   📝 Editor: No active clusters — skipping report generation")
            continue

        # Generate report (runs in thread pool to avoid blocking event loop)
        loop   = asyncio.get_event_loop()
        report = await loop.run_in_executor(None, generate_report, clusters)

        if report:
            # Store the report — FastAPI will serve it via GET /api/report
            update_report_fn(report)