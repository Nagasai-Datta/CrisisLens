# =============================================================================
# FILE: backend/scripts/prepare_data.py
#
# PURPOSE:
#   This is a ONE-TIME setup script. It takes two completely different
#   disaster tweet datasets (Kaggle + CrisisLexT26) and merges them into
#   one clean, unified CSV file that our model training script can read.
#
# WHY THIS FILE EXISTS:
#   Kaggle dataset  → 1 CSV file, labels are 0 and 1 (numbers)
#   CrisisLexT26    → 26 CSV files, labels are text words like
#                     "Related and informative" or "Not related"
#   You can't just concatenate these — they speak different languages.
#   This script is the translator.
#
# INPUT:
#   backend/data/raw/kaggle/train.csv
#   backend/data/raw/crisislex/<26 event folders>/
#
# OUTPUT:
#   backend/data/processed/training_data.csv
#   Columns: text (string), label (0 or 1)
#
# HOW TO RUN (from the crisislens/ root folder):
#   source venv/bin/activate
#   python3 backend/scripts/prepare_data.py
#
# RUN THIS: Once only. Never again.
# =============================================================================

import os           # for walking through folders and file paths
import pandas as pd # for reading CSVs and manipulating data tables
import re           # for cleaning tweet text (removing URLs etc.)


# =============================================================================
# SECTION 1 — FILE PATH CONFIGURATION
# All paths relative to the crisislens/ root folder.
# =============================================================================

# Get the absolute path to the crisislens/ root folder.
# __file__ is this script's path. We go up two levels (scripts/ → backend/ → root)
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Where the raw Kaggle CSV lives
KAGGLE_PATH = os.path.join(ROOT_DIR, "backend", "data", "raw", "kaggle", "train.csv")

# Where the 26 CrisisLex event folders live
CRISISLEX_DIR = os.path.join(ROOT_DIR, "backend", "data", "raw", "crisislex")

# Where we save the final merged CSV
OUTPUT_PATH = os.path.join(ROOT_DIR, "backend", "data", "processed", "training_data.csv")


# =============================================================================
# SECTION 2 — TEXT CLEANING FUNCTION
#
# WHY CLEAN TEXT?
#   Raw tweets contain noise that confuses the model:
#   - URLs like "http://t.co/abc123" → meaningless tokens
#   - @mentions like "@FEMA" → not useful for classification
#   - #hashtags → keep the word, remove the # symbol
#   - HTML entities like "&amp;" → should be "&"
#   - Extra whitespace
#
# This SAME cleaning is applied to BOTH datasets so the training data
# matches exactly what runtime tweets will look like after cleaning.
# Consistency is critical — if you clean training data differently from
# runtime data, the model's vocabulary won't match and predictions degrade.
# =============================================================================

def clean_text(text: str) -> str:
    """
    Cleans a single tweet string.
    Returns the cleaned string.

    Called on every tweet from both datasets before saving.
    Also called identically in bouncer.py at runtime.
    """
    if not isinstance(text, str):
        # Some rows have NaN (missing values) instead of text
        # Return empty string — these get dropped later
        return ""

    text = text.lower()                          # LOWERCASE everything
    text = re.sub(r"http\S+|www\S+", "", text)  # REMOVE URLs
    text = re.sub(r"@\w+", "", text)            # REMOVE @mentions
    text = re.sub(r"#", "", text)               # REMOVE # but keep the word
    text = re.sub(r"&amp;", "&", text)          # FIX HTML entity
    text = re.sub(r"&lt;", "<", text)           # FIX HTML entity
    text = re.sub(r"&gt;", ">", text)           # FIX HTML entity
    text = re.sub(r"[^\w\s]", " ", text)        # REMOVE punctuation (keep words + spaces)
    text = re.sub(r"\s+", " ", text).strip()    # COLLAPSE multiple spaces

    return text


# =============================================================================
# SECTION 3 — LOAD KAGGLE DATASET
#
# Kaggle CSV columns:
#   id       → tweet ID (we don't need this)
#   keyword  → disaster keyword (we don't need this)
#   location → user location (we don't need this)
#   text     → the actual tweet (WE NEED THIS)
#   target   → 0 = not disaster, 1 = disaster (WE NEED THIS)
# =============================================================================

def load_kaggle() -> pd.DataFrame:
    """
    Reads the Kaggle disaster tweets CSV.
    Returns a DataFrame with raw_text, text, label columns.
    """
    print("\n Loading Kaggle dataset...")

    # Safety check — tell user clearly if file is missing
    if not os.path.exists(KAGGLE_PATH):
        raise FileNotFoundError(
            f"Kaggle CSV not found at: {KAGGLE_PATH}\n"
            f"Please download train.csv from kaggle.com/competitions/nlp-getting-started\n"
            f"and place it in backend/data/raw/kaggle/"
        )

    df = pd.read_csv(KAGGLE_PATH)

    print(f"   Raw shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Columns found: {list(df.columns)}")

    # Keep only the columns we need for cleaning + labeling
    df = df[["text", "target"]].copy()

    # Keep the original tweet text for safer deduplication later
    df.rename(columns={"text": "raw_text", "target": "label"}, inplace=True)

    # Apply text cleaning to every row
    df["text"] = df["raw_text"].apply(clean_text)

    # Drop rows where cleaning produced an empty string
    df = df[df["text"].str.len() > 0]

    print(f"   After cleaning: {df.shape[0]} rows")
    print(f"   Class distribution → 0: {(df['label']==0).sum()}  |  1: {(df['label']==1).sum()}")

    return df


# =============================================================================
# SECTION 4 — LOAD CRISISLEXT26 DATASET
#
# CrisisLexT26 structure:
#   26 folders, one per crisis event
#   Each folder contains one CSV file named: {event}-tweets_labeled.csv
#
# CSV columns inside each file:
#   Tweet ID          → tweet ID (we don't need this)
#   Tweet Text        → the actual tweet (WE NEED THIS)
#   Information Source → source type (we don't need)
#   Information Type   → information category (we don't need)
#   Informativeness    → relevance label (WE NEED THIS)
#
# Informativeness label values and what we do with them:
#   "Related and informative"       → label 1  (it's a real crisis report)
#   "Related - but not informative" → label 1  (still crisis-related, keep it)
#   "Not related"                   → label 0  (noise, not about the crisis)
#   "Not applicable"                → label 0  (noise)
#
# WHY map "Related - but not informative" to 1?
#   A tweet like "Praying for everyone in Colorado" is related to the crisis
#   even if it doesn't provide actionable information. It still contains
#   disaster vocabulary. Including it improves the Bouncer's understanding
#   of the full range of crisis language, not just rescue requests.
# =============================================================================

# This dictionary maps every possible CrisisLexT26 label to 0 or 1.
# It's defined at module level so you can see exactly what mapping happens.
CRISISLEX_LABEL_MAP = {
    "related and informative"       : 1,
    "related - but not informative" : 1,
    "not related"                   : 0,
    "not applicable"                : 0,
}

def load_crisislex() -> pd.DataFrame:
    """
    Walks all 26 CrisisLexT26 event folders.
    Reads each CSV, extracts text + label.
    Returns a single merged DataFrame with columns: raw_text, text, label
    """
    print("\n Loading CrisisLexT26 dataset...")

    # Safety check — tell user clearly if folder is missing
    if not os.path.exists(CRISISLEX_DIR):
        raise FileNotFoundError(
            f"CrisisLexT26 folder not found at: {CRISISLEX_DIR}\n"
            f"Please download CrisisLexT26 and place the 26 event folders\n"
            f"inside backend/data/raw/crisislex/"
        )

    all_frames = []       # we'll collect one dataframe per event, then merge
    events_loaded = 0     # counter for logging
    label_counts = {}     # tracks how many of each label type we saw (for Bug 14 audit)

    # Walk the event folders deterministically so repeated runs stay stable.
    for root, dirs, files in os.walk(CRISISLEX_DIR):
        dirs.sort()
        for filename in sorted(files):
            # Only load the labeled tweet files, e.g.
            # "2012_Colorado_wildfires-tweets_labeled.csv".
            if not filename.endswith("-tweets_labeled.csv"):
                continue

            filepath = os.path.join(root, filename)

            try:
                # CrisisLex labeled files in this dataset decode cleanly as UTF-8.
                df = pd.read_csv(filepath, encoding="utf-8")

                # ── Find the text column ──────────────────────────────────
                # Column might be named "Tweet Text" or "tweet_text"
                # We search case-insensitively to be safe
                text_col = None
                for col in df.columns:
                    if "tweet" in col.lower() and "text" in col.lower():
                        text_col = col
                        break

                if text_col is None:
                    print(f"     Skipping {filename} — no tweet text column found")
                    print(f"      Columns available: {list(df.columns)}")
                    continue

                # ── Find the label column ─────────────────────────────────
                # Column might be named "Informativeness" or "informativeness"
                label_col = None
                for col in df.columns:
                    if "informative" in col.lower():
                        label_col = col
                        break

                if label_col is None:
                    print(f"     Skipping {filename} — no informativeness column found")
                    print(f"      Columns available: {list(df.columns)}")
                    continue

                # ── Audit label variants (Bug 14 fix) ────────────────────
                # Log EVERY unique label value we encounter
                # This is how we know if there are unexpected label variants
                for raw_label in df[label_col].dropna().unique():
                    key = str(raw_label).strip()
                    label_counts[key] = label_counts.get(key, 0) + (df[label_col] == raw_label).sum()

                # ── Map labels to 0 / 1 ──────────────────────────────────
                def map_label(raw):
                    """Converts a CrisisLexT26 label string to 0 or 1"""
                    if not isinstance(raw, str):
                        return 0  # NaN or unexpected type → treat as noise
                    normalized = raw.strip().lower()
                    # Look up in our mapping dict
                    # If label is unknown/unexpected → default to 0 (safe choice)
                    return CRISISLEX_LABEL_MAP.get(normalized, 0)

                # ── Build clean dataframe for this event ─────────────────
                event_df = pd.DataFrame({
                    "raw_text": df[text_col],
                    "text"    : df[text_col].apply(clean_text),
                    "label": df[label_col].apply(map_label)
                })

                # Drop rows where cleaning produced empty string
                event_df = event_df[event_df["text"].str.len() > 0]

                all_frames.append(event_df)
                events_loaded += 1

            except Exception as e:
                raise ValueError(f"Failed to read CrisisLex file {filepath}: {e}") from e

    if events_loaded == 0:
        raise ValueError(
            "No CrisisLexT26 CSV files were loaded. "
            "Check that your crisislex/ folder contains the 26 event subfolders."
        )

    # Merge all 26 event dataframes into one
    combined = pd.concat(all_frames, ignore_index=True)

    print(f"   Events loaded: {events_loaded}/26")
    print(f"   Total rows before dedup: {combined.shape[0]}")
    print(f"\n   Label variant audit (Bug 14 — all variants logged):")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        mapped_to = CRISISLEX_LABEL_MAP.get(label.lower(), "UNKNOWN → mapped to 0")
        print(f"      '{label}': {count:,} rows → {mapped_to}")

    print(f"\n   After label mapping:")
    print(f"   Class distribution → 0: {(combined['label']==0).sum():,}  |  1: {(combined['label']==1).sum():,}")

    return combined


# =============================================================================
# SECTION 5 — MERGE, DEDUPLICATE, AND SAVE
# =============================================================================

def prepare():
    """
    Main function. Orchestrates everything:
    1. Load Kaggle data
    2. Load CrisisLexT26 data
    3. Merge both into one dataframe
    4. Deduplicate
    5. Save to processed/training_data.csv
    6. Print a summary report
    """
    print("=" * 60)
    print("CrisisLens — Data Preparation Script")
    print("=" * 60)

    # ── Step 1 + 2: Load both datasets ───────────────────────────
    kaggle_df    = load_kaggle()
    crisislex_df = load_crisislex()

    kaggle_count    = len(kaggle_df)
    crisislex_count = len(crisislex_df)

    # ── Step 3: Merge ─────────────────────────────────────────────
    print("\n Merging both datasets...")
    merged = pd.concat([kaggle_df, crisislex_df], ignore_index=True)
    print(f"   Combined size: {len(merged):,} rows")

    # ── Step 4: Deduplicate ───────────────────────────────────────
    # First remove exact duplicate examples without collapsing label conflicts.
    before_exact_dedup = len(merged)
    merged.drop_duplicates(subset=["raw_text", "label"], inplace=True)
    exact_duplicates_removed = before_exact_dedup - len(merged)

    # If different raw tweets normalize to the same cleaned text but disagree on
    # the label, drop the whole group instead of keeping one arbitrarily.
    label_conflicts = merged.groupby("text")["label"].nunique()
    conflicting_texts = label_conflicts[label_conflicts > 1].index
    conflict_rows_removed = int(merged["text"].isin(conflicting_texts).sum())
    conflict_groups_removed = len(conflicting_texts)
    if conflict_rows_removed:
        merged = merged[~merged["text"].isin(conflicting_texts)].copy()

    # After conflicts are removed, keep one row per cleaned text + label pair.
    before_clean_dedup = len(merged)
    merged.drop_duplicates(subset=["text", "label"], inplace=True)
    cleaned_duplicates_removed = before_clean_dedup - len(merged)

    after_dedup = len(merged)
    duplicates_removed = (
        exact_duplicates_removed +
        conflict_rows_removed +
        cleaned_duplicates_removed
    )

    print(f"   Exact duplicates removed: {exact_duplicates_removed:,}")
    print(f"   Conflicting cleaned-text groups removed: {conflict_groups_removed:,}")
    print(f"   Rows removed from conflicts: {conflict_rows_removed:,}")
    print(f"   Same-label cleaned duplicates removed: {cleaned_duplicates_removed:,}")
    print(f"   Total rows removed: {duplicates_removed:,}")
    print(f"   Final dataset size: {after_dedup:,} rows")

    # Shuffle rows so Kaggle and CrisisLex tweets are mixed
    # (not all Kaggle tweets first, all CrisisLex tweets second)
    # This prevents the train/test split from being accidentally ordered
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    output_df = merged[["text", "label"]].copy()

    # ── Step 5: Save ──────────────────────────────────────────────
    # Ensure output directory exists (it should from setup, but be safe)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n Saved to: {OUTPUT_PATH}")

    # ── Step 6: Final Summary Report ──────────────────────────────
    label_0 = (output_df["label"] == 0).sum()
    label_1 = (output_df["label"] == 1).sum()
    balance  = label_1 / len(output_df) * 100

    print("\n" + "=" * 60)
    print("✅ Data Preparation Complete — Summary")
    print("=" * 60)
    print(f"  Source 1 — Kaggle:        {kaggle_count:>8,} tweets")
    print(f"  Source 2 — CrisisLexT26:  {crisislex_count:>8,} tweets")
    print(f"  Duplicates removed:       {duplicates_removed:>8,}")
    print(f"  ─────────────────────────────────────")
    print(f"  Final training set:       {after_dedup:>8,} tweets")
    print(f"  Label 0 (not disaster):   {label_0:>8,} ({100-balance:.1f}%)")
    print(f"  Label 1 (disaster):       {label_1:>8,} ({balance:.1f}%)")
    print(f"\n  ⚠️  Class balance note:")
    if balance > 60:
        print(f"  Dataset is {balance:.0f}% disaster — moderately imbalanced.")
        print(f"  train_bouncer.py uses class_weight='balanced' to compensate.")
    else:
        print(f"  Dataset is reasonably balanced at {balance:.0f}% disaster.")
    print("=" * 60)


# =============================================================================
# ENTRY POINT
# Python runs this block only when you execute the script directly.
# Not when another file imports it.
# =============================================================================

if __name__ == "__main__":
    prepare()
