# =============================================================================
# FILE: backend/scripts/train_bouncer.py
#
# PURPOSE:
#   Trains the TF-IDF vectoriser + LinearSVC classifier (the Bouncer).
#   Saves both to backend/models/ as .pkl files.
#
# WHY TWO SEPARATE FILES ARE SAVED:
#   tfidf_vectoriser.pkl  →  The vocabulary + word-to-column mapping
#                            "flood" is always column 3,847
#                            "trapped" is always column 7,291
#
#   svm_classifier.pkl    →  The decision rules learned from data
#                            "if column 7,291 is high → disaster"
#
#   At runtime, BOTH must be loaded. If you only save the SVM and create
#   a fresh TF-IDF at runtime, "trapped" gets a random new column number.
#   The SVM's rules then point to the wrong columns → garbage predictions.
#
# INPUT:
#   backend/data/processed/training_data.csv
#
# OUTPUT:
#   backend/models/tfidf_vectoriser.pkl
#   backend/models/svm_classifier.pkl
#
# HOW TO RUN:
#   source venv/bin/activate
#   python backend/scripts/train_bouncer.py
#
# RUN THIS: Once only. Re-run only if you want to retrain with new data.
# =============================================================================

import os
import pandas as pd
import numpy  as np
import joblib   # for saving/loading Python objects to/from disk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm                     import LinearSVC
from sklearn.model_selection         import train_test_split
from sklearn.metrics                 import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline                import Pipeline

import re   # for the same text cleaning function used in prepare_data.py


# =============================================================================
# SECTION 1 — PATH CONFIGURATION
# =============================================================================

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH    = os.path.join(ROOT_DIR, "backend", "data", "processed", "training_data.csv")
MODELS_DIR   = os.path.join(ROOT_DIR, "backend", "models")
TFIDF_PATH   = os.path.join(MODELS_DIR, "tfidf_vectoriser.pkl")
SVM_PATH     = os.path.join(MODELS_DIR, "svm_classifier.pkl")


# =============================================================================
# SECTION 2 — TEXT CLEANING
# CRITICAL: This must be IDENTICAL to the clean_text() in prepare_data.py
# AND identical to the clean_text() in bouncer.py (runtime).
# All three files must clean text the same way.
# If they diverge, training data and runtime data look different → bad model.
# =============================================================================

def clean_text(text: str) -> str:
    """
    Cleans a tweet string.
    MUST be identical to the same function in prepare_data.py and bouncer.py.
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
# SECTION 3 — LOAD AND VALIDATE DATA
# =============================================================================

def load_data() -> tuple:
    """
    Loads training_data.csv.
    Returns (X, y) where:
        X = list of cleaned tweet strings
        y = numpy array of labels (0 or 1)
    """
    print(" Loading training data...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"training_data.csv not found at {DATA_PATH}\n"
            f"Run prepare_data.py first."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df):,} rows")

    # Verify expected columns exist
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Expected columns 'text' and 'label' not found.\n"
            f"Columns in file: {list(df.columns)}\n"
            f"Re-run prepare_data.py to fix this."
        )

    # Drop any rows with missing text or label
    before = len(df)
    df.dropna(subset=["text", "label"], inplace=True)
    df = df[df["text"].str.len() > 0]
    after  = len(df)
    if before != after:
        print(f"   Dropped {before - after} rows with missing values")

    # Apply text cleaning (should already be clean from prepare_data.py, but be safe)
    X = df["text"].apply(clean_text).tolist()
    y = np.asarray(df["label"].astype(int).values)

    print(f"   Final: {len(X):,} samples")
    print(f"   Label 0: {(y==0).sum():,}  |  Label 1: {(y==1).sum():,}")

    return X, y


# =============================================================================
# SECTION 4 — TRAIN
#
# Two key decisions explained:
#
# TF-IDF max_features=10000:
#   Limits vocabulary to 10,000 most important words.
#   More features = more memory + slower at runtime.
#   10,000 is a sweet spot for tweet-length text.
#
# class_weight='balanced' (Bug 1 fix):
#   Tells SVM: penalise misclassifying the minority class more heavily.
#   Without this, if 65% of data is disaster, SVM learns that predicting
#   "disaster" on everything gives 65% accuracy — and it lazily does that.
#   With 'balanced', it must learn to actually distinguish the two classes.
# =============================================================================

def train(X_train, y_train):
    """
    Trains TF-IDF vectoriser and LinearSVC.
    Returns (fitted_tfidf, fitted_svm)
    """
    print("\n  Training TF-IDF vectoriser...")

    # TF-IDF: converts text → sparse matrix of numbers
    # max_features=10000 → only keep top 10,000 most informative words
    # ngram_range=(1,2)  → captures single words AND two-word phrases
    #                       "flash flood" is more specific than just "flood"
    # sublinear_tf=True  → dampens very frequent words (log scale)
    tfidf = TfidfVectorizer(
        max_features = 10_000,
        ngram_range  = (1, 2),    # unigrams + bigrams
        sublinear_tf = True,      # log normalization
        strip_accents = "unicode",
        analyzer     = "word",
        min_df       = 2,         # ignore words appearing in fewer than 2 docs
    )

    # Fit ONLY on training data (never on test data — that would be data leakage)
    X_train_tfidf = tfidf.fit_transform(X_train)
    print(f"   Vocabulary size: {len(tfidf.vocabulary_):,} words")
    print(f"   Training matrix: {X_train_tfidf.shape[0]:,} tweets × {X_train_tfidf.shape[1]:,} features")

    print("\n  Training LinearSVC classifier...")

    # LinearSVC: fast linear Support Vector Machine
    # class_weight='balanced' → corrects for class imbalance (Bug 1 fix)
    # max_iter=2000           → give it enough iterations to converge
    # C=1.0                   → regularisation strength (default, works well)
    svm = LinearSVC(
        class_weight = "balanced",   # Bug 1 fix
        max_iter     = 2000,
        C            = 1.0,
        random_state = 42
    )

    svm.fit(X_train_tfidf, y_train)
    print("   Training complete ✓")

    return tfidf, svm


# =============================================================================
# SECTION 5 — EVALUATE
# =============================================================================

def evaluate(tfidf, svm, X_test, y_test):
    """
    Evaluates the trained model on the held-out test set.
    Prints accuracy, F1, precision, recall, confusion matrix.
    """
    print("\n Evaluating on test set...")

    # Transform test set using the ALREADY FITTED tfidf (not a new one)
    X_test_tfidf = tfidf.transform(X_test)

    # Predict
    y_pred = svm.predict(X_test_tfidf)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="weighted")

    print(f"\n   Accuracy : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"   F1 Score : {f1:.4f}")

    # Full classification report (precision + recall per class)
    print("\n   Full Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Not Disaster (0)", "Disaster (1)"]
    ))

    # Confusion matrix — helps understand error patterns
    cm = confusion_matrix(y_test, y_pred)
    print("   Confusion Matrix:")
    print(f"   ┌─────────────────────────────────┐")
    print(f"   │              Predicted           │")
    print(f"   │         Not Disaster  Disaster   │")
    print(f"   │ Actual                           │")
    print(f"   │ Not Disaster  {cm[0][0]:>6,}    {cm[0][1]:>6,}   │")
    print(f"   │ Disaster      {cm[1][0]:>6,}    {cm[1][1]:>6,}   │")
    print(f"   └─────────────────────────────────┘")
    print(f"\n   False Positives (noise passed as disaster): {cm[0][1]:,}")
    print(f"   False Negatives (disasters missed):         {cm[1][0]:,}")

    # Flag if model is underperforming
    if f1 < 0.85:
        print(f"\n     F1 score {f1:.3f} is below target of 0.85")
        print(f"   Consider: more data, tuning C parameter, or different ngram range")
    else:
        print(f"\n    F1 score {f1:.3f} meets target of ≥0.85")

    return accuracy, f1


# =============================================================================
# SECTION 6 — SAVE MODELS
# =============================================================================

def save_models(tfidf, svm):
    """
    Saves both fitted objects to backend/models/ as .pkl files.
    Both MUST be saved. One without the other is useless at runtime.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(tfidf, TFIDF_PATH)
    print(f"\n Saved TF-IDF vectoriser → {TFIDF_PATH}")

    joblib.dump(svm, SVM_PATH)
    print(f" Saved SVM classifier   → {SVM_PATH}")

    # Verify files exist and log sizes
    tfidf_size = os.path.getsize(TFIDF_PATH) / (1024 * 1024)
    svm_size   = os.path.getsize(SVM_PATH)   / (1024 * 1024)
    print(f"\n   tfidf_vectoriser.pkl : {tfidf_size:.2f} MB")
    print(f"   svm_classifier.pkl   : {svm_size:.2f} MB")


# =============================================================================
# SECTION 7 — MAIN ORCHESTRATION
# =============================================================================

def train_bouncer():
    """
    Orchestrates the full training pipeline:
    load → split → train → evaluate → save
    """
    print("=" * 60)
    print("CrisisLens — Bouncer Training Script")
    print("=" * 60)

    # Load data
    X, y = load_data()

    # Split into train (80%) and test (20%)
    # stratify=y ensures both splits have the same class ratio (Bug 1 support)
    # random_state=42 ensures reproducible splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.20,
        random_state = 42,
        stratify     = y      # maintain class balance in both splits
    )

    print(f"\n Split:")
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test:  {len(X_test):,} samples")

    # Train
    tfidf, svm = train(X_train, y_train)

    # Evaluate
    accuracy, f1 = evaluate(tfidf, svm, X_test, y_test)

    # Save
    save_models(tfidf, svm)

    print("\n" + "=" * 60)
    print(" Bouncer Training Complete")
    print(f"   Accuracy : {accuracy*100:.1f}%")
    print(f"   F1 Score : {f1:.4f}")
    print("   Next step: verify .pkl files exist in backend/models/")
    print("   Then run:  bouncer.py is ready to use (no extra run needed)")
    print("=" * 60)


if __name__ == "__main__":
    train_bouncer()