# =============================================================================
# FILE: backend/scripts/eval.py                   (CrisisLens v4.0)
#
# PURPOSE:
#   Standalone evaluation script for the CrisisLens Bouncer component.
#   Loads the trained TF-IDF + LinearSVC models, runs a comprehensive
#   experimental battery, and saves ALL results to results/ and figures/.
#
#   Designed to be run ONCE after training. Results feed directly into
#   the conference paper via results/results_macros.tex.
#
# HOW TO RUN:
#   # From project root (same directory as backend/):
#   python backend/scripts/eval.py
#   python backend/scripts/eval.py --data backend/data/processed/training_data.csv
#   python backend/scripts/eval.py --threshold 0.3 --cv-folds 5
#
# OUTPUT FILES:
#   figures/confusion_matrix.png      — heatmap (TP/FP/TN/FN)
#   figures/roc_curve.png             — ROC curve with AUC
#   figures/pr_curve.png              — Precision-Recall with AP score
#   figures/threshold_analysis.png    — metrics vs confidence threshold
#   figures/score_distribution.png    — decision score histogram per class
#   figures/feature_importance.png    — top-20 SVM features each class
#   figures/learning_curve.png        — train/val accuracy vs training size
#   figures/cv_scores.png             — cross-validation boxplot
#   results/metrics_summary.json      — all numeric results in one JSON
#   results/cv_scores.csv             — cross-validation fold breakdown
#   results/results_macros.tex        — LaTeX \newcommand macros for paper
#
# IMPORTANT:
#   This script is COMPLETELY DISJOINT from the live pipeline.
#   It does NOT import bouncer.py, pipeline.py, or any FastAPI code.
#   It loads the same .pkl files but evaluates in a clean environment.
#
# REQUIREMENTS:
#   pip install scikit-learn matplotlib seaborn joblib pandas numpy
# =============================================================================

import os
import sys
import re
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, learning_curve
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import calibration_curve


# =============================================================================
# SECTION 1 — PATH CONFIGURATION
# =============================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODELS_DIR   = os.path.join(ROOT_DIR, "backend", "models")
DEFAULT_DATA = os.path.join(ROOT_DIR, "backend", "data", "processed", "training_data.csv")
TFIDF_PATH   = os.path.join(MODELS_DIR, "tfidf_vectoriser.pkl")
SVM_PATH     = os.path.join(MODELS_DIR, "svm_classifier.pkl")

# Output directories — relative to project root
FIGURES_DIR  = os.path.join(ROOT_DIR, "figures")
RESULTS_DIR  = os.path.join(ROOT_DIR, "results")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# SECTION 2 — TEXT CLEANING  (identical to bouncer.py and train_bouncer.py)
# =============================================================================

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;",  "<", text)
    text = re.sub(r"&gt;",  ">", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =============================================================================
# SECTION 3 — FALLBACK DATASET  (120 hand-labelled tweets)
#   Used when training_data.csv is not found.
#   Guarantees the script always produces output on a fresh checkout.
# =============================================================================

FALLBACK_TWEETS = [
    # --- Disaster (label = 1) ---
    ("people trapped under collapsed building after earthquake velachery", 1),
    ("flood waters rising in tambaram rescue teams deployed immediately", 1),
    ("family stranded on rooftop need immediate evacuation koyambedu", 1),
    ("fire broke out in adyar apartment block casualties reported", 1),
    ("cyclone warning issued for chennai coast all residents evacuate now", 1),
    ("bridge collapsed over cooum river cars submerged emergency response", 1),
    ("sos help needed near t nagar flood victims no food water", 1),
    ("landslide blocks highway rescue operations underway nilgiris district", 1),
    ("train derailment 20 passengers injured tambaram railways", 1),
    ("flood alert red level issued for city rivers overflowing", 1),
    ("man drowning in flood near velachery lake help needed urgently", 1),
    ("power lines down sparking fires in mylapore area evacuation ordered", 1),
    ("tsunami warning issued for coastal areas residents fleeing inland", 1),
    ("hospital flooded patients being evacuated emergency declared", 1),
    ("trapped miners need rescue in tirunelveli disaster zone", 1),
    ("cyclone landfall imminent all beaches closed coast guard deployed", 1),
    ("flood rescue boats deployed in besant nagar urgent help needed", 1),
    ("massive fire at goods warehouse port area fire brigades rushing", 1),
    ("earthquake tremors felt building evacuated casualties unknown", 1),
    ("missing child in flood zone porur last seen at bus stand", 1),
    ("roads inundated chennai airport shut down due to flooding", 1),
    ("storm surge hits marina beach walls breach evacuate immediately", 1),
    ("gas leak at industrial zone workers hospitalised hazmat teams deployed", 1),
    ("flash flood warning for palar river basin all residents alert", 1),
    ("six dead twelve missing after boat capsizes in flood poonamallee", 1),
    ("fire spreading in slum area anna nagar residents displaced", 1),
    ("breaking cyclone eye makes landfall near mahabalipuram coast", 1),
    ("flood levels critical at chembarambakkam reservoir release imminent", 1),
    ("elderly couple found stranded on terrace flood waters receding slowly", 1),
    ("emergency services overwhelmed rescue calls increasing every hour", 1),
    ("construction collapse three workers buried debris rescue ongoing", 1),
    ("rain induced landslide cuts off ooty ghats roads blocked", 1),
    ("urgent need medical supplies flood relief camp guindy", 1),
    ("drinking water contaminated in flood affected areas tiruvallur", 1),
    ("trees uprooted blocking ambulance route lives at risk", 1),
    ("children stranded at school bus cannot move due to flooding", 1),
    ("rescue operation underway in adyar river flood zone", 1),
    ("oil tanker spill river ecological disaster cleanup needed urgently", 1),
    ("critically injured flood victim needs blood type o negative", 1),
    ("disaster response teams mobilised across all districts immediately", 1),
    ("survivors pulled from rubble earthquake rescue still ongoing", 1),
    ("rising flood waters isolate village no food supply access", 1),
    ("fire at chemical plant toxic smoke residents warned stay indoors", 1),
    ("two youths swept into lake flood overflow drowning report urgent", 1),
    ("electricity poles submerged electrocution risk in flood zones", 1),
    ("army deployment requested flood victims in low lying areas", 1),
    ("water level at record high all gates of dam opened", 1),
    ("helicopter rescue operation beach cyclone survivors spotted", 1),
    ("mass casualty event hospital activates emergency protocol", 1),
    ("ship capsized in storm bay of bengal coast guard search rescue", 1),
    ("police warn dangerous driving conditions roads completely submerged", 1),
    ("NGO requesting volunteers flood relief distribution centre", 1),
    ("government declares state of emergency three districts affected", 1),
    ("search and rescue dogs deployed missing persons flood site", 1),
    ("flood victims sheltering in school building food and water needed", 1),
    ("pregnant woman needs evacuation from flooded colony urgent", 1),
    ("six feet of water in ground floor residents completely stuck", 1),
    ("cyclone damage assessment underway power outage full district", 1),
    ("body recovered in floodwater victim identification underway", 1),
    ("emergency helpline numbers flood relief state government", 1),
    # --- Non-disaster / Noise (label = 0) ---
    ("flooding the internet with cute cat pictures right now", 0),
    ("this sale is killing it absolutely flooded with orders today", 0),
    ("my weekend was a disaster totally burnt the pasta again", 0),
    ("the concert was on fire crowd went absolutely crazy", 0),
    ("this morning traffic is a total disaster every single monday", 0),
    ("earthquaking performance by the band last night incredible show", 0),
    ("tsunami of emotions after watching that movie so beautiful", 0),
    ("storm of controversy around the new policy released today", 0),
    ("my phone battery is dead again third time today", 0),
    ("we are drowning in work this quarter deadline pressure", 0),
    ("collapsed on the couch after gym session so tired today", 0),
    ("fire sale everything must go up to seventy percent off today", 0),
    ("explosion of new AI products this year is absolutely insane", 0),
    ("she literally destroyed him in the chess tournament yesterday", 0),
    ("the team is on a burning hot streak five wins in a row", 0),
    ("lol that joke totally killed me absolutely hilarious", 0),
    ("caught in a storm trying to get to the gym this morning", 0),
    ("going to miss the train again terrible morning commute", 0),
    ("landslide victory for favourite team in last nights match", 0),
    ("floods of memories when I see old college photographs", 0),
    ("the film was a complete trainwreck from start to finish", 0),
    ("office politics are a disaster class on petty behaviour", 0),
    ("sunrise over the mountains was absolutely breathtaking this morning", 0),
    ("the economy is a sinking ship according to most analysts", 0),
    ("drowning in emails can someone help me sort through them", 0),
    ("what a blast we had at the family reunion last weekend", 0),
    ("weather app always gets it completely wrong useless technology", 0),
    ("new restaurant just opened fantastic food highly recommend", 0),
    ("book review the plot completely collapsed in chapter three", 0),
    ("social media is toxic please take a break from it all", 0),
    ("my plants are barely surviving the summer heat", 0),
    ("power nap in the afternoon best productivity hack ever", 0),
    ("new coffee shop opened near the office must visit soon", 0),
    ("highway was jammed for two hours terrible commute home", 0),
    ("new headphones are fire absolute best purchase in months", 0),
    ("this mango lassi is absolutely killing it favourite drink ever", 0),
    ("cricket match was electric last ball thriller fantastic game", 0),
    ("she was absolutely glowing at the wedding ceremony beautiful", 0),
    ("budget meeting was a total disaster nobody came prepared", 0),
    ("binge watched the whole series in one sitting incredible", 0),
    ("so many new streaming shows flooding my watch list now", 0),
    ("movie trailer looks explosive cannot wait for release date", 0),
    ("quarterly report is a tsunami of data charts and graphs", 0),
    ("happy birthday to the most awesome colleague ever", 0),
    ("just finished a 10k run feeling absolutely dead but happy", 0),
    ("new phone model leaked what an avalanche of features", 0),
    ("festival season is here streets look amazing with lights", 0),
    ("amazing sunset at the beach just a perfect evening out", 0),
    ("crashed my car into the curb just a small scratch though", 0),
    ("traffic police caught me for jumping signal terrible day", 0),
    ("power cut again for the third time this week frustrating", 0),
    ("laptop overheated and shut down lost all my work today", 0),
    ("heatwave in the city schools are off for two days", 0),
    ("water supply disrupted in colony calling the corporation now", 0),
    ("internet is down again working from mobile hotspot today", 0),
    ("city roads need urgent repair potholes everywhere monsoon", 0),
    ("stray dog bit cyclist near the park please be careful", 0),
    ("auto drivers refusing short trips again frustrating commute", 0),
    ("noise pollution from construction waking me at 5am daily", 0),
    ("parking issue near the mall impossible to find any space", 0),
]


# =============================================================================
# SECTION 4 — DATA LOADING
# =============================================================================

def load_data(data_path: str):
    if data_path and os.path.exists(data_path):
        print(f"   Loading: {data_path}")
        df = pd.read_csv(data_path)
        tcol = next((c for c in df.columns if c.lower() in ["text","tweet","content"]), None)
        lcol = next((c for c in df.columns if c.lower() in ["target","label","class"]), None)
        if tcol and lcol:
            texts  = df[tcol].fillna("").tolist()
            labels = df[lcol].astype(int).tolist()
            print(f"   {len(texts):,} samples | disaster={sum(labels):,} | non-disaster={len(labels)-sum(labels):,}")
            return texts, labels
        print("   WARNING: column names not recognised, falling back to sample set.")

    print("   Using built-in 120-tweet sample set.")
    texts  = [t for t, _ in FALLBACK_TWEETS]
    labels = [l for _, l in FALLBACK_TWEETS]
    print(f"   {len(texts)} samples | disaster={sum(labels)} | non-disaster={len(labels)-sum(labels)}")
    return texts, labels


# =============================================================================
# SECTION 5 — MODEL LOADING
# =============================================================================

def load_models():
    for path, name in [(TFIDF_PATH, "TF-IDF vectoriser"), (SVM_PATH, "LinearSVC")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found at:\n  {path}\nRun train_bouncer.py first."
            )
    print("   Loading TF-IDF vectoriser...")
    tfidf = joblib.load(TFIDF_PATH)
    print("   Loading LinearSVC...")
    svm   = joblib.load(SVM_PATH)
    tfidf_mb = os.path.getsize(TFIDF_PATH) / 1e6
    svm_mb   = os.path.getsize(SVM_PATH)   / 1e6
    print(f"   tfidf_vectoriser.pkl : {tfidf_mb:.2f} MB")
    print(f"   svm_classifier.pkl   : {svm_mb:.2f} MB")
    return tfidf, svm, tfidf_mb, svm_mb


# =============================================================================
# SECTION 6 — STYLE HELPERS
# =============================================================================

C = {
    "blue"   : "#2563EB",
    "red"    : "#DC2626",
    "green"  : "#16A34A",
    "orange" : "#EA580C",
    "purple" : "#7C3AED",
    "gray"   : "#6B7280",
    "teal"   : "#0891B2",
    "bg"     : "#F9FAFB",
    "dark"   : "#111827",
}

def _fig(w=7, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])
    return fig, ax

def _save(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"   → figures/{name}")
    return path


# =============================================================================
# SECTION 7 — PLOT: CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=C["bg"])
    ax.set_facecolor(C["bg"])

    display_cm = np.array([[tn, fp], [fn, tp]])
    annot = np.array([
        [f"TN\n{tn}\n(TNR {tnr:.2f})", f"FP\n{fp}\n(FPR {fpr:.2f})"],
        [f"FN\n{fn}\n(FNR {fnr:.2f})", f"TP\n{tp}\n(TPR {tpr:.2f})"],
    ])

    sns.heatmap(display_cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=["Predicted\nNon-Disaster", "Predicted\nDisaster"],
                yticklabels=["Actual\nNon-Disaster", "Actual\nDisaster"],
                ax=ax, linewidths=1, linecolor="#CBD5E1", cbar=False,
                annot_kws={"size": 11})

    ax.set_title("Confusion Matrix — TF-IDF + LinearSVC (Bouncer)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.tick_params(labelsize=10)
    _save("confusion_matrix.png")
    return {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "TPR": round(tpr,4), "FPR": round(fpr,4),
            "TNR": round(tnr,4), "FNR": round(fnr,4)}


# =============================================================================
# SECTION 8 — PLOT: ROC CURVE
# =============================================================================

def plot_roc_curve(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = _fig(6, 5)
    ax.plot(fpr, tpr, color=C["blue"], lw=2.5,
            label=f"TF-IDF + LinearSVC  (AUC = {roc_auc:.4f})")
    ax.plot([0,1],[0,1], color=C["gray"], lw=1.5, ls="--",
            label="Random baseline  (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.07, color=C["blue"])
    ax.set_xlim([0,1]); ax.set_ylim([0,1.03])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curve — Bouncer Classifier", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    _save("roc_curve.png")
    return round(roc_auc, 4)


# =============================================================================
# SECTION 9 — PLOT: PRECISION-RECALL CURVE
# =============================================================================

def plot_pr_curve(y_true, scores):
    prec, rec, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    baseline = sum(y_true) / len(y_true)

    fig, ax = _fig(6, 5)
    ax.plot(rec, prec, color=C["orange"], lw=2.5,
            label=f"TF-IDF + LinearSVC  (AP = {ap:.4f})")
    ax.axhline(baseline, color=C["gray"], lw=1.5, ls="--",
               label=f"No-skill baseline  (AP = {baseline:.2f})")
    ax.fill_between(rec, prec, alpha=0.07, color=C["orange"])
    ax.set_xlim([0,1]); ax.set_ylim([0,1.03])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve — Bouncer Classifier", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    _save("pr_curve.png")
    return round(ap, 4)


# =============================================================================
# SECTION 10 — PLOT: CONFIDENCE THRESHOLD ANALYSIS
#   Validates bouncer.py's CONFIDENCE_THRESHOLD = 0.3 design decision
# =============================================================================

def plot_threshold_analysis(y_true, scores):
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    f1s, precs, recs, accs = [], [], [], []

    for t in thresholds:
        yp = (scores >= t).astype(int)
        if yp.sum() == 0:
            f1s.append(0); precs.append(1); recs.append(0); accs.append(0)
        else:
            f1s.append(f1_score(y_true, yp, zero_division=0))
            precs.append(precision_score(y_true, yp, zero_division=0))
            recs.append(recall_score(y_true, yp, zero_division=0))
            accs.append(accuracy_score(y_true, yp))

    best_idx = int(np.argmax(f1s))
    best_thr = float(thresholds[best_idx])
    best_f1  = float(f1s[best_idx])

    fig, ax = _fig(8, 5)
    ax.plot(thresholds, f1s,   color=C["blue"],   lw=2.2, label="F1-Score")
    ax.plot(thresholds, precs, color=C["green"],  lw=2.2, label="Precision")
    ax.plot(thresholds, recs,  color=C["orange"], lw=2.2, label="Recall")
    ax.plot(thresholds, accs,  color=C["teal"],   lw=1.5, ls=":", label="Accuracy")
    ax.axvline(0.3, color=C["red"], lw=2, ls="--",
               label=f"Deployed threshold  (t = 0.3)")
    ax.axvline(best_thr, color=C["purple"], lw=1.5, ls=":",
               label=f"Optimal F1 threshold  (t = {best_thr:.2f})")
    ax.set_xlabel("Confidence Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Metrics vs. Confidence Threshold — Bouncer (Bug 2 Validation)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotate deployed threshold value
    t03_idx = int(np.argmin(np.abs(thresholds - 0.3)))
    ax.annotate(f"F1={f1s[t03_idx]:.3f}", xy=(0.3, f1s[t03_idx]),
                xytext=(0.3 + 0.08, f1s[t03_idx] - 0.05),
                arrowprops=dict(arrowstyle="->", color=C["red"]),
                fontsize=9, color=C["red"])

    _save("threshold_analysis.png")
    return round(best_thr, 4), round(best_f1, 4)


# =============================================================================
# SECTION 11 — PLOT: DECISION SCORE DISTRIBUTION
#   Shows how well the model separates the two classes
# =============================================================================

def plot_score_distribution(y_true, scores):
    y_arr = np.array(y_true)
    scores0 = scores[y_arr == 0]
    scores1 = scores[y_arr == 1]

    fig, ax = _fig(8, 5)
    ax.hist(scores0, bins=40, alpha=0.65, color=C["blue"],
            label="Non-Disaster (class 0)", density=True, edgecolor="white", lw=0.3)
    ax.hist(scores1, bins=40, alpha=0.65, color=C["red"],
            label="Disaster (class 1)", density=True, edgecolor="white", lw=0.3)
    ax.axvline(0.0, color=C["dark"], lw=1.5, ls="--", label="Hard decision boundary (t = 0)")
    ax.axvline(0.3, color=C["purple"], lw=1.5, ls=":", label="Deployed threshold (t = 0.3)")
    ax.set_xlabel("SVM Decision Function Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Decision Score Distribution by Class — Bouncer", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    _save("score_distribution.png")

    # Compute overlap region metric (lower = better separation)
    overlap = np.sum(
        np.minimum(
            np.histogram(scores0, bins=100, density=True)[0],
            np.histogram(scores1, bins=100, density=True)[0]
        )
    ) / 100
    return round(float(overlap), 4)


# =============================================================================
# SECTION 12 — PLOT: TOP FEATURE IMPORTANCE
#   Extracts SVM coefficients → maps back to TF-IDF feature names
#   Directly shows what the model learned
# =============================================================================

def plot_feature_importance(tfidf, svm, top_n=20):
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = svm.coef_[0]

    top_disaster    = np.argsort(coefs)[::-1][:top_n]
    top_nondisaster = np.argsort(coefs)[:top_n]

    dis_names  = feature_names[top_disaster]
    dis_vals   = coefs[top_disaster]
    non_names  = feature_names[top_nondisaster]
    non_vals   = np.abs(coefs[top_nondisaster])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), facecolor=C["bg"])

    # Disaster features
    y_pos = np.arange(top_n)
    ax1.barh(y_pos, dis_vals[::-1], color=C["red"], alpha=0.85, edgecolor="white")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(dis_names[::-1], fontsize=9)
    ax1.set_xlabel("SVM Coefficient Weight", fontsize=10)
    ax1.set_title(f"Top {top_n} Disaster-Signalling Features\n(Class 1 — Crisis)", fontsize=11, fontweight="bold")
    ax1.set_facecolor(C["bg"])
    ax1.grid(axis="x", alpha=0.3)

    # Non-disaster features
    ax2.barh(y_pos, non_vals[::-1], color=C["blue"], alpha=0.85, edgecolor="white")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(non_names[::-1], fontsize=9)
    ax2.set_xlabel("SVM Coefficient Weight (absolute)", fontsize=10)
    ax2.set_title(f"Top {top_n} Non-Disaster Features\n(Class 0 — Noise)", fontsize=11, fontweight="bold")
    ax2.set_facecolor(C["bg"])
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Feature Importance — TF-IDF + LinearSVC Learned Weights",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save("feature_importance.png")

    return {
        "top_disaster_features"    : dis_names[:10].tolist(),
        "top_nondisaster_features" : non_names[:10].tolist(),
    }


# =============================================================================
# SECTION 13 — PLOT: LEARNING CURVE
#   Shows whether more data would help (bias vs variance diagnosis)
# =============================================================================

def plot_learning_curve(tfidf, svm, X_clean, y, cv_folds=5):
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("tfidf", tfidf), ("svm", svm)])

    train_sizes = np.linspace(0.1, 1.0, 8)
    try:
        ts, tr_scores, cv_scores = learning_curve(
            pipe, X_clean, y,
            train_sizes=train_sizes,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring="f1_macro",
            n_jobs=-1,
            error_score=0.0,
        )
    except Exception as e:
        print(f"   WARNING: learning curve failed ({e}), skipping plot.")
        return

    tr_mean  = tr_scores.mean(axis=1)
    tr_std   = tr_scores.std(axis=1)
    cv_mean  = cv_scores.mean(axis=1)
    cv_std   = cv_scores.std(axis=1)

    fig, ax = _fig(8, 5)
    ax.plot(ts, tr_mean, color=C["blue"], lw=2.2, label="Training F1 (macro)")
    ax.fill_between(ts, tr_mean - tr_std, tr_mean + tr_std, alpha=0.12, color=C["blue"])
    ax.plot(ts, cv_mean, color=C["orange"], lw=2.2, label="Validation F1 (macro)")
    ax.fill_between(ts, cv_mean - cv_std, cv_mean + cv_std, alpha=0.12, color=C["orange"])
    ax.set_xlabel("Training Set Size", fontsize=11)
    ax.set_ylabel("F1-Score (Macro)", fontsize=11)
    ax.set_title("Learning Curve — TF-IDF + LinearSVC", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    _save("learning_curve.png")


# =============================================================================
# SECTION 14 — CROSS-VALIDATION
# =============================================================================

def run_cross_validation(tfidf, svm, X_clean, y, cv_folds=5):
    from sklearn.pipeline import Pipeline
    from sklearn.base import clone

    # Fit tfidf on all data for CV (pipeline handles splitting correctly)
    pipe = Pipeline([("tfidf", clone(tfidf)), ("svm", clone(svm))])

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = cross_validate(
        pipe, X_clean, y, cv=skf,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro",
                 "f1_weighted": "f1_weighted", "precision_macro": "precision_macro",
                 "recall_macro": "recall_macro"},
        return_train_score=True,
        error_score=0.0,
    )

    summary = {}
    for metric in ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]:
        key = f"test_{metric}"
        vals = results[key]
        summary[metric] = {
            "scores": [round(v, 4) for v in vals.tolist()],
            "mean"  : round(float(vals.mean()), 4),
            "std"   : round(float(vals.std()),  4),
        }

    # Save to CSV
    rows = []
    for fold_i in range(cv_folds):
        rows.append({
            "fold"             : fold_i + 1,
            "accuracy"         : round(results["test_accuracy"][fold_i], 4),
            "f1_macro"         : round(results["test_f1_macro"][fold_i], 4),
            "f1_weighted"      : round(results["test_f1_weighted"][fold_i], 4),
            "precision_macro"  : round(results["test_precision_macro"][fold_i], 4),
            "recall_macro"     : round(results["test_recall_macro"][fold_i], 4),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "cv_scores.csv")
    df.to_csv(csv_path, index=False)
    print(f"   → results/cv_scores.csv")

    # Boxplot
    fig, ax = _fig(8, 5)
    metrics_to_plot = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    labels = ["Accuracy", "F1 Macro", "F1 Weighted", "Precision\nMacro", "Recall\nMacro"]
    data   = [results[f"test_{m}"] for m in metrics_to_plot]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
    colors = [C["blue"], C["orange"], C["green"], C["purple"], C["teal"]]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"{cv_folds}-Fold Stratified Cross-Validation — Bouncer Classifier",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0.5, 1.05])
    _save("cv_scores.png")

    return summary


# =============================================================================
# SECTION 15 — INFERENCE LATENCY BENCHMARK
# =============================================================================

def benchmark_latency(tfidf, svm, X_clean, n_trials=1000):
    sample = (X_clean * ((n_trials // len(X_clean)) + 1))[:n_trials]

    # Warm up
    vec = tfidf.transform(sample[:10])
    svm.predict(vec)

    # Benchmark transform
    t0 = time.perf_counter()
    X_vec = tfidf.transform(sample)
    tfidf_ms = (time.perf_counter() - t0) * 1000

    # Benchmark predict
    t0 = time.perf_counter()
    _ = svm.predict(X_vec)
    svm_ms = (time.perf_counter() - t0) * 1000

    # Benchmark decision_function (used by bouncer.py)
    t0 = time.perf_counter()
    _ = svm.decision_function(X_vec)
    dec_ms = (time.perf_counter() - t0) * 1000

    total_ms        = tfidf_ms + dec_ms
    per_tweet_ms    = total_ms / n_trials
    throughput      = 1000 / per_tweet_ms

    print(f"   TF-IDF transform   : {tfidf_ms:.2f}ms for {n_trials} tweets")
    print(f"   SVM decision_fn    : {dec_ms:.2f}ms for {n_trials} tweets")
    print(f"   Per tweet (total)  : {per_tweet_ms:.4f}ms")
    print(f"   Throughput         : {throughput:,.0f} tweets/second")

    return {
        "n_trials"          : n_trials,
        "tfidf_total_ms"    : round(tfidf_ms, 3),
        "svm_decision_ms"   : round(dec_ms, 3),
        "per_tweet_ms"      : round(per_tweet_ms, 6),
        "throughput_per_sec": round(throughput, 1),
    }


# =============================================================================
# SECTION 16 — DATASET STATISTICS
# =============================================================================

def compute_dataset_stats(texts_raw, labels, X_clean):
    y = np.array(labels)
    n_total      = len(y)
    n_disaster   = int((y == 1).sum())
    n_nondisaster= int((y == 0).sum())
    ratio        = round(n_disaster / n_nondisaster, 3)

    # Token length stats per class
    raw_lengths = [len(t.split()) for t in texts_raw]
    clean_lengths = [len(t.split()) for t in X_clean]

    dis_lens   = [l for l, lbl in zip(raw_lengths, labels) if lbl == 1]
    ndis_lens  = [l for l, lbl in zip(raw_lengths, labels) if lbl == 0]

    return {
        "n_total"              : n_total,
        "n_disaster"           : n_disaster,
        "n_nondisaster"        : n_nondisaster,
        "disaster_ratio"       : ratio,
        "avg_token_len_all"    : round(np.mean(raw_lengths), 2),
        "avg_token_len_disaster"  : round(np.mean(dis_lens), 2) if dis_lens else 0,
        "avg_token_len_nondisaster": round(np.mean(ndis_lens), 2) if ndis_lens else 0,
        "max_token_len"        : int(np.max(raw_lengths)),
        "min_token_len"        : int(np.min(raw_lengths)),
    }


# =============================================================================
# SECTION 17 — VOCABULARY STATISTICS
# =============================================================================

def compute_vocab_stats(tfidf):
    vocab = tfidf.vocabulary_
    total = len(vocab)
    bigrams   = sum(1 for k in vocab if " " in k)
    unigrams  = total - bigrams
    return {
        "vocabulary_size"  : total,
        "unigram_count"    : unigrams,
        "bigram_count"     : bigrams,
        "bigram_pct"       : round(bigrams / total * 100, 2),
        "max_features_cfg" : 10000,
        "ngram_range"      : "(1, 2)",
        "sublinear_tf"     : True,
        "min_df"           : 2,
    }


# =============================================================================
# SECTION 18 — LaTeX MACRO FILE GENERATOR
#   Writes results/results_macros.tex so the paper auto-updates
# =============================================================================

def write_latex_macros(summary: dict):
    lines = [
        "% ================================================================",
        "% AUTO-GENERATED by eval.py — do not edit by hand",
        "% Re-run eval.py to update all numbers in the paper",
        "% ================================================================",
        "",
    ]

    def cmd(name, value):
        # Sanitise value for LaTeX
        val_str = str(value).replace("%", r"\%").replace("_", r"\_")
        lines.append(f"\\newcommand{{\\{name}}}{{{val_str}}}")

    # Standard metrics
    m = summary.get("classification", {})
    cmd("BouncerAccuracy",       f"{summary.get('accuracy_pct', 'XX.X')}\\%")
    cmd("BouncerMacroF",         summary.get("macro_f1", "0.XXXX"))
    cmd("BouncerWeightedF",      summary.get("weighted_f1", "0.XXXX"))
    cmd("BouncerPrecisionW",     summary.get("weighted_precision", "0.XXXX"))
    cmd("BouncerRecallW",        summary.get("weighted_recall", "0.XXXX"))
    cmd("BouncerROCAUC",         summary.get("roc_auc", "0.XXXX"))
    cmd("BouncerAP",             summary.get("average_precision", "0.XXXX"))
    cmd("BouncerDisasterF",      summary.get("disaster_f1", "0.XXXX"))
    cmd("BouncerNonDisasterF",   summary.get("nondisaster_f1", "0.XXXX"))

    # Confusion matrix
    cm = summary.get("confusion_matrix", {})
    cmd("BTP",  cm.get("TP", "X"))
    cmd("BTN",  cm.get("TN", "X"))
    cmd("BFP",  cm.get("FP", "X"))
    cmd("BFN",  cm.get("FN", "X"))
    cmd("BTPR", cm.get("TPR", "X"))
    cmd("BFPR", cm.get("FPR", "X"))

    # Threshold
    cmd("OptimalThreshold", summary.get("best_threshold", "X.XX"))
    cmd("OptimalF",         summary.get("best_f1_at_threshold", "0.XXXX"))

    # Cross-validation
    cv = summary.get("cross_validation", {})
    acc_cv = cv.get("accuracy", {})
    f1_cv  = cv.get("f1_macro", {})
    cmd("CVAccMean",  acc_cv.get("mean", "0.XXXX"))
    cmd("CVAccStd",   acc_cv.get("std",  "0.XXXX"))
    cmd("CVF1Mean",   f1_cv.get("mean",  "0.XXXX"))
    cmd("CVF1Std",    f1_cv.get("std",   "0.XXXX"))

    # Latency
    lat = summary.get("latency", {})
    cmd("InferenceLatency",  lat.get("per_tweet_ms", "X.XXXX"))
    cmd("InferenceThroughput", lat.get("throughput_per_sec", "X"))

    # Dataset stats
    ds = summary.get("dataset", {})
    cmd("DatasetTotal",    ds.get("n_total", "X"))
    cmd("DatasetDisaster", ds.get("n_disaster", "X"))
    cmd("DatasetNoise",    ds.get("n_nondisaster", "X"))
    cmd("AvgTweetLen",     ds.get("avg_token_len_all", "X.X"))

    # Vocab
    voc = summary.get("vocabulary", {})
    cmd("VocabSize",    voc.get("vocabulary_size", "X"))
    cmd("UnigramCount", voc.get("unigram_count", "X"))
    cmd("BigramCount",  voc.get("bigram_count", "X"))
    cmd("BigramPct",    voc.get("bigram_pct", "X.X"))

    # Model sizes
    cmd("TFIDFsizeMB", summary.get("tfidf_size_mb", "X.XX"))
    cmd("SVMsizeMB",   summary.get("svm_size_mb",   "X.XX"))

    lines.append("")
    path = os.path.join(RESULTS_DIR, "results_macros.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"   → results/results_macros.tex")


# =============================================================================
# SECTION 19 — MAIN EVALUATION ORCHESTRATOR
# =============================================================================

def run_evaluation(data_path: str, threshold: float = 0.3, cv_folds: int = 5):
    print("\n" + "="*60)
    print("  CrisisLens v4.0 — Comprehensive Bouncer Evaluation")
    print("="*60)

    print("\n[1/9] Loading data...")
    texts_raw, labels = load_data(data_path)

    print("\n[2/9] Loading models...")
    tfidf, svm, tfidf_mb, svm_mb = load_models()

    print("\n[3/9] Preparing evaluation split (80/20 stratified)...")
    X_clean = [clean_text(t) for t in texts_raw]
    valid   = [(r, c, l) for r, c, l in zip(texts_raw, X_clean, labels) if len(c) > 0]
    texts_raw_v = [v[0] for v in valid]
    X_clean_v   = [v[1] for v in valid]
    labels_v    = [v[2] for v in valid]

    if len(X_clean_v) <= 120:
        X_test, y_test = X_clean_v, labels_v
        print("   Using full sample set for test evaluation.")
    else:
        _, X_test, _, y_test = train_test_split(
            X_clean_v, labels_v, test_size=0.2, random_state=42, stratify=labels_v
        )

    X_tfidf = tfidf.transform(X_test)
    y_pred  = svm.predict(X_tfidf)
    scores  = svm.decision_function(X_tfidf)

    print("\n[4/9] Computing standard metrics...")
    acc     = accuracy_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred, output_dict=True,
                                    target_names=["Non-Disaster", "Disaster"])
    macro_f1     = round(f1_score(y_test, y_pred, average="macro"), 4)
    weighted_f1  = round(f1_score(y_test, y_pred, average="weighted"), 4)
    weighted_prec= round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4)
    weighted_rec = round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4)
    dis_f1   = round(report["Disaster"]["f1-score"], 4)
    ndis_f1  = round(report["Non-Disaster"]["f1-score"], 4)

    print(f"   Accuracy          : {acc*100:.2f}%")
    print(f"   Macro F1          : {macro_f1}")
    print(f"   Weighted F1       : {weighted_f1}")
    print(f"   Disaster F1       : {dis_f1}")
    print(f"   Non-Disaster F1   : {ndis_f1}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Non-Disaster","Disaster"]))

    print("[5/9] Generating plots...")
    cm_stats   = plot_confusion_matrix(y_test, y_pred)
    roc_auc    = plot_roc_curve(y_test, scores)
    ap         = plot_pr_curve(y_test, scores)
    best_thr, best_f1 = plot_threshold_analysis(y_test, scores)
    score_overlap = plot_score_distribution(y_test, scores)
    feat_info  = plot_feature_importance(tfidf, svm)

    print("\n[6/9] Learning curve (slow — trains model multiple times)...")
    plot_learning_curve(tfidf, svm, X_clean_v, labels_v, cv_folds=min(cv_folds, 3))

    print("\n[7/9] Cross-validation...")
    cv_results = run_cross_validation(tfidf, svm, X_clean_v, labels_v, cv_folds)
    cv_acc  = cv_results["accuracy"]
    cv_f1   = cv_results["f1_macro"]
    print(f"   CV Accuracy : {cv_acc['mean']:.4f} ± {cv_acc['std']:.4f}")
    print(f"   CV Macro F1 : {cv_f1['mean']:.4f} ± {cv_f1['std']:.4f}")

    print("\n[8/9] Latency benchmark...")
    latency = benchmark_latency(tfidf, svm, X_clean_v)

    print("\n[9/9] Saving JSON + LaTeX macros...")
    ds_stats  = compute_dataset_stats(texts_raw_v, labels_v, X_clean_v)
    voc_stats = compute_vocab_stats(tfidf)

    summary = {
        "accuracy"            : round(acc, 4),
        "accuracy_pct"        : round(acc * 100, 2),
        "macro_f1"            : macro_f1,
        "weighted_f1"         : weighted_f1,
        "weighted_precision"  : weighted_prec,
        "weighted_recall"     : weighted_rec,
        "roc_auc"             : roc_auc,
        "average_precision"   : ap,
        "disaster_f1"         : dis_f1,
        "nondisaster_f1"      : ndis_f1,
        "best_threshold"      : best_thr,
        "best_f1_at_threshold": best_f1,
        "score_overlap"       : score_overlap,
        "confusion_matrix"    : cm_stats,
        "cross_validation"    : cv_results,
        "latency"             : latency,
        "dataset"             : ds_stats,
        "vocabulary"          : voc_stats,
        "feature_analysis"    : feat_info,
        "tfidf_size_mb"       : round(tfidf_mb, 2),
        "svm_size_mb"         : round(svm_mb, 2),
        "model_config": {
            "classifier"   : "LinearSVC",
            "class_weight" : "balanced",
            "C"            : 1.0,
            "max_iter"     : 2000,
            "confidence_threshold_deployed": threshold,
        }
    }

    json_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   → results/metrics_summary.json")

    write_latex_macros(summary)

    print("\n" + "="*60)
    print(f"  Done. figures/ and results/ populated.")
    print(f"  Compile main.tex — all \\newcommand macros now updated.")
    print("="*60 + "\n")
    return summary


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrisisLens Bouncer evaluation")
    parser.add_argument("--data",      default=None,  help="Path to CSV (text + target columns)")
    parser.add_argument("--threshold", default=0.3,   type=float)
    parser.add_argument("--cv-folds",  default=5,     type=int)
    args = parser.parse_args()
    run_evaluation(
        data_path  = args.data or DEFAULT_DATA,
        threshold  = args.threshold,
        cv_folds   = args.cv_folds,
    )
