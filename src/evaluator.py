"""
src/evaluator.py
Evaluation metrics, confusion matrix, and ablation study.
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (confusion_matrix, classification_report,
                              f1_score, precision_score, recall_score)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LABEL_NAMES, NUM_LABELS, BASE_DIR

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", filename="confusion_matrix.png"):
    """Plot and save a normalised confusion matrix."""
    labels = [LABEL_NAMES[i] for i in range(NUM_LABELS)]
    cm     = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    path = RESULTS_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Saved {path}")
    return path


def plot_class_distribution(df: pd.DataFrame, filename="class_distribution.png"):
    """Bar chart of class distribution in a dataframe."""
    counts = df["label"].map(LABEL_NAMES).value_counts().reindex(
        [LABEL_NAMES[i] for i in range(NUM_LABELS)])

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#22c55e", "#ef4444", "#f97316", "#a855f7"]
    counts.plot(kind="bar", ax=ax, color=colors, edgecolor="black")
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
    plt.tight_layout()
    path = RESULTS_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Saved {path}")
    return path


def compute_metrics(y_true, y_pred, stage="Classifier") -> dict:
    """Return a dict of macro precision / recall / F1."""
    return {
        "stage":     stage,
        "F1":        round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


def run_ablation(test_df: pd.DataFrame,
                 classifier,
                 rag_pipeline,
                 filename="ablation.png") -> pd.DataFrame:
    """
    Compare four ablation variants and plot results.

    Variants:
      1. Classifier alone
      2. Classifier + RAG (MiniLM, k=3) — default
      3. Classifier + RAG (k=1)
      4. Classifier + RAG (k=5)
    """
    from src.rag_pipeline import N_RESULTS, SIMILARITY_THRESHOLD

    print("\n🔬 Running ablation study...")
    rows = []

    # ── Variant 1: Classifier alone ───────────────────────────────────────────
    clf_results = classifier.predict(test_df["text"].tolist())
    y_pred_clf  = [r["label_id"] for r in clf_results]
    rows.append(compute_metrics(test_df["label"].tolist(), y_pred_clf,
                                "Classifier only"))

    # ── Variants 2-4: Classifier + RAG with different k ──────────────────────
    for k in [1, 3, 5]:
        y_pred_rag = []
        for text, clf_res in zip(test_df["text"].tolist(), clf_results):
            hits    = rag_pipeline.retrieve(text, n=k)
            top_hit = hits[0] if hits else None
            conf    = clf_res["confidence"]

            if conf >= 0.90:
                verdict = clf_res["label"]
            elif top_hit and top_hit["distance"] <= SIMILARITY_THRESHOLD:
                verdict = top_hit["label"]
            else:
                verdict = clf_res["label"]

            y_pred_rag.append(
                next(i for i, n in LABEL_NAMES.items() if n == verdict))

        rows.append(compute_metrics(test_df["label"].tolist(), y_pred_rag,
                                    f"Classifier + RAG (k={k})"))

    results_df = pd.DataFrame(rows)
    print(results_df.to_string(index=False))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x       = np.arange(len(results_df))
    width   = 0.25
    ax.bar(x - width, results_df["F1"],        width, label="F1",        color="#3b82f6")
    ax.bar(x,          results_df["Precision"], width, label="Precision", color="#22c55e")
    ax.bar(x + width,  results_df["Recall"],   width, label="Recall",    color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["stage"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Classifier vs Classifier + RAG")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = RESULTS_DIR / filename
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Saved {path}")

    results_df.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)
    return results_df
