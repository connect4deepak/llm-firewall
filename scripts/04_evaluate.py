"""
scripts/04_evaluate.py
Step 4 — Evaluate classifier + RAG pipeline, produce ablation plots
"""
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier   import FirewallClassifier
from src.rag_pipeline import RAGPipeline
from src.evaluator    import (plot_confusion_matrix, plot_class_distribution,
                               run_ablation, compute_metrics)
from config import SAVED_MODEL_DIR, PROCESSED_DIR, LABEL_NAMES

if __name__ == "__main__":
    print("📦 Loading models...")
    clf = FirewallClassifier(SAVED_MODEL_DIR)
    rag = RAGPipeline()

    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    print(f"Test set: {len(test)} samples\n")

    # 1. Classifier-only evaluation
    print("📊 Evaluating classifier...")
    eval_result = clf.evaluate(test)
    y_true = eval_result["y_true"]
    y_pred = eval_result["y_pred"]

    print("\nClassification Report:")
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred,
          target_names=[LABEL_NAMES[i] for i in range(4)]))

    plot_confusion_matrix(y_true, y_pred,
        title="DistilBERT Classifier — Test Set",
        filename="confusion_matrix_classifier.png")

    plot_class_distribution(test, filename="class_distribution.png")

    # 2. Ablation study
    ablation_df = run_ablation(test, clf, rag)

    print("\n✅ Evaluation complete. Results saved to results/")
    print("   Open results/*.png to view plots.")
