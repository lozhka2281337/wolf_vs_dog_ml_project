import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)

import config as cfg
from data import get_test_generator


def plot_pretty_metrics(y_true, y_prob, class_names=("dog", "wolf")):
    y_pred = (y_prob >= 0.5).astype(int)

    # Ошибки
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    # Стили
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dog vs Wolf — Model Evaluation", fontsize=16, fontweight="bold")

    # ROC
    ax = axes[0, 0]
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    # PR curve
    ax = axes[0, 1]
    ax.plot(recall, precision, linewidth=2, label=f"AP = {ap:.3f}", color="tab:green")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()

    # Вероятности по классам
    ax = axes[1, 0]
    ax.hist(y_prob[y_true == 0], bins=20, alpha=0.7, label=class_names[0], color="tab:blue")
    ax.hist(y_prob[y_true == 1], bins=20, alpha=0.7, label=class_names[1], color="tab:orange")
    ax.axvline(0.5, linestyle="--", linewidth=1.5, color="black", label="threshold=0.5")
    ax.set_title("Predicted Probability Distribution (P(wolf))")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    ax.legend()

    # Ошибки
    ax = axes[1, 1]
    labels = ["TN", "FP", "FN", "TP"]
    values = [tn, fp, fn, tp]
    colors = ["#4CAF50", "#F44336", "#FF9800", "#2196F3"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Prediction Outcome Counts")
    ax.set_ylabel("Count")

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h, f"{int(h)}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


def main():
    test_data = get_test_generator(cfg.TEST_DIR)
    model = keras.models.load_model(cfg.MODEL_PATH)

    loss, acc = model.evaluate(test_data, verbose=1)
    print(f"Test loss: {loss:.4f}")
    print(f"Test acc : {acc:.4f}")

    test_data.reset()
    y_prob = model.predict(test_data, verbose=1).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = test_data.classes

    idx_to_class = {v: k for k, v in test_data.class_indices.items()}
    target_names = [idx_to_class[0], idx_to_class[1]]

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    plot_pretty_metrics(y_true, y_prob, class_names=target_names)


if __name__ == "__main__":
    main()