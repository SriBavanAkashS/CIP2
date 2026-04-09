import os
from datetime import datetime
import glob

import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        roc_curve,
    )
except Exception as e:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for Module 8. Install it (e.g., `pip install scikit-learn`) and retry."
    ) from e

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
except Exception as e:  # pragma: no cover
    raise ImportError(
        "TensorFlow/Keras is required for Module 8. "
        "Install it (e.g., `pip install tensorflow`) and retry."
    ) from e

from src.module6_classification import (
    prepare_end_to_end_dataset_with_subjects,
    build_end_to_end_model,
    prepare_dataset,
    build_module6_classifier,
)
from src.module7_explainability import predict_and_explain_end_to_end


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "module8_performance")

CLASS_NAMES = {0: "low valence", 1: "medium valence", 2: "high valence"}


def _load_best_weights_path(weights_dir=None):
    """
    Find the end-to-end .keras model file and extract weights from it.
    Returns the path to a temporary weights file extracted from the .keras archive.
    """
    if weights_dir is None:
        weights_dir = os.path.join(PROJECT_ROOT, "outputs", "module6_classification")
    keras_path = os.path.join(weights_dir, "end_to_end_model.keras")
    if not os.path.exists(keras_path):
        raise FileNotFoundError(
            f"End-to-end model not found at: {keras_path}. "
            "Train Module 6 in e2e mode first."
        )
    return keras_path


def _build_and_load_model(seg, channels, window, num_classes=3, weights_path=None):
    """
    Rebuild the end-to-end model and load trained weights from .keras archive.
    """
    if weights_path is None:
        weights_path = _load_best_weights_path()
    model, _ = build_end_to_end_model(seg=seg, channels=channels, window=window, num_classes=num_classes)

    # Extract weights from the .keras archive (zip format)
    import zipfile, tempfile, shutil
    tmpdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(weights_path, 'r') as zf:
            zf.extractall(tmpdir)
        h5_path = os.path.join(tmpdir, "model.weights.h5")
        if os.path.exists(h5_path):
            model.load_weights(h5_path)
        else:
            raise FileNotFoundError("Could not find model.weights.h5 inside the .keras archive.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return model, weights_path


def compute_metrics(y_true, y_pred, y_proba):
    """
    Implements AlgoModule8:
      - confusion matrix
      - accuracy
      - precision
      - recall
      - F1-score
      - ROC-AUC (macro, one-vs-rest)
    """
    labels = sorted(np.unique(y_true).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    # ROC-AUC: one-vs-rest macro
    num_classes = y_proba.shape[1]
    y_true_ovr = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)
    try:
        roc_auc = roc_auc_score(y_true_ovr, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    return {
        "confusion_matrix": cm,
        "labels": labels,
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "roc_auc_macro": roc_auc,
    }


def plot_confusion_matrix(cm, labels, output_dir=OUTPUT_DIR, normalize=True):
    """
    Confusion Matrix Viewer (Module 8 diagram).
    """
    os.makedirs(output_dir, exist_ok=True)
    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        cm_norm = cm / row_sums
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proportion")

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([CLASS_NAMES.get(int(l), str(l)) for l in labels], rotation=45, ha="right")
    ax.set_yticklabels([CLASS_NAMES.get(int(l), str(l)) for l in labels])
    ax.set_ylabel("True class")
    ax.set_xlabel("Predicted class")
    ax.set_title("Module 8: Confusion Matrix (Counts & %)")

    # Print values in cells (Raw Count as primary, with optional normalization for color)
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            # Main text: Raw trial count (integer)
            val = int(cm[i, j])
            # Secondary text: Percentage (optional, but makes it look premium)
            pct = cm_norm[i, j] * 100
            
            ax.text(
                j,
                i,
                f"{val}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=9,
                fontweight='bold'
            )

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix_module8.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def plot_roc_curves(y_true, y_proba, labels, output_dir=OUTPUT_DIR):
    """
    ROC–AUC Analysis Viewer (Module 8 diagram).
    """
    os.makedirs(output_dir, exist_ok=True)
    num_classes = y_proba.shape[1]
    y_true_ovr = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, lab in enumerate(labels):
        try:
            fpr, tpr, _ = roc_curve(y_true_ovr[:, i], y_proba[:, i])
            auc_val = roc_auc_score(y_true_ovr[:, i], y_proba[:, i])
        except ValueError:
            continue
        ax.plot(fpr, tpr, linewidth=2, label=f"{CLASS_NAMES.get(int(lab), lab)} (AUC = {auc_val:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Module 8: ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves_module8.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def plot_class_confidence_dashboard(y_true, y_proba, output_dir=OUTPUT_DIR):
    """
    Class Confidence Dashboard (Module 8 diagram).
    Shows mean predicted probability per class and distribution of confidences.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_classes = y_proba.shape[1]

    mean_conf = y_proba.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Mean confidence per class
    x = np.arange(num_classes)
    axes[0].bar(x, mean_conf, color="steelblue", alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([CLASS_NAMES.get(i, str(i)) for i in range(num_classes)], rotation=20)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Mean predicted probability")
    axes[0].set_title("Mean Class Confidence (all evaluated trials)")
    axes[0].grid(True, axis="y", alpha=0.25)

    # Confidence for true class per trial
    true_conf = y_proba[np.arange(len(y_true)), y_true]
    axes[1].hist(true_conf, bins=15, range=(0.0, 1.0), color="darkorange", alpha=0.85)
    axes[1].set_xlabel("Predicted probability of true class")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of Confidence for True Class")
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(output_dir, "class_confidence_dashboard.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def evaluate_module8(test_subject=None, output_dir=OUTPUT_DIR, weights_path=None):
    """
    High-level entrypoint for Module 8 (Performance Evaluation).

    If test_subject is provided (0-based), only that subject's trials are evaluated.
    Otherwise, all trials are used.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect DE mode
    de_features_path = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial", "spatial_features.npy")
    _use_de = os.path.exists(de_features_path)

    X, y, subj_idx, trial_idx, (Seg, C, W) = prepare_end_to_end_dataset_with_subjects(use_de=_use_de)

    if test_subject is not None:
        if test_subject < 0 or test_subject > int(subj_idx.max()):
            raise ValueError(
                f"test_subject={test_subject} is out of range. "
                f"Valid range is [0, {int(subj_idx.max())}]."
            )
        mask = subj_idx == test_subject
        X_eval = X[mask]
        y_eval = y[mask]
    else:
        X_eval = X
        y_eval = y

    model, used_weights = _build_and_load_model(Seg, C, W, num_classes=3, weights_path=weights_path)

    y_proba = model.predict(X_eval, batch_size=32, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    metrics = compute_metrics(y_eval, y_pred, y_proba)

    cm_path = plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"], output_dir=output_dir)
    roc_path = plot_roc_curves(y_eval, y_proba, metrics["labels"], output_dir=output_dir)
    dashboard_path = plot_class_confidence_dashboard(y_eval, y_proba, output_dir=output_dir)

    # Save textual summary
    summary_path = os.path.join(output_dir, "performance_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MODULE 8: PERFORMANCE EVALUATION - SUMMARY\n")
        f.write("=" * 56 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Evaluated trials: {len(y_eval)}\n")
        f.write(f"Used weights: {used_weights}\n")
        if test_subject is not None:
            f.write(f"Held-out subject index: {test_subject}\n")
        else:
            f.write("Evaluation set: all subjects and trials\n")
        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(str(metrics["confusion_matrix"]) + "\n\n")
        f.write(f"Accuracy:       {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (macro): {metrics['precision_macro']:.4f}\n")
        f.write(f"Recall (macro):    {metrics['recall_macro']:.4f}\n")
        f.write(f"F1-score (macro):  {metrics['f1_macro']:.4f}\n")
        f.write(f"ROC-AUC (macro, OVR): {metrics['roc_auc_macro']:.4f}\n")
        f.write("\nArtifacts:\n")
        f.write(f"  - Confusion matrix: {cm_path}\n")
        f.write(f"  - ROC curves:       {roc_path}\n")
        f.write(f"  - Confidence dashboard: {dashboard_path}\n")

    return {
        "metrics": metrics,
        "summary_path": summary_path,
        "confusion_matrix_path": cm_path,
        "roc_curves_path": roc_path,
        "confidence_dashboard_path": dashboard_path,
    }


def _detect_test_subject():
    """
    Auto-detect the held-out LOSO test subject from the Module 6 summary file.
    Returns the test subject index (int) or None if not found.
    """
    summary_path = os.path.join(PROJECT_ROOT, "outputs", "module6_classification", "summary_end_to_end.txt")
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path, "r") as f:
            for line in f:
                if "test_subject=" in line:
                    # e.g. "Split: subject-wise hold-out, test_subject=8"
                    part = line.split("test_subject=")[1].strip()
                    return int(part)
    except Exception:
        pass
    return None


def evaluate_held_out_subject(output_dir=OUTPUT_DIR, weights_path=None):
    """
    Evaluate ONLY the held-out LOSO test subject (the one left out during training).
    Generates a separate confusion matrix, ROC curves, and summary specifically
    for the unseen test subject — giving an honest generalization benchmark.
    """
    test_subject = _detect_test_subject()
    if test_subject is None:
        print("  [SKIP] Could not auto-detect the held-out test subject from Module 6 summary.")
        return None

    print(f"\n  -> Evaluating held-out LOSO test subject: {test_subject}")

    # Create a subdirectory for test-subject-specific outputs
    test_dir = os.path.join(output_dir, f"test_subject_{test_subject}")
    os.makedirs(test_dir, exist_ok=True)

    # Auto-detect DE mode
    de_features_path = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial", "spatial_features.npy")
    _use_de = os.path.exists(de_features_path)

    X, y, subj_idx, trial_idx, (Seg, C, W) = prepare_end_to_end_dataset_with_subjects(use_de=_use_de)

    # Isolate ONLY the test subject's trials
    mask = subj_idx == test_subject
    X_test = X[mask]
    y_test = y[mask]
    print(f"     Test trials: {len(y_test)}")

    model, used_weights = _build_and_load_model(Seg, C, W, num_classes=3, weights_path=weights_path)

    y_proba = model.predict(X_test, batch_size=32, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    metrics = compute_metrics(y_test, y_pred, y_proba)

    # Generate test-subject-specific plots
    cm_path = plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"], output_dir=test_dir)
    roc_path = plot_roc_curves(y_test, y_proba, metrics["labels"], output_dir=test_dir)
    dashboard_path = plot_class_confidence_dashboard(y_test, y_proba, output_dir=test_dir)

    # Save summary
    summary_path = os.path.join(test_dir, "test_subject_performance.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"MODULE 8: HELD-OUT TEST SUBJECT {test_subject} - PERFORMANCE\n")
        f.write("=" * 56 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test subject (LOSO held-out): {test_subject}\n")
        f.write(f"Evaluated trials: {len(y_test)}\n")
        f.write(f"Used weights: {used_weights}\n")
        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(str(metrics["confusion_matrix"]) + "\n\n")
        f.write(f"Accuracy:          {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (macro): {metrics['precision_macro']:.4f}\n")
        f.write(f"Recall (macro):    {metrics['recall_macro']:.4f}\n")
        f.write(f"F1-score (macro):  {metrics['f1_macro']:.4f}\n")
        f.write(f"ROC-AUC (macro):   {metrics['roc_auc_macro']:.4f}\n")
        f.write("\nArtifacts:\n")
        f.write(f"  - Confusion matrix: {cm_path}\n")
        f.write(f"  - ROC curves:       {roc_path}\n")
        f.write(f"  - Confidence dashboard: {dashboard_path}\n")

    print(f"     Accuracy:  {metrics['accuracy']:.4f}")
    print(f"     F1-score:  {metrics['f1_macro']:.4f}")
    print(f"     ROC-AUC:   {metrics['roc_auc_macro']:.4f}")

    return {
        "test_subject": test_subject,
        "metrics": metrics,
        "summary_path": summary_path,
        "confusion_matrix_path": cm_path,
        "roc_curves_path": roc_path,
        "confidence_dashboard_path": dashboard_path,
    }



def build_module8_dashboard(subject_index, trial_index, weights_path=None, output_dir=OUTPUT_DIR, use_sd_model=False, sd_eval_result=None):
    """
    Generate a comprehensive Module 8 performance dashboard for a single trial.
    """
    import tensorflow as tf
    tf.keras.backend.clear_session()
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect DE mode
    de_features_path = os.path.join(PROJECT_ROOT, "outputs", "module3_spatial", "spatial_features.npy")
    _use_de = os.path.exists(de_features_path)

    # 1) Global performance (all subjects) – reuse evaluate_module8
    if use_sd_model and sd_eval_result:
        eval_result = sd_eval_result
    else:
        eval_result = evaluate_module8(test_subject=None, output_dir=output_dir, weights_path=weights_path)

    # 2) Per-trial explanation from Module 7
    X, y, subj_idx, trial_idx, (Seg, C, W) = prepare_end_to_end_dataset_with_subjects(use_de=_use_de)
    mask = (subj_idx == subject_index) & (trial_idx == trial_index)
    matches = np.where(mask)[0]
    if matches.size == 0:
        raise ValueError(f"No trial found for subject {subject_index} with trial index {trial_index}.")
    i = int(matches[0])
    one_sequence = X[i]  # (Seg, C, W, 1)
    explain_result = predict_and_explain_end_to_end(
        one_sequence, 
        use_sd_model=use_sd_model
    )

    elec_imp = explain_result["electrode_importance"]
    time_sal = explain_result["temporal_saliency"]
    pred_class = explain_result["predicted_class"]
    pred_name = explain_result.get("class_name", str(pred_class))

    # 3) Build a combined dashboard figure
    fig = plt.figure(figsize=(16, 10))

    # Top-left: confusion matrix image
    cm_img = plt.imread(eval_result["confusion_matrix_path"])
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(cm_img)
    ax1.axis("off")
    ax1.set_title("Global Confusion Matrix")

    # Top-right: ROC curves image
    roc_img = plt.imread(eval_result["roc_curves_path"])
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(roc_img)
    ax2.axis("off")
    ax2.set_title("ROC Curves")

    # Bottom: per-trial electrode importance
    ax3 = fig.add_subplot(2, 1, 2)
    x_e = np.arange(len(elec_imp))
    ax3.bar(x_e, elec_imp, color="steelblue", alpha=0.9)
    ax3.set_xlabel("EEG Electrode Index (0–31)")
    ax3.set_ylabel("|∂y/∂x| (per electrode)")
    ax3.set_title(
        f"Per-trial Electrode Importance\n"
        f"Subject {subject_index}, Trial {trial_index}, Predicted: {pred_name} ({pred_class})"
    )
    ax3.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    dashboard_path = os.path.join(output_dir, f"module8_dashboard_subject{subject_index}_trial{trial_index}.png")
    fig.savefig(dashboard_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    tf.keras.backend.clear_session()
    return {
        "dashboard_path": dashboard_path,
        "eval_result": eval_result,
        "explain_result": explain_result,
    }



def evaluate_sd(output_dir=None, weights_dir=None):
    """
    Subject-Dependent evaluation: load the SD-trained model,
    pool all trials, do a random 80/20 stratified split (same seed as training),
    and evaluate only the 20% test portion.
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "outputs", "module8_sd")
    if weights_dir is None:
        weights_dir = os.path.join(PROJECT_ROOT, "outputs", "module6_sd")
    os.makedirs(output_dir, exist_ok=True)

    X, y, subj_idx = prepare_dataset()

    # Reproduce the SAME 80/20 split used during training (random_state=42)
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)

    X_test = X[test_idx]
    y_test = y[test_idx]
    print(f"  Subject-Dependent test set: {len(y_test)} trials")

    # Load scaler and transform test data
    scaler_path = os.path.join(weights_dir, "scaler_params.npz")
    with np.load(scaler_path) as data:
        mean_arr = data["mean"]
        scale_arr = data["scale"]
    X_test = (X_test - mean_arr) / scale_arr

    weights_path = os.path.join(weights_dir, "module6_classifier_weights.h5")
    
    # Build the lightweight DNN and load weights
    model = build_module6_classifier(input_dim=X_test.shape[1], num_classes=3)
    model.load_weights(weights_path)
    used_weights = weights_path

    y_proba = model.predict(X_test, batch_size=32, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    metrics = compute_metrics(y_test, y_pred, y_proba)

    cm_path = plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"], output_dir=output_dir)
    roc_path = plot_roc_curves(y_test, y_proba, metrics["labels"], output_dir=output_dir)
    dashboard_path = plot_class_confidence_dashboard(y_test, y_proba, output_dir=output_dir)

    summary_path = os.path.join(output_dir, "performance_summary_sd.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MODULE 8: SUBJECT-DEPENDENT PERFORMANCE EVALUATION\n")
        f.write("=" * 56 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Split: 80/20 random stratified (subject-dependent)\n")
        f.write(f"Test trials: {len(y_test)}\n")
        f.write(f"Used weights: {used_weights}\n")
        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(str(metrics["confusion_matrix"]) + "\n\n")
        f.write(f"Accuracy:          {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (macro): {metrics['precision_macro']:.4f}\n")
        f.write(f"Recall (macro):    {metrics['recall_macro']:.4f}\n")
        f.write(f"F1-score (macro):  {metrics['f1_macro']:.4f}\n")
        f.write(f"ROC-AUC (macro):   {metrics['roc_auc_macro']:.4f}\n")
        f.write("\nArtifacts:\n")
        f.write(f"  - Confusion matrix: {cm_path}\n")
        f.write(f"  - ROC curves:       {roc_path}\n")
        f.write(f"  - Confidence dashboard: {dashboard_path}\n")

    print(f"\n  Subject-Dependent Results:")
    print(f"     Accuracy:  {metrics['accuracy']:.4f}")
    print(f"     F1-score:  {metrics['f1_macro']:.4f}")
    print(f"     ROC-AUC:   {metrics['roc_auc_macro']:.4f}")

    return {
        "metrics": metrics,
        "summary_path": summary_path,
        "confusion_matrix_path": cm_path,
        "roc_curves_path": roc_path,
        "confidence_dashboard_path": dashboard_path,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Module 8: Performance Evaluation of EEG Emotion Recognition Model"
    )
    parser.add_argument(
        "--test_subject",
        type=int,
        default=-1,
        help=(
            "If provided (0-based), evaluate only this subject's trials. "
            "If -1, evaluate all trials."
        ),
    )
    parser.add_argument(
        "--dashboard_subject",
        type=int,
        default=-1,
        help="If >=0, also build a Module 8 dashboard using this subject index.",
    )
    parser.add_argument(
        "--mode",
        choices=["loso", "sd"],
        default="loso",
        help="Evaluation mode: 'loso' (default, subject-independent) or 'sd' (subject-dependent 80/20 split)",
    )
    parser.add_argument(
        "--dashboard_trial",
        type=int,
        default=0,
        help="Trial index (0-based) to visualize in the Module 8 dashboard.",
    )
    args = parser.parse_args()

    if args.mode == "sd":
        # =============================================
        # SUBJECT-DEPENDENT EVALUATION
        # =============================================
        print("\n[Subject-Dependent Mode] Evaluating with 80/20 random stratified split...")
        SD_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "module8_sd")
        sd_result = evaluate_sd(output_dir=SD_OUTPUT_DIR)
        print("\nSaved Module 8 SD performance outputs:")
        print(f"  - {sd_result['summary_path']}")
        print(f"  - {sd_result['confusion_matrix_path']}")
        print(f"  - {sd_result['roc_curves_path']}")
        print(f"  - {sd_result['confidence_dashboard_path']}")
        
        if args.dashboard_subject >= 0:
            print(f"\nBuilding comprehensive SD dashboard for subject {args.dashboard_subject}, trial {args.dashboard_trial}...")
            dash = build_module8_dashboard(
                subject_index=args.dashboard_subject,
                trial_index=args.dashboard_trial,
                weights_path=None,
                output_dir=SD_OUTPUT_DIR,
                use_sd_model=True,
                sd_eval_result=sd_result
            )
            print(f"\nSD Dashboard saved to: {dash['dashboard_path']}")
            
        print("\nModule 8 completed (subject-dependent mode).")
    else:
        ts = args.test_subject if args.test_subject >= 0 else None
        print("\n" + "=" * 60)
        print("MODULE 8: PERFORMANCE EVALUATION")
        print("=" * 60)

        result = evaluate_module8(test_subject=ts, output_dir=OUTPUT_DIR)

        print("\nSaved Module 8 performance outputs:")
        print(f"  - {result['summary_path']}")
        print(f"  - {result['confusion_matrix_path']}")
        print(f"  - {result['roc_curves_path']}")
        print(f"  - {result['confidence_dashboard_path']}")

        # Automatically evaluate the held-out LOSO test subject
        held_out_result = evaluate_held_out_subject(output_dir=OUTPUT_DIR)
        if held_out_result is not None:
            print(f"\n  Held-out test subject outputs:")
            print(f"  - {held_out_result['summary_path']}")
            print(f"  - {held_out_result['confusion_matrix_path']}")
            print(f"  - {held_out_result['roc_curves_path']}")

        # Decide which subject to use for the dashboard.
        # If user passed --dashboard_subject, use that.
        # Otherwise, if --test_subject was given, use it automatically.
        dash_subject = args.dashboard_subject
        if dash_subject < 0 and ts is not None:
            dash_subject = ts

        if dash_subject is not None and dash_subject >= 0:
            dash = build_module8_dashboard(
                subject_index=dash_subject,
                trial_index=args.dashboard_trial,
                output_dir=OUTPUT_DIR,
            )
            print(f"\nDashboard saved to: {dash['dashboard_path']}")

        print("\nModule 8 completed.")
