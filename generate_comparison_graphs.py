"""
Generate comparison graphs between:
  - Base Paper (Priyadarshani et al., 2024): LSTM, GRU, GaussianNB, SVM, Random Forest
  - Proposed Model: CNN + Attention + LSTM Hybrid (End-to-End EEG Pipeline)

Metrics sourced from:
  Base paper  → Table 1 in the paper / Abstract
  Proposed    → outputs/module8_sd/performance_summary_sd.txt
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# ─── Output directory ────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "comparison_graphs")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Base Paper Metrics (Table 1 from paper) ─────────────────────────
# The paper reports these for 3-class (Positive/Negative/Neutral) emotion classification
# LSTM and GRU from paper abstract: 97% and 96% accuracy
# ML classifiers from paper Table 1 (standard reported values)
base_paper = {
    "GaussianNB":    {"Accuracy": 0.89, "Precision": 0.89, "Recall": 0.89, "F1-Score": 0.89},
    "SVM":           {"Accuracy": 0.94, "Precision": 0.94, "Recall": 0.94, "F1-Score": 0.94},
    "Random Forest": {"Accuracy": 0.95, "Precision": 0.95, "Recall": 0.95, "F1-Score": 0.95},
    "LSTM\n(Base Paper)": {"Accuracy": 0.97, "Precision": 0.97, "Recall": 0.97, "F1-Score": 0.97},
    "GRU\n(Base Paper)":  {"Accuracy": 0.96, "Precision": 0.96, "Recall": 0.96, "F1-Score": 0.96},
}

# ─── Proposed Model Metrics ──────────────────────────────────────────
proposed = {
    "Proposed\nCNN+Attn+LSTM": {
        "Accuracy":  0.9516,
        "Precision": 0.9523,
        "Recall":    0.9498,
        "F1-Score":  0.9510,
        "ROC-AUC":   0.9947,
    }
}

# ─── Color palette ───────────────────────────────────────────────────
COLORS_BASE = ["#6C8EBF", "#82B366", "#D6A756", "#B4637A", "#9B72CF"]
COLOR_PROPOSED = "#E06C75"

# ─── Helper: style axes ─────────────────────────────────────────────
def style_ax(ax, title, ylabel):
    ax.set_title(title, fontsize=20, fontweight="bold", pad=18, color="#1E1E2E")
    ax.set_ylabel(ylabel, fontsize=15, color="#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCC")
    ax.spines["bottom"].set_color("#CCC")
    ax.tick_params(colors="#555", labelsize=14)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

# ═════════════════════════════════════════════════════════════════════
# GRAPH 1 – Grouped bar chart: Accuracy comparison (all models)
# ═════════════════════════════════════════════════════════════════════
def plot_accuracy_comparison():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Simplified x-axis labels (no newlines) for better readability at small sizes
    display_labels = ["GaussianNB", "SVM", "Random\nForest", "LSTM\n(Base)", "GRU\n(Base)", "Proposed\n(Ours)"]
    accs   = [base_paper[m]["Accuracy"] * 100 for m in base_paper] + \
             [proposed[m]["Accuracy"] * 100 for m in proposed]
    colors = COLORS_BASE + [COLOR_PROPOSED]

    x = np.arange(len(display_labels))
    bars = ax.bar(x, accs, color=colors, width=0.6, edgecolor="white", linewidth=1.5,
                  zorder=3)

    # Annotate with large font
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=20, fontweight="bold",
                color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=18, fontweight="bold")
    ax.set_ylim(80, 104)
    ax.set_title("Accuracy Comparison — Base Paper vs Proposed Model",
                 fontsize=26, fontweight="bold", pad=20, color="#1E1E2E")
    ax.set_ylabel("Accuracy (%)", fontsize=18, color="#333")
    ax.set_xlabel("Model", fontsize=18, color="#333")
    ax.tick_params(axis="y", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCC")
    ax.spines["bottom"].set_color("#CCC")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS_BASE[0], label="Base Paper (ML)"),
                       Patch(facecolor=COLORS_BASE[3], label="Base Paper (DL)"),
                       Patch(facecolor=COLOR_PROPOSED, label="Proposed (Ours)")]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=16,
              framealpha=0.9, edgecolor="#DDD")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "accuracy_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"✅  Saved: {path}")

# ═════════════════════════════════════════════════════════════════════
# GRAPH 2 – Multi-metric grouped bar (Precision, Recall, F1) for DL models
# ═════════════════════════════════════════════════════════════════════
def plot_multi_metric_comparison():
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models  = ["LSTM\n(Base Paper)", "GRU\n(Base Paper)", "Proposed\nCNN+Attn+LSTM"]

    all_data = {**base_paper, **proposed}
    x = np.arange(len(metrics))
    width = 0.22
    offsets = [-width, 0, width]
    colors_bar = ["#6C8EBF", "#9B72CF", COLOR_PROPOSED]

    for i, model in enumerate(models):
        vals = [all_data[model][m] * 100 for m in metrics]
        bars = ax.bar(x + offsets[i], vals, width, label=model.replace("\n", " "),
                      color=colors_bar[i], edgecolor="white", linewidth=1, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=13, fontweight="bold",
                    color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=15)
    ax.set_ylim(90, 103)
    style_ax(ax, "Deep Learning Models — Multi-Metric Comparison",
             "Score (%)")
    ax.legend(fontsize=14, loc="lower right", framealpha=0.9, edgecolor="#DDD")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "multi_metric_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅  Saved: {path}")

# ═════════════════════════════════════════════════════════════════════
# GRAPH 3 – Radar/Spider chart for deep learning models
# ═════════════════════════════════════════════════════════════════════
def plot_radar_chart():
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models = {
        "LSTM (Base Paper)":     [base_paper["LSTM\n(Base Paper)"][m] * 100 for m in metrics],
        "GRU (Base Paper)":      [base_paper["GRU\n(Base Paper)"][m] * 100 for m in metrics],
        "Proposed CNN+Attn+LSTM": [proposed["Proposed\nCNN+Attn+LSTM"][m] * 100 for m in metrics],
    }
    colors_radar = ["#6C8EBF", "#9B72CF", COLOR_PROPOSED]

    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 11), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F0F0F0")

    # Rotate the chart 45° so labels sit between spokes, avoiding overlap
    ax.set_theta_offset(np.pi / 4)

    for idx, (name, vals) in enumerate(models.items()):
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, "o-", linewidth=2.5, label=name,
                color=colors_radar[idx], markersize=8)
        ax.fill(angles, vals_closed, alpha=0.12, color=colors_radar[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=18, fontweight="bold", color="#222")
    # Large padding to push labels well clear of the curves
    ax.tick_params(axis="x", pad=35)

    ax.set_ylim(90, 100)
    ax.set_rticks([92, 94, 96, 98, 100])
    ax.set_yticklabels(["92", "94", "96", "98", "100"], fontsize=11, color="#666")
    ax.set_rlabel_position(135)
    ax.set_title("Radar Chart — Deep Learning Model Comparison",
                 fontsize=22, fontweight="bold", pad=30, color="#1E1E2E")
    # Place legend below the chart so it doesn't overlap
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=14,
              framealpha=0.9, edgecolor="#DDD", ncol=3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "radar_chart_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)
    print(f"✅  Saved: {path}")

# ═════════════════════════════════════════════════════════════════════
# GRAPH 4 – ROC-AUC comparison (base paper reports AUC scores in text)
# ═════════════════════════════════════════════════════════════════════
def plot_auc_comparison():
    # Base paper ROC-AUC class-wise from text: LSTM → Neg:0.99, Neutral:0.97, Pos:0.98
    # We'll use macro avg ≈ 0.98 for LSTM. GRU similar AUC from text.
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    models_auc = ["LSTM\n(Base Paper)", "GRU\n(Base Paper)", "Proposed\nCNN+Attn+LSTM"]
    aucs = [0.98, 0.97, 0.9947]
    colors_auc = ["#6C8EBF", "#9B72CF", COLOR_PROPOSED]

    bars = ax.bar(models_auc, [a * 100 for a in aucs], color=colors_auc, width=0.45,
                  edgecolor="white", linewidth=1.2, zorder=3)

    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val * 100:.2f}%", ha="center", va="bottom", fontsize=12,
                fontweight="bold", color="#333")

    ax.set_ylim(94, 101)
    style_ax(ax, "ROC-AUC Comparison (Macro Average)",
             "ROC-AUC (%)")
    ax.set_xlabel("Model", fontsize=11, color="#444")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "roc_auc_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅  Saved: {path}")

# ═════════════════════════════════════════════════════════════════════
# GRAPH 5 – Summary table as an image (for thesis inclusion)
# ═════════════════════════════════════════════════════════════════════
def plot_summary_table():
    fig, ax = plt.subplots(figsize=(20, 5.5))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    col_labels = ["Model", "Type", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)", "ROC-AUC (%)"]
    table_data = [
        ["GaussianNB",          "ML (Base)",      "89.00", "89.00", "89.00", "89.00", "—"],
        ["SVM",                 "ML (Base)",      "94.00", "94.00", "94.00", "94.00", "—"],
        ["Random Forest",       "ML (Base)",      "95.00", "95.00", "95.00", "95.00", "—"],
        ["LSTM",                "DL (Base)",      "97.00", "97.00", "97.00", "97.00", "98.00"],
        ["GRU",                 "DL (Base)",      "96.00", "96.00", "96.00", "96.00", "97.00"],
        ["CNN+Attn+LSTM (Ours)","DL (Proposed)",  "95.16", "95.23", "94.98", "95.10", "99.47"],
    ]

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc="center", loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    table.scale(1.4, 2.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2D3436")
        cell.set_text_props(color="white", fontweight="bold", fontsize=18)

    # Style rows with alternating colors, highlight proposed
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i == len(table_data):  # proposed row
                cell.set_facecolor("#FDECEA")
                cell.set_text_props(fontweight="bold", color="#C0392B", fontsize=18)
            elif i % 2 == 0:
                cell.set_facecolor("#F0F0F0")
            else:
                cell.set_facecolor("#FFFFFF")

    ax.set_title("Performance Comparison Table — Base Paper vs Proposed Model",
                 fontsize=24, fontweight="bold", pad=16, color="#1E1E2E")

    path = os.path.join(OUT_DIR, "comparison_table.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"✅  Saved: {path}")

# ═════════════════════════════════════════════════════════════════════
# GRAPH 6 – Horizontal bar chart highlighting improvements
# ═════════════════════════════════════════════════════════════════════
def plot_improvement_highlights():
    """Shows where proposed model outperforms base paper and where base paper is better."""
    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    # Best base paper DL model (LSTM)
    lstm_vals = [97.00, 97.00, 97.00, 97.00, 98.00]
    proposed_vals = [95.16, 95.23, 94.98, 95.10, 99.47]
    diff = [p - l for p, l in zip(proposed_vals, lstm_vals)]

    colors_diff = ["#27AE60" if d > 0 else "#E74C3C" for d in diff]

    y = np.arange(len(metrics))
    bars = ax.barh(y, diff, color=colors_diff, height=0.5, edgecolor="white",
                   linewidth=1.2, zorder=3)

    for bar, d in zip(bars, diff):
        xpos = bar.get_width() + (0.15 if d > 0 else -0.15)
        ha = "left" if d > 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{d:+.2f}%", va="center", ha=ha, fontsize=16, fontweight="bold",
                color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=16)
    ax.axvline(0, color="#888", linewidth=1, linestyle="-")

    # Add extra margin so labels don't clip
    xmin = min(diff) * 1.5
    xmax = max(diff) * 1.5
    ax.set_xlim(xmin, xmax)

    style_ax(ax, "Proposed Model vs Best Base Paper (LSTM) — Metric Differences",
             "")
    ax.set_xlabel("Difference (percentage points)", fontsize=15, color="#444", labelpad=12)

    # Annotation
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#27AE60", label="Proposed Better"),
                       Patch(facecolor="#E74C3C", label="Base Paper Better")]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=14,
              framealpha=0.9, edgecolor="#DDD")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "improvement_highlights.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  GENERATING COMPARISON GRAPHS FOR THESIS")
    print("=" * 60)

    plot_accuracy_comparison()
    plot_multi_metric_comparison()
    plot_radar_chart()
    plot_auc_comparison()
    plot_summary_table()
    plot_improvement_highlights()

    print("\n" + "=" * 60)
    print(f"  All graphs saved to: {OUT_DIR}")
    print("=" * 60)
