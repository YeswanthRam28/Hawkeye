# explainability/risk_bar.py
import matplotlib.pyplot as plt
import numpy as np

def risk_bar_chart(contributions, save_path):
    """
    contributions: dict-like with keys "vision","audio","motion" (values 0..1)
    Backward compatible: if contributions already in 0..100, function will detect and handle.
    """
    # Read values with defaults 0
    v = contributions.get("vision", 0)
    a = contributions.get("audio", 0)
    m = contributions.get("motion", 0)

    # If values likely in 0..1, scale to 0..100
    def _scale(x):
        if x <= 1.0:
            return x * 100
        return x
    values = np.array([_scale(v), _scale(a), _scale(m)])
    labels = ["Vision", "Audio", "Motion"]
    colors = ["#e63946", "#457b9d", "#2a9d8f"]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2)

    # value labels above bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.0f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylabel("Risk (%)", fontsize=12, fontweight="bold")
    ax.set_title("Risk Contribution", fontsize=14, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
