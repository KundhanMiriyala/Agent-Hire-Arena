import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme(style="darkgrid")

OUTPUT_DIR = os.path.join("outputs", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_gullibility_collapse() -> None:
    x_labels = [
        "Easy\n(Honest)",
        "Medium\n(Noisy)",
        "Hard\n(Coached Decoys)",
        "Adversarial\n(+ Angry Boss)",
        "Nightmare\n(Max Pressure)",
    ]
    y = [0.9576, 0.9163, 0.6938, 0.5715, 0.0000]
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_title(
        "AgentHire Arena: The Baseline Sycophancy Collapse",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    ax.plot(
        x,
        y,
        color="#ef4444",
        linewidth=4,
        marker="o",
        markersize=10,
        label="Baseline AI (GPT-4o / Gemma)",
    )
    ax.fill_between(x, y, 0, color="#ef4444", alpha=0.1)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Final Ground-Truth Score", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.2, len(x_labels) - 0.8)

    ax.annotate(
        "Sycophancy Trap:\nAI caves to NPC pressure\nand hires decoys.",
        xy=(3, 0.5715),
        xytext=(2.1, 0.32),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=2),
        fontsize=10,
        color="#7f1d1d",
        ha="right",
        va="center",
    )

    ax.annotate(
        "Analysis Paralysis:\nAI panics, burns budget\non interviews, scores 0.",
        xy=(4, 0.0),
        xytext=(3.1, 0.18),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=2),
        fontsize=10,
        color="#7f1d1d",
        ha="right",
        va="center",
    )

    ax.legend(loc="upper right", frameon=True, shadow=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "gullibility_collapse_curve.png"), dpi=150)
    plt.close(fig)


def plot_analysis_paralysis() -> None:
    steps = list(range(1, 20))
    budget = [180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_title(
        "Nightmare Trajectory: AI Analysis Paralysis",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    ax.fill_between(steps, 0, 50, color="#ef4444", alpha=0.1, label="Bankruptcy Zone")
    ax.axhline(50, color="#ef4444", linestyle="--", linewidth=2, label="Minimum Hire Cost (50 Units)")
    ax.plot(
        steps,
        budget,
        color="#3b82f6",
        linewidth=3,
        marker="s",
        markersize=7,
        label="Agent Budget Remaining",
    )

    ax.set_xlabel("Episode Step Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("Budget Remaining (Units)", fontsize=12, fontweight="bold")
    ax.set_ylim(-5, 190)
    ax.set_xlim(1, 19)
    ax.set_xticks(steps)

    ax.annotate(
        "Point of No Return:\nAgent can no longer\nafford to hire anyone.",
        xy=(14, 50),
        xytext=(8, 80),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#991b1b", lw=2),
        fontsize=10,
        color="#7f1d1d",
        ha="left",
        va="center",
    )

    ax.annotate(
        "Episode Ends:\nBankrupt (Score: 0)",
        xy=(19, 0),
        xytext=(13.2, 20),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#1e3a8a", lw=2),
        fontsize=10,
        color="#1e3a8a",
        ha="left",
        va="center",
    )

    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "analysis_paralysis_trajectory.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    plot_gullibility_collapse()
    plot_analysis_paralysis()
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
