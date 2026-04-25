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
    # Baseline-from-run scores parsed from terminal output (before training).
    y = [0.9466, 0.6263, 0.6883, 0.4452, 0.2110]
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_title(
        "AgentHire Arena: Baselines-Before Training Collapse",
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
        label="Baselines-Before Training (Gemma-4-26B)",
    )
    ax.fill_between(x, y, 0, color="#ef4444", alpha=0.1)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Final Ground-Truth Score", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.2, len(x_labels) - 0.8)

    ax.annotate(
        "Constraint failure zone:\nAdversarial pressure + role constraints\ncut score sharply.",
        xy=(3, 0.4452),
        xytext=(2.15, 0.36),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=2),
        fontsize=10,
        color="#7f1d1d",
        ha="right",
        va="center",
    )

    ax.annotate(
        "Nightmare knapsack trap:\nBudget/role constraints force\na low terminal score.",
        xy=(4, 0.2110),
        xytext=(3.15, 0.24),
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
    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "baselines_before_training_gullibility_collapse_curve.png",
        ),
        dpi=150,
    )
    plt.close(fig)


def plot_analysis_paralysis() -> None:
    # Nightmare baseline trajectory from the latest inference terminal log.
    steps = list(range(1, 15))
    budget = [170, 170, 170, 160, 150, 120, 70, 70, 60, 60, 50, 40, 30, 30]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_title(
        "Nightmare Trajectory: Baselines-Before Training",
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
    ax.set_xlim(1, 14)
    ax.set_xticks(steps)

    ax.annotate(
        "Point of No Return:\nAgent can no longer\nafford to hire anyone.",
        xy=(11, 50),
        xytext=(7.2, 88),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#991b1b", lw=2),
        fontsize=10,
        color="#7f1d1d",
        ha="left",
        va="center",
    )

    ax.annotate(
        "Episode Ends:\nFinal score = 0.2110",
        xy=(14, 30),
        xytext=(10.4, 24),
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
    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "baselines_before_training_analysis_paralysis_trajectory.png",
        ),
        dpi=150,
    )
    plt.close(fig)


def main() -> None:
    plot_gullibility_collapse()
    plot_analysis_paralysis()
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
