import argparse
import csv
import os
import re
import subprocess
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

TASK_ORDER = ["easy", "medium", "hard", "adversarial", "nightmare"]


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_scores(output: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    # Parse from FINAL RESULTS block lines like:
    #   easy      0.7876  █████
    pattern = re.compile(r"^\s*(easy|medium|hard|adversarial|nightmare)\s+([0-9]*\.?[0-9]+)", re.IGNORECASE)
    for line in output.splitlines():
        m = pattern.match(line)
        if m:
            scores[m.group(1).lower()] = float(m.group(2))
    return scores


def _run_one_seed(repo_root: str, seed: int, model_name: str) -> Dict[str, float]:
    env = os.environ.copy()
    env["MODEL_SEED"] = str(seed)
    env["MODEL_NAME"] = model_name

    run_dir = os.path.join(repo_root, "logs", "seed_runs")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"run_seed_{seed}.txt")

    proc = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(output)

    if proc.returncode != 0:
        print(f"[WARN] Seed {seed} run exited with code {proc.returncode}. See {log_path}")

    scores = _parse_scores(output)
    missing = [t for t in TASK_ORDER if t not in scores]
    if missing:
        print(f"[WARN] Seed {seed} missing task scores: {missing}. Check {log_path}")
    return scores


def _save_csv(out_csv: str, rows: List[Dict[str, float]], seeds: List[int]) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed"] + TASK_ORDER + ["avg"])
        for seed, row in zip(seeds, rows):
            vals = [row.get(t, float("nan")) for t in TASK_ORDER]
            avg = float(np.nanmean(vals)) if vals else float("nan")
            writer.writerow([seed] + vals + [avg])


def _plot_curves(out_dir: str, rows: List[Dict[str, float]], seeds: List[int]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    mat = np.array([[r.get(t, np.nan) for t in TASK_ORDER] for r in rows], dtype=float)
    x = np.arange(len(TASK_ORDER))

    # Plot 1: per-seed task curves + mean
    plt.figure(figsize=(10, 6), dpi=150)
    for i, seed in enumerate(seeds):
        plt.plot(x, mat[i], marker="o", linewidth=2, label=f"seed={seed}")

    mean_vals = np.nanmean(mat, axis=0)
    std_vals = np.nanstd(mat, axis=0)
    plt.plot(x, mean_vals, marker="s", linewidth=3, color="black", label="mean")
    plt.fill_between(x, mean_vals - std_vals, mean_vals + std_vals, alpha=0.15, color="black", label="mean±std")

    plt.xticks(x, TASK_ORDER)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Final Score")
    plt.title("Gemma Seed Sweep (3 seeds) - Task Curves")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "seed_task_curves.png"))
    plt.close()

    # Plot 2: average score per seed
    seed_avgs = np.nanmean(mat, axis=1)
    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.bar([str(s) for s in seeds], seed_avgs)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Average Score Across Tasks")
    plt.xlabel("Seed")
    plt.title("Gemma Seed Sweep - Average Score by Seed")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "seed_avg_scores.png"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3-seed Gemma evaluation and generate curves.")
    parser.add_argument("--seeds", default="11,22,33", help="Comma-separated integer seeds.")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "google/gemma-4-31b-it"), help="Model name.")
    parser.add_argument("--out-dir", default=os.path.join("outputs", "figures", "seed_sweep"), help="Output directory for csv/plots.")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    repo_root = _repo_root()

    all_rows: List[Dict[str, float]] = []
    for seed in seeds:
        print(f"[INFO] Running seed={seed} model={args.model}")
        scores = _run_one_seed(repo_root=repo_root, seed=seed, model_name=args.model)
        all_rows.append(scores)

    out_dir_abs = os.path.join(repo_root, args.out_dir)
    csv_path = os.path.join(out_dir_abs, "seed_scores.csv")
    _save_csv(csv_path, all_rows, seeds)
    _plot_curves(out_dir_abs, all_rows, seeds)

    print("\n[INFO] Seed sweep completed.")
    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] Plot: {os.path.join(out_dir_abs, 'seed_task_curves.png')}")
    print(f"[INFO] Plot: {os.path.join(out_dir_abs, 'seed_avg_scores.png')}")

    # Console summary
    mat = np.array([[r.get(t, np.nan) for t in TASK_ORDER] for r in all_rows], dtype=float)
    mean_vals = np.nanmean(mat, axis=0)
    print("\n[SUMMARY] Mean by task:")
    for task, val in zip(TASK_ORDER, mean_vals):
        print(f"  {task:<12} {val:.4f}")
    print(f"  {'avg':<12} {np.nanmean(mean_vals):.4f}")


if __name__ == "__main__":
    main()
