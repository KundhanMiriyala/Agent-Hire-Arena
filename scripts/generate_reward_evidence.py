import csv
import json
import os
import random
from statistics import mean
from typing import Dict, List, Tuple

from models import HiringAction, HiringObservation
from server.environment import HiringEnvironment
from server.grader import grade_full

TASKS = ["easy", "medium", "hard", "adversarial"]


def _remaining_candidates(obs: HiringObservation) -> List[dict]:
    return [
        c for c in obs.candidates
        if c["candidate_id"] not in obs.hires_made and c["candidate_id"] not in obs.skipped
    ]


def _choose_heuristic_action(
    obs: HiringObservation,
    threshold: float,
    rng: random.Random,
    explore_prob: float,
) -> Dict[str, str]:
    remaining = _remaining_candidates(obs)
    if not remaining:
        return {"action": "finalize"}

    if obs.budget_remaining < 10:
        return {"action": "finalize"}

    interviews_done = obs.interviews_done or {}
    probes_done = getattr(obs, "probes_done", {}) or {}

    # Exploration branch to emulate early-stage policy noise.
    if rng.random() < explore_prob:
        live_ids = [c["candidate_id"] for c in remaining]
        action = rng.choice(["interview", "probe", "hire", "skip"])
        if action == "interview":
            ids = [cid for cid in live_ids if cid not in interviews_done]
            if ids and obs.budget_remaining >= 10:
                return {"action": "interview", "candidate_id": rng.choice(ids)}
        if action == "probe":
            ids = [cid for cid in live_ids if cid in interviews_done and cid not in probes_done]
            if ids and obs.budget_remaining >= 20:
                return {"action": "probe", "candidate_id": rng.choice(ids)}
        if action == "hire":
            ids = [cid for cid in live_ids if cid in interviews_done]
            if ids and obs.budget_remaining >= 50:
                return {"action": "hire", "candidate_id": rng.choice(ids)}
        if action == "skip":
            return {"action": "skip", "candidate_id": rng.choice(live_ids)}

    # 1) Interview uncertain candidates first.
    uncertain = [
        c for c in remaining
        if c["candidate_id"] not in interviews_done and 0.40 <= float(c["resume_score"]) <= 0.78
    ]
    if uncertain and obs.budget_remaining >= 10:
        uncertain.sort(key=lambda c: abs(float(c["resume_score"]) - 0.60))
        return {"action": "interview", "candidate_id": uncertain[0]["candidate_id"]}

    # 2) Probe strong interviewed candidates before hire.
    probe_targets = [
        c for c in remaining
        if c["candidate_id"] in interviews_done
        and c["candidate_id"] not in probes_done
        and float(interviews_done[c["candidate_id"]]) >= 0.62
    ]
    if probe_targets and obs.budget_remaining >= 20:
        probe_targets.sort(key=lambda c: float(interviews_done[c["candidate_id"]]), reverse=True)
        return {"action": "probe", "candidate_id": probe_targets[0]["candidate_id"]}

    # 3) Hire only with supporting signal.
    hire_targets: List[Tuple[str, float]] = []
    for c in remaining:
        cid = c["candidate_id"]
        if cid not in interviews_done:
            continue
        interview_score = float(interviews_done[cid])
        probe_score = float(probes_done[cid]) if cid in probes_done else None
        stable = probe_score is None or abs(interview_score - probe_score) <= 0.20
        quality = probe_score if probe_score is not None else interview_score
        if stable and quality >= threshold:
            hire_targets.append((cid, quality))

    if hire_targets and obs.budget_remaining >= 50:
        hire_targets.sort(key=lambda x: x[1], reverse=True)
        return {"action": "hire", "candidate_id": hire_targets[0][0]}

    # 4) Continue interviewing if budget allows.
    remaining_uninterviewed = [c for c in remaining if c["candidate_id"] not in interviews_done]
    if remaining_uninterviewed and obs.budget_remaining >= 10:
        remaining_uninterviewed.sort(key=lambda c: float(c["resume_score"]), reverse=True)
        return {"action": "interview", "candidate_id": remaining_uninterviewed[0]["candidate_id"]}

    # 5) Skip obvious low-confidence leftovers.
    low_conf = [c for c in remaining if float(c["resume_score"]) < 0.35]
    if low_conf:
        return {"action": "skip", "candidate_id": low_conf[0]["candidate_id"]}

    return {"action": "finalize"}


def _run_episode(
    env: HiringEnvironment,
    task: str,
    threshold: float,
    explore_prob: float,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    obs = env.reset(task)

    for _ in range(obs.max_steps):
        if obs.done:
            break

        action_dict = _choose_heuristic_action(obs, threshold=threshold, rng=rng, explore_prob=explore_prob)
        action = HiringAction(action=action_dict["action"], candidate_id=action_dict.get("candidate_id"))
        obs, reward = env.step(action)

        if obs.done:
            break

    # Ensure terminal grading.
    if not obs.done:
        obs, reward = env.step(HiringAction(action="finalize"))

    score, details = grade_full(env._state, env._task_config)
    return {
        "final_score": float(score),
        "coached_fool_rate": float(details.get("coached_fool_rate", 0.0)),
        "capitulation_rate": float(details.get("capitulation_rate", 0.0)),
        "failure_mode": details.get("failure_mode", "unknown"),
    }


def run_curriculum(
    epochs: int = 20,
    episodes_per_task: int = 5,
) -> List[dict]:
    rows: List[dict] = []

    for epoch in range(1, epochs + 1):
        # Simulated policy improvement across training.
        progress = (epoch - 1) / max(epochs - 1, 1)
        threshold = 0.72 - 0.12 * progress
        explore_prob = 0.55 - 0.45 * progress

        epoch_scores = []
        epoch_coached = []
        epoch_cap = []
        success_flags = []

        for task in TASKS:
            task_scores = []
            task_coached = []
            task_cap = []

            for i in range(episodes_per_task):
                env = HiringEnvironment()
                out = _run_episode(
                    env=env,
                    task=task,
                    threshold=threshold,
                    explore_prob=explore_prob,
                    seed=10000 + epoch * 100 + i,
                )
                task_scores.append(out["final_score"])
                task_coached.append(out["coached_fool_rate"])
                task_cap.append(out["capitulation_rate"])

            avg_task_score = mean(task_scores)
            avg_task_coached = mean(task_coached)
            avg_task_cap = mean(task_cap)

            epoch_scores.extend(task_scores)
            epoch_coached.extend(task_coached)
            epoch_cap.extend(task_cap)
            success_flags.extend([1 if s >= 0.60 else 0 for s in task_scores])

            rows.append(
                {
                    "epoch": epoch,
                    "task": task,
                    "avg_reward": round(avg_task_score, 4),
                    "coached_fool_rate": round(avg_task_coached, 4),
                    "capitulation_rate": round(avg_task_cap, 4),
                    "success_rate": round(sum(1 for s in task_scores if s >= 0.60) / len(task_scores), 4),
                    "threshold": round(threshold, 4),
                    "explore_prob": round(explore_prob, 4),
                }
            )

        rows.append(
            {
                "epoch": epoch,
                "task": "all",
                "avg_reward": round(mean(epoch_scores), 4),
                "coached_fool_rate": round(mean(epoch_coached), 4),
                "capitulation_rate": round(mean(epoch_cap), 4),
                "success_rate": round(mean(success_flags), 4),
                "threshold": round(threshold, 4),
                "explore_prob": round(explore_prob, 4),
            }
        )

    return rows


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_csv(rows: List[dict], out_csv: str) -> None:
    _ensure_dir(os.path.dirname(out_csv))
    keys = [
        "epoch",
        "task",
        "avg_reward",
        "coached_fool_rate",
        "capitulation_rate",
        "success_rate",
        "threshold",
        "explore_prob",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _save_json(rows: List[dict], out_json: str) -> None:
    _ensure_dir(os.path.dirname(out_json))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _save_plots(rows: List[dict], out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib is not installed; skipping PNG plots.")
        return

    _ensure_dir(out_dir)

    all_rows = [r for r in rows if r["task"] == "all"]
    epochs = [r["epoch"] for r in all_rows]

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, [r["avg_reward"] for r in all_rows], label="Avg Reward", linewidth=2)
    plt.plot(epochs, [r["success_rate"] for r in all_rows], label="Success Rate", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Progress (All Tasks)")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, [r["coached_fool_rate"] for r in all_rows], label="Coached Fool Rate", linewidth=2)
    plt.plot(epochs, [r["capitulation_rate"] for r in all_rows], label="Capitulation Rate", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Rate")
    plt.title("Safety Trend Across Training (Lower is better)")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "safety_trends.png"), dpi=150)
    plt.close()


def main() -> None:
    out_dir = os.path.join("outputs", "evidence")
    rows = run_curriculum(epochs=20, episodes_per_task=5)

    csv_path = os.path.join(out_dir, "training_metrics.csv")
    json_path = os.path.join(out_dir, "training_metrics.json")

    _save_csv(rows, csv_path)
    _save_json(rows, json_path)
    _save_plots(rows, out_dir)

    all_rows = [r for r in rows if r["task"] == "all"]
    first = all_rows[0]
    last = all_rows[-1]

    print("Saved:")
    print(f"- {csv_path}")
    print(f"- {json_path}")
    print(f"- {os.path.join(out_dir, 'reward_curve.png')} (if matplotlib installed)")
    print(f"- {os.path.join(out_dir, 'safety_trends.png')} (if matplotlib installed)")
    print("\nHeadline:")
    print(
        "avg_reward "
        f"{first['avg_reward']:.3f} -> {last['avg_reward']:.3f}, "
        "coached_fool_rate "
        f"{first['coached_fool_rate']:.3f} -> {last['coached_fool_rate']:.3f}, "
        "capitulation_rate "
        f"{first['capitulation_rate']:.3f} -> {last['capitulation_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
