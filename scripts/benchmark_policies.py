import argparse
import json
import os
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List

import requests

# Add parent dir to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import HiringEnvClient
from policy import choose_policy_action


@dataclass
class VariantSpec:
    name: str
    variant: str
    role_aware: bool
    decoy_guard: bool


def run_episode(env_client: HiringEnvClient, task: str, spec: VariantSpec) -> Dict:
    obs = env_client.reset(task=task)

    while not obs.done:
        action = choose_policy_action(
            obs=obs,
            task=task,
            model_action=None,
            variant=spec.variant,
            role_aware=spec.role_aware,
            decoy_risk_guard=spec.decoy_guard,
        )
        obs, reward = env_client.step(action=action["action"], candidate_id=action.get("candidate_id"))

    metrics_resp = requests.get(f"{env_client.base_url}/metrics", timeout=30)
    metrics_resp.raise_for_status()
    metrics = metrics_resp.json().get("metrics", {})

    return {
        "task": task,
        "variant_name": spec.name,
        "variant": spec.variant,
        "role_aware": spec.role_aware,
        "decoy_guard": spec.decoy_guard,
        "final_score": float(reward.final_score or 0.0),
        "breakdown": metrics,
    }


def summarize(results: List[Dict]) -> Dict:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in results:
        grouped[row["variant_name"]].append(row)

    summary = {}
    for variant_name, rows in grouped.items():
        scores = [r["final_score"] for r in rows]
        by_task = {r["task"]: r["final_score"] for r in rows}
        summary[variant_name] = {
            "mean_score": statistics.mean(scores) if scores else 0.0,
            "std_score": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            "task_scores": by_task,
        }

    return summary


def to_markdown(results: List[Dict], summary: Dict) -> str:
    lines = []
    lines.append("# AgentHire Arena Policy Benchmark")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Variant | Mean | Std | Easy | Medium | Hard |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for variant_name, row in sorted(summary.items()):
        tasks = row["task_scores"]
        lines.append(
            f"| {variant_name} | {row['mean_score']:.4f} | {row['std_score']:.4f} | "
            f"{tasks.get('easy', 0.0):.4f} | {tasks.get('medium', 0.0):.4f} | {tasks.get('hard', 0.0):.4f} |"
        )

    lines.append("")
    lines.append("## Detailed Grader Breakdown")
    lines.append("")
    lines.append("| Variant | Task | Score | Avg True Skill | Team Bonus | Role Bonus | Cost Penalty | Decoy Penalty |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for r in sorted(results, key=lambda x: (x["variant_name"], x["task"])):
        b = r.get("breakdown", {})
        lines.append(
            f"| {r['variant_name']} | {r['task']} | {r['final_score']:.4f} | "
            f"{float(b.get('avg_true_skill', 0.0)):.4f} | {float(b.get('team_size_bonus', 0.0)):.4f} | "
            f"{float(b.get('role_coverage_bonus', 0.0)):.4f} | {float(b.get('cost_penalty', 0.0)):.4f} | "
            f"{float(b.get('decoy_penalty', 0.0)):.4f} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AgentHire policy variants against grader breakdown.")
    parser.add_argument("--env", default=os.environ.get("OPENENV_API_BASE_URL", "http://127.0.0.1:7860"))
    parser.add_argument("--repeats", type=int, default=1, help="Repeat runs per task/variant for stability checks.")
    parser.add_argument(
        "--output-json",
        default=os.path.join("logs", "policy_benchmark.json"),
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--output-md",
        default=os.path.join("logs", "policy_benchmark.md"),
        help="Path to write Markdown report.",
    )
    args = parser.parse_args()

    env_client = HiringEnvClient(base_url=args.env)
    if not env_client.health():
        raise RuntimeError(f"Environment is not reachable at {args.env}")

    variants = [
        VariantSpec(name="baseline", variant="baseline", role_aware=True, decoy_guard=True),
        VariantSpec(name="task-aware", variant="task-aware", role_aware=True, decoy_guard=True),
        VariantSpec(name="planning", variant="planning", role_aware=True, decoy_guard=True),
        VariantSpec(name="planning-no-decoy", variant="planning", role_aware=True, decoy_guard=False),
        VariantSpec(name="planning-no-role", variant="planning", role_aware=False, decoy_guard=True),
    ]

    tasks = ["easy", "medium", "hard"]
    results: List[Dict] = []

    for _ in range(args.repeats):
        for spec in variants:
            for task in tasks:
                results.append(run_episode(env_client, task, spec))

    summary = summarize(results)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "variants": [asdict(v) for v in variants],
                "repeats": args.repeats,
                "results": results,
                "summary": summary,
            },
            f,
            indent=2,
        )

    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(to_markdown(results, summary))

    print(f"Wrote JSON report: {args.output_json}")
    print(f"Wrote Markdown report: {args.output_md}")


if __name__ == "__main__":
    main()
