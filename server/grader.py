from typing import List

from models import CandidateProfile, HiringState
from server.tasks import TaskConfig


def grade(state: HiringState, task_config: TaskConfig) -> float:
    """
    Compute the final score [0.0, 1.0] for a completed episode.

    Easy / Medium:
        score = avg(true_skill of hires) - (total_cost / budget)

    Hard:
        score = avg(true_skill of hires)
                - (total_cost / budget)
                - 0.20 * (decoy_hires / total_hires)

    Returns 0.0 if no hires were made.
    """
    if not state.hires_made:
        return 0.0

    # Build a lookup: candidate_id -> CandidateProfile (with hidden fields)
    candidate_map = {c.candidate_id: c for c in state.candidates}

    # Collect true skills of hired candidates
    hired_profiles: List[CandidateProfile] = [
        candidate_map[cid] for cid in state.hires_made
        if cid in candidate_map
    ]

    if not hired_profiles:
        return 0.0

    avg_true_skill = sum(c.true_skill for c in hired_profiles) / len(hired_profiles)

    total_cost = state.budget_total - state.budget_remaining
    cost_ratio = total_cost / task_config.budget

    score = avg_true_skill - cost_ratio

    # Hard task: apply decoy penalty
    if task_config.name == "hard":
        decoy_hires = sum(1 for c in hired_profiles if c.is_decoy)
        total_hires = len(hired_profiles)
        decoy_ratio = decoy_hires / total_hires if total_hires > 0 else 0.0
        score -= 0.20 * decoy_ratio

    # Clip to [0.0, 1.0]
    return float(max(0.0, min(1.0, score)))


def explain_grade(state: HiringState, task_config: TaskConfig) -> dict:
    """
    Returns a detailed breakdown of the final score.
    Useful for debugging and README baseline reporting.
    """
    if not state.hires_made:
        return {
            "final_score": 0.0,
            "reason": "No hires made before finalize().",
            "hires": [],
            "avg_true_skill": 0.0,
            "cost_ratio": 0.0,
            "decoy_penalty": 0.0,
        }

    candidate_map = {c.candidate_id: c for c in state.candidates}
    hired_profiles = [
        candidate_map[cid] for cid in state.hires_made if cid in candidate_map
    ]

    avg_true_skill = sum(c.true_skill for c in hired_profiles) / len(hired_profiles)
    total_cost = state.budget_total - state.budget_remaining
    cost_ratio = total_cost / task_config.budget

    decoy_penalty = 0.0
    if task_config.name == "hard":
        decoy_hires = sum(1 for c in hired_profiles if c.is_decoy)
        decoy_ratio = decoy_hires / len(hired_profiles)
        decoy_penalty = 0.20 * decoy_ratio

    raw_score = avg_true_skill - cost_ratio - decoy_penalty
    final_score = float(max(0.0, min(1.0, raw_score)))

    return {
        "final_score": round(final_score, 4),
        "avg_true_skill": round(avg_true_skill, 4),
        "cost_ratio": round(cost_ratio, 4),
        "total_cost": round(total_cost, 2),
        "budget": task_config.budget,
        "decoy_penalty": round(decoy_penalty, 4),
        "hires": [
            {
                "candidate_id": c.candidate_id,
                "name": c.name,
                "true_skill": c.true_skill,
                "is_decoy": c.is_decoy,
                "resume_score": c.resume_score,
            }
            for c in hired_profiles
        ],
    }
