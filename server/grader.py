from typing import List

from models import CandidateProfile, HiringState
from server.tasks import TaskConfig


def grade(state: HiringState, task_config: TaskConfig) -> float:
    """
    Compute the final score [0.0, 1.0] with aligned multi-objective grading.

    score = avg_true_skill
            + team_size_bonus
            + role_coverage_bonus
            - cost_penalty
            - decoy_penalty

    Early finalize is softly penalized to discourage trivial one-step exits.
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
    cost_ratio = total_cost / task_config.budget if task_config.budget > 0 else 1.0

    team_size_bonus = min(len(hired_profiles) / 3.0, 1.0) * 0.25
    cost_penalty = (cost_ratio ** 1.3) * 0.40

    decoy_hires = sum(1 for c in hired_profiles if c.is_decoy)
    decoy_ratio = decoy_hires / len(hired_profiles) if hired_profiles else 0.0
    decoy_penalty = 0.50 * decoy_ratio  # Doubled penalty to force careful decoy detection

    req = task_config.role_requirements or {}
    if req:
        hired_roles = {c.role for c in hired_profiles}
        covered = sum(1 for role in req if role in hired_roles)
        role_coverage_bonus = 0.20 * (covered / len(req))
    else:
        role_coverage_bonus = 0.0

    score = avg_true_skill + team_size_bonus + role_coverage_bonus - cost_penalty - decoy_penalty

    # Anti-gaming: very early finalize receives a soft multiplier.
    if state.step_num < 3:
        score *= 0.70

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
            "team_size_bonus": 0.0,
            "role_coverage_bonus": 0.0,
            "cost_penalty": 0.0,
            "decoy_penalty": 0.0,
        }

    candidate_map = {c.candidate_id: c for c in state.candidates}
    hired_profiles = [
        candidate_map[cid] for cid in state.hires_made if cid in candidate_map
    ]

    avg_true_skill = sum(c.true_skill for c in hired_profiles) / len(hired_profiles)
    total_cost = state.budget_total - state.budget_remaining
    cost_ratio = total_cost / task_config.budget

    team_size_bonus = min(len(hired_profiles) / 3.0, 1.0) * 0.25
    cost_penalty = (cost_ratio ** 1.3) * 0.40
    decoy_hires = sum(1 for c in hired_profiles if c.is_decoy)
    decoy_ratio = decoy_hires / len(hired_profiles)
    decoy_penalty = 0.25 * decoy_ratio

    req = task_config.role_requirements or {}
    if req:
        hired_roles = {c.role for c in hired_profiles}
        covered = sum(1 for role in req if role in hired_roles)
        role_coverage_bonus = 0.20 * (covered / len(req))
    else:
        covered = 0
        role_coverage_bonus = 0.0

    raw_score = avg_true_skill + team_size_bonus + role_coverage_bonus - cost_penalty - decoy_penalty
    early_finalize_multiplier = 0.70 if state.step_num < 3 else 1.0
    raw_score *= early_finalize_multiplier
    final_score = float(max(0.0, min(1.0, raw_score)))

    return {
        "final_score": round(final_score, 4),
        "avg_true_skill": round(avg_true_skill, 4),
        "cost_ratio": round(cost_ratio, 4),
        "total_cost": round(total_cost, 2),
        "budget": task_config.budget,
        "team_size_bonus": round(team_size_bonus, 4),
        "role_coverage_bonus": round(role_coverage_bonus, 4),
        "role_coverage": {
            "covered_roles": covered,
            "required_roles": len(req),
            "requirements": req,
        },
        "cost_penalty": round(cost_penalty, 4),
        "decoy_penalty": round(decoy_penalty, 4),
        "early_finalize_multiplier": early_finalize_multiplier,
        "hires": [
            {
                "candidate_id": c.candidate_id,
                "name": c.name,
                "role": c.role,
                "true_skill": c.true_skill,
                "is_decoy": c.is_decoy,
                "resume_score": c.resume_score,
            }
            for c in hired_profiles
        ],
    }
