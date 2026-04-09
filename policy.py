from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from models import HiringObservation

VALID_ACTIONS = {"interview", "hire", "skip", "finalize"}

ROLE_REQUIREMENTS: Dict[str, List[str]] = {
    "easy": ["ML Engineer"],
    "medium": ["ML Engineer", "Backend"],
    "hard": ["ML Engineer", "Backend", "Data Scientist"],
}


@dataclass(frozen=True)
class TaskPolicyConfig:
    target_interviews: int
    min_interview_coverage: float
    min_interviews_before_hire: int
    target_hires: int
    hire_interview_threshold: float
    finalize_best_score: float
    max_effective_interviews: int
    min_budget_to_continue: float
    hard_disagreement_gate: float = 0.28
    medium_disagreement_gate: float = 0.35


BASELINE_CONFIGS: Dict[str, TaskPolicyConfig] = {
    "easy": TaskPolicyConfig(
        target_interviews=3,
        min_interview_coverage=0.40,
        min_interviews_before_hire=2,
        target_hires=1,
        hire_interview_threshold=0.65,
        finalize_best_score=0.78,
        max_effective_interviews=3,
        min_budget_to_continue=60,
    ),
    "medium": TaskPolicyConfig(
        target_interviews=2,
        min_interview_coverage=0.60,
        min_interviews_before_hire=4,
        target_hires=2,
        hire_interview_threshold=0.65,
        finalize_best_score=0.72,
        max_effective_interviews=3,
        min_budget_to_continue=60,
    ),
    "hard": TaskPolicyConfig(
        target_interviews=4,
        min_interview_coverage=0.70,
        min_interviews_before_hire=8,
        target_hires=2,
        hire_interview_threshold=0.70,
        finalize_best_score=0.72,
        max_effective_interviews=4,
        min_budget_to_continue=60,
        hard_disagreement_gate=0.28,
        medium_disagreement_gate=0.35,
    ),
}


TASK_AWARE_CONFIGS: Dict[str, TaskPolicyConfig] = {
    "easy": TaskPolicyConfig(
        target_interviews=2,
        min_interview_coverage=0.35,
        min_interviews_before_hire=2,
        target_hires=1,
        hire_interview_threshold=0.64,
        finalize_best_score=0.76,
        max_effective_interviews=3,
        min_budget_to_continue=55,
    ),
    "medium": TaskPolicyConfig(
        target_interviews=4,
        min_interview_coverage=0.60,
        min_interviews_before_hire=4,
        target_hires=2,
        hire_interview_threshold=0.68,
        finalize_best_score=0.75,
        max_effective_interviews=6,
        min_budget_to_continue=60,
    ),
    "hard": TaskPolicyConfig(
        target_interviews=6,
        min_interview_coverage=0.70,
        min_interviews_before_hire=6,
        target_hires=2,
        hire_interview_threshold=0.72,
        finalize_best_score=0.76,
        max_effective_interviews=8,
        min_budget_to_continue=65,
        hard_disagreement_gate=0.24,
        medium_disagreement_gate=0.32,
    ),
}


@dataclass(frozen=True)
class PolicyFeatures:
    role_aware: bool = True
    decoy_risk_guard: bool = True
    planning_depth: int = 0
    beam_width: int = 4


def _signal_disagreement(resume_score: float, interview_score: Optional[float]) -> float:
    if interview_score is None:
        return 0.0
    return abs(float(resume_score) - float(interview_score))


def _decoy_risk(resume_score: float, interview_score: Optional[float]) -> float:
    if interview_score is None:
        return 0.20 if resume_score > 0.85 else 0.0

    gap = resume_score - interview_score
    if gap > 0.25:
        return 0.50
    if gap > 0.15:
        return 0.30
    return 0.0


class DecisionCore:
    def __init__(
        self,
        obs: HiringObservation,
        task: str,
        config: TaskPolicyConfig,
        features: PolicyFeatures,
    ):
        self.obs = obs
        self.task = task
        self.config = config
        self.features = features

        self.interviewed = obs.interviews_done or {}
        self.remaining: List[dict] = [
            c
            for c in obs.candidates
            if c["candidate_id"] not in obs.hires_made and c["candidate_id"] not in obs.skipped
        ]
        self.by_id: Dict[str, dict] = {c["candidate_id"]: c for c in self.remaining}
        self.all_by_id: Dict[str, dict] = {c["candidate_id"]: c for c in obs.candidates}
        self.hired_roles: Set[str] = {
            self.all_by_id[cid].get("role", "Unknown")
            for cid in obs.hires_made
            if cid in self.all_by_id
        }

    def can_interview(self) -> bool:
        return self.obs.budget_remaining >= 10

    def can_hire(self) -> bool:
        return self.obs.budget_remaining >= 50

    def min_interviews_needed(self) -> int:
        return max(
            self.config.min_interviews_before_hire,
            int(round(len(self.obs.candidates) * self.config.min_interview_coverage)),
        )

    def role_bonus(self, candidate: dict) -> float:
        if not self.features.role_aware:
            return 0.0
        role = candidate.get("role")
        required_roles = ROLE_REQUIREMENTS.get(self.task, [])
        if role in required_roles and role not in self.hired_roles:
            return 0.12
        if role not in self.hired_roles:
            return 0.05
        return 0.0

    def candidate_value(self, candidate: dict, interview_score: Optional[float]) -> float:
        resume = float(candidate.get("resume_score", 0.0))
        if interview_score is None:
            uncertain = 0.08 if 0.45 <= resume <= 0.80 else 0.0
            hard_skeptic = -0.10 if self.task == "hard" and resume > 0.88 else -0.03
            return resume + uncertain + hard_skeptic + self.role_bonus(candidate)

        disagreement = _signal_disagreement(resume, interview_score)
        task_penalty = 0.45 if self.task == "hard" else 0.20 if self.task == "medium" else 0.10
        return interview_score - (task_penalty * disagreement) + self.role_bonus(candidate)

    def interview_value(self, candidate: dict) -> float:
        resume = float(candidate.get("resume_score", 0.0))
        uncertainty = 0.10 if 0.50 <= resume <= 0.78 else 0.0
        hard_bias = 0.07 if self.task == "hard" and resume < 0.82 else 0.0
        return self.candidate_value(candidate, None) + uncertainty + hard_bias

    def should_hire(self, candidate: dict, interview_score: Optional[float]) -> bool:
        if interview_score is None:
            return False

        resume = float(candidate.get("resume_score", 0.0))
        disagreement = _signal_disagreement(resume, interview_score)

        if self.features.decoy_risk_guard:
            if self.task == "hard" and disagreement > self.config.hard_disagreement_gate:
                return False
            if (
                self.task == "medium"
                and disagreement > self.config.medium_disagreement_gate
                and interview_score < (self.config.hire_interview_threshold + 0.08)
            ):
                return False
            if self.task == "hard" and _decoy_risk(resume, interview_score) >= 0.30:
                return False

        threshold = self.config.hire_interview_threshold
        if self.features.role_aware and self.role_bonus(candidate) > 0:
            threshold -= 0.02
        return interview_score >= threshold

    def should_finalize(self, best_score: float, interviews_done: int, hires_count: int) -> bool:
        if interviews_done < self.min_interviews_needed():
            return False

        if hires_count >= self.config.target_hires:
            return True

        if hires_count >= 1 and best_score >= self.config.finalize_best_score:
            return True

        if interviews_done >= self.config.max_effective_interviews:
            return hires_count >= 1

        if self.obs.budget_remaining < self.config.min_budget_to_continue:
            return hires_count >= 1

        return False

    def ranked_interviewed(self) -> List[Tuple[str, float, float]]:
        ranked: List[Tuple[str, float, float]] = []
        for cid, score in self.interviewed.items():
            c = self.by_id.get(cid)
            if c is None:
                continue
            value = self.candidate_value(c, score)
            ranked.append((cid, value, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def ranked_uninterviewed(self) -> List[dict]:
        interviewed_ids = set(self.interviewed.keys())
        candidates = [c for c in self.remaining if c["candidate_id"] not in interviewed_ids]
        candidates.sort(key=lambda c: self.interview_value(c), reverse=True)
        return candidates


def _sanitize_model_action(model_action: Optional[dict], obs: HiringObservation) -> Optional[dict]:
    if not model_action:
        return None

    action = model_action.get("action")
    cid = model_action.get("candidate_id")
    if action not in VALID_ACTIONS:
        return None
    if action == "finalize":
        return {"action": "finalize"}
    if not cid:
        return None

    live_ids = {
        c["candidate_id"]
        for c in obs.candidates
        if c["candidate_id"] not in obs.hires_made and c["candidate_id"] not in obs.skipped
    }
    if cid not in live_ids:
        return None

    if action == "interview" and cid in (obs.interviews_done or {}):
        return None

    return {"action": action, "candidate_id": cid}


def _simulate_utility(core: DecisionCore, action: dict) -> float:
    remaining_budget = core.obs.budget_remaining
    hires_count = len(core.obs.hires_made)
    interviewed_count = len(core.interviewed)
    estimated_best = 0.0

    ranked_interviewed = core.ranked_interviewed()
    if ranked_interviewed:
        estimated_best = ranked_interviewed[0][1]

    action_type = action["action"]
    if action_type == "finalize":
        return 0.35 + 0.30 * max(estimated_best, 0.0) - (0.20 if hires_count == 0 else 0.0)

    if action_type == "interview":
        cid = action["candidate_id"]
        candidate = core.by_id.get(cid)
        if candidate is None or remaining_budget < 10:
            return -1.0
        resume = float(candidate.get("resume_score", 0.0))
        expected_interview = resume - (0.05 if core.task == "hard" and resume > 0.85 else 0.0)
        disagreement = _signal_disagreement(resume, expected_interview)
        info_gain = 0.30 if 0.45 <= resume <= 0.80 else 0.12
        return info_gain - 0.06 + 0.15 * (1.0 - disagreement) - 0.02 * interviewed_count

    if action_type == "hire":
        cid = action["candidate_id"]
        candidate = core.by_id.get(cid)
        if candidate is None or remaining_budget < 50:
            return -1.0
        interview = core.interviewed.get(cid)
        if not core.should_hire(candidate, interview):
            return -0.5
        value = core.candidate_value(candidate, interview)
        cost_term = (remaining_budget - 50.0) / max(core.obs.budget_remaining, 1.0)
        return 0.45 * value + 0.10 * cost_term + core.role_bonus(candidate)

    if action_type == "skip":
        cid = action["candidate_id"]
        candidate = core.by_id.get(cid)
        if candidate is None:
            return -1.0
        interview = core.interviewed.get(cid)
        value = core.candidate_value(candidate, interview)
        return 0.08 if value < 0.55 else -0.12

    return -1.0


def _planning_action(core: DecisionCore, safe_model_action: Optional[dict]) -> dict:
    interviewed_ranked = core.ranked_interviewed()
    uninterviewed_ranked = core.ranked_uninterviewed()

    action_pool: List[dict] = [{"action": "finalize"}]

    if core.can_interview():
        for c in uninterviewed_ranked[:2]:
            action_pool.append({"action": "interview", "candidate_id": c["candidate_id"]})

    if core.can_hire():
        for cid, _, _ in interviewed_ranked[:2]:
            action_pool.append({"action": "hire", "candidate_id": cid})

    for c in core.remaining[:1]:
        action_pool.append({"action": "skip", "candidate_id": c["candidate_id"]})

    if safe_model_action:
        action_pool.append(safe_model_action)

    # Deduplicate while preserving order
    deduped: List[dict] = []
    seen: Set[Tuple[str, str]] = set()
    for action in action_pool:
        key = (action["action"], action.get("candidate_id", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)

    scored = sorted(((a, _simulate_utility(core, a)) for a in deduped), key=lambda x: x[1], reverse=True)

    if not scored:
        return {"action": "finalize"}
    return scored[0][0]


def _greedy_action(core: DecisionCore, safe_model_action: Optional[dict]) -> dict:
    if not core.remaining:
        return {"action": "finalize"}

    interviewed_ranked = core.ranked_interviewed()
    uninterviewed_ranked = core.ranked_uninterviewed()

    best_candidate_id = None
    best_candidate_score = -1.0
    best_candidate_interview = None
    if interviewed_ranked:
        best_candidate_id, best_candidate_score, best_candidate_interview = interviewed_ranked[0]

    interviews_done = len(core.interviewed)
    hires_count = len(core.obs.hires_made)

    if core.should_finalize(best_candidate_score, interviews_done, hires_count):
        return {"action": "finalize"}

    min_interviews_needed = core.min_interviews_needed()

    if safe_model_action and safe_model_action.get("action") == "interview" and core.can_interview():
        top_ids = [c["candidate_id"] for c in uninterviewed_ranked[:3]]
        if safe_model_action.get("candidate_id") in top_ids:
            return safe_model_action

    if safe_model_action and safe_model_action.get("action") == "hire" and core.can_hire():
        cid = safe_model_action.get("candidate_id")
        candidate = core.by_id.get(cid)
        if candidate and interviews_done >= min_interviews_needed and core.should_hire(candidate, core.interviewed.get(cid)):
            return safe_model_action

    if core.can_interview() and interviews_done < core.config.target_interviews and uninterviewed_ranked:
        return {"action": "interview", "candidate_id": uninterviewed_ranked[0]["candidate_id"]}

    if core.can_interview() and interviews_done < min_interviews_needed and uninterviewed_ranked:
        return {"action": "interview", "candidate_id": uninterviewed_ranked[0]["candidate_id"]}

    if core.can_hire() and best_candidate_id and interviews_done >= min_interviews_needed:
        best_candidate = core.by_id.get(best_candidate_id)
        if best_candidate and core.should_hire(best_candidate, best_candidate_interview):
            return {"action": "hire", "candidate_id": best_candidate_id}

    if core.can_interview() and uninterviewed_ranked:
        return {"action": "interview", "candidate_id": uninterviewed_ranked[0]["candidate_id"]}

    return {"action": "finalize"}


def choose_policy_action(
    obs: HiringObservation,
    task: str,
    model_action: Optional[dict] = None,
    variant: str = "baseline",
    role_aware: bool = True,
    decoy_risk_guard: bool = True,
) -> dict:
    safe_model_action = _sanitize_model_action(model_action, obs)

    if variant == "baseline":
        config = BASELINE_CONFIGS[task]
        features = PolicyFeatures(role_aware=role_aware, decoy_risk_guard=decoy_risk_guard, planning_depth=0)
    elif variant == "task-aware":
        config = TASK_AWARE_CONFIGS[task]
        features = PolicyFeatures(role_aware=role_aware, decoy_risk_guard=decoy_risk_guard, planning_depth=0)
    elif variant == "planning":
        config = TASK_AWARE_CONFIGS[task]
        features = PolicyFeatures(role_aware=role_aware, decoy_risk_guard=decoy_risk_guard, planning_depth=2, beam_width=4)
    else:
        raise ValueError(f"Unknown policy variant '{variant}'. Choose from: baseline, task-aware, planning")

    core = DecisionCore(obs=obs, task=task, config=config, features=features)

    if not core.remaining:
        return {"action": "finalize"}

    if features.planning_depth > 0:
        return _planning_action(core, safe_model_action)

    return _greedy_action(core, safe_model_action)
