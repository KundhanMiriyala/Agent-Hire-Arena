import numpy as np
import hashlib
import json
import os
from typing import Tuple

import requests

from models import (
    CandidateProfile,
    HiringAction,
    HiringObservation,
    HiringReward,
    HiringState,
)
from server.tasks import TaskConfig, get_task
from server.candidate_generator import generate_candidates
from server.grader import grade, explain_grade


# Shaped step rewards
REWARD_INTERVIEW_UNCERTAIN = +0.05   # interview in uncertainty zone (resume 0.4-0.75)
REWARD_HIRE_INFORMED       = +0.10   # hire after interviewing
PENALTY_HIRE_BLIND         = -0.05   # hire without interviewing
PENALTY_BUDGET_EXHAUSTED   = -0.10   # budget runs out mid-episode
REWARD_PROBE_AUDIT         = +0.03   # probe after interview to audit coaching risk


class HiringEnvironment:
    """
    Core environment logic. One instance per session.
    Exposes reset(), step(), and state() following the OpenEnv spec.
    """

    def __init__(self):
        self._state: HiringState | None = None
        self._task_config: TaskConfig | None = None

    # ------------------------------------------------------------------ #
    #  reset()                                                             #
    # ------------------------------------------------------------------ #
    def reset(self, task: str = "easy") -> HiringObservation:
        config = get_task(task)
        candidates = generate_candidates(config)

        self._task_config = config
        self._state = HiringState(
            task_name=task,
            candidates=candidates,
            budget_remaining=config.budget,
            budget_total=config.budget,
            interviews_done={},
            probes_done={},
            probe_gaps={},
            hires_made=[],
            skipped=[],
            step_rewards=[],
            step_num=0,
            max_steps=config.max_steps,
            last_action_result="Episode started. Examine candidates and begin hiring.",
            done=False,
            final_score=None,
        )
        return self._build_observation()

    # ------------------------------------------------------------------ #
    #  step()                                                              #
    # ------------------------------------------------------------------ #
    def step(self, action: HiringAction) -> Tuple[HiringObservation, HiringReward]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            return self._build_observation(), HiringReward(
                step_reward=0.0, reason="Episode already done."
            )

        s = self._state
        s.step_num += 1
        step_reward = 0.0
        reason_parts = []

        # ---- FINALIZE ------------------------------------------------ #
        if action.action == "finalize":
            s.done = True
            final_score = grade(s, self._task_config)
            s.final_score = final_score
            breakdown = explain_grade(s, self._task_config)
            s.last_action_result = (
                f"Episode finalized. Final score: {final_score:.4f}. "
                f"Avg true skill: {breakdown['avg_true_skill']:.3f}. "
                f"Cost ratio: {breakdown['cost_ratio']:.3f}."
            )
            # record the finalize step as zero step reward
            s.step_rewards.append(0.0)
            return self._build_observation(), HiringReward(
                step_reward=0.0,
                final_score=final_score,
                reason=f"Finalized. Score={final_score:.4f}",
            )

        # ---- Validate candidate_id for non-finalize actions ---------- #
        cid = action.candidate_id
        if not cid:
            s.last_action_result = "Error: candidate_id is required for this action."
            return self._build_observation(), HiringReward(
                step_reward=0.0, reason="Missing candidate_id."
            )

        candidate = self._get_candidate(cid)
        if candidate is None:
            s.last_action_result = f"Error: candidate '{cid}' not found."
            return self._build_observation(), HiringReward(
                step_reward=0.0, reason=f"Unknown candidate {cid}."
            )

        if cid in s.hires_made:
            s.last_action_result = f"Error: {candidate.name} is already hired."
            return self._build_observation(), HiringReward(
                step_reward=0.0, reason="Already hired."
            )

        if cid in s.skipped:
            s.last_action_result = f"Error: {candidate.name} was already skipped."
            return self._build_observation(), HiringReward(
                step_reward=0.0, reason="Already skipped."
            )

        # ---- INTERVIEW ----------------------------------------------- #
        if action.action == "interview":
            if cid in s.interviews_done:
                s.last_action_result = f"Error: {candidate.name} was already interviewed."
                return self._build_observation(), HiringReward(
                    step_reward=0.0, reason="Already interviewed."
                )
            cost = 10.0
            if s.budget_remaining < cost:
                s.last_action_result = "Error: insufficient budget to interview."
                return self._build_observation(), HiringReward(
                    step_reward=0.0, reason="Insufficient budget."
                )

            s.budget_remaining -= cost

            # Reveal interview_score: true_skill + small noise (more reliable than resume)
            # Use a stable, cross-process hash to seed interview RNG so
            # interview noise is reproducible across hosts and runs.
            seed_bytes = hashlib.sha256(
                f"{self._task_config.seed}-{cid}-interview".encode()
            ).digest()
            seed_int = int.from_bytes(seed_bytes[:4], "big") % (2 ** 31)
            rng = np.random.default_rng(seed_int)
            noise = float(rng.normal(0.0, 0.10))
            raw_interview = (candidate.true_skill + noise) * float(candidate.interview_difficulty)
            interview_score = float(np.clip(raw_interview, 0.0, 1.0))
            s.interviews_done[cid] = round(interview_score, 3)

            # Step reward: interviewing in the uncertain zone is smart
            if 0.40 <= candidate.resume_score <= 0.75:
                step_reward += REWARD_INTERVIEW_UNCERTAIN
                reason_parts.append("bonus: interviewed uncertain candidate")

            s.last_action_result = (
                f"Interviewed {candidate.name} ({cid}). "
                f"Interview score: {interview_score:.3f}. "
                f"Budget remaining: {s.budget_remaining:.0f}."
            )

        # ---- PROBE --------------------------------------------------- #
        elif action.action == "probe":
            if cid not in s.interviews_done:
                s.last_action_result = f"Error: {candidate.name} must be interviewed before probing."
                return self._build_observation(), HiringReward(
                    step_reward=0.0, reason="Must interview before probing."
                )
            if cid in s.probes_done:
                s.last_action_result = f"Error: {candidate.name} was already probed."
                return self._build_observation(), HiringReward(
                    step_reward=0.0, reason="Already probed."
                )

            cost = 20.0
            if s.budget_remaining < cost:
                s.last_action_result = "Error: insufficient budget to probe."
                return self._build_observation(), HiringReward(
                    step_reward=0.0, reason="Insufficient budget."
                )

            s.budget_remaining -= cost

            interview_score = float(s.interviews_done[cid])
            probe_score, judge_note = self._probe_score(candidate, interview_score)
            probe_gap = round(interview_score - probe_score, 3)
            s.probes_done[cid] = round(probe_score, 3)
            s.probe_gaps[cid] = probe_gap

            step_reward += REWARD_PROBE_AUDIT
            reason_parts.append("bonus: probed interviewed candidate")

            s.last_action_result = (
                f"Probed {candidate.name} ({cid}). "
                f"Probe score: {probe_score:.3f}. "
                f"Interview-probe gap: {probe_gap:+.3f}. "
                f"{judge_note} "
                f"Budget remaining: {s.budget_remaining:.0f}."
            )

        # ---- HIRE ---------------------------------------------------- #
        elif action.action == "hire":
            cost = 50.0
            if s.budget_remaining < cost:
                s.last_action_result = "Error: insufficient budget to hire."
                return self._build_observation(), HiringReward(
                    step_reward=0.0, reason="Insufficient budget."
                )

            s.budget_remaining -= cost
            s.hires_made.append(cid)

            if cid in s.interviews_done:
                step_reward += REWARD_HIRE_INFORMED
                reason_parts.append("bonus: hired after interviewing")
                s.last_action_result = (
                    f"Hired {candidate.name} ({cid}) — informed hire. "
                    f"Budget remaining: {s.budget_remaining:.0f}."
                )
            else:
                step_reward += PENALTY_HIRE_BLIND
                reason_parts.append("penalty: blind hire (no interview)")
                s.last_action_result = (
                    f"Hired {candidate.name} ({cid}) without interview (blind hire). "
                    f"Budget remaining: {s.budget_remaining:.0f}."
                )

        # ---- SKIP ---------------------------------------------------- #
        elif action.action == "skip":
            s.skipped.append(cid)
            s.last_action_result = f"Skipped {candidate.name} ({cid}). No cost."

        else:
            s.last_action_result = (
                f"Error: unknown action '{action.action}'. "
                "Valid actions: interview, probe, hire, skip, finalize."
            )
            return self._build_observation(), HiringReward(
                step_reward=0.0, reason="Unknown action."
            )

        # ---- Check budget exhaustion --------------------------------- #
        if s.budget_remaining < 10.0 and not s.done:
            # Can't do anything useful — auto-end
            if s.budget_remaining < 50.0:
                step_reward += PENALTY_BUDGET_EXHAUSTED
                reason_parts.append("penalty: budget effectively exhausted")

        # ---- Check max steps ---------------------------------------- #
        if s.step_num >= s.max_steps and not s.done:
            s.done = True
            s.final_score = grade(s, self._task_config)
            s.last_action_result += (
                f" | Max steps reached. Episode ended automatically. "
                f"Final score: {s.final_score:.4f}."
            )

        reason = "; ".join(reason_parts) if reason_parts else action.action

        # record the shaped step reward for metrics
        s.step_rewards.append(round(step_reward, 4))

        return self._build_observation(), HiringReward(
            step_reward=round(step_reward, 4),
            final_score=s.final_score if s.done else None,
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    #  state()                                                             #
    # ------------------------------------------------------------------ #
    def state(self) -> dict:
        """Returns the full internal state (for debugging/judging only)."""
        if self._state is None:
            return {"error": "No active episode. Call reset() first."}
        s = self._state
        return {
            "task_name": s.task_name,
            "budget_remaining": s.budget_remaining,
            "budget_total": s.budget_total,
            "interviews_done": s.interviews_done,
            "probes_done": s.probes_done,
            "probe_gaps": s.probe_gaps,
            "hires_made": s.hires_made,
            "skipped": s.skipped,
            "step_rewards": s.step_rewards,
            "step_num": s.step_num,
            "max_steps": s.max_steps,
            "done": s.done,
            "final_score": s.final_score,
            "last_action_result": s.last_action_result,
            # Include true_skill and is_decoy here for judge inspection
            "candidates_full": [
                {
                    "candidate_id": c.candidate_id,
                    "name": c.name,
                    "role": c.role,
                    "resume_score": c.resume_score,
                    "true_skill": c.true_skill,
                    "is_decoy": c.is_decoy,
                    "interview_difficulty": c.interview_difficulty,
                }
                for c in s.candidates
            ],
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _build_observation(self) -> HiringObservation:
        s = self._state
        return HiringObservation(
            candidates=[c.to_agent_view() for c in s.candidates],
            budget_remaining=s.budget_remaining,
            interviews_done=s.interviews_done,
            probes_done=s.probes_done,
            probe_gaps=s.probe_gaps,
            hires_made=s.hires_made,
            skipped=s.skipped,
            step_num=s.step_num,
            max_steps=s.max_steps,
            last_action_result=s.last_action_result,
            done=s.done,
        )

    def _get_candidate(self, candidate_id: str) -> CandidateProfile | None:
        for c in self._state.candidates:
            if c.candidate_id == candidate_id:
                return c
        return None

    def _probe_score(self, candidate: CandidateProfile, interview_score: float) -> tuple[float, str]:
        """Return a truth-aligned probe score, with optional HF/OpenAI judge support."""
        fallback_score = float(np.clip(candidate.true_skill, 0.0, 1.0))
        if os.environ.get("DISABLE_LIVE_PROBE_JUDGE") == "1":
            return fallback_score, "Judge=fallback."

        token = self._env_value("HF_TOKEN")
        model = self._env_value("HF_MODEL_NAME") or self._env_value("MODEL_NAME")
        if not token or not model:
            return fallback_score, "Judge=fallback."

        base_url = (self._env_value("HF_API_BASE_URL") or "https://router.huggingface.co/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        prompt = (
            "You are judging a hiring benchmark probe. Return JSON only with "
            'a numeric "probe_score" from 0 to 1 and a short "coaching_risk". '
            "The probe score should reflect true ability, not interview coaching.\n"
            f"Candidate role: {candidate.role}\n"
            f"Resume score: {candidate.resume_score}\n"
            f"Interview score: {interview_score:.3f}\n"
            f"Ground-truth skill for judge calibration: {candidate.true_skill:.4f}\n"
            f"Is decoy: {candidate.is_decoy}\n"
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 80,
            "temperature": 0,
        }
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=8)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            score = float(data.get("probe_score", fallback_score))
            risk = str(data.get("coaching_risk", "unknown"))[:80]
            return float(np.clip(score, 0.0, 1.0)), f"Judge=llm risk={risk}."
        except Exception:
            return fallback_score, "Judge=fallback."

    def _env_value(self, key: str) -> str:
        value = os.environ.get(key, "").strip()
        if value:
            return value

        env_path = os.path.join(os.getcwd(), ".env")
        try:
            with open(env_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw or raw.startswith("#") or "=" not in raw:
                        continue
                    name, raw_value = raw.split("=", 1)
                    if name.strip() == key:
                        return raw_value.strip().strip('"').strip("'")
        except OSError:
            return ""
        return ""
