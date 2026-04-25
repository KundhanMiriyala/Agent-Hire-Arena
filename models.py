from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
    candidate_id: str
    name: str
    resume_score: float          # noisy proxy, visible to agent
    years_experience: int        # noisy at medium/hard
    role: str                    # visible role for team-composition reasoning
    skills: List[str]            # partially noisy on hard

    # Hidden from agent — excluded from serialization
    true_skill: float = Field(exclude=True)
    is_decoy: bool = Field(exclude=True, default=False)
    interview_difficulty: float = Field(exclude=True, default=1.0)
    is_coached: bool = Field(exclude=True, default=False)

    class Config:
        # Allow the full object (with hidden fields) to exist in memory
        # but exclude hidden fields when serializing to JSON/dict for the agent
        pass

    def to_agent_view(self) -> dict:
        """Returns only the fields the agent is allowed to see."""
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "resume_score": round(self.resume_score, 3),
            "years_experience": self.years_experience,
            "role": self.role,
            "skills": self.skills,
        }


class HiringAction(BaseModel):
    action: str                          # "interview" | "probe" | "hire" | "skip" | "finalize"
    candidate_id: Optional[str] = None   # required for interview/probe/hire/skip


class HiringObservation(BaseModel):
    candidates: List[dict]               # agent-view dicts (no hidden fields)
    budget_remaining: float
    interviews_done: Dict[str, float]    # candidate_id -> interview_score
    probes_done: Dict[str, float]        # candidate_id -> truth-aligned probe_score
    probe_gaps: Dict[str, float]         # candidate_id -> interview_score - probe_score
    hires_made: List[str]                # candidate_ids
    skipped: List[str]                   # candidate_ids
    step_num: int
    max_steps: int
    last_action_result: str
    done: bool
    candidates_remaining: int             # count of candidates not hired/skipped
    hiring_manager_message: Optional[str] = None  # NPC adversarial message
    adversarial_strategy: Optional[str] = None    # NPC strategy type


class HiringReward(BaseModel):
    step_reward: float
    final_score: Optional[float] = None  # set only on finalize()
    reason: str


class HiringState(BaseModel):
    """Full internal state — never sent to agent directly."""
    task_name: str
    candidates: List[CandidateProfile]   # includes hidden fields
    budget_remaining: float
    budget_total: float
    interviews_done: Dict[str, float]
    probes_done: Dict[str, float]
    probe_gaps: Dict[str, float]
    hires_made: List[str]
    skipped: List[str]
    step_rewards: List[float]
    step_num: int
    max_steps: int
    last_action_result: str
    done: bool
    final_score: Optional[float] = None
