"""
inference.py — Baseline LLM agent for AgentHire Arena.

Reads from environment variables:
    API_BASE_URL  — OpenAI-compatible model endpoint injected by the evaluator
    API_KEY       — API key for the OpenAI-compatible model endpoint
    MODEL_NAME    — Model name for the OpenAI-compatible API
    OPENENV_API_BASE_URL / ENV_API_BASE_URL / ENV_BASE_URL — optional environment server URL
    LLM_PROVIDER  — Optional override: "mock" for local testing
    OPENAI_API_KEY / HF_TOKEN — optional local fallbacks

Run:
    API_BASE_URL=https://router.example/v1 \
    API_KEY=sk-xxx \
    MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct \
    OPENENV_API_BASE_URL=http://localhost:7860 \
    python inference.py
"""

import os
import re
import json
import time
from typing import Optional
import numpy as np
import requests

from openai import OpenAI
from requests import post as requests_post

# If a local .env exists, load it so HF_TOKEN / API_BASE_URL can be set there
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from models import HiringObservation


# Minimal HTTP client for environment communication
class HiringEnvClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/tasks", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def reset(self, task: str = "easy") -> HiringObservation:
        response = requests_post(f"{self.base_url}/reset", json={"task": task}, timeout=30)
        response.raise_for_status()
        return HiringObservation(**response.json())

    def step(self, action: str, candidate_id: str = None):
        payload = {"action": action}
        if candidate_id:
            payload["candidate_id"] = candidate_id
        response = requests_post(f"{self.base_url}/step", json=payload, timeout=30)
        if response.status_code >= 400:
            detail = None
            try:
                body = response.json()
                detail = body.get("detail") if isinstance(body, dict) else body
            except Exception:
                detail = response.text.strip()
            raise RuntimeError(
                f"Step API error {response.status_code} for payload={payload}: {detail}"
            )
        data = response.json()
        obs = HiringObservation(**data.get("observation", {}))
        
        from models import HiringReward
        reward = HiringReward(**data.get("reward", {}))
        return obs, reward


# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini").strip().strip('"').strip("'")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

TASKS_TO_RUN = ["easy", "medium", "hard", "adversarial", "nightmare"]
MODEL_MAX_RETRIES = 3
STEP_MAX_RETRIES = 5


def _is_retriable_model_exception(exc: Exception) -> bool:
    """Retry only transient API/network failures, not invalid requests (e.g., model_not_found)."""
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code in {408, 409, 429, 500, 502, 503, 504}

    msg = str(exc).lower()
    if "model_not_found" in msg:
        return False
    if "invalid_request_error" in msg:
        return False
    return True


def _looks_like_llm_endpoint(url: str) -> bool:
    """Detect common model-router URLs that are not the environment API."""
    u = (url or "").lower()
    return any(token in u for token in ["litellm", "openai", "huggingface", "router", "/v1"])


def _resolve_env_base_url() -> str:
    """Pick a reliable environment base URL across local and judge runtimes.

    Some runners export API_BASE_URL for model backends (e.g., LiteLLM).
    In that case we must not use it as the environment URL.
    """
    candidates = [
        os.environ.get("OPENENV_API_BASE_URL", ""),
        os.environ.get("ENV_API_BASE_URL", ""),
        os.environ.get("ENV_BASE_URL", ""),
        os.environ.get("API_BASE_URL", ""),
    ]

    for c in candidates:
        c = (c or "").strip()
        if c and not _looks_like_llm_endpoint(c):
            return c.rstrip("/")

    return "http://127.0.0.1:7860"

# ------------------------------------------------------------------ #
#  System prompt (the most important piece — agents read this)         #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """You are an expert hiring manager making sequential decisions to assemble the best team within a strict budget.

ACTIONS — respond with exactly one JSON object per turn:
  {"action": "interview", "candidate_id": "C03"}   → costs 10 units, reveals interview_score
  {"action": "probe",     "candidate_id": "C03"}   → costs 20 units, bypasses coaching to reveal true skill
  {"action": "hire",      "candidate_id": "C03"}   → costs 50 units, adds to your team
  {"action": "skip",      "candidate_id": "C03"}   → free, permanently rejects
  {"action": "finalize"}                           → ends episode, scores your team

RULES:
- You MUST call finalize to receive your score. Forgetting = score of 0.
- resume_score is noisy and unreliable — never hire based on it alone.
- interview_score is more reliable but costs 10 units.
- On hard tasks, candidates may be "coached" to fake high interview scores. If an interview score seems suspiciously high compared to the resume, use the `probe` action to uncover their true skill.
- Hiring without interviewing is allowed but risky (blind hire).
- Once you skip a candidate, that decision is permanent.
- Budget exhaustion ends the episode with whatever hires you have.

STRATEGY:
1. Scan all resumes first.
2. Interview candidates in the uncertain range (resume 0.4-0.75) where you need more signal.
3. Hire candidates with strong interview scores.
4. Skip clearly weak candidates to clear mental space.
5. Finalize when your team looks solid or budget is running low.

Think step by step. Reason about: budget remaining, how many more hires you can afford, which candidates are worth interviewing vs skipping. Then output your JSON action on the final line.
Note:
Do not waste tokens, be concise in your reasoning, and always end with a valid JSON action and also make the responses as short as possible. The evaluator will parse only the last JSON object in your response, so if you change your mind or want to update your action, just output a new JSON object on a new  line, give a short reasoning.

example response:
[Step 2]
  Reasoning: Noah (C03) has a very high resume score (0.98) and a perfect interview score (1.000). This is a strong signal for a high-quality hire.
  Model action:  {"action":"hire","candidate_id":"C03"}
  Safe action:   {"action":"hire","candidate_id":"C03"}
  Final action:  {"action": "hire", "candidate_id": "C03"}
  Decision:      Executed model action directly.
  Reward:    -20.00  (rejected: high-confidence hire without probe required)
  Result:    Cannot hire Noah (C03): High-confidence candidate (resume=0.98, interview=1.00) must be probed first to detect coaching. Probe to verify true skill.
[STEP]  step=2 action={"action": "hire", "candidate_id": "C03"} reward=-20.00 done=false error=rejected: high-confidence hire without probe required

Note Reasoning must be in a line! in a sentence! short and precise!


"""


# ------------------------------------------------------------------ #
#  Observation → natural language renderer                             #
# ------------------------------------------------------------------ #

def render_observation(obs: HiringObservation) -> str:
    lines = [
        f"=== HIRING DASHBOARD ===",
        f"Step: {obs.step_num} / {obs.max_steps}",
        f"Budget remaining: {obs.budget_remaining:.0f} units  "
        f"(interview=10, hire=50)",
        f"Candidates remaining: {obs.candidates_remaining}",
        f"Hired so far: {obs.hires_made if obs.hires_made else 'none'}",
        f"Skipped: {obs.skipped if obs.skipped else 'none'}",
        "",
        "CANDIDATES (resume_score is noisy — interview to get better signal):",
    ]

    for c in obs.candidates:
        cid = c["candidate_id"]

        # Skip already-acted-on candidates
        if cid in obs.hires_made:
            continue
        if cid in obs.skipped:
            continue

        interview_str = ""
        if cid in obs.interviews_done:
            score = obs.interviews_done[cid]
            confidence = _confidence_score(c["resume_score"], score)
            disagreement = _signal_disagreement(c["resume_score"], score)
            interview_str = f" | INTERVIEWED: {score:.3f} | conf={confidence:.2f} | disagree={disagreement:.2f}"

        lines.append(
            f"  {cid}: {c['name']:<10} "
            f"resume={c['resume_score']:.2f}  "
            f"exp={c['years_experience']}yrs  "
            f"role={c.get('role', 'Unknown')}  "
            f"skills=[{', '.join(c['skills'][:3])}]"
            + interview_str
        )

    if obs.last_action_result:
        lines.append(f"\nLast result: {obs.last_action_result}")

    lines.append(
        f"\nBudget math: you can afford "
        f"{int(obs.budget_remaining // 10)} more interviews OR "
        f"{int(obs.budget_remaining // 50)} more hires."
    )

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Action parser — robust, never crashes the loop                      #
# ------------------------------------------------------------------ #

VALID_ACTIONS = {"interview", "hire", "skip", "probe", "finalize"}


def parse_action(text: str) -> dict:
    """
    Extract the JSON action from the LLM's response text.
    Searches for the last valid JSON object containing an 'action' key.
    Falls back to finalize if nothing parseable is found.
    """
    # Try direct parse first for strict-JSON responses
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and obj.get("action") in VALID_ACTIONS:
            return obj
    except Exception:
        pass

    # Find all JSON-like blocks in the response
    matches = re.findall(r'\{[^{}]+\}', text, re.DOTALL)

    for raw in reversed(matches):   # last match = most likely the action
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and obj.get("action") in VALID_ACTIONS:
                return obj
        except json.JSONDecodeError:
            continue

    # Last-resort keyword parse so parsing never fails.
    lower = (text or "").lower()
    action = None
    for candidate in ["finalize", "interview", "probe", "hire", "skip"]:
        if candidate in lower:
            action = candidate
            break

    if not action:
        return {"action": "finalize"}

    if action == "finalize":
        return {"action": "finalize"}

    cid_match = re.search(r"\bC\d{2}\b", text or "", flags=re.IGNORECASE)
    if cid_match:
        return {"action": action, "candidate_id": cid_match.group(0).upper()}

    return {"action": action}


def _fallback_action_from_obs(obs: HiringObservation) -> dict:
    """Deterministic emergency fallback when model output is invalid for current state."""
    live = [
        c for c in obs.candidates
        if c["candidate_id"] not in obs.hires_made and c["candidate_id"] not in obs.skipped
    ]
    if not live:
        return {"action": "finalize"}

    interviewed = set((obs.interviews_done or {}).keys())
    can_interview = obs.budget_remaining >= 10
    can_hire = obs.budget_remaining >= 50

    # Prefer collecting signal first if possible.
    if can_interview:
        for c in live:
            if c["candidate_id"] not in interviewed:
                return {"action": "interview", "candidate_id": c["candidate_id"]}

    if can_hire:
        for c in live:
            if c["candidate_id"] in interviewed:
                return {"action": "hire", "candidate_id": c["candidate_id"]}

    return {"action": "finalize"}


TASK_POLICY = {
    "easy": {
        "target_interviews": 3,
        "min_interview_coverage": 0.40,
        "min_interviews_before_hire": 2,
        "target_hires": 1,
        "hire_interview_threshold": 0.65,
        "finalize_best_score": 0.78,
        "max_effective_interviews": 3,
        "min_budget_to_continue": 60,
    },
    "medium": {
        "target_interviews": 2,
        "min_interview_coverage": 0.60,
        "min_interviews_before_hire": 4,
        "target_hires": 2,
        "hire_interview_threshold": 0.65,
        "finalize_best_score": 0.72,
        "max_effective_interviews": 3,
        "min_budget_to_continue": 60,
    },
    "hard": {
        "target_interviews": 4,
        "min_interview_coverage": 0.70,
        "min_interviews_before_hire": 8,
        "target_hires": 2,
        "hire_interview_threshold": 0.70,
        "early_finalize_best_score": 0.75,
        "late_finalize_best_score": 0.65,
        "max_effective_interviews": 4,
        "min_budget_to_continue": 60,
    },
}


def _interview_priority(candidate: dict) -> float:
    """Value-of-information priority for selecting next interview target."""
    resume = float(candidate.get("resume_score", 0.0))
    role = candidate.get("role", "")
    if 0.50 <= resume <= 0.75:
        return 1.0
    if resume > 0.85:
        return 0.8
    if role in {"ML Engineer", "Backend", "Data Scientist"}:
        return 0.65
    return 0.3


def _expected_interview_value(candidate: dict, task: str, hired_roles: set[str]) -> float:
    """Score interview candidates by expected value per cost."""
    resume = float(candidate.get("resume_score", 0.0))
    priority = _interview_priority(candidate)
    role_bonus = 0.10 if candidate.get("role") not in hired_roles else 0.0
    task_bias = 0.08 if task == "hard" and resume < 0.8 else 0.0
    uncertainty_bonus = 0.10 if 0.45 <= resume <= 0.80 else 0.0
    return resume + priority + role_bonus + task_bias + uncertainty_bonus


def _signal_disagreement(resume_score: float, interview_score: Optional[float]) -> float:
    """Quantifies how much resume and interview signals disagree (0..1)."""
    if interview_score is None:
        return 0.0
    return abs(float(resume_score) - float(interview_score))


def _confidence_score(resume_score: float, interview_score: Optional[float]) -> float:
    """Simple confidence derived from signal agreement (higher is better)."""
    if interview_score is None:
        return 0.0
    return max(0.0, 1.0 - _signal_disagreement(resume_score, interview_score))


def _candidate_value(candidate: dict, interview_score: Optional[float], task: str) -> float:
    """Selective value function optimized for score-per-cost under this grader."""
    resume = float(candidate.get("resume_score", 0.0))
    if interview_score is None:
        # Prior estimate: very high resumes are suspicious on hard, slight prior penalty elsewhere.
        return resume - (0.15 if resume > 0.85 else 0.05)

    disagreement = _signal_disagreement(resume, interview_score)
    if task == "hard":
        return interview_score - (0.45 * disagreement)
    if task == "medium":
        return interview_score - (0.20 * disagreement)
    return interview_score - (0.10 * disagreement)


def _should_hire(candidate: dict, interview_score: Optional[float], task: str, threshold: float) -> bool:
    """Strict hire gate: never blind-hire, only hire high-confidence candidates."""
    if interview_score is None:
        return False
    resume = float(candidate.get("resume_score", 0.0))
    disagreement = _signal_disagreement(resume, interview_score)

    # Anti-decoy gate: reject highly inconsistent signals.
    if task == "hard" and disagreement > 0.28:
        return False
    if task == "medium" and disagreement > 0.35 and interview_score < (threshold + 0.08):
        return False

    if task == "hard":
        return interview_score >= max(0.70, threshold)
    return interview_score >= threshold


def _should_finalize(
    obs: HiringObservation,
    task: str,
    best_score: float,
    interviews_done: int,
    hires_count: int,
    policy: dict,
) -> bool:
    """Stop when marginal value is low and score-per-cost objective is satisfied."""
    min_interviews_before_hire = policy["min_interviews_before_hire"]
    min_interview_coverage = policy["min_interview_coverage"]
    min_interviews_needed = max(
        min_interviews_before_hire,
        int(round(len(obs.candidates) * min_interview_coverage)),
    )

    if interviews_done < min_interviews_needed:
        return False

    if hires_count >= policy["target_hires"]:
        return True

    if task == "hard":
        if hires_count >= 1 and best_score >= policy["early_finalize_best_score"] and interviews_done >= 2:
            return True
        if hires_count >= 1 and interviews_done >= policy["max_effective_interviews"] and best_score >= policy["late_finalize_best_score"]:
            return True
        if obs.budget_remaining < policy["min_budget_to_continue"]:
            return hires_count >= 1
        return False

    if hires_count >= 1 and best_score >= policy["finalize_best_score"]:
        return True
    if interviews_done >= policy["max_effective_interviews"]:
        return hires_count >= 1
    if obs.budget_remaining < policy["min_budget_to_continue"]:
        return hires_count >= 1
    return False


def _sanitize_model_action(model_action: Optional[dict], obs: HiringObservation) -> Optional[dict]:
    """Accept model action only when valid for current state."""
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
        c["candidate_id"] for c in obs.candidates
        if c["candidate_id"] not in obs.hires_made and c["candidate_id"] not in obs.skipped
    }
    if cid not in live_ids:
        return None

    if action == "interview" and cid in (obs.interviews_done or {}):
        return None

    return {"action": action, "candidate_id": cid}


def _format_action(action: Optional[dict]) -> str:
    if not action:
        return "<none>"
    return json.dumps(action, separators=(",", ":"))


def _explain_action_selection(
    obs: HiringObservation,
    task: str,
    safe_model_action: Optional[dict],
    policy_action: dict,
) -> str:
    """Explain why final action matches or overrides model suggestion."""
    if not safe_model_action:
        return "Model action unavailable or invalid; policy selected deterministic safe action."

    if safe_model_action == policy_action:
        return "Model action accepted by policy."

    policy = TASK_POLICY[task]
    interviews_done = len(obs.interviews_done or {})
    min_interviews_needed = max(
        policy["min_interviews_before_hire"],
        int(round(len(obs.candidates) * policy["min_interview_coverage"])),
    )

    model_action = safe_model_action.get("action")
    policy_kind = policy_action.get("action")

    if model_action == "hire" and policy_kind == "interview":
        if interviews_done < min_interviews_needed:
            return (
                f"Policy override: requires broader interview coverage "
                f"({interviews_done}/{min_interviews_needed}) before hiring."
            )
        return "Policy override: selected higher expected-value interview before committing hire."

    if model_action == "interview" and policy_kind == "interview":
        return "Policy override: selected a higher-priority interview target by deterministic ranking."

    if policy_kind == "finalize":
        return "Policy override: finalize condition met (enough evidence and low marginal value from more actions)."

    return "Policy override: deterministic scorer-selected action for reliability."


def choose_heuristic_action(obs: HiringObservation, task: str, safe_model_action: Optional[dict] = None) -> dict:
    """Deterministic policy for strong baseline performance.

    We keep the agent lightweight and reproducible by using the model for
    narrative suggestions, but the actual submitted action comes from this
    rule-based policy.
    """
    policy = TASK_POLICY[task]

    remaining = [
        c for c in obs.candidates
        if c["candidate_id"] not in obs.hires_made
        and c["candidate_id"] not in obs.skipped
    ]

    if not remaining:
        return {"action": "finalize"}

    interviewed = obs.interviews_done or {}
    interviewed_ids = set(interviewed.keys())
    uninterviewed = [c for c in remaining if c["candidate_id"] not in interviewed_ids]

    can_interview = obs.budget_remaining >= 10
    can_hire = obs.budget_remaining >= 50
    hires_count = len(obs.hires_made)

    # Build candidate lookups
    by_id = {c["candidate_id"]: c for c in remaining}
    all_by_id = {c["candidate_id"]: c for c in obs.candidates}
    hired_roles = {
        all_by_id[cid].get("role", "Unknown")
        for cid in obs.hires_made
        if cid in all_by_id
    }

    min_interviews_before_hire = policy["min_interviews_before_hire"]
    min_interview_coverage = policy["min_interview_coverage"]
    min_interviews_needed = max(
        min_interviews_before_hire,
        int(round(len(obs.candidates) * min_interview_coverage)),
    )

    # Ranked interviewed candidates by value
    interviewed_ranked = []
    for cid, score in interviewed.items():
        c = by_id.get(cid)
        if c is None:
            continue
        role_bonus = 0.08 if (policy["target_hires"] > 1 and c.get("role") not in hired_roles) else 0.0
        interviewed_ranked.append((cid, _candidate_value(c, score, task) + role_bonus, score, c["resume_score"]))
    interviewed_ranked.sort(key=lambda x: x[1], reverse=True)

    best_candidate_id = None
    best_candidate_score = -1.0
    best_candidate_interview = None
    if interviewed_ranked:
        best_candidate_id, best_candidate_score, best_candidate_interview, _ = interviewed_ranked[0]

    # Interview queue: peak information zone first, then strong resumes.
    sorted_uninterviewed = sorted(
        uninterviewed,
        key=lambda c: (
            0.10 if (policy["target_hires"] > 1 and c.get("role") not in hired_roles) else 0.0,
            _expected_interview_value(c, task, hired_roles),
        ),
        reverse=True,
    )

    if not can_interview and not can_hire:
        return {"action": "finalize"}

    interviews_done = len(interviewed)
    top_interview_ids = [c["candidate_id"] for c in sorted_uninterviewed[:3]]

    # Finalize when marginal value is low.
    if _should_finalize(obs, task, best_candidate_score, interviews_done, hires_count, policy):
        return {"action": "finalize"}

    # Let LLM shape high-value interview/hire choices when suggestions are safe.
    if safe_model_action and safe_model_action.get("action") == "interview" and can_interview:
        model_cid = safe_model_action.get("candidate_id")
        if model_cid in top_interview_ids:
            return {"action": "interview", "candidate_id": model_cid}

    if safe_model_action and safe_model_action.get("action") == "hire" and can_hire:
        model_cid = safe_model_action.get("candidate_id")
        model_candidate = by_id.get(model_cid)
        model_interview = interviewed.get(model_cid)
        if (
            model_candidate
            and interviews_done >= min_interviews_needed
            and _should_hire(model_candidate, model_interview, task, policy["hire_interview_threshold"])
        ):
            return {"action": "hire", "candidate_id": model_cid}

    # Interview high-value candidates first.
    if can_interview and interviews_done < policy["target_interviews"] and sorted_uninterviewed:
        return {"action": "interview", "candidate_id": sorted_uninterviewed[0]["candidate_id"]}

    # Keep interviewing until enough of the pool has been evaluated.
    if can_interview and interviews_done < min_interviews_needed and sorted_uninterviewed:
        return {"action": "interview", "candidate_id": sorted_uninterviewed[0]["candidate_id"]}

    # Then hire best interviewed candidate if it passes strict threshold.
    if can_hire and best_candidate_id and interviews_done >= min_interviews_needed:
        best_candidate = by_id.get(best_candidate_id)
        if best_candidate and _should_hire(best_candidate, best_candidate_interview, task, policy["hire_interview_threshold"]):
            return {"action": "hire", "candidate_id": best_candidate_id}

    # If we still can interview and have unexplored candidates, continue exploration.
    if can_interview and sorted_uninterviewed:
        return {"action": "interview", "candidate_id": sorted_uninterviewed[0]["candidate_id"]}

    return {"action": "finalize"}


# ------------------------------------------------------------------ #
#  Single episode runner                                               #
# ------------------------------------------------------------------ #

def run_episode(openai_client: OpenAI, env_client: HiringEnvClient, task: str, seed: int = 42) -> float:
    """
    Run a full episode for the given task.
    Returns the final_score from the grader.
    """
    print(f"\n{'='*60}")
    print(f"  TASK: {task.upper()}")
    print(f"{'='*60}")

    # Set seed for reproducibility
    np.random.seed(seed)
    print(f"  [INFO] Set random seed to {seed}")

    # Emit machine-parseable start line required by the validator
    print(f"[START] task={task} env=AgentHire-Arena model={MODEL_NAME}")

    obs = env_client.reset(task=task)
    step = 0
    rewards_list = []

    while not obs.done:
        step += 1
        user_content = render_observation(obs)

        # Keep prompts short and stateless to avoid context-length failures.
        model_call_ok = False
        try:
            last_exc = None
            assistant_msg = ""
            for attempt in range(MODEL_MAX_RETRIES + 1):
                try:
                    response = openai_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        max_tokens=300,
                        temperature=0.2,
                    )
                    assistant_msg = response.choices[0].message.content or ""
                    model_call_ok = True
                    break
                except Exception as exc:
                    last_exc = exc
                    retriable = _is_retriable_model_exception(exc)
                    if not retriable:
                        print(
                            f"  [ERROR] Non-retriable model API error at step {step}: {exc}. "
                            f"Request used model='{MODEL_NAME}'."
                        )
                        break
                    if attempt < MODEL_MAX_RETRIES:
                        # Retry 3 times with 1-2s spacing to handle transient overload/rate-limit errors.
                        retry_delay = 1.0 + (attempt * 0.5)
                        print(
                            f"  [WARN] Model call failed at step {step} (attempt {attempt + 1}/{MODEL_MAX_RETRIES + 1}): {exc}. "
                            f"Retrying in {retry_delay:.1f}s..."
                        )
                        time.sleep(retry_delay)
                    else:
                        print(f"  [WARN] Model call failed at step {step} after {MODEL_MAX_RETRIES} retries: {last_exc}")
        except Exception as exc:
            print(f"  [WARN] Model call failed at step {step}: {exc}")
            assistant_msg = ""

        # Parse model action and sanitize it. If invalid for current state,
        # apply a deterministic emergency fallback so action execution never fails.
        model_action = parse_action(assistant_msg) if model_call_ok else None
        safe_model_action = _sanitize_model_action(model_action, obs)
        used_fallback = False
        if safe_model_action is None:
            safe_model_action = _fallback_action_from_obs(obs)
            used_fallback = True

        # Model-first execution path: run model action directly.
        action = safe_model_action

        action_str = json.dumps(action)

        print(f"\n[Step {step}]")
        # Print the LLM's reasoning (everything before the JSON)
        reasoning = re.sub(r'\{[^{}]+\}', '', assistant_msg).strip()
        if reasoning:
            print(f"  Reasoning: {reasoning[:200]}{'...' if len(reasoning)>200 else ''}")
        print(f"  Model action:  {_format_action(model_action)}")
        print(f"  Safe action:   {_format_action(safe_model_action)}")
        print(f"  Final action:  {action_str}")
        if used_fallback:
            print("  Decision:      Model action invalid for current state; used safe fallback action.")
        else:
            print("  Decision:      Executed model action directly.")

        # Submit action to environment with retry logic for transient failures
        step_submission_ok = False
        last_step_exc = None
        obs = None
        reward = None
        
        for attempt in range(STEP_MAX_RETRIES + 1):
            try:
                obs, reward = env_client.step(
                    action=action["action"],
                    candidate_id=action.get("candidate_id"),
                )
                step_submission_ok = True
                break
            except Exception as exc:
                last_step_exc = exc
                if attempt < STEP_MAX_RETRIES:
                    # Retry 3 times with 1-2s spacing to handle transient server errors (500, 429, timeouts, etc).
                    retry_delay = 1.0 + (attempt * 0.5)
                    print(
                        f"  [WARN] Step submission failed at step {step} (attempt {attempt + 1}/{STEP_MAX_RETRIES + 1}): {exc}. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    print(f"  [WARN] Step submission failed at step {step} after {STEP_MAX_RETRIES} retries: {last_step_exc}")
        
        if not step_submission_ok:
            print(f"  [ERROR] Step submission failed at step {step}: {last_step_exc}")
            # Emit step line for debugging consistency, then abort this task cleanly.
            print(
                f"[STEP]  step={step} action={action_str} reward=0.00 done=true error=step_submission_failed"
            )
            raise last_step_exc

        print(f"  Reward:    {reward.step_reward:+.2f}  ({reward.reason})")
        print(f"  Result:    {obs.last_action_result}")

        # record reward for final reporting
        try:
            rewards_list.append(float(reward.step_reward))
        except Exception:
            rewards_list.append(0.0)

        # Emit machine-parseable step line (validator requires this exact format)
        error_field = reward.reason if reward.reason else "null"
        done_str = "true" if obs.done else "false"
        print(
            f"[STEP]  step={step} action={action_str} reward={reward.step_reward:.2f} done={done_str} error={error_field}"
        )

        if obs.done:
            break

        # Small delay to avoid rate limiting on hosted endpoints
        time.sleep(0.5)

    final_score = reward.final_score if reward.final_score is not None else 0.0
    print(f"\n  FINAL SCORE [{task}]: {final_score:.4f}")

    # Emit machine-parseable end line with full per-step rewards collected
    success = "true" if final_score is not None else "false"
    rewards_strs = [f"{r:.2f}" for r in rewards_list]
    print(f"[END]   success={success} steps={step} score={final_score:.2f} rewards={','.join(rewards_strs)}")
    return final_score


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    # Validate env vars
    if not MODEL_NAME:
        raise ValueError("MODEL_NAME environment variable is required.")

    # Initialize OpenAI-compatible client.
    # Primary path for the evaluator: API_BASE_URL + API_KEY.
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    injected_api_base_url = os.environ.get("API_BASE_URL", "").strip()
    injected_api_key = os.environ.get("API_KEY", "").strip()

    if injected_api_base_url and injected_api_key:
        print(f"[INFO] Using injected OpenAI-compatible API at {injected_api_base_url}.")
        openai_client = OpenAI(api_key=injected_api_key, base_url=injected_api_base_url)
    else:
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        hf_token = os.environ.get("HF_TOKEN") or HF_TOKEN

        if openai_key:
            openai_client = OpenAI(api_key=openai_key)
        elif provider == "hf" and hf_token:
            hf_base_url = os.environ.get("HF_API_BASE_URL", "https://router.huggingface.co/v1")
            print(f"[INFO] Using Hugging Face router at {hf_base_url}.")
            openai_client = OpenAI(api_key=hf_token, base_url=hf_base_url)
        else:
            raise ValueError(
                "No valid LLM configuration found. Please set one of: "
                "API_BASE_URL + API_KEY, OPENAI_API_KEY, or HF_TOKEN"
            )

    env_base_url = _resolve_env_base_url()
    env_client = HiringEnvClient(base_url=env_base_url)

    # Health check
    if not env_client.health():
        raise ConnectionError(
            f"Cannot reach environment at {env_base_url}. "
            "Is the server running?"
        )
    print(f"Connected to environment at {env_base_url}")
    print(f"Model: {MODEL_NAME}")
    active_base_url = getattr(getattr(openai_client, "_client", None), "base_url", None)
    if active_base_url:
        print(f"[INFO] LLM request target: {active_base_url}chat/completions")
    print(
        f"[INFO] LLM request payload summary: "
        f"model={MODEL_NAME}, max_tokens=300, temperature=0.2"
    )

    # Fetch available tasks
    response = requests.get(f"{env_base_url}/tasks", timeout=5)
    all_tasks = response.json() if response.status_code == 200 else ["easy", "medium", "hard", "adversarial", "nightmare"]

    # Get seed from environment variable or use default 42
    seed = os.environ.get("SEED", "42").strip()
    seed_value = int(seed) if seed else 42
    print(f"\n[INFO] Running with seed={seed_value}\n")

    # Run all tasks and collect results
    print(f"\n{'='*70}")
    print(f"  RUNNING {len(all_tasks)} TASKS")
    print(f"{'='*70}")

    final_scores = {}
    for task in all_tasks:
        try:
            final_score = run_episode(openai_client, env_client, task, seed=seed_value)
            final_scores[task] = final_score
        except Exception as e:
            print(f"[ERROR] Task {task} failed: {e}")
            final_scores[task] = 0.0

    # Print final summary in machine-readable format
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    for task, score in final_scores.items():
        print(f"  {task:15s}: {score:.4f}")
    avg_score = sum(final_scores.values()) / len(final_scores)
    print(f"\n  AVERAGE SCORE: {avg_score:.4f}")


if __name__ == "__main__":
    main()
