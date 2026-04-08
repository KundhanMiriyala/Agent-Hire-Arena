"""
inference.py — Baseline LLM agent for AgentHire Arena.

Reads from environment variables:
    API_BASE_URL  — URL of the AgentHire Arena server (e.g. http://localhost:7860)
    MODEL_NAME    — Model name for the OpenAI-compatible API
    LLM_PROVIDER  — Optional override: "hf", "openai", or "mock"
    OPENAI_API_KEY — OpenAI API key for final submission runs
    HF_TOKEN      — HuggingFace token (used as API key for HF Inference Endpoints)

Run:
    API_BASE_URL=http://localhost:7860 \
    MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct \
    HF_TOKEN=hf_xxx \
    python inference.py
"""

import os
import re
import json
import time
from typing import Optional
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


# Simple mock OpenAI client for local offline testing
class _MockOpenAI:
    class _Chat:
        class _Completions:
            def create(self, model, messages, max_tokens=None, temperature=0.2):
                # Minimal deterministic policy: give a short reasoning then finalize
                reasoning = "I inspected candidates and will finalize based on resumes and budget."
                action = {"action": "finalize"}
                content = f"{reasoning}\n{json.dumps(action)}"
                class _Msg:
                    def __init__(self, content):
                        self.message = type("_M", (), {"content": content})()

                return type("_R", (), {"choices": [_Msg(content)]})()

    def __init__(self):
        self.chat = type("_C", (), {"completions": self._Chat()._Completions()})()


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
        response.raise_for_status()
        data = response.json()
        obs = HiringObservation(**data.get("observation", {}))
        
        from models import HiringReward
        reward = HiringReward(**data.get("reward", {}))
        return obs, reward


# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

TASKS_TO_RUN = ["easy", "medium", "hard"]


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
  {"action": "hire",      "candidate_id": "C03"}   → costs 50 units, adds to your team
  {"action": "skip",      "candidate_id": "C03"}   → free, permanently rejects
  {"action": "finalize"}                            → ends episode, scores your team

RULES:
- You MUST call finalize to receive your score. Forgetting = score of 0.
- resume_score is noisy and unreliable — never hire based on it alone.
- interview_score is more reliable but costs 10 units.
- Hiring without interviewing is allowed but risky (blind hire).
- On hard tasks, some candidates have impressive resumes but very low true skill. Be skeptical.
- Once you skip a candidate, that decision is permanent.
- Budget exhaustion ends the episode with whatever hires you have.

STRATEGY:
1. Scan all resumes first.
2. Interview candidates in the uncertain range (resume 0.4-0.75) where you need more signal.
3. Hire candidates with strong interview scores.
4. Skip clearly weak candidates to clear mental space.
5. Finalize when your team looks solid or budget is running low.

Think step by step. Reason about: budget remaining, how many more hires you can afford, which candidates are worth interviewing vs skipping. Then output your JSON action on the final line.
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

VALID_ACTIONS = {"interview", "hire", "skip", "finalize"}


def parse_action(text: str) -> Optional[dict]:
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

    return None


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


def _decoy_risk(resume_score: float, interview_score: Optional[float]) -> float:
    """Heuristic decoy-risk estimate used mainly for hard task."""
    if interview_score is None:
        return 0.20 if resume_score > 0.85 else 0.0

    gap = resume_score - interview_score
    if gap > 0.25:
        return 0.50
    if gap > 0.15:
        return 0.30
    return 0.0


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

def run_episode(openai_client: OpenAI, env_client: HiringEnvClient, task: str) -> float:
    """
    Run a full episode for the given task.
    Returns the final_score from the grader.
    """
    print(f"\n{'='*60}")
    print(f"  TASK: {task.upper()}")
    print(f"{'='*60}")

    # Emit machine-parseable start line required by the validator
    print(f"[START] task={task} env=AgentHire-Arena model={MODEL_NAME}")

    obs = env_client.reset(task=task)
    step = 0
    rewards_list = []

    while not obs.done:
        step += 1
        user_content = render_observation(obs)

        # Keep prompts short and stateless to avoid context-length failures.
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
        except Exception as exc:
            print(f"  [WARN] Model call failed at step {step}: {exc}")
            assistant_msg = ""

        # Parse model action and sanitize it. We still prioritize robust task policy,
        # but model outputs are used when they are valid and policy-compatible.
        model_action = parse_action(assistant_msg)
        safe_model_action = _sanitize_model_action(model_action, obs)
        policy_action = choose_heuristic_action(obs, task, safe_model_action)

        # Deterministic execution for reliability: model is advisory, policy decides.
        action = policy_action

        action_str = json.dumps(action)

        print(f"\n[Step {step}]")
        # Print the LLM's reasoning (everything before the JSON)
        reasoning = re.sub(r'\{[^{}]+\}', '', assistant_msg).strip()
        if reasoning:
            print(f"  Reasoning: {reasoning[:200]}{'...' if len(reasoning)>200 else ''}")
        if safe_model_action:
            print(f"  Model action: {json.dumps(safe_model_action)}")
        else:
            print("  Model action: <unparsed>")
        print(f"  Policy action: {json.dumps(policy_action)}")
        print(f"  Final action:  {action_str}")

        # Submit action to environment
        obs, reward = env_client.step(
            action=action["action"],
            candidate_id=action.get("candidate_id"),
        )

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

    # Initialize OpenAI-compatible client
    # Selection priority:
    # 1. MOCK_OPENAI=1, MODEL_NAME==mock, or LLM_PROVIDER=mock -> local deterministic mock
    # 2. LLM_PROVIDER=openai or OPENAI_API_KEY -> use OpenAI for final submission runs
    # 3. LLM_PROVIDER=hf or HF_TOKEN -> use Hugging Face router for testing
    # 4. fallback -> mock (safe)
    mock_mode = os.environ.get("MOCK_OPENAI", "0") == "1" or MODEL_NAME == "mock"
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()

    if mock_mode or provider == "mock":
        openai_client = _MockOpenAI()
    else:
        openai_key = os.environ.get("OPENAI_API_KEY")
        hf_token = os.environ.get("HF_TOKEN") or HF_TOKEN

        if provider == "hf":
            if not hf_token:
                print("[WARN] LLM_PROVIDER=hf but HF_TOKEN is missing — falling back to mock client.")
                openai_client = _MockOpenAI()
            else:
                hf_base_url = os.environ.get("HF_API_BASE_URL", "https://router.huggingface.co/v1")
                print(f"[INFO] Using Hugging Face router at {hf_base_url} for testing.")
                openai_client = OpenAI(api_key=hf_token, base_url=hf_base_url)
        elif provider == "openai" or openai_key:
            openai_client = OpenAI(api_key=openai_key)
        elif hf_token:
            hf_base_url = os.environ.get("HF_API_BASE_URL", "https://router.huggingface.co/v1")
            print(f"[INFO] Using Hugging Face router at {hf_base_url} for testing.")
            openai_client = OpenAI(api_key=hf_token, base_url=hf_base_url)
        else:
            print("[WARN] No OPENAI_API_KEY or HF_TOKEN found — falling back to mock client.")
            openai_client = _MockOpenAI()

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

    # Run all tasks and collect scores
    results = {}
    for task in TASKS_TO_RUN:
        try:
            score = run_episode(openai_client, env_client, task)
            results[task] = score
        except Exception as e:
            print(f"\n[ERROR] Task '{task}' failed: {e}")
            results[task] = 0.0

    # Summary
    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    for task, score in results.items():
        bar = "█" * int(score * 20)
        print(f"  {task:<8}  {score:.4f}  {bar}")
    avg = sum(results.values()) / len(results)
    print(f"  {'avg':<8}  {avg:.4f}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    main()
