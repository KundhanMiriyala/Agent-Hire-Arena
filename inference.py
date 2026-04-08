"""
inference.py — Baseline LLM agent for AgentHire Arena.

Reads from environment variables:
    API_BASE_URL  — URL of the AgentHire Arena server (e.g. http://localhost:7860)
    MODEL_NAME    — Model name for the OpenAI-compatible API
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
            response = requests_post(f"{self.base_url}/tasks", timeout=5)
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
            interview_str = f" | INTERVIEWED: {score:.3f}"

        lines.append(
            f"  {cid}: {c['name']:<10} "
            f"resume={c['resume_score']:.2f}  "
            f"exp={c['years_experience']}yrs  "
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


def parse_action(text: str) -> dict:
    """
    Extract the JSON action from the LLM's response text.
    Searches for the last valid JSON object containing an 'action' key.
    Falls back to finalize if nothing parseable is found.
    """
    # Find all JSON-like blocks in the response
    matches = re.findall(r'\{[^{}]+\}', text, re.DOTALL)

    for raw in reversed(matches):   # last match = most likely the action
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and obj.get("action") in VALID_ACTIONS:
                return obj
        except json.JSONDecodeError:
            continue

    # If we can't parse, finalize safely rather than looping forever
    print(f"  [WARN] Could not parse action from response. Falling back to finalize.")
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
    messages = []
    step = 0
    rewards_list = []

    while not obs.done:
        step += 1
        user_content = render_observation(obs)
        messages.append({"role": "user", "content": user_content})

        # LLM call
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *messages,
            ],
            max_tokens=600,
            temperature=0.2,   # low temp for consistent, reasoned decisions
        )

        assistant_msg = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_msg})

        # Parse action
        action = parse_action(assistant_msg)
        action_str = json.dumps(action)

        print(f"\n[Step {step}]")
        # Print the LLM's reasoning (everything before the JSON)
        reasoning = re.sub(r'\{[^{}]+\}', '', assistant_msg).strip()
        if reasoning:
            print(f"  Reasoning: {reasoning[:200]}{'...' if len(reasoning)>200 else ''}")
        print(f"  Action:    {action_str}")

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
    # 1. MOCK_OPENAI=1 or MODEL_NAME==mock -> local deterministic mock
    # 2. USE_GEMINI=1 -> Google Generative API (Gemini)
    # 3. HF_TOKEN + HF router in API_BASE_URL -> use HF Inference via OpenAI-compatible client
    # 4. OPENAI_API_KEY -> use OpenAI
    # 5. fallback -> mock (safe)
    mock_mode = os.environ.get("MOCK_OPENAI", "0") == "1" or MODEL_NAME == "mock"
    use_gemini = os.environ.get("USE_GEMINI", "0") == "1" or os.environ.get("MODEL_PROVIDER", "") == "gemini"

    if mock_mode:
        openai_client = _MockOpenAI()
    elif use_gemini:
        google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not google_key:
            print("[WARN] USE_GEMINI=1 but no GOOGLE_API_KEY found — falling back to mock client.")
            openai_client = _MockOpenAI()
        else:
            openai_client = GeminiClient(api_key=google_key)
    else:
        hf_token = os.environ.get("HF_TOKEN") or HF_TOKEN
        openai_key = os.environ.get("OPENAI_API_KEY")

        # Use Hugging Face router when HF_TOKEN is set and API_BASE_URL looks like a HF router
        if hf_token and "huggingface" in (API_BASE_URL or "").lower():
            openai_client = OpenAI(api_key=hf_token, base_url=API_BASE_URL)
        elif openai_key:
            openai_client = OpenAI(api_key=openai_key)
        elif hf_token:
            # HF token provided but API_BASE_URL not set to router; attempt using HF router by default
            print("[WARN] HF_TOKEN provided but API_BASE_URL doesn't look like HF router. Using HF router base_url by default.")
            openai_client = OpenAI(api_key=hf_token, base_url="https://router.huggingface.co/v1")
        else:
            print("[WARN] No HF_TOKEN or OPENAI_API_KEY found — falling back to mock client. Set HF_TOKEN or OPENAI_API_KEY for real runs.")
            openai_client = _MockOpenAI()

    env_client = HiringEnvClient(base_url=API_BASE_URL)

    # Health check
    if not env_client.health():
        raise ConnectionError(
            f"Cannot reach environment at {API_BASE_URL}. "
            "Is the server running?"
        )
    print(f"Connected to environment at {API_BASE_URL}")
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
