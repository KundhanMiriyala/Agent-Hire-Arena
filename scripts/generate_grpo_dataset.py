import argparse
import json
import os
import uuid
from typing import Any, Dict, List

# Add repo root to import path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from inference import (
    HF_TOKEN,
    MODEL_NAME,
    SYSTEM_PROMPT,
    HiringEnvClient,
    _resolve_env_base_url,
    _sanitize_model_action,
    choose_heuristic_action,
    parse_action,
    render_observation,
)


def _build_client() -> OpenAI:
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    injected_api_base_url = os.environ.get("API_BASE_URL", "").strip()
    injected_api_key = os.environ.get("API_KEY", "").strip()

    if injected_api_base_url and injected_api_key:
        return OpenAI(api_key=injected_api_key, base_url=injected_api_base_url)

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    hf_token = os.environ.get("HF_TOKEN") or HF_TOKEN

    if openai_key:
        return OpenAI(api_key=openai_key)

    if provider == "hf" and hf_token:
        hf_base_url = os.environ.get("HF_API_BASE_URL", "https://router.huggingface.co/v1")
        return OpenAI(api_key=hf_token, base_url=hf_base_url)

    raise ValueError(
        "No valid LLM configuration found. Set API_BASE_URL+API_KEY, OPENAI_API_KEY, or HF_TOKEN with LLM_PROVIDER=hf."
    )


def _sample_group(
    client: OpenAI,
    user_prompt: str,
    group_size: int,
    temperature: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    for _ in range(group_size):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            text = f"ERROR: {exc}"

        parsed = parse_action(text) if not text.startswith("ERROR:") else None
        samples.append(
            {
                "completion": text,
                "parsed_action": parsed,
            }
        )

    return samples


def _run_episode(
    client: OpenAI,
    env: HiringEnvClient,
    task: str,
    episode_id: int,
    group_size: int,
    temperature: float,
    max_tokens: int,
    action_mode: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    obs = env.reset(task=task)
    step = 0

    while not obs.done:
        step += 1
        user_prompt = render_observation(obs)
        group_id = str(uuid.uuid4())
        samples = _sample_group(
            client=client,
            user_prompt=user_prompt,
            group_size=group_size,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        first_parsed = samples[0]["parsed_action"] if samples else None
        safe_model_action = _sanitize_model_action(first_parsed, obs)
        policy_action = choose_heuristic_action(obs, task, safe_model_action)

        if action_mode == "model" and safe_model_action is not None:
            executed_action = safe_model_action
            selection_reason = "model"
        elif action_mode == "model" and safe_model_action is None:
            executed_action = policy_action
            selection_reason = "fallback_policy_invalid_model"
        else:
            executed_action = policy_action
            selection_reason = "policy"

        obs, reward = env.step(
            action=executed_action["action"],
            candidate_id=executed_action.get("candidate_id"),
        )

        rows.append(
            {
                "dataset": "agenthire-grpo-rollout",
                "group_id": group_id,
                "task": task,
                "episode_id": episode_id,
                "step": step,
                "model": MODEL_NAME,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "samples": samples,
                "selected_action": executed_action,
                "selection_reason": selection_reason,
                "policy_action": policy_action,
                "safe_model_action": safe_model_action,
                "step_reward": float(reward.step_reward),
                "final_score": float(reward.final_score) if reward.final_score is not None else None,
                "done": bool(obs.done),
                "next_obs": {
                    "budget_remaining": obs.budget_remaining,
                    "interviews_done": obs.interviews_done,
                    "probes_done": obs.probes_done,
                    "probe_gaps": obs.probe_gaps,
                    "hires_made": obs.hires_made,
                    "skipped": obs.skipped,
                    "step_num": obs.step_num,
                    "max_steps": obs.max_steps,
                    "last_action_result": obs.last_action_result,
                },
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grouped rollout dataset for GRPO training.")
    parser.add_argument(
        "--tasks",
        default="easy,medium,hard,adversarial,nightmare",
        help="Comma-separated tasks to include.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=4, help="Number of completions sampled per state.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument(
        "--action-mode",
        choices=["model", "policy"],
        default="model",
        help="Action source used to step the env. model mode falls back to policy when model action is invalid.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("outputs", "grpo", "rollouts.jsonl"),
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    client = _build_client()
    env = HiringEnvClient(base_url=_resolve_env_base_url())

    if not env.health():
        raise RuntimeError(f"Environment not reachable at {env.base_url}. Start server first.")

    total_rows = 0
    episode_counter = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for task in tasks:
            for _ in range(args.episodes_per_task):
                episode_counter += 1
                rows = _run_episode(
                    client=client,
                    env=env,
                    task=task,
                    episode_id=episode_counter,
                    group_size=args.group_size,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    action_mode=args.action_mode,
                )
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=True) + "\n")
                total_rows += len(rows)

    print(f"Saved {total_rows} grouped rollout rows to {args.output}")
    print("Use action-mode=model for GRPO-style actor learning; action-mode=policy for behavior-cloned baseline data.")


if __name__ == "__main__":
    main()
