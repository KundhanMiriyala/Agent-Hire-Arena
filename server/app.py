import sys
import os

# Ensure project root is on the path so `models` and `server` resolve correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from models import HiringAction, HiringObservation, HiringReward
from server.environment import HiringEnvironment
from server.tasks import TASKS
from server.grader import explain_grade


# ------------------------------------------------------------------ #
#  App setup                                                           #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="AgentHire Arena",
    description=(
        "OpenEnv environment for evaluating AI agents on hiring decisions "
        "under uncertainty, budget constraints, and delayed feedback."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per server process (single-session for HF Spaces)
env = HiringEnvironment()
# Static files will be mounted at the end to avoid shadowing API endpoints


# ------------------------------------------------------------------ #
#  Request / Response schemas                                          #
# ------------------------------------------------------------------ #

class ResetRequest(BaseModel):
    task: str = "easy"   # "easy" | "medium" | "hard"


class StepRequest(BaseModel):
    action: str                          # "interview" | "probe" | "hire" | "skip" | "finalize"
    candidate_id: Optional[str] = None


class StepResponse(BaseModel):
    observation: HiringObservation
    reward: HiringReward


# ------------------------------------------------------------------ #
#  Endpoints                                                           #
# ------------------------------------------------------------------ #

@app.get("/info")
def root():
    return {
        "name": "AgentHire Arena",
        "version": "1.0.0",
        "tasks": list(TASKS.keys()),
        "endpoints": ["/reset", "/step", "/state", "/tasks"],
    }


@app.post("/reset", response_model=HiringObservation)
def reset(request: Optional[ResetRequest] = None):
    """
    Start a new episode for the given task.
    Returns the initial HiringObservation.
    """
    task = request.task if request is not None else "easy"
    if task not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task}'. Choose from: {list(TASKS.keys())}",
        )
    observation = env.reset(task=task)
    return observation


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Submit one action and receive the next observation + reward.
    """
    action = HiringAction(
        action=request.action,
        candidate_id=request.candidate_id,
    )
    try:
        observation, reward = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=observation, reward=reward)


@app.get("/state")
def state():
    """
    Returns full internal state including hidden fields (true_skill, is_decoy).
    For judge inspection and debugging only — not shown to the agent.
    """
    return env.state()


@app.get("/tasks")
def list_tasks():
    """Returns all available task configs."""
    return {
        name: {
            "name": cfg.name,
            "num_candidates": cfg.num_candidates,
            "noise_level": cfg.noise_level,
            "budget": cfg.budget,
            "decoy_fraction": cfg.decoy_fraction,
            "coached_fraction": cfg.coached_fraction,
            "adversarial": cfg.adversarial,
            "adversarial_start_step": cfg.adversarial_start_step,
            "max_steps": cfg.max_steps,
            "role_requirements": cfg.role_requirements,
            "description": cfg.description,
        }
        for name, cfg in TASKS.items()
    }


@app.get("/metrics")
def metrics():
    """Returns a diagnostics/metrics blob for judge inspection.

    Includes the grader breakdown, per-step rewards, and budget telemetry.
    """
    if env._state is None:
        raise HTTPException(status_code=400, detail="No active episode. Call reset() first.")

    task_cfg = env._task_config
    breakdown = explain_grade(env._state, task_cfg)
    # Expose step-level rewards and a few quick telemetry fields
    return {
        "metrics": breakdown,
        "step_rewards": env._state.step_rewards,
        "budget_remaining": env._state.budget_remaining,
        "budget_total": env._state.budget_total,
    }


@app.post("/agent_step")
def agent_step():
    """
    Executes one step of the baseline agent and returns the observation, reward, and reasoning.
    """
    if env._state is None:
        raise HTTPException(status_code=400, detail="No active episode. Call reset() first.")
        
    if env._state.done:
        return {"observation": env._state, "reward": {}, "reasoning": "Episode already complete.", "action": {"action": "finalize"}}

    # Import inference logic dynamically to avoid circular imports
    from inference import choose_heuristic_action, _explain_action_selection
    
    task_name = env._task_config.name if env._task_config else "easy"
    
    # Calculate next action using the baseline heuristic policy
    action = choose_heuristic_action(env._state, task_name, None)
    reasoning = _explain_action_selection(env._state, task_name, None, action)
    
    # Execute action
    hiring_action = HiringAction(action=action["action"], candidate_id=action.get("candidate_id"))
    obs, reward = env.step(hiring_action)
    
    return {
        "observation": obs,
        "reward": reward,
        "action": action,
        "reasoning": reasoning
    }

# Mount static files at the root
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #


def main(host: str = "0.0.0.0", port: int = 7860, reload: bool = False) -> None:
    """Start a Uvicorn server serving this FastAPI `app`.

    Exposed so the project can provide a script entrypoint for OpenEnv
    validation and containerized runs (`server.app:main`).
    """
    import uvicorn

    uvicorn.run("server.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
