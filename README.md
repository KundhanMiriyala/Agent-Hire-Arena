# AgentHire Arena

AgentHire Arena is an OpenEnv hiring simulation for evaluating agent decision-making under uncertainty, budget constraints, and delayed final rewards.

The agent must build a high-quality team while deciding when to interview, hire, skip, and finalize. Candidate resumes are noisy proxies, interview signals cost budget, and true skill is hidden until grading.

## Why This Benchmark

- Sequential decision-making with irreversible actions.
- Budget-constrained exploration vs exploitation.
- Hidden-signal setting with realistic decoys (hard mode).
- Deterministic tasks and seeds for fully reproducible runs.
- Judge-friendly introspection endpoints and score breakdown metrics.

## Submission Snapshot

- Environment: FastAPI OpenEnv-compatible server.
- Tasks: easy, medium, hard.
- Determinism: fixed task seeds and stable candidate generation.
- Scoring: deterministic [0.0, 1.0] with transparent breakdown.
- Baseline policy: deterministic heuristic loop in [inference.py](inference.py).
- Extended benchmark policies: modular decision core in [policy.py](policy.py).

## Task Levels

| Task | Candidates | Noise (sigma) | Budget | Decoy Fraction | Main Challenge |
|---|---:|---:|---:|---:|---|
| easy | 5 | 0.05 | 300 | 0.0 | Basic quality selection |
| medium | 10 | 0.15 | 220 | 0.0 | Cost-quality tradeoff |
| hard | 20 | 0.30 | 200 | 0.25 | Decoy resistance with zero slack |

Hard-mode budget math: 5 interviews x 10 + 3 hires x 50 = 200.

## Action Space

| Action | Input | Cost | Outcome |
|---|---|---:|---|
| interview | candidate_id | 10 | Reveals interview_score (once per candidate) |
| hire | candidate_id | 50 | Adds candidate to team |
| skip | candidate_id | 0 | Permanently rejects candidate |
| finalize | - | 0 | Ends episode and computes final score |

Important: the agent must call finalize to receive the final score.

## Observation Model

The agent receives:

- candidate profiles with visible fields (resume_score, role, skills, etc.)
- budget_remaining
- interviews_done
- hires_made
- skipped
- step_num and max_steps
- last_action_result
- done

The agent does not receive true_skill or is_decoy.

## Scoring (Implemented Grader)

Final grading is implemented in [server/grader.py](server/grader.py) and uses a multi-objective formula:

score = avg_true_skill + team_size_bonus + role_coverage_bonus - cost_penalty - decoy_penalty

Where:

- avg_true_skill = average true skill of hired candidates
- team_size_bonus = min(team_size / 3, 1) x 0.25
- role_coverage_bonus = 0.20 x (covered_required_roles / total_required_roles)
- cost_penalty = (cost_ratio ^ 1.3) x 0.40
- decoy_penalty = 0.25 x decoy_hire_ratio
- early finalize multiplier = 0.70 when step_num < 3
- final score is clipped to [0.0, 1.0]

Step-level shaped rewards are implemented in [server/environment.py](server/environment.py):

- +0.05 interview in uncertain resume zone (0.40 to 0.75)
- +0.10 informed hire (after interview)
- -0.05 blind hire
- -0.10 budget effectively exhausted

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | /reset | Start episode for task |
| POST | /step | Apply one action |
| GET | /state | Full internal state for judges/debug |
| GET | /tasks | Task metadata |
| GET | /metrics | Grader breakdown and telemetry |

## Project Structure

| Path | Purpose |
|---|---|
| [server/app.py](server/app.py) | FastAPI server and endpoints |
| [server/environment.py](server/environment.py) | Core reset/step/state environment logic |
| [server/tasks.py](server/tasks.py) | Task definitions and budgets |
| [server/candidate_generator.py](server/candidate_generator.py) | Deterministic candidate generation |
| [server/grader.py](server/grader.py) | Final score computation and explanation |
| [models.py](models.py) | Shared action/observation/reward/state models |
| [inference.py](inference.py) | Baseline agent loop |
| [policy.py](policy.py) | Modular policy core and variants |
| [scripts/benchmark_policies.py](scripts/benchmark_policies.py) | Variant and ablation benchmark runner |

## Quickstart

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Start environment server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3) Run baseline inference

Bash:

```bash
export OPENENV_API_BASE_URL=http://127.0.0.1:7860
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...
python inference.py
```

PowerShell:

```powershell
$env:OPENENV_API_BASE_URL = "http://127.0.0.1:7860"
$env:MODEL_NAME = "gpt-4o-mini"
$env:OPENAI_API_KEY = "sk-..."
python inference.py
```

## Policy Variants

Core environment behavior is unchanged. Policy improvements are opt-in and preserve baseline comparability.

Environment variables:

| Variable | Values | Default | Purpose |
|---|---|---|---|
| POLICY_VARIANT | baseline, task-aware, planning | baseline | Select policy family |
| POLICY_ROLE_AWARE | 0 or 1 | 1 | Enable role composition-aware selection |
| POLICY_DECOY_GUARD | 0 or 1 | 1 | Enable decoy-risk guardrails |

Examples:

```bash
POLICY_VARIANT=baseline python inference.py
POLICY_VARIANT=task-aware python inference.py
POLICY_VARIANT=planning python inference.py
```

## Benchmark and Ablations

Run benchmark variants and ablation toggles with grader-aligned reporting:

```bash
python scripts/benchmark_policies.py --env http://127.0.0.1:7860 --repeats 3
```

Outputs:

- logs/policy_benchmark.json
- logs/policy_benchmark.md

The report includes:

- per-task and overall scores
- mean and standard deviation over repeats
- grader component breakdown (avg_true_skill, team_size_bonus, role_coverage_bonus, cost_penalty, decoy_penalty)

## Reproducibility

- Task generation is deterministic per task seed.
- Interview noise uses stable candidate/task-derived seeds.
- Policy execution is deterministic for a fixed code version and settings.
- Evaluations are reproducible across runs on the same commit.

## Hackathon Judge Guide

Recommended review flow:

1. Start server and run baseline once.
2. Run policy benchmark with repeats.
3. Inspect logs/policy_benchmark.md for variant deltas.
4. Inspect /metrics and /state for transparent score attribution.

## Pre-Submission Checklist

```bash
pip install -r requirements.txt
python -m pytest
openenv validate
docker build -t agenthire-arena .
```

Then verify:

- /tasks, /reset, /step, /state, /metrics endpoints are healthy.
- baseline run completes all tasks and finalizes.
- benchmark script writes JSON and Markdown reports.

## Deployment Notes (Hugging Face Spaces)

The repository is Docker-ready.

1. Create a Docker Space.
2. Connect this repository.
3. Ensure port 7860 is exposed.
4. App entrypoint is [app.py](app.py), which exposes the FastAPI app from [server/app.py](server/app.py).

## License

MIT
