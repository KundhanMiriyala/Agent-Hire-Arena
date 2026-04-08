# AgentHire Arena 🎯

**Evaluate AI agents on hiring decisions under uncertainty, cost constraints, and delayed feedback.**

AgentHire Arena is an [OpenEnv](https://openenv.dev) environment that challenges an AI agent to build the best team possible — without ever seeing a candidate's true skill. Resumes are noisy, interviews are costly, and performance is only revealed at the end.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# 3. In a new terminal, run the LLM agent baseline
export API_BASE_URL=http://localhost:7860
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...
python inference.py
```

---

## Tasks

| Task | Candidates | Noise (σ) | Budget | Decoys | Key Challenge |
|------|-----------|-----------|--------|--------|--------------|
| 🟢 easy | 5 | 0.05 | 300 | 0% | Pick best from clean signals |
| 🟡 medium | 10 | 0.15 | 220 | 0% | Balance interview cost vs quality |
| 🔴 hard | 20 | 0.30 | 200 | 25% | Avoid decoys, zero budget slack |

**Hard task budget math:** 5 interviews (×10) + 3 hires (×50) = 200 exactly. Zero slack.

---

## Action Space

| Action | Parameters | Cost | Effect |
|--------|-----------|------|--------|
| `interview` | `candidate_id` | 10 | Reveals `interview_score`. Once per candidate. |
| `hire` | `candidate_id` | 50 | Hires the candidate. Blind hire = penalty. |
| `skip` | `candidate_id` | 0 | Permanently rejects. Free. |
| `finalize` | — | 0 | Ends episode, triggers grader. **Must call to get score.** |

---

## Observation Space

```json
{
  "candidates": [
    {
      "candidate_id": "C01",
      "name": "Alice",
      "resume_score": 0.82,
      "years_experience": 7,
      "skills": ["Python", "Machine Learning", "SQL"]
    }
  ],
  "budget_remaining": 220.0,
  "interviews_done": {"C03": 0.741},
  "hires_made": ["C03"],
  "skipped": ["C07", "C09"],
  "step_num": 4,
  "max_steps": 30,
  "last_action_result": "Hired Carol (C03) — informed hire. Budget remaining: 160.",
  "done": false
}
```

**Hidden from agent:** `true_skill`, `is_decoy` — revealed only at `/state` for judge inspection.

---

## Reward Function

### Step-level (trajectory signal)

| Event | Reward |
|-------|--------|
| Interview a candidate in uncertain zone (resume 0.4–0.75) | +0.05 |
| Hire after interviewing | +0.10 |
| Blind hire (hire without interview) | −0.05 |
| Budget exhausted mid-episode | −0.10 |
| Skip | 0.00 |

### Final score (on `finalize`)

**Easy / Medium:**
```
score = avg(true_skill of hires) − (total_cost / budget)
```

**Hard:**
```
score = avg(true_skill of hires) − (total_cost / budget) − 0.20 × (decoy_hires / total_hires)
```

All scores clipped to `[0.0, 1.0]`.

---

## API Endpoints

```
POST /reset        Body: {"task": "easy"}
POST /step         Body: {"action": "interview", "candidate_id": "C01"}
GET  /state        Full internal state (true_skill visible — for judges only)
GET  /tasks        List all task configs
```

---

## What Makes It Hard

- **Partial observability** — `true_skill` is never revealed
- **Misleading signals** — decoys (Hard) have high resume scores but low true skill
- **Delayed rewards** — hire quality only scored at `finalize()`
- **Budget pressure** — every interview and hire is irreversible
- **Exploration vs exploitation** — interviewing reduces hiring capacity

---

## Environment Variables for `inference.py`

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | URL of the environment server |
| `MODEL_NAME` | Model name for the OpenAI-compatible API |
| `HF_TOKEN` | HuggingFace token (used as API key for HF endpoints) |

---

## Project Structure

```
agenthire-arena/
├── models.py                  # Pydantic dataclasses
├── client.py                  # HTTP client for the environment
├── inference.py               # LLM agent baseline (agentic loop)
├── openenv.yaml               # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
├── README.md
└── server/
    ├── __init__.py
    ├── app.py                 # FastAPI application
    ├── environment.py         # Core reset/step/state logic
    ├── tasks.py               # Task configs (easy/medium/hard)
    ├── candidate_generator.py # Seeded candidate pool generator
    └── grader.py              # Deterministic final scorer
```

---

## Baseline Scores

Run `python inference.py` with GPT-4o-mini to reproduce:

| Task | Baseline Score |
|------|---------------|
| easy | ~0.62 |
| medium | ~0.48 |
| hard | ~0.31 |

*(Scores are deterministic given fixed seeds and same model.)*

---

## Pre-submit Checklist

Run these commands before submitting to the hackathon to ensure the repo passes validation and builds cleanly.

1. Install dependencies and validator

```bash
pip install -r requirements.txt
pip install openenv-core
```

2. Run tests

```bash
pytest
```

3. Validate OpenEnv spec

```bash
openenv validate
```

4. Regenerate `uv.lock` (run on a machine with `uv` available)

```bash
uv lock
git add uv.lock
git commit -m "Regenerate uv.lock"
```

5. Build Docker image

```bash
docker build -t agenthire-arena .
```

6. Sanity-run the server and check endpoints

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
curl -s http://localhost:7860/tasks
curl -s -X POST http://localhost:7860/reset -H 'Content-Type: application/json' -d '{"task":"easy"}'
```

7. Run the baseline (smoke)

```bash
API_BASE_URL=http://localhost:7860 MODEL_NAME=local-dummy HF_TOKEN=token python inference.py
```

8. Verify metrics endpoint

```bash
curl -s http://localhost:7860/metrics | jq .
```

If all of the above succeed, the repository is ready for submission.
