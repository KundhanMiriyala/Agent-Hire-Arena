from server.environment import HiringEnvironment
from models import HiringAction


def test_interview_reduces_budget_and_records_interview():
    env = HiringEnvironment()
    obs = env.reset(task="easy")

    # Pick first candidate id
    cid = obs.candidates[0]["candidate_id"]

    before_budget = obs.budget_remaining
    observation, reward = env.step(HiringAction(action="interview", candidate_id=cid))

    # Budget should be decreased by 10
    assert observation.budget_remaining == before_budget - 10.0
    # Interview should be recorded
    assert cid in observation.interviews_done


def test_hire_without_interview_applies_penalty():
    env = HiringEnvironment()
    obs = env.reset(task="easy")
    cid = obs.candidates[0]["candidate_id"]

    # Hire without interviewing
    observation, reward = env.step(HiringAction(action="hire", candidate_id=cid))

    # Reward step_reward should include negative blind hire penalty (-0.05)
    assert reward.step_reward <= 0.0
    assert cid in observation.hires_made


def test_skip_records_candidate_without_budget_cost():
    env = HiringEnvironment()
    obs = env.reset(task="easy")
    cid = obs.candidates[0]["candidate_id"]

    before_budget = obs.budget_remaining
    observation, reward = env.step(HiringAction(action="skip", candidate_id=cid))

    assert observation.budget_remaining == before_budget
    assert cid in observation.skipped
    assert reward.step_reward == 0.0


def test_probe_before_interview_is_blocked(monkeypatch):
    monkeypatch.setenv("DISABLE_LIVE_PROBE_JUDGE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = HiringEnvironment()
    obs = env.reset(task="easy")
    cid = obs.candidates[0]["candidate_id"]

    observation, reward = env.step(HiringAction(action="probe", candidate_id=cid))

    assert reward.step_reward == 0.0
    assert "must interview" in reward.reason.lower()
    assert cid not in observation.probes_done


def test_probe_after_interview_records_probe_score_and_gap(monkeypatch):
    monkeypatch.setenv("DISABLE_LIVE_PROBE_JUDGE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = HiringEnvironment()
    obs = env.reset(task="easy")
    cid = obs.candidates[0]["candidate_id"]

    obs, _ = env.step(HiringAction(action="interview", candidate_id=cid))
    before_budget = obs.budget_remaining
    observation, reward = env.step(HiringAction(action="probe", candidate_id=cid))

    assert observation.budget_remaining == before_budget - 20.0
    assert cid in observation.probes_done
    assert cid in observation.probe_gaps
    assert observation.probe_gaps[cid] == round(
        observation.interviews_done[cid] - observation.probes_done[cid], 3
    )
    assert reward.step_reward >= 0.03


def test_probe_twice_is_blocked(monkeypatch):
    monkeypatch.setenv("DISABLE_LIVE_PROBE_JUDGE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = HiringEnvironment()
    obs = env.reset(task="easy")
    cid = obs.candidates[0]["candidate_id"]

    env.step(HiringAction(action="interview", candidate_id=cid))
    env.step(HiringAction(action="probe", candidate_id=cid))
    observation, reward = env.step(HiringAction(action="probe", candidate_id=cid))

    assert reward.step_reward == 0.0
    assert "already probed" in reward.reason.lower()
    assert cid in observation.probes_done
