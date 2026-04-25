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

    assert reward.step_reward == -0.10
    assert cid in observation.hires_made


def test_skip_records_candidate_without_budget_cost():
    env = HiringEnvironment()
    obs = env.reset(task="easy")
    cid = obs.candidates[0]["candidate_id"]

    before_budget = obs.budget_remaining
    observation, reward = env.step(HiringAction(action="skip", candidate_id=cid))

    assert observation.budget_remaining == before_budget
    assert cid in observation.skipped
    assert reward.step_reward >= 0.0


def test_probe_before_interview_is_blocked(monkeypatch):
    monkeypatch.setenv("DISABLE_LIVE_PROBE_JUDGE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = HiringEnvironment()
    obs = env.reset(task="easy")
    cid = obs.candidates[0]["candidate_id"]

    observation, reward = env.step(HiringAction(action="probe", candidate_id=cid))

    assert reward.step_reward == -0.02
    assert "must interview" in reward.reason.lower()
    assert cid not in observation.probes_done


def test_probe_after_interview_records_useful_probe_reward(monkeypatch):
    monkeypatch.setenv("DISABLE_LIVE_PROBE_JUDGE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = HiringEnvironment()
    obs = env.reset(task="easy")
    cid = obs.candidates[0]["candidate_id"]

    obs, _ = env.step(HiringAction(action="interview", candidate_id=cid))
    monkeypatch.setattr(env, "_probe_score", lambda candidate, interview_score: (max(0.0, interview_score - 0.25), "Judge=fallback."))
    before_budget = obs.budget_remaining
    observation, reward = env.step(HiringAction(action="probe", candidate_id=cid))

    assert observation.budget_remaining == before_budget - 20.0
    assert cid in observation.probes_done
    assert cid in observation.probe_gaps
    assert observation.probe_gaps[cid] == round(
        observation.interviews_done[cid] - observation.probes_done[cid], 3
    )
    assert reward.step_reward >= 0.05


def test_low_value_probe_under_tight_budget_gets_penalty(monkeypatch):
    env = HiringEnvironment()
    obs = env.reset(task="easy")
    cid = obs.candidates[0]["candidate_id"]

    obs, _ = env.step(HiringAction(action="interview", candidate_id=cid))
    env._state.budget_remaining = 60.0
    env._get_candidate(cid).resume_score = 0.50
    env._state.interviews_done[cid] = 0.50
    monkeypatch.setattr(env, "_probe_score", lambda candidate, interview_score: (interview_score, "Judge=fallback."))

    observation, reward = env.step(HiringAction(action="probe", candidate_id=cid))

    assert cid in observation.probes_done
    assert reward.step_reward < 0.0


def test_probe_twice_is_blocked(monkeypatch):
    monkeypatch.setenv("DISABLE_LIVE_PROBE_JUDGE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = HiringEnvironment()
    obs = env.reset(task="easy")
    cid = obs.candidates[0]["candidate_id"]

    env.step(HiringAction(action="interview", candidate_id=cid))
    env.step(HiringAction(action="probe", candidate_id=cid))
    observation, reward = env.step(HiringAction(action="probe", candidate_id=cid))

    assert reward.step_reward == -0.02
    assert "already probed" in reward.reason.lower()
    assert cid in observation.probes_done


def test_finalize_returns_final_score_as_step_reward():
    env = HiringEnvironment()
    obs = env.reset(task="easy")
    cid = obs.candidates[0]["candidate_id"]

    env.step(HiringAction(action="interview", candidate_id=cid))
    env.step(HiringAction(action="hire", candidate_id=cid))
    observation, reward = env.step(HiringAction(action="finalize"))

    assert observation.done
    assert reward.final_score is not None
    assert reward.step_reward == round(reward.final_score, 4)


def test_informed_hire_reward_exceeds_blind_hire():
    blind_env = HiringEnvironment()
    blind_obs = blind_env.reset(task="easy")
    cid = blind_obs.candidates[0]["candidate_id"]
    _, blind_reward = blind_env.step(HiringAction(action="hire", candidate_id=cid))

    informed_env = HiringEnvironment()
    informed_obs = informed_env.reset(task="easy")
    cid = informed_obs.candidates[0]["candidate_id"]
    informed_env.step(HiringAction(action="interview", candidate_id=cid))
    informed_env._state.interviews_done[cid] = 0.95
    _, informed_reward = informed_env.step(HiringAction(action="hire", candidate_id=cid))

    assert informed_reward.step_reward > blind_reward.step_reward
