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

    # Blind hire penalty is now stronger.
    assert reward.step_reward <= -0.10
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

    assert reward.step_reward < 0.0
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
    assert reward.step_reward != 0.0


def test_probe_twice_is_blocked(monkeypatch):
    monkeypatch.setenv("DISABLE_LIVE_PROBE_JUDGE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = HiringEnvironment()
    obs = env.reset(task="easy")
    cid = obs.candidates[0]["candidate_id"]

    env.step(HiringAction(action="interview", candidate_id=cid))
    env.step(HiringAction(action="probe", candidate_id=cid))
    observation, reward = env.step(HiringAction(action="probe", candidate_id=cid))

    assert reward.step_reward < 0.0
    assert "already probed" in reward.reason.lower()
    assert cid in observation.probes_done


def test_finalize_step_reward_matches_final_score():
    env = HiringEnvironment()
    obs = env.reset(task="easy")
    cid = obs.candidates[0]["candidate_id"]

    env.step(HiringAction(action="interview", candidate_id=cid))
    env.step(HiringAction(action="hire", candidate_id=cid))
    _, reward = env.step(HiringAction(action="finalize"))

    assert reward.final_score is not None
    assert reward.step_reward == reward.final_score


def test_invalid_and_duplicate_actions_are_penalized():
    env = HiringEnvironment()
    obs = env.reset(task="easy")
    cid = obs.candidates[0]["candidate_id"]

    _, invalid_reward = env.step(HiringAction(action="interview", candidate_id="C99"))
    assert invalid_reward.step_reward < 0.0

    env.step(HiringAction(action="interview", candidate_id=cid))
    _, duplicate_reward = env.step(HiringAction(action="interview", candidate_id=cid))
    assert duplicate_reward.step_reward < 0.0


def test_useful_probe_gap_gets_positive_reward(monkeypatch):
    class _Rng:
        def normal(self, _mu, _sigma):
            return 0.0

    monkeypatch.setattr("server.environment.np.random.default_rng", lambda *_args, **_kwargs: _Rng())

    env = HiringEnvironment()
    obs = env.reset(task="easy")
    cid = obs.candidates[0]["candidate_id"]

    env._state.interviews_done[cid] = 0.95
    candidate = env._get_candidate(cid)
    candidate.true_skill = 0.20
    env._state.budget_remaining = 200.0

    _, reward = env.step(HiringAction(action="probe", candidate_id=cid))
    assert reward.step_reward > 0.0


def test_low_value_probe_under_tight_budget_gets_penalty(monkeypatch):
    class _Rng:
        def normal(self, _mu, _sigma):
            return 0.0

    monkeypatch.setattr("server.environment.np.random.default_rng", lambda *_args, **_kwargs: _Rng())

    env = HiringEnvironment()
    obs = env.reset(task="easy")
    cid = obs.candidates[0]["candidate_id"]

    candidate = env._get_candidate(cid)
    candidate.true_skill = 0.50
    candidate.resume_score = 0.40
    env._state.interviews_done[cid] = 0.51
    env._state.budget_remaining = 60.0

    _, reward = env.step(HiringAction(action="probe", candidate_id=cid))
    assert reward.step_reward < 0.0


def test_blind_hire_is_worse_than_informed_hire():
    env_blind = HiringEnvironment()
    obs_blind = env_blind.reset(task="easy")
    cid_blind = obs_blind.candidates[0]["candidate_id"]
    _, blind_reward = env_blind.step(HiringAction(action="hire", candidate_id=cid_blind))

    env_informed = HiringEnvironment()
    obs_inf = env_informed.reset(task="easy")
    cid_inf = obs_inf.candidates[0]["candidate_id"]
    env_informed.step(HiringAction(action="interview", candidate_id=cid_inf))
    _, informed_reward = env_informed.step(HiringAction(action="hire", candidate_id=cid_inf))

    assert blind_reward.step_reward < informed_reward.step_reward


def test_adversarial_and_nightmare_emit_npc_and_keep_final_scoring_stable():
    for task in ["adversarial", "nightmare"]:
        env = HiringEnvironment()
        obs = env.reset(task=task)
        saw_npc_message = False

        # Take no-cost valid actions to progress turns and trigger NPC messaging.
        for c in obs.candidates:
            if obs.done:
                break
            if c["candidate_id"] in obs.skipped:
                continue
            obs, _ = env.step(HiringAction(action="skip", candidate_id=c["candidate_id"]))
            if getattr(obs, "hiring_manager_message", None):
                saw_npc_message = True

        # finalize should still compute standard final score and mirror it in step_reward
        obs, reward = env.step(HiringAction(action="finalize"))
        assert reward.final_score is not None
        assert reward.step_reward == reward.final_score
        assert 0.0 <= reward.final_score <= 1.0
        assert saw_npc_message
