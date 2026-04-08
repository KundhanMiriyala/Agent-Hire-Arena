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
