import pytest

from models import HiringState, CandidateProfile
from server.tasks import get_task
from server.grader import grade


def make_state_with_hires(candidates, hires_made, budget_total, budget_remaining, task_name="easy"):
    return HiringState(
        task_name=task_name,
        candidates=candidates,
        budget_remaining=budget_remaining,
        budget_total=budget_total,
        step_rewards=[],
        interviews_done={},
        hires_made=hires_made,
        skipped=[],
        step_num=0,
        max_steps=10,
        last_action_result="",
        done=True,
        final_score=None,
    )


def test_grade_no_hires_returns_zero():
    task = get_task("easy")
    state = make_state_with_hires([], [], task.budget, task.budget)
    assert grade(state, task) == 0.0


def test_grade_easy_formula():
    # Two hired candidates with known true skills
    c1 = CandidateProfile(candidate_id="C01", name="A", resume_score=0.8, years_experience=5, skills=["Python"], true_skill=0.8)
    c2 = CandidateProfile(candidate_id="C02", name="B", resume_score=0.6, years_experience=3, skills=["SQL"], true_skill=0.6)

    task = get_task("easy")
    # Use the task's budget so cost ratio calculation is correct.
    # To get cost_ratio=0.5 (score = 0.7 - 0.5 = 0.2) we set total_cost=0.5*b
    state = make_state_with_hires(
        [c1, c2], ["C01", "C02"], task.budget, task.budget - (0.5 * task.budget), task_name="easy"
    )

    sc = grade(state, task)
    # avg_true_skill = 0.7, cost_ratio = 0.5 -> score = 0.2
    assert pytest.approx(sc, rel=1e-3) == 0.2
