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
    # Two hired candidates with known true skills and distinct roles.
    c1 = CandidateProfile(
        candidate_id="C01",
        name="A",
        resume_score=0.8,
        years_experience=5,
        role="ML Engineer",
        skills=["Python"],
        true_skill=0.8,
    )
    c2 = CandidateProfile(
        candidate_id="C02",
        name="B",
        resume_score=0.6,
        years_experience=3,
        role="Backend",
        skills=["SQL"],
        true_skill=0.6,
    )

    task = get_task("easy")
    # Use the task's budget so cost ratio calculation is correct.
    # To get cost_ratio=0.5 (score = 0.7 - 0.5 = 0.2) we set total_cost=0.5*b
    state = make_state_with_hires(
        [c1, c2], ["C01", "C02"], task.budget, task.budget - (0.5 * task.budget), task_name="easy"
    )

    sc = grade(state, task)
    # Updated grader is multi-objective; assert bounded and meaningfully positive.
    assert 0.0 <= sc <= 1.0
    assert sc > 0.2


def test_grade_role_coverage_and_finalize_penalty():
    task = get_task("easy")

    required_role_candidate = CandidateProfile(
        candidate_id="C01",
        name="A",
        resume_score=0.75,
        years_experience=4,
        role="ML Engineer",
        skills=["Python"],
        true_skill=0.75,
    )
    non_required_role_candidate = CandidateProfile(
        candidate_id="C01",
        name="A",
        resume_score=0.75,
        years_experience=4,
        role="Backend",
        skills=["Python"],
        true_skill=0.75,
    )

    covered_state = make_state_with_hires(
        [required_role_candidate],
        ["C01"],
        task.budget,
        task.budget - 50.0,
        task_name="easy",
    )
    uncovered_state = make_state_with_hires(
        [non_required_role_candidate],
        ["C01"],
        task.budget,
        task.budget - 50.0,
        task_name="easy",
    )

    covered_score = grade(covered_state, task)
    uncovered_score = grade(uncovered_state, task)

    assert covered_score > uncovered_score

    early_state = make_state_with_hires(
        [required_role_candidate],
        ["C01"],
        task.budget,
        task.budget - 50.0,
        task_name="easy",
    )
    early_state.step_num = 1

    late_state = make_state_with_hires(
        [required_role_candidate],
        ["C01"],
        task.budget,
        task.budget - 50.0,
        task_name="easy",
    )
    late_state.step_num = 5

    assert grade(early_state, task) < grade(late_state, task)
