from inference import parse_action, _sanitize_model_action
from models import HiringObservation
from policy import choose_policy_action


def make_observation(*, interviews_done=None, probes_done=None, probe_gaps=None, budget_remaining=200.0):
    return HiringObservation(
        candidates=[
            {
                "candidate_id": "C01",
                "name": "High Interview",
                "resume_score": 0.92,
                "years_experience": 8,
                "role": "ML Engineer",
                "skills": ["Python", "Machine Learning"],
            },
            {
                "candidate_id": "C02",
                "name": "Normal Interview",
                "resume_score": 0.62,
                "years_experience": 4,
                "role": "Backend",
                "skills": ["Go", "SQL"],
            },
        ],
        budget_remaining=budget_remaining,
        interviews_done=interviews_done or {},
        probes_done=probes_done or {},
        probe_gaps=probe_gaps or {},
        hires_made=[],
        skipped=[],
        step_num=0,
        max_steps=10,
        last_action_result="Episode started.",
        done=False,
    )


def test_parse_action_accepts_probe():
    assert parse_action('{"action": "probe", "candidate_id": "C01"}') == {
        "action": "probe",
        "candidate_id": "C01",
    }


def test_sanitize_rejects_probe_without_interview():
    obs = make_observation()
    assert _sanitize_model_action({"action": "probe", "candidate_id": "C01"}, obs) is None


def test_sanitize_accepts_probe_after_interview():
    obs = make_observation(interviews_done={"C01": 0.91})
    assert _sanitize_model_action({"action": "probe", "candidate_id": "C01"}, obs) == {
        "action": "probe",
        "candidate_id": "C01",
    }


def test_policy_interviews_before_probe():
    obs = make_observation()
    action = choose_policy_action(obs=obs, task="hard", variant="baseline")
    assert action["action"] == "interview"


def test_policy_can_probe_interviewed_suspicious_candidate():
    obs = make_observation(
        interviews_done={
            "C01": 0.93,
            "C02": 0.61,
            "C03": 0.62,
            "C04": 0.63,
            "C05": 0.64,
            "C06": 0.65,
            "C07": 0.66,
            "C08": 0.67,
        }
    )
    action = choose_policy_action(obs=obs, task="hard", variant="baseline")
    assert action == {"action": "probe", "candidate_id": "C01"}


def test_policy_hire_is_not_overridden_by_probe():
    obs = make_observation(
        interviews_done={
            "C01": 0.93,
            "C02": 0.61,
            "C03": 0.62,
            "C04": 0.63,
            "C05": 0.64,
            "C06": 0.65,
            "C07": 0.66,
            "C08": 0.67,
        },
        probes_done={"C01": 0.92},
        probe_gaps={"C01": 0.01},
    )
    action = choose_policy_action(obs=obs, task="hard", variant="baseline")
    assert action == {"action": "hire", "candidate_id": "C01"}
