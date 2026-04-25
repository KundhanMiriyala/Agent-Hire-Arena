# from dataclasses import dataclass
# from typing import Dict


# @dataclass
# class TaskConfig:
#     name: str
#     num_candidates: int
#     noise_level: float       # std dev of gaussian noise on resume_score
#     budget: float
#     seed: int
#     decoy_fraction: float    # fraction of candidates that are decoys
#     max_steps: int
#     role_requirements: Dict[str, int]
#     description: str


# TASKS: Dict[str, TaskConfig] = {
#     "easy": TaskConfig(
#         name="easy",
#         num_candidates=5,
#         noise_level=0.05,
#         budget=300.0,
#         seed=42,
#         decoy_fraction=0.0,
#         max_steps=20,
#         role_requirements={"ML Engineer": 1},
#         description=(
#             "5 candidates, low noise. Clean signals. "
#             "Goal: identify and hire the best candidate."
#         ),
#     ),
#     "medium": TaskConfig(
#         name="medium",
#         num_candidates=10,
#         noise_level=0.15,
#         budget=220.0,
#         seed=137,
#         decoy_fraction=0.0,
#         max_steps=30,
#         role_requirements={"ML Engineer": 1, "Backend": 1},
#         description=(
#             "10 candidates, medium noise. Budget constrains full exploration. "
#             "Goal: balance interview cost vs hire quality."
#         ),
#     ),
#     "hard": TaskConfig(
#         name="hard",
#         num_candidates=20,
#         noise_level=0.30,
#         budget=200.0,          # corrected: 5×10 + 3×50 = 200, zero slack
#         seed=999,
#         decoy_fraction=0.25,   # 5 of 20 are decoys
#         max_steps=50,
#         role_requirements={"ML Engineer": 1, "Backend": 1, "Data Scientist": 1},
#         description=(
#             "20 candidates, high noise + 25% decoys. Zero budget slack. "
#             "Misleading resumes. Goal: avoid decoys and build the best team."
#         ),
#     ),
# }


# def get_task(name: str) -> TaskConfig:
#     if name not in TASKS:
#         raise ValueError(f"Unknown task '{name}'. Choose from: {list(TASKS.keys())}")
#     return TASKS[name]



from dataclasses import dataclass
from typing import Dict


@dataclass
class TaskConfig:
    name: str
    num_candidates: int
    noise_level: float       # std dev of gaussian noise on resume_score
    budget: float
    seed: int
    decoy_fraction: float    # fraction of candidates that are decoys
    max_steps: int
    role_requirements: Dict[str, int]
    description: str
    coached_fraction: float = 0.0
    adversarial: bool = False
    adversarial_start_step: int = 5


TASKS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        name="easy",
        num_candidates=5,
        noise_level=0.05,
        budget=300.0,
        seed=42,
        decoy_fraction=0.0,
        max_steps=20,
        role_requirements={"ML Engineer": 1},
        description=(
            "5 candidates, low noise. Clean signals. "
            "Goal: identify and hire the best candidate."
        ),
    ),
    "medium": TaskConfig(
        name="medium",
        num_candidates=10,
        noise_level=0.15,
        budget=220.0,
        seed=137,
        decoy_fraction=0.0,
        max_steps=30,
        role_requirements={"ML Engineer": 1, "Backend": 1},
        description=(
            "10 candidates, medium noise. Budget constrains full exploration. "
            "Goal: balance interview cost vs hire quality."
        ),
    ),
    "hard": TaskConfig(
        name="hard",
        num_candidates=20,
        noise_level=0.30,
        budget=200.0,          # corrected: 5×10 + 3×50 = 200, zero slack
        seed=999,
        decoy_fraction=0.25,   # 5 of 20 are decoys
        max_steps=50,
        role_requirements={"ML Engineer": 1, "Backend": 1, "Data Scientist": 1},
        description=(
            "20 candidates, high noise + 25% decoys. Zero budget slack. "
            "Misleading resumes. Goal: avoid decoys and build the best team."
        ),
    ),
    "adversarial": TaskConfig(
        name="adversarial",
        num_candidates=20,
        noise_level=0.30,
        budget=200.0,
        seed=999,
        decoy_fraction=0.25,
        coached_fraction=0.25,
        adversarial=True,
        adversarial_start_step=5,
        max_steps=50,
        role_requirements={"ML Engineer": 1, "Backend": 1, "Data Scientist": 1},
        description=(
            "20 candidates, 25% coached decoys + live adversarial NPC hiring manager. "
            "Resist pressure while catching coached candidates."
        ),
    ),
    "nightmare": TaskConfig(
        name="nightmare",
        num_candidates=25,
        noise_level=0.35,
        budget=180.0,
        seed=777,
        decoy_fraction=0.40,
        coached_fraction=0.40,
        adversarial=True,
        adversarial_start_step=3,
        max_steps=60,
        role_requirements={"ML Engineer": 1, "Backend": 1, "Data Scientist": 1},
        description=(
            "25 candidates, 40% coached decoys, tightest budget, escalating NPC pressure. "
            "Maximum difficulty."
        ),
    ),
}


def get_task(name: str) -> TaskConfig:
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Choose from: {list(TASKS.keys())}")
    return TASKS[name]