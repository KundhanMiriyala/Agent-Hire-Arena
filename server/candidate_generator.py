import numpy as np
from typing import List

from models import CandidateProfile
from server.tasks import TaskConfig

# Candidate name pool (seeded selection keeps names deterministic)
FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zoe", "Aaron", "Bella", "Carlos", "Diana",
]

SKILL_POOL = [
    "Python", "Java", "Go", "Rust", "TypeScript",
    "Machine Learning", "Data Analysis", "SQL", "React", "Node.js",
    "System Design", "Leadership", "Communication", "Testing", "DevOps",
    "Cloud (AWS)", "Cloud (GCP)", "Kubernetes", "Security", "Product Thinking",
]

ROLE_SKILLS = {
    "ML Engineer": ["Python", "Machine Learning", "Data Analysis", "SQL"],
    "Backend": ["Java", "Go", "System Design", "DevOps"],
    "Data Scientist": ["Python", "Data Analysis", "Machine Learning", "SQL"],
}

ROLES = list(ROLE_SKILLS.keys())


def generate_candidates(task_config: TaskConfig) -> List[CandidateProfile]:
    """
    Deterministically generate a candidate pool using the task's fixed seed.
    The same task config always produces the same pool — guaranteeing
    reproducible baseline scores across runs.
    """
    rng = np.random.default_rng(task_config.seed)

    n = task_config.num_candidates
    name_indices = rng.choice(len(FIRST_NAMES), size=n, replace=False)

    # Decide which candidates are decoys upfront
    num_decoys = int(n * task_config.decoy_fraction)
    decoy_indices = set(rng.choice(n, size=num_decoys, replace=False).tolist())

    num_coached = int(n * task_config.coached_fraction)
    coached_indices = set(rng.choice(list(decoy_indices), size=min(num_coached, len(decoy_indices)), replace=False))

    candidates = []
    for i in range(n):
        candidate_id = f"C{i+1:02d}"
        name = FIRST_NAMES[name_indices[i]]
        is_decoy = i in decoy_indices
        is_coached = i in coached_indices
        role = ROLES[int(rng.integers(0, len(ROLES)))]
        interview_difficulty = float(rng.uniform(0.8, 1.2))

        if is_decoy:
            # Decoy: impressive resume, low true skill
            true_skill = float(rng.uniform(0.10, 0.30))
            resume_score = float(rng.uniform(0.75, 0.95))
            # Experience also inflated
            years_experience = int(rng.integers(6, 14))
        else:
            # Normal candidate: noisy resume around true skill
            true_skill = float(rng.uniform(0.25, 0.95))
            noise = float(rng.normal(0.0, task_config.noise_level))
            resume_score = float(np.clip(true_skill + noise, 0.05, 0.98))
            # Experience correlated with skill, with noise
            base_exp = int(true_skill * 12) + 1
            exp_noise = int(rng.integers(-2, 3))
            years_experience = max(0, base_exp + exp_noise)

            # Hard task adversarial twist: some truly strong candidates can look weaker on resume.
            if task_config.name == "hard" and rng.random() < 0.20:
                resume_score = float(np.clip(resume_score - 0.20, 0.05, 0.98))

        # Skills: pick 3-5 from pool; on hard, add 1-2 fake/inflated skills
        num_skills = int(rng.integers(3, 6))
        skill_indices = rng.choice(len(SKILL_POOL), size=num_skills, replace=False)
        skills = [SKILL_POOL[j] for j in skill_indices]

        # Bias visible skills toward the candidate's primary role.
        for rs in ROLE_SKILLS[role]:
            if rs not in skills and len(skills) < 5:
                skills.append(rs)

        candidates.append(
            CandidateProfile(
                candidate_id=candidate_id,
                name=name,
                resume_score=round(resume_score, 3),
                years_experience=years_experience,
                role=role,
                skills=skills,
                true_skill=round(true_skill, 4),
                is_decoy=is_decoy,
                is_coached=is_coached,
                interview_difficulty=round(interview_difficulty, 3),
            )
        )

    return candidates
