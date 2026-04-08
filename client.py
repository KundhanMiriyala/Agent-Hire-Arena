"""
client.py — Python client for the AgentHire Arena environment.

Usage:
    from client import HiringEnvClient

    client = HiringEnvClient(base_url="http://localhost:7860")
    obs = client.reset(task="easy")
    obs, reward = client.step("interview", candidate_id="C01")
    obs, reward = client.step("hire", candidate_id="C01")
    obs, reward = client.step("finalize")
    print(reward.final_score)
"""

import requests
from typing import Optional, Tuple

from models import HiringObservation, HiringReward


class HiringEnvClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ---------------------------------------------------------------- #

    def reset(self, task: str = "easy") -> HiringObservation:
        """Start a new episode. Returns the initial observation."""
        resp = self._session.post(
            f"{self.base_url}/reset",
            json={"task": task},
            timeout=30,
        )
        resp.raise_for_status()
        return HiringObservation(**resp.json())

    def step(
        self,
        action: str,
        candidate_id: Optional[str] = None,
    ) -> Tuple[HiringObservation, HiringReward]:
        """
        Submit one action. Returns (observation, reward).

        Args:
            action: One of "interview", "hire", "skip", "finalize"
            candidate_id: Required for interview/hire/skip. None for finalize.
        """
        payload = {"action": action}
        if candidate_id:
            payload["candidate_id"] = candidate_id

        resp = self._session.post(
            f"{self.base_url}/step",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            HiringObservation(**data["observation"]),
            HiringReward(**data["reward"]),
        )

    def state(self) -> dict:
        """Fetch full internal state (includes hidden fields for debugging)."""
        resp = self._session.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        """Returns True if the server is reachable."""
        try:
            resp = self._session.get(f"{self.base_url}/", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
