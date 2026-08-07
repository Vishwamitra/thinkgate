from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stats:
    requests_seen: int = 0
    requests_healed: int = 0

    def record(self, healed: bool) -> None:
        self.requests_seen += 1
        if healed:
            self.requests_healed += 1

    def as_dict(self) -> dict:
        return {
            "requests_seen": self.requests_seen,
            "requests_healed": self.requests_healed,
        }


# Module-level singleton -- fine for v1's one proxy process, one counter
# set. Can become dependency-injected state later if that's ever needed.
stats = Stats()
