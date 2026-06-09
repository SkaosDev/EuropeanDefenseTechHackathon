"""Compteur de FPS glissant (moyenne sur les N dernières frames)."""

from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """Mesure le débit réel du pipeline à partir des temps inter-frames."""

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        """À appeler une fois par frame traitée."""
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """FPS moyen sur la fenêtre glissante (0.0 tant que < 2 frames)."""
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / span if span > 0 else 0.0


if __name__ == "__main__":
    # Test standalone : simule ~50 fps pendant un court instant.
    counter = FPSCounter()
    for _ in range(40):
        counter.tick()
        time.sleep(0.02)
    print(f"FPS mesure ~ {counter.fps:.1f} (attendu ~ 50)")
