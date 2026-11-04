"""Lightweight context-manager timer for profiling named steps."""
import time
import numpy as np
from typing import Dict, List, Optional


class StepTimer:
    """
    Accumulates per-key wall-clock durations via a context manager.

    Usage:
        timer = StepTimer(["unpack", "clone", "inner"])

        with timer("unpack"):
            result = do_work()

        print(timer.avg_max_report("unpack"))       # '412.0 ms / 426.0 ms'
        print(timer.total_pct_report("loss", 2800)) # '137.0 ms total (5%)'
        timer.reset()
    """

    def __init__(self, keys: List[str]) -> None:
        self._data: Dict[str, List[float]] = {k: [] for k in keys}
        self._active_key: Optional[str] = None
        self._t0: Optional[float] = None

    # ── Context-manager interface ──────────────────────────────────
    def __call__(self, key: str) -> "StepTimer":
        if key not in self._data:
            raise KeyError(f"Unknown timer key {key!r}. Valid: {list(self._data)}")
        self._active_key = key
        return self

    def __enter__(self) -> "StepTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self._data[self._active_key].append(time.perf_counter() - self._t0)

    # ── Queries (all return ms) ────────────────────────────────────
    def avg_ms(self, key: str) -> float:
        v = self._data[key]
        return float(np.mean(v)) * 1e3 if v else 0.0

    def max_ms(self, key: str) -> float:
        v = self._data[key]
        return float(np.max(v)) * 1e3 if v else 0.0

    def total_ms(self, key: str) -> float:
        return sum(self._data[key]) * 1e3

    def has_data(self, key: str) -> bool:
        return len(self._data[key]) > 0

    # ── Lifecycle ──────────────────────────────────────────────────
    def reset(self) -> None:
        for k in self._data:
            self._data[k].clear()

    # ── Formatted report helpers ───────────────────────────────────
    def avg_max_report(self, key: str) -> str:
        """'412.0 ms / 426.0 ms'"""
        return f"{self.avg_ms(key):.1f} ms / {self.max_ms(key):.1f} ms"

    def total_pct_report(self, key: str, pool_ms: float) -> str:
        """'137.0 ms total  (5%)'"""
        t = self.total_ms(key)
        pct = 100 * t / pool_ms if pool_ms > 0 else 0.0
        return f"{t:.1f} ms total  ({pct:.0f}%)"
