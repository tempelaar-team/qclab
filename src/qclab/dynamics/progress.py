"""Utility helpers for displaying dual progress bars for dynamics drivers."""

from __future__ import annotations

import threading
from typing import Optional

from tqdm import tqdm


class BatchProgressBars:
    """Render two stacked progress bars for tracking batches.

    The top bar tracks the number of completed batches. The bottom bar tracks the
    progress of the slowest running batch measured in integration steps.
    """

    def __init__(
        self,
        total_batches: int,
        steps_per_batch: int,
        enabled: bool = True,
        *,
        position: int = 0,
    ) -> None:
        self.enabled = enabled and total_batches > 0 and steps_per_batch > 0
        self.steps_per_batch = steps_per_batch
        self._lock = threading.Lock()
        if self.enabled:
            self.total_bar = tqdm(
                total=total_batches,
                desc="Batches",
                position=position,
                leave=False,
            )
            self.slowest_bar = tqdm(
                total=steps_per_batch,
                desc="Slowest batch",
                position=position + 1,
                leave=False,
            )
        else:
            self.total_bar = None
            self.slowest_bar = None

    def set_total(self, completed_batches: int) -> None:
        """Update the overall batch bar to the provided completion count."""

        if not self.enabled or self.total_bar is None:
            return
        with self._lock:
            completed_batches = min(completed_batches, self.total_bar.total)
            delta = completed_batches - self.total_bar.n
            if delta > 0:
                self.total_bar.update(delta)

    def set_slowest(self, slowest_progress: int) -> None:
        """Update the slowest batch bar to ``slowest_progress`` steps."""

        if not self.enabled or self.slowest_bar is None:
            return
        with self._lock:
            progress = max(0, min(slowest_progress, self.steps_per_batch))
            self.slowest_bar.n = progress
            self.slowest_bar.refresh()

    def close(self) -> None:
        """Close both progress bars."""

        if not self.enabled:
            return
        if self.total_bar is not None:
            self.total_bar.close()
        if self.slowest_bar is not None:
            self.slowest_bar.close()


class SerialProgressReporter:
    """Reporter used by drivers that execute batches sequentially."""

    def __init__(self, bars: Optional[BatchProgressBars], steps_per_batch: int) -> None:
        self.bars = bars
        self.steps_per_batch = steps_per_batch
        self.progress = 0
        if self.bars is not None:
            self.bars.set_slowest(0)

    def update(self, steps: int = 1) -> None:
        self.progress += steps
        if self.bars is not None:
            self.bars.set_slowest(self.progress)

    def close(self) -> None:
        if self.bars is not None:
            self.bars.set_slowest(self.steps_per_batch)


class SharedValueProgressReporter:
    """Reporter that writes progress to a multiprocessing shared value."""

    def __init__(self, shared_value, steps_per_batch: int) -> None:
        self.shared_value = shared_value
        self.steps_per_batch = steps_per_batch

    def update(self, steps: int = 1) -> None:
        with self.shared_value.get_lock():
            self.shared_value.value += steps

    def close(self) -> None:
        with self.shared_value.get_lock():
            self.shared_value.value = self.steps_per_batch
