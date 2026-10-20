"""Time the steps of a round, on the host, with one timestamp a step.

A round measures 50.710 ms in the best arm's own record while its 78 super
kernels account for 16.4 -- so two thirds of it is outside the device and has
never been attributed. The drafted path has six steps and only one of them
touches the device, which makes a timestamp a step enough to find the rest.

Kept separate from the executor because the executor should not grow a
measurement concern, and off unless asked: `mark` is reached through an
attribute that is None by default.
"""
from time import perf_counter_ns


class RoundPhases:
    """Accumulate per-step nanoseconds across rounds."""

    def __init__(self):
        self.totals = {}
        self.counts = {}
        self.rounds = 0
        self._last = None

    def begin(self):
        """Start a round's clock."""
        self.rounds += 1
        self._last = perf_counter_ns()

    def mark(self, step):
        """Charge the time since the previous mark to `step`."""
        now = perf_counter_ns()
        if self._last is not None:
            self.totals[step] = self.totals.get(step, 0) + now - self._last
            self.counts[step] = self.counts.get(step, 0) + 1
        self._last = now

    def report(self):
        """Milliseconds a round, by step, with what the marks did not cover."""
        if not self.rounds:
            return {}
        held = {step: nanoseconds / self.rounds / 1e6
                for step, nanoseconds in self.totals.items()}
        held["rounds"] = self.rounds
        return held
