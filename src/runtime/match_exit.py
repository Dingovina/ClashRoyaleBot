from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MatchExitTracker:
    """Counts consecutive CNN probes where battlefield probability stays below the end threshold."""

    consecutive_below: int = field(default=0, init=False)

    def observe_probability(
        self,
        *,
        prob: float,
        end_threshold: float,
        confirm_ticks: int,
        did_check: bool,
    ) -> bool:
        """
        Update state from one probe. Returns True when the match should end.

        ``did_check`` is False on ticks where the CNN was not run (no pixels); streak is unchanged.
        """
        if not did_check:
            return False
        if prob < end_threshold:
            self.consecutive_below += 1
        else:
            self.consecutive_below = 0
        return self.consecutive_below >= confirm_ticks
