"""Night-level star eligibility for reference and comparison stars.

Combines the border cut (:mod:`relphot.border`) and the tail cut (:mod:`relphot.tails`)
into the single ``star_eligible`` mask that
:func:`relphot.reference.select_candidates` and
:func:`relphot.comparison.select_comparison_pool` take. Both cuts are whole-star and
night-level, so the reference of a tile stays one fixed star set in every kept frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from relphot.border import BorderInfo, border_eligibility
from relphot.tails import TailInfo, tail_eligibility

if TYPE_CHECKING:
    from relphot.config import Settings
    from relphot.match import MatchedNight

__all__ = ["EligibilityInfo", "star_eligibility"]


@dataclass(frozen=True, slots=True)
class EligibilityInfo:
    """``eligible`` is ``border.eligible & tails.eligible``, ``(n_stars,)`` bool."""

    eligible: np.ndarray
    border: BorderInfo
    tails: TailInfo


def star_eligibility(night: MatchedNight, settings: Settings) -> EligibilityInfo:
    """Border and tail eligibility of every star of ``night`` and their conjunction."""
    border = border_eligibility(night, settings.border)
    tails = tail_eligibility(night, settings.tails)
    return EligibilityInfo(
        eligible=border.eligible & tails.eligible, border=border, tails=tails
    )
