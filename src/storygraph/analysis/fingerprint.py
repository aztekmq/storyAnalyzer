"""
Fingerprint feature extraction for stories.

A fingerprint is a concise numeric representation of the narrative's
structural properties that can be compared across stories. The
implementation focuses on descriptive statistics derived from emotion,
pacing, and tension series.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Fingerprint:
    """Container for computed fingerprint metrics."""

    story_id: str
    emotion_mean: float
    emotion_std: float
    emotion_slope: float
    emotion_peak: float
    tension_mean: float
    tension_std: float
    pace_mean: float
    pace_std: float

    def to_frame(self) -> pd.DataFrame:
        """Return a single-row dataframe representation."""

        data: Dict[str, float | str] = {
            "story_id": self.story_id,
            "emotion_mean": self.emotion_mean,
            "emotion_std": self.emotion_std,
            "emotion_slope": self.emotion_slope,
            "emotion_peak": self.emotion_peak,
            "tension_mean": self.tension_mean,
            "tension_std": self.tension_std,
            "pace_mean": self.pace_mean,
            "pace_std": self.pace_std,
        }
        return pd.DataFrame([data])


def compute_fingerprint(story_id: str, frame: pd.DataFrame) -> Fingerprint:
    """Compute a fingerprint vector from the story dataframe."""

    def slope(series: pd.Series) -> float:
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series.to_numpy(), 1)
        return float(coeffs[0])

    fingerprint = Fingerprint(
        story_id=story_id,
        emotion_mean=float(frame["emotion"].mean()),
        emotion_std=float(frame["emotion"].std(ddof=0)),
        emotion_slope=slope(frame["emotion"]),
        emotion_peak=float(frame["emotion"].max()),
        tension_mean=float(frame["tension"].mean()),
        tension_std=float(frame["tension"].std(ddof=0)),
        pace_mean=float(frame["pace"].mean()),
        pace_std=float(frame["pace"].std(ddof=0)),
    )

    logger.info("Fingerprint computed for story '%s'", story_id)
    logger.debug("Fingerprint details: %s", fingerprint)
    return fingerprint
