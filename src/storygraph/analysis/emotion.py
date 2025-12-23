"""
Emotion and pacing time-series utilities.

The functions in this module convert validated :class:`~storygraph.io.story_loader.Scene`
objects into pandas structures for further processing or visualization.
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from storygraph.io.story_loader import Scene

logger = logging.getLogger(__name__)


def scenes_to_frame(scenes: Iterable[Scene]) -> pd.DataFrame:
    """Convert scenes to a tidy :class:`pandas.DataFrame`.

    The resulting dataframe contains the columns ``index``, ``label``,
    ``emotion``, ``pace``, and ``tension``. An ``index`` column is added to
    preserve chronological ordering for plotting and slope calculations.
    """

    rows = []
    for idx, scene in enumerate(scenes):
        logger.debug(
            "Serializing scene %s: emotion=%s, pace=%s, tension=%s", scene.label, scene.emotion, scene.pace, scene.tension
        )
        rows.append(
            {
                "index": idx,
                "label": scene.label,
                "emotion": scene.emotion,
                "pace": scene.pace,
                "tension": scene.tension,
            }
        )

    frame = pd.DataFrame(rows)
    logger.info("Constructed time-series dataframe with %d rows", len(frame))
    return frame
