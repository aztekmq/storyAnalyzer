"""
Story loading and validation utilities.

The module defines a minimal schema for story JSON inputs using
Pydantic models to guarantee predictable downstream processing. The
schema balances flexibility with explicit field validation so that the
pipeline can emit rich errors when inputs are incomplete.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


class Character(BaseModel):
    """A character appearing in the narrative."""

    id: str = Field(..., description="Unique identifier for the character.")
    name: str = Field(..., description="Display name for the character.")
    role: Optional[str] = Field(None, description="Optional narrative role or title.")


class Interaction(BaseModel):
    """Interaction between two characters within a scene."""

    source: str = Field(..., description="Origin character identifier.")
    target: str = Field(..., description="Destination character identifier.")
    strength: float = Field(..., ge=0.0, description="Interaction intensity.")

    @field_validator("source", "target")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            msg = "Interaction endpoints must be non-empty strings."
            raise ValueError(msg)
        return value


class Scene(BaseModel):
    """A scene or chapter with emotional and pacing metadata."""

    label: str = Field(..., description="Scene label or identifier.")
    emotion: float = Field(..., description="Normalized emotional intensity (0-1 range recommended).")
    pace: float = Field(..., description="Normalized pacing metric (0-1 range recommended).")
    tension: float = Field(..., description="Narrative tension metric.")
    interactions: List[Interaction] = Field(default_factory=list)


class Story(BaseModel):
    """Validated story payload."""

    story_id: str = Field(..., description="Unique story identifier used for exports.")
    title: str = Field(..., description="Human-readable story title.")
    author: Optional[str] = Field(None, description="Original author or creator.")
    characters: List[Character] = Field(default_factory=list)
    scenes: List[Scene] = Field(default_factory=list)

    @field_validator("story_id", "title")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value.strip():
            msg = "Story identifiers and titles must be non-empty strings."
            raise ValueError(msg)
        return value

    @field_validator("scenes")
    @classmethod
    def require_scenes(cls, scenes: List[Scene]) -> List[Scene]:
        if not scenes:
            msg = "At least one scene must be provided to compute time-series features."
            raise ValueError(msg)
        return scenes


class StoryLoader:
    """Helper class for reading and validating story JSON inputs."""

    def __init__(self, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root or ".").resolve()
        logger.debug("StoryLoader initialized with repo root: %s", self.repo_root)

    def load_story(self, story_path: str | Path) -> Story:
        """Load a story JSON file and return a validated :class:`Story`.

        Parameters
        ----------
        story_path:
            Path to the JSON file relative to ``repo_root`` or absolute.

        Returns
        -------
        Story
            Parsed and validated story representation.
        """

        resolved_path = (self.repo_root / story_path).resolve()
        logger.info("Loading story file from %s", resolved_path)

        if not resolved_path.exists():
            msg = f"Story file not found at {resolved_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        with resolved_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            logger.debug("Raw story payload keys: %s", list(payload.keys()))

        try:
            story = Story.model_validate(payload)
        except ValidationError as exc:
            logger.exception("Story validation failed: %s", exc)
            raise

        logger.info("Story '%s' validated with %d scenes and %d characters", story.title, len(story.scenes), len(story.characters))
        return story


def load_story(story_path: str | Path, repo_root: str | Path | None = None) -> Story:
    """Convenience function to load a story using :class:`StoryLoader`."""

    return StoryLoader(repo_root=repo_root).load_story(story_path)
