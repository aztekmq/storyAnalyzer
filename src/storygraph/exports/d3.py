"""
D3-compatible graph export helpers.

The functions convert interaction data into the canonical ``nodes`` and
``links`` structure expected by D3 force-directed layouts.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import List

from storygraph.io.story_loader import Story

logger = logging.getLogger(__name__)


def build_d3_graph(story: Story) -> dict:
    """Create a D3-ready graph dictionary from a :class:`Story`."""

    character_lookup = {character.id: character for character in story.characters}
    nodes: List[dict] = [
        {"id": character.id, "name": character.name, "role": character.role}
        for character in story.characters
    ]

    link_strengths = defaultdict(float)
    for scene in story.scenes:
        for interaction in scene.interactions:
            key = (interaction.source, interaction.target)
            link_strengths[key] += float(interaction.strength)
            logger.debug(
                "Accumulating interaction %s -> %s (total=%s)",
                interaction.source,
                interaction.target,
                link_strengths[key],
            )

    links = [
        {"source": source, "target": target, "value": strength}
        for (source, target), strength in link_strengths.items()
        if source in character_lookup and target in character_lookup
    ]

    logger.info("Built D3 graph with %d nodes and %d links", len(nodes), len(links))
    return {"nodes": nodes, "links": links}


def export_d3_graph(graph: dict, output_path: str | Path) -> Path:
    """Serialize a graph dictionary to JSON for D3 consumption."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(graph, handle, indent=2)
        logger.info("D3 graph exported to %s", output_path)
    return output_path
