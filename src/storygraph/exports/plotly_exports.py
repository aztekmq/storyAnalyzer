"""
Plotly export helpers for StoryGraph Lab visualizations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from storygraph.io.story_loader import Interaction, Story

logger = logging.getLogger(__name__)


_DEF_LAYOUT = dict(titlefont_size=16, font=dict(size=12))


def _aggregate_interactions(story: Story) -> nx.Graph:
    graph = nx.Graph()
    for character in story.characters:
        graph.add_node(character.id, label=character.name, role=character.role)

    for scene in story.scenes:
        for interaction in scene.interactions:
            weight = graph.get_edge_data(interaction.source, interaction.target, {}).get("weight", 0)
            graph.add_edge(interaction.source, interaction.target, weight=weight + interaction.strength)
            logger.debug(
                "Graph edge %s-%s updated to weight %s",
                interaction.source,
                interaction.target,
                weight + interaction.strength,
            )
    logger.info("Aggregated interactions into graph with %d nodes and %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def _network_trace(graph: nx.Graph) -> Tuple[go.Scatter, go.Scatter]:
    pos = nx.spring_layout(graph, seed=42)
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    text = []
    for node, attrs in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(attrs.get("label", node))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=text,
        textposition="bottom center",
        hovertext=text,
        marker=dict(
            showscale=False,
            color="#1f77b4",
            size=14,
            line_width=2,
        ),
    )

    logger.debug("Generated Plotly network traces with %d nodes and %d edges", len(node_x), len(edge_x) // 3)
    return edge_trace, node_trace


def export_character_network(story: Story, output_path: str | Path) -> Path:
    """Create and save the interactive character network visualization."""

    graph = _aggregate_interactions(story)
    edge_trace, node_trace = _network_trace(graph)

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title=f"Character Network — {story.title}", showlegend=False, **_DEF_LAYOUT)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    logger.info("Character network exported to %s", output_path)
    return output_path


def export_emotion_chart(frame: pd.DataFrame, story_title: str, output_path: str | Path) -> Path:
    """Export an emotional arc line chart."""

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["index"], y=frame["emotion"], mode="lines+markers", name="Emotion"))
    fig.add_trace(go.Scatter(x=frame["index"], y=frame["tension"], mode="lines+markers", name="Tension"))

    fig.update_layout(title=f"Emotion & Tension — {story_title}", xaxis_title="Scene", yaxis_title="Value", **_DEF_LAYOUT)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    logger.info("Emotion chart exported to %s", output_path)
    return output_path


def export_tension_pace_chart(frame: pd.DataFrame, story_title: str, output_path: str | Path) -> Path:
    """Export a combined tension vs. pace chart."""

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["index"], y=frame["tension"], mode="lines+markers", name="Tension"))
    fig.add_trace(go.Scatter(x=frame["index"], y=frame["pace"], mode="lines+markers", name="Pace"))

    fig.update_layout(title=f"Tension vs Pace — {story_title}", xaxis_title="Scene", yaxis_title="Value", **_DEF_LAYOUT)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    logger.info("Tension vs Pace chart exported to %s", output_path)
    return output_path
