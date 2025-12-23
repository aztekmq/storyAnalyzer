"""
Pipeline entrypoints for StoryGraph Lab.

This module orchestrates loading, validation, analysis, and export steps
for a single story JSON payload. Outputs include Plotly HTML charts,
D3-compatible graph JSON, and a CSV fingerprint vector suitable for
machine learning pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from storygraph.analysis.emotion import scenes_to_frame
from storygraph.analysis.fingerprint import compute_fingerprint
from storygraph.exports.d3 import build_d3_graph, export_d3_graph
from storygraph.exports.plotly_exports import (
    export_character_network,
    export_emotion_chart,
    export_tension_pace_chart,
)
from storygraph.io.story_loader import load_story
from storygraph.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


class PipelineResult(Dict[str, Path]):
    """Dictionary-like container for pipeline outputs."""


_DEF_EXPORTS = {
    "html": Path("exports/html"),
    "d3": Path("exports/d3"),
    "fingerprints": Path("exports/fingerprints"),
}


def run_pipeline(story_path: str, prefix: str, repo_root: str | Path | None = None) -> PipelineResult:
    """Execute the end-to-end story analysis pipeline."""

    configure_logging()
    repo_root = Path(repo_root or ".").resolve()
    logger.info("Running pipeline for %s with prefix '%s'", story_path, prefix)

    story = load_story(story_path, repo_root=repo_root)
    frame = scenes_to_frame(story.scenes)

    html_dir = repo_root / _DEF_EXPORTS["html"]
    d3_dir = repo_root / _DEF_EXPORTS["d3"]
    fp_dir = repo_root / _DEF_EXPORTS["fingerprints"]

    results: PipelineResult = PipelineResult()
    results["network_html"] = export_character_network(story, html_dir / f"{prefix}_network.html")
    results["emotion_html"] = export_emotion_chart(frame, story.title, html_dir / f"{prefix}_emotion.html")
    results["tension_pace_html"] = export_tension_pace_chart(frame, story.title, html_dir / f"{prefix}_tension_pace.html")

    d3_graph = build_d3_graph(story)
    results["d3_json"] = export_d3_graph(d3_graph, d3_dir / f"{prefix}_graph.json")

    fingerprint = compute_fingerprint(story.story_id, frame)
    fp_path = fp_dir / f"{prefix}_fingerprint.csv"
    fp_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint.to_frame().to_csv(fp_path, index=False)
    results["fingerprint_csv"] = fp_path
    logger.info("Fingerprint exported to %s", fp_path)

    logger.info("Pipeline completed successfully")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the StoryGraph Lab pipeline on a story JSON file.")
    parser.add_argument("story_path", help="Path to the story JSON file.")
    parser.add_argument("prefix", help="Output filename prefix.")
    parser.add_argument("--repo-root", default=".", help="Repository root for resolving paths.")
    args = parser.parse_args()

    output = run_pipeline(args.story_path, args.prefix, repo_root=args.repo_root)
    for name, path in output.items():
        print(f"{name}: {path}")
