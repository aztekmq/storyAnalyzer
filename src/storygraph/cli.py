"""
Command-line interface for StoryGraph Lab.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from storygraph.pipeline.run_pipeline import run_pipeline
from storygraph.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StoryGraph Lab CLI")
    parser.add_argument("story_path", help="Path to the story JSON file.")
    parser.add_argument("prefix", help="Prefix for export filenames.")
    parser.add_argument("--repo-root", default=".", help="Repository root for resolving paths.")
    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    logger.info("CLI invoked with story=%s prefix=%s", args.story_path, args.prefix)
    outputs = run_pipeline(args.story_path, args.prefix, repo_root=Path(args.repo_root))
    for name, path in outputs.items():
        logger.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()
