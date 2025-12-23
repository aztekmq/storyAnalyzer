"""
Logging configuration utilities for StoryGraph Lab.

The module centralizes the verbose logging setup requested for debugging
and analysis workflows. Call :func:`configure_logging` as early as
possible in an application entrypoint to ensure consistent formatting
and verbosity.
"""

from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level: int = logging.DEBUG, log_format: Optional[str] = None) -> None:
    """Configure application-wide logging with a verbose formatter.

    Parameters
    ----------
    level:
        The minimum severity level to capture. Defaults to :data:`logging.DEBUG`
        for rich observability during development.
    log_format:
        Optional custom log format string. When omitted a detailed format with
        timestamps, module names, and line numbers is used.

    Notes
    -----
    The configuration is idempotent; repeated invocations will not attach
    duplicate handlers. This behavior keeps downstream scripts simple while
    ensuring consistent debug-friendly output.
    """

    if logging.getLogger().handlers:
        # Logging already configured; avoid duplicating handlers.
        logging.getLogger(__name__).debug("Logging already configured; skipping duplicate setup.")
        return

    verbose_format = log_format or (
        "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
    )
    logging.basicConfig(level=level, format=verbose_format)

    logging.getLogger(__name__).debug("Verbose logging configured with level %s", level)
