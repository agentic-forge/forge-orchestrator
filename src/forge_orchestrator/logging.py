"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(json_output: bool = True, log_level: str = "INFO") -> None:
    """Configure structlog for structured JSON logging.

    Args:
        json_output: If True, output JSON logs. If False, output human-readable logs.
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    # Shared processors for all loggers
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.ExtraAdder(),
    ]

    if json_output:
        # JSON output for production
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Human-readable output for development
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger.

    Args:
        name: Optional logger name. If not provided, uses the caller's module name.

    Returns:
        A bound structlog logger.
    """
    return structlog.get_logger(name)


# Convenience function to bind context variables
def bind_context(**kwargs: Any) -> None:
    """Bind context variables that will be included in all subsequent log messages.

    Args:
        **kwargs: Key-value pairs to bind to the logging context.
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()
