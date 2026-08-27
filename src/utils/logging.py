"""
    Simple Logging Helpers.
"""

import logging

from rich.logging import RichHandler


def get_logger(name: str = "house_price") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger(name)