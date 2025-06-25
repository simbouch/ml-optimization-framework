"""
Utility modules for the ML optimization framework.

This package contains utility functions and configurations used
across the framework.
"""

from .logging_config import (
    logger,
    LoggingConfig,
    setup_logging,
    setup_development_logging,
    setup_production_logging,
    setup_testing_logging,
    get_logger
)

__all__ = [
    "logger",
    "LoggingConfig",
    "setup_logging", 
    "setup_development_logging",
    "setup_production_logging",
    "setup_testing_logging",
    "get_logger"
]
