"""
Logging configuration using loguru.

This module provides centralized logging configuration for the ML optimization framework
using loguru for better performance and more features than standard logging.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger


class LoggingConfig:
    """
    Centralized logging configuration using loguru.
    
    Provides consistent logging setup across the entire framework with
    different log levels, formats, and output destinations.
    """
    
    def __init__(self):
        """Initialize logging configuration."""
        self.is_configured = False
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
    
    def setup_logging(
        self,
        level: str = "INFO",
        log_to_file: bool = True,
        log_to_console: bool = True,
        rotation: str = "10 MB",
        retention: str = "1 week",
        format_string: Optional[str] = None
    ) -> None:
        """
        Setup loguru logging configuration.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
            rotation: Log file rotation policy
            retention: Log file retention policy
            format_string: Custom format string
        """
        if self.is_configured:
            return
        
        # Remove default handler
        logger.remove()
        
        # Default format
        if format_string is None:
            format_string = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
        
        # Console handler
        if log_to_console:
            logger.add(
                sys.stdout,
                level=level,
                format=format_string,
                colorize=True,
                backtrace=True,
                diagnose=True
            )
        
        # File handler
        if log_to_file:
            logger.add(
                self.log_dir / "optimization.log",
                level=level,
                format=format_string,
                rotation=rotation,
                retention=retention,
                compression="zip",
                backtrace=True,
                diagnose=True
            )
            
            # Separate error log
            logger.add(
                self.log_dir / "errors.log",
                level="ERROR",
                format=format_string,
                rotation=rotation,
                retention=retention,
                compression="zip",
                backtrace=True,
                diagnose=True
            )
        
        self.is_configured = True
        logger.info("Logging configuration initialized")
    
    def setup_development_logging(self) -> None:
        """Setup logging for development environment."""
        self.setup_logging(
            level="DEBUG",
            log_to_file=True,
            log_to_console=True,
            format_string=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
                "<level>{message}</level>"
            )
        )
    
    def setup_production_logging(self) -> None:
        """Setup logging for production environment."""
        self.setup_logging(
            level="INFO",
            log_to_file=True,
            log_to_console=False,
            format_string=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            )
        )
    
    def setup_testing_logging(self) -> None:
        """Setup logging for testing environment."""
        self.setup_logging(
            level="WARNING",
            log_to_file=False,
            log_to_console=True,
            format_string="<level>{level: <8}</level> | <level>{message}</level>"
        )
    
    def get_logger(self, name: str) -> Any:
        """
        Get a logger instance for a specific module.
        
        Args:
            name: Logger name (usually __name__)
            
        Returns:
            Logger instance
        """
        if not self.is_configured:
            self.setup_logging()
        
        return logger.bind(name=name)
    
    def log_optimization_start(
        self,
        model_name: str,
        n_trials: int,
        dataset_info: Dict[str, Any]
    ) -> None:
        """
        Log optimization start with context information.
        
        Args:
            model_name: Name of the model being optimized
            n_trials: Number of optimization trials
            dataset_info: Dataset information
        """
        logger.info("🚀 Starting optimization", 
                   model=model_name, 
                   trials=n_trials,
                   **dataset_info)
    
    def log_optimization_complete(
        self,
        model_name: str,
        best_score: float,
        best_params: Dict[str, Any],
        optimization_time: float
    ) -> None:
        """
        Log optimization completion with results.
        
        Args:
            model_name: Name of the optimized model
            best_score: Best achieved score
            best_params: Best parameters found
            optimization_time: Total optimization time
        """
        logger.success("✅ Optimization completed",
                      model=model_name,
                      best_score=best_score,
                      optimization_time=f"{optimization_time:.2f}s",
                      best_params=best_params)
    
    def log_trial_result(
        self,
        trial_number: int,
        score: float,
        params: Dict[str, Any],
        is_best: bool = False
    ) -> None:
        """
        Log individual trial result.
        
        Args:
            trial_number: Trial number
            score: Trial score
            params: Trial parameters
            is_best: Whether this is the best trial so far
        """
        if is_best:
            logger.info("🎯 New best trial",
                       trial=trial_number,
                       score=score,
                       params=params)
        else:
            logger.debug("Trial completed",
                        trial=trial_number,
                        score=score)
    
    def log_error_with_context(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> None:
        """
        Log error with additional context.
        
        Args:
            error: Exception that occurred
            context: Additional context information
        """
        logger.error("❌ Error occurred: {error}",
                    error=str(error),
                    error_type=type(error).__name__,
                    **context)


# Global logging configuration instance
_logging_config = LoggingConfig()

# Convenience functions
def setup_logging(level: str = "INFO", **kwargs) -> None:
    """Setup logging with specified level."""
    _logging_config.setup_logging(level=level, **kwargs)

def setup_development_logging() -> None:
    """Setup development logging."""
    _logging_config.setup_development_logging()

def setup_production_logging() -> None:
    """Setup production logging."""
    _logging_config.setup_production_logging()

def setup_testing_logging() -> None:
    """Setup testing logging."""
    _logging_config.setup_testing_logging()

def get_logger(name: str) -> Any:
    """Get logger for module."""
    return _logging_config.get_logger(name)

# Export the main logger
__all__ = [
    "logger",
    "LoggingConfig", 
    "setup_logging",
    "setup_development_logging",
    "setup_production_logging", 
    "setup_testing_logging",
    "get_logger"
]
