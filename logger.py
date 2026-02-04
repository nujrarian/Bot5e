"""Logging configuration for Bot5e application."""
import logging
import sys
from config import config


def setup_logger(name: str = 'bot5e') -> logging.Logger:
    """
    Set up and return a logger with the configured settings.

    Args:
        name: Logger name (defaults to 'bot5e')

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

        # Create formatter
        formatter = logging.Formatter(config.log_format)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        try:
            file_handler = logging.FileHandler(config.log_file, encoding='utf-8')
            file_handler.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create log file handler: {e}")

        # Prevent propagation to root logger
        logger.propagate = False

    return logger


# Global logger instance
logger = setup_logger()
