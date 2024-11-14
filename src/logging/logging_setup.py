# src/logging/logging_setup.py

import logging
from logging.handlers import RotatingFileHandler


def setup_logging(log_file: str, log_level: int = logging.INFO, max_bytes: int = 5 * 1024 * 1024,
                  backup_count: int = 5) -> logging.Logger:
    """
    Sets up logging for the application.

    Parameters:
    - log_file (str): Path to the log file.
    - log_level (int): Logging level (e.g., logging.INFO).
    - max_bytes (int): Maximum size of the log file before rotation (in bytes).
    - backup_count (int): Number of backup log files to keep.

    Returns:
    - logger (logging.Logger): Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Prevent adding multiple handlers if they already exist
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler with rotation
        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        # Stream handler (console)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
