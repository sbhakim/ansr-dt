# utils/logging_setup.py

import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(log_file: str = 'logs/nexus_dt.log',
                  log_level: int = logging.INFO,
                  max_bytes: int = 5 * 1024 * 1024,  # 5 MB
                  backup_count: int = 5):
    """
    Sets up logging for the application.

    Parameters:
    - log_file (str): Path to the log file.
    - log_level (int): Logging level.
    - max_bytes (int): Maximum size of the log file before rotation.
    - backup_count (int): Number of backup log files to keep.
    """
    # Ensure the logs directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Define log format
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create handlers
    # File Handler with Rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)

    # Avoid adding multiple handlers if they already exist
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
