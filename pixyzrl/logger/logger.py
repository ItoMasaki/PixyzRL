"""Logger class to handle logging for training."""

import logging
from pathlib import Path


class Logger:
    """Logger class to handle logging for training."""

    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO) -> None:
        """Initialize the logger."""
        self.log_dir = log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        log_file = Path(log_dir) / "training.log"

        logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
        self.logger = logging.getLogger("PixyzRLLogger")

    def log(self, message: str, level: int = logging.INFO) -> None:
        """Log a message."""
        self.logger.log(level, message)
