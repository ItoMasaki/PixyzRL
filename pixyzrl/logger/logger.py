import csv
import json
import logging
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from tensorboardX import SummaryWriter


class Logger:
    """Logger class to handle logging for training, including TensorBoard support.

    Attributes:
        log_dir (str): Directory where logs will be stored.
        log_types (set): Types of logging enabled (e.g., file, print, tensorboard, json, csv).
        lock (threading.Lock): Lock for thread-safe operations.
        json_file (Path): Path to the JSON log file.
        csv_file (Path): Path to the CSV log file.
        logger (logging.Logger): Logger instance for file and console logging.
        writer (SummaryWriter or None): TensorBoard writer instance if enabled.
        global_step (int): Step counter for logging.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        log_level: int = logging.INFO,
        log_types: list[str] | None = None,
        log_format: str = "%(asctime)s - %(levelname)s - %(message)s",
        max_log_size: int = 5 * 1024 * 1024,
        backup_count: int = 3,
    ) -> None:
        """Initializes the logger, TensorBoard writer, and log types.

        Args:
            log_dir (str, optional): Directory where logs will be stored. Defaults to "logs".
            log_level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.
            log_types (list[str] | None, optional): List of logging types to enable. Defaults to all types.
            log_format (str, optional): Format for log messages. Defaults to "%(asctime)s - %(levelname)s - %(message)s".
            max_log_size (int, optional): Maximum size of log files before rotating. Defaults to 5MB.
            backup_count (int, optional): Number of backup log files to keep. Defaults to 3.

        Examples:
            >>> logger = Logger(log_dir="logs", log_types=["print", "file"])
        """
        self.log_dir = log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        if log_types is None:
            log_types = ["file", "print", "tensorboard", "json", "csv"]
        self.log_types = set(log_types)
        self.lock = threading.Lock()

        log_file = Path(log_dir) / "training.log"
        self.json_file = Path(log_dir) / "training.json"
        self.csv_file = Path(log_dir) / "training.csv"

        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                RotatingFileHandler(log_file, maxBytes=max_log_size, backupCount=backup_count),
                logging.StreamHandler(),
            ],
        )

        self.logger = logging.getLogger("PixyzRLLogger")
        self.writer = SummaryWriter(log_dir) if "tensorboard" in self.log_types else None
        self.global_step = 0

    def log(self, message: str | dict[str, Any], level: int = logging.INFO, step: int | None = None) -> None:
        """Logs a message or a dictionary of values based on selected log types.

        Args:
            message (str | dict[str, Any]): The message to log or a dictionary of values to log.
            level (int, optional): Logging level (e.g., logging.INFO, logging.ERROR). Defaults to logging.INFO.
            step (int | None, optional): Step index for logging. Defaults to internal counter.

        Examples:
            >>> logger = Logger(log_types=["print"])
            >>> logger.log("Training started")
            >>> logger.log({"loss": 0.01, "accuracy": 99.0})
        """
        with self.lock:
            if step is None:
                step = self.global_step

            if isinstance(message, str):
                self._log_message(message, level)
            else:
                self._log_dict(message, step)

            self.global_step += 1

    def _log_message(self, message: str, level: int) -> None:
        """Logs a string message."""
        if "file" in self.log_types or "print" in self.log_types:
            self.logger.log(level, message)

    def _log_dict(self, message: dict[str, Any], step: int) -> None:
        """Logs a dictionary of values."""
        for key, value in message.items():
            if isinstance(value, int | float):
                self._log_tensorboard(key, value, step)
                self._log_file(key, value, step)
                self._log_print(key, value, step)
                self._log_json(key, value, step)
                self._log_csv(key, value, step)

    def _log_tensorboard(self, key: str, value: float, step: int) -> None:
        """Logs a value to TensorBoard."""
        if "tensorboard" in self.log_types and self.writer:
            self.writer.add_scalar(key, value, step)

    def _log_file(self, key: str, value: float, step: int) -> None:
        """Logs a value to file and print."""
        log_message = f"{key}: {value} (step {step})"
        if "file" in self.log_types:
            self.logger.info(log_message)

    def _log_print(self, key: str, value: float, step: int) -> None:
        """Logs a value to the console."""
        if "print" in self.log_types:
            self.logger.info("%s: %s (step %d)", key, value, step)

    def _log_json(self, key: str, value: float, step: int) -> None:
        """Logs a value to a JSON file."""
        if "json" in self.log_types:
            with open(self.json_file, "a") as f:
                json.dump({"step": step, key: value}, f)
                f.write("\n")

    def _log_csv(self, key: str, value: float, step: int) -> None:
        """Logs a value to a CSV file."""
        if "csv" in self.log_types:
            with open(self.csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, key, value])

    def set_log_level(self, level: int) -> None:
        """Dynamically changes the logging level.

        Args:
            level (int): New logging level (e.g., logging.DEBUG, logging.WARNING).

        Examples:
            >>> logger = Logger()
            >>> logger.set_log_level(logging.DEBUG)
        """
        with self.lock:
            self.logger.setLevel(level)

    def start_epoch(self, epoch: int) -> None:
        """Logs the start of an epoch.

        Args:
            epoch (int): The epoch number.

        Examples:
            >>> logger = Logger()
            >>> logger.start_epoch(1)
        """
        self.log(f"Starting epoch {epoch}", logging.INFO)

    def end_epoch(self, epoch: int) -> None:
        """Logs the end of an epoch.

        Args:
            epoch (int): The epoch number.

        Examples:
            >>> logger = Logger()
            >>> logger.end_epoch(1)
        """
        self.log(f"Ending epoch {epoch}", logging.INFO)

    def close(self) -> None:
        """Closes the TensorBoard writer if enabled.

        Examples:
            >>> logger = Logger()
            >>> logger.close()
        """
        with self.lock:
            if self.writer:
                self.writer.close()
