import logging
import os
from pathlib import Path
import sys
from typing import Optional, Union
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from lavoisier.core.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """
    Set up logging with the provided configuration
    
    Args:
        config: Logging configuration
        
    Returns:
        Root logger configured according to the provided settings
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(config.file), exist_ok=True)
    
    # Reset root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level
    level = getattr(logging, config.level.upper())
    root_logger.setLevel(level)
    
    # Configure file handler
    file_handler = logging.FileHandler(config.file, encoding='utf-8')
    file_formatter = logging.Formatter(config.format)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Configure console handler if enabled
    if config.console_output:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True, 
            show_path=True
        )
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Return the configured logger
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance with the given name
    """
    return logging.getLogger(name)


# Rich progress bar for CLI
def create_progress_bar(description: str = "Processing", total: int = 100) -> Progress:
    """
    Create a Rich progress bar for CLI output
    
    Args:
        description: Description of the progress operation
        total: Total number of steps
        
    Returns:
        Rich Progress instance
    """
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeRemainingColumn()
    )


class ProgressLogger:
    """A logger that also manages a progress bar for CLI interfaces"""
    
    def __init__(
        self, 
        name: str, 
        total: int = 100, 
        description: str = "Processing",
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or get_logger(name)
        self.console = Console()
        self.progress = create_progress_bar(description, total)
        self.task_id = None
    
    def __enter__(self):
        """Start the progress bar when used as a context manager"""
        self.progress.start()
        self.task_id = self.progress.add_task(description=self.progress.columns[0].text, total=100)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the progress bar when exiting the context manager"""
        self.progress.stop()
    
    def update(self, advance: Union[int, float] = 1, description: Optional[str] = None):
        """Update the progress bar
        
        Args:
            advance: Number of steps to advance
            description: New description text (optional)
        """
        if self.task_id is not None:
            if description:
                self.progress.update(self.task_id, description=description, advance=advance)
            else:
                self.progress.update(self.task_id, advance=advance)
    
    def info(self, message: str):
        """Log an info message
        
        Args:
            message: Message to log
        """
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log a warning message
        
        Args:
            message: Message to log
        """
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log an error message
        
        Args:
            message: Message to log
        """
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log a debug message
        
        Args:
            message: Message to log
        """
        self.logger.debug(message)


# Initialize rich console for CLI output
console = Console()
