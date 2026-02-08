import logging
import sys
from pathlib import Path

class Logger:
    """
    Centralized logging utility for the entire project.
    
    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Processing started")
        logger.error("Failed to connect", exc_info=True)
    """
    
    _loggers = {}
    
    @staticmethod
    def setup(log_level: str = "INFO", log_file: str = None):
        """
        Configure the root logger.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for log output
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # Format: [2024-02-05 15:40:59] [INFO] [module_name] Message
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File Handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get or create a logger for a specific module.
        
        Args:
            name: Module name (use __name__)
        
        Returns:
            Configured logger instance
        """
        if name not in Logger._loggers:
            Logger._loggers[name] = logging.getLogger(name)
        return Logger._loggers[name]


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger."""
    return Logger.get_logger(name)


# Auto-setup on import with default INFO level
Logger.setup(log_level="INFO")
