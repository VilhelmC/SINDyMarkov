import os
import logging

def setup_logging(log_file='logs/sindy_model.log', console_level=logging.WARNING):
    """
    Set up logging configuration for the SINDy Markov Chain Model.
    
    Parameters:
    -----------
    log_file : str
        Path to the log file
    console_level : int
        Logging level for console output (default: WARNING)
        
    Returns:
    --------
    logger : Logger
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    # Create and configure logger for SINDy module
    logger = logging.getLogger('SINDyMarkovModel')
    
    # Set different levels for file and console
    for handler in logging.root.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG)  # Detailed output to file
        elif isinstance(handler, logging.StreamHandler):
            handler.setLevel(console_level)  # Less verbose output to console
    
    return logger

def get_logger():
    """Get the logger for SINDy module."""
    return logging.getLogger('SINDyMarkovModel')