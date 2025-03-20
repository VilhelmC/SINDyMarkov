import os
import logging
import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform colored terminal output
colorama.init()

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log levels and preserves ANSI color codes
    already present in log messages.
    """
    
    # Define colors for different log levels
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }
    
    def __init__(self, fmt=None, datefmt=None, style='%', validate=True):
        super().__init__(fmt, datefmt, style, validate)
    
    def format(self, record):
        # Save the original message
        original_msg = record.msg
        
        # Color the log level
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        
        # Format the record
        result = super().format(record)
        
        # Restore the original message
        record.msg = original_msg
        record.levelname = levelname
        
        return result

class ColorPreservingFileHandler(logging.FileHandler):
    """
    File handler that preserves ANSI color codes in log files.
    """
    
    def __init__(self, filename, mode='a', encoding='utf-8'):
        super().__init__(filename, mode, encoding)

def setup_enhanced_logging(log_file='logs/sindy_model.log', console_level=logging.INFO, file_level=logging.DEBUG, ansi_in_file=True):
    """
    Set up logging with colored output and proper formatting.
    
    Parameters:
    -----------
    log_file : str
        Path to the log file
    console_level : int
        Logging level for console output
    file_level : int
        Logging level for file output
    ansi_in_file : bool
        Whether to preserve ANSI color codes in log files
        
    Returns:
    --------
    logger : Logger
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to lowest level to capture everything
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    
    # Create file handler
    if ansi_in_file:
        file_handler = ColorPreservingFileHandler(log_file, mode='w')
    else:
        file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(file_level)
    
    # Create formatters
    console_formatter = ColoredFormatter(
        '%(levelname)s - %(message)s'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set formatters
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name='SINDyMarkovModel'):
    """Get a named logger."""
    return logging.getLogger(name)

def color_text(text, color=Fore.GREEN, style=Style.NORMAL):
    """
    Add color and style to text.
    
    Parameters:
    -----------
    text : str
        Text to colorize
    color : str
        Color code from colorama.Fore
    style : str
        Style code from colorama.Style
        
    Returns:
    --------
    colored_text : str
        Colorized text
    """
    return f"{style}{color}{text}{Style.RESET_ALL}"

# Define shortcut functions for common colors
def green(text, bright=False):
    """Green colored text."""
    style = Style.BRIGHT if bright else Style.NORMAL
    return color_text(text, Fore.GREEN, style)

def red(text, bright=False):
    """Red colored text."""
    style = Style.BRIGHT if bright else Style.NORMAL
    return color_text(text, Fore.RED, style)

def yellow(text, bright=False):
    """Yellow colored text."""
    style = Style.BRIGHT if bright else Style.NORMAL
    return color_text(text, Fore.YELLOW, style)

def blue(text, bright=False):
    """Blue colored text."""
    style = Style.BRIGHT if bright else Style.NORMAL
    return color_text(text, Fore.BLUE, style)

def cyan(text, bright=False):
    """Cyan colored text."""
    style = Style.BRIGHT if bright else Style.NORMAL
    return color_text(text, Fore.CYAN, style)

def magenta(text, bright=False):
    """Magenta colored text."""
    style = Style.BRIGHT if bright else Style.NORMAL
    return color_text(text, Fore.MAGENTA, style)

def white(text, bright=False):
    """White colored text."""
    style = Style.BRIGHT if bright else Style.NORMAL
    return color_text(text, Fore.WHITE, style)

def bold(text):
    """Bold text."""
    return f"{Style.BRIGHT}{text}{Style.NORMAL}"

def header(text, width=80, char='=', color=Fore.CYAN):
    """Create a header with the given text."""
    padding = (width - len(text) - 2) // 2
    return color_text(f"{char*padding} {text} {char*padding}", color, Style.BRIGHT)

def section(text, width=80, color=Fore.GREEN):
    """Create a section header."""
    return color_text(f"{'-'*width}\n{text}\n{'-'*width}", color, Style.BRIGHT)