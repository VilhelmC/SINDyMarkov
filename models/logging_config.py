import os
import logging
import re

# Regular expression to match ANSI escape sequences
ANSI_ESCAPE_PATTERN = re.compile(r'\x1b\[[0-9;]*m')

class ColoredConsoleFormatter(logging.Formatter):
    """Formatter that adds colors for console but keeps messages unmodified."""
    COLORS = {
        'DEBUG': '\033[94m',      # Blue
        'INFO': '\033[92m',       # Green
        'WARNING': '\033[93m',    # Yellow
        'ERROR': '\033[91m',      # Red
        'CRITICAL': '\033[91;1m'  # Bright Red
    }
    RESET = '\033[0m'
    
    def format(self, record):
        colored_record = logging.makeLogRecord(vars(record))
        levelname = colored_record.levelname
        if levelname in self.COLORS:
            colored_record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(colored_record)

class PlainFileFormatter(logging.Formatter):
    """Formatter that strips ANSI color codes for file output."""
    def format(self, record):
        plain_record = logging.makeLogRecord(vars(record))
        if isinstance(plain_record.msg, str):
            # Strip ANSI escape sequences from the message
            plain_record.msg = ANSI_ESCAPE_PATTERN.sub('', plain_record.msg)
        return super().format(plain_record)

def setup_logging(log_file='logs/sindy_model.log', console_level=logging.INFO, file_level=logging.DEBUG):
    """Set up logging configuration once for the entire application."""
    # Create logs directory if needed
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with colors - force UTF-8 encoding 
    try:
        # Try to create a handler with UTF-8 encoding
        console = logging.StreamHandler()
        console.stream.reconfigure(encoding='utf-8')  # Python 3.7+ method
    except (AttributeError, UnicodeError):
        # Fallback to standard handler if reconfigure is not available or fails
        console = logging.StreamHandler()
        
    console.setLevel(console_level)
    console.setFormatter(ColoredConsoleFormatter('%(levelname)s - %(message)s'))
    
    # File handler without colors - explicitly set UTF-8 encoding
    try:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    except (ValueError, UnicodeError):
        # Fallback to default encoding if UTF-8 fails
        file_handler = logging.FileHandler(log_file, mode='w')
        
    file_handler.setLevel(file_level)
    file_handler.setFormatter(PlainFileFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Add handlers to logger
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)
    
    # Set higher log levels for third-party libraries to reduce noise
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return root_logger

def get_logger(name):
    """Get a named logger that inherits from the root configuration."""
    return logging.getLogger(name)

# Simple color functions that will be colored in console but plain in log files
def colored(text, color_code):
    """Add ANSI color to text (shows in console, stripped in log files)."""
    return f"{color_code}{text}\033[0m"

def bold(text): 
    return colored(text, '\033[1m')

def red(text): 
    return colored(text, '\033[91m')

def green(text): 
    return colored(text, '\033[92m')

def yellow(text): 
    return colored(text, '\033[93m')

def blue(text): 
    return colored(text, '\033[94m')

def magenta(text): 
    return colored(text, '\033[95m')

def cyan(text): 
    return colored(text, '\033[96m')

# Compound formatting functions
def bold_red(text):
    return bold(red(text))

def bold_green(text):
    return bold(green(text))

def bold_yellow(text):
    return bold(yellow(text))

def bold_blue(text):
    return bold(blue(text))

def bold_cyan(text):
    return bold(cyan(text))

def bold_magenta(text):
    return bold(magenta(text))

# Header/section formatting
def header(text, width=80, char='=', color_func=cyan):
    """Create a header with the given text."""
    header_text = f"\n{char * width}\n{text}\n{char * width}"
    return color_func(header_text)

def section(text, width=80, char='-', color_func=green):
    """Create a section divider with text."""
    return color_func(f"\n{char * width}\n{text}\n{char * width}")