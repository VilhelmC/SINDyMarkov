import os
import logging
import re
import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform colored terminal output
colorama.init(autoreset=False)

# Regular expression to match ANSI escape sequences
ANSI_ESCAPE_PATTERN = re.compile(r'\x1b\[[0-9;]*m')

class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors for console but keeps messages unmodified."""
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }
    
    def format(self, record):
        colored_record = logging.makeLogRecord(vars(record))
        levelname = colored_record.levelname
        if levelname in self.COLORS:
            colored_record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
            
        # Apply formatting to the message if it doesn't already have color codes
        if isinstance(colored_record.msg, str) and not colored_record.msg.startswith('\033'):
            # Special formatting for headers and sections
            if '=' * 20 in colored_record.msg:
                colored_record.msg = cyan(colored_record.msg, bright=True)
            elif '-' * 20 in colored_record.msg:
                colored_record.msg = green(colored_record.msg)
            elif colored_record.msg.startswith('TRUE MODEL STATE'):
                colored_record.msg = bold_yellow(colored_record.msg)
            elif 'Discrepancy:' in colored_record.msg and 'Large' not in colored_record.msg:
                # Regular discrepancy
                colored_record.msg = colored_record.msg.replace('Discrepancy:', f"Discrepancy: {yellow('')}")
            elif 'Large Discrepancy:' in colored_record.msg:
                # Large discrepancy
                colored_record.msg = colored_record.msg.replace('Large Discrepancy:', f"Large Discrepancy: {bold_red('')}")
                
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
    """
    Set up logging configuration for the SINDy Markov Chain Model.
    
    Parameters:
    -----------
    log_file : str
        Path to the log file
    console_level : int
        Logging level for console output (default: INFO)
    file_level : int
        Logging level for file output (default: DEBUG)
        
    Returns:
    --------
    logger : Logger
        Configured logger instance
    """
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
    
    # Console handler with colors
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(ColoredFormatter('%(levelname)s - %(message)s'))
    
    # File handler without colors
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(PlainFileFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Add handlers to logger
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)
    
    # Create a module-specific logger
    logger = logging.getLogger('SINDyMarkovModel')
    
    return logger

def get_logger(name='SINDyMarkovModel'):
    """Get the logger for SINDy module."""
    return logging.getLogger(name)

def suppress_common_warnings():
    """Suppress common warnings that clutter the terminal output."""
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.validation")
    warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn._oldcore")
    warnings.filterwarnings("ignore", message=".*is_sparse.*")
    warnings.filterwarnings("ignore", message=".*is_categorical_dtype.*")

# Color utility functions
def colored(text, color_code):
    """Add ANSI color to text."""
    return f"{color_code}{text}{Style.RESET_ALL}"

def bold(text): 
    return colored(text, Style.BRIGHT)

def red(text, bright=False): 
    return colored(text, Fore.RED + (Style.BRIGHT if bright else ""))

def green(text, bright=False): 
    return colored(text, Fore.GREEN + (Style.BRIGHT if bright else ""))

def yellow(text, bright=False): 
    return colored(text, Fore.YELLOW + (Style.BRIGHT if bright else ""))

def blue(text, bright=False): 
    return colored(text, Fore.BLUE + (Style.BRIGHT if bright else ""))

def magenta(text, bright=False): 
    return colored(text, Fore.MAGENTA + (Style.BRIGHT if bright else ""))

def cyan(text, bright=False): 
    return colored(text, Fore.CYAN + (Style.BRIGHT if bright else ""))

# Compound formatting functions
def bold_red(text):
    return colored(text, Style.BRIGHT + Fore.RED)

def bold_green(text):
    return colored(text, Style.BRIGHT + Fore.GREEN)

def bold_yellow(text):
    return colored(text, Style.BRIGHT + Fore.YELLOW)

def bold_blue(text):
    return colored(text, Style.BRIGHT + Fore.BLUE)

def bold_magenta(text):
    return colored(text, Style.BRIGHT + Fore.MAGENTA)

def bold_cyan(text):
    return colored(text, Style.BRIGHT + Fore.CYAN)

def header(text, width=80, char='=', color_func=cyan):
    """Create a header with the given text."""
    padding = max(0, (width - len(text) - 2) // 2)
    header_text = f"\n{char * padding} {text} {char * padding}"
    return color_func(header_text, bright=True)

def section(text, width=80, char='-', color_func=green):
    """Create a section divider with text."""
    padding = max(0, (width - len(text) - 2) // 2)
    section_text = f"\n{char * padding} {text} {char * padding}"
    return color_func(section_text)

def log_to_markdown(log_file, markdown_file=None):
    """
    Convert a log file to a formatted Markdown document.
    This makes log files more readable when viewed in a Markdown viewer.
    
    Parameters:
    -----------
    log_file : str
        Path to the log file
    markdown_file : str, optional
        Path to the output Markdown file. If None, uses the same name with .md extension.
    
    Returns:
    --------
    markdown_file : str
        Path to the created markdown file
    """
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return None
    
    # Create default markdown file name if not provided
    if markdown_file is None:
        markdown_file = os.path.splitext(log_file)[0] + '.md'
    
    # Create directory for markdown file if it doesn't exist
    md_dir = os.path.dirname(markdown_file)
    if md_dir and not os.path.exists(md_dir):
        os.makedirs(md_dir)
    
    with open(log_file, 'r', encoding='utf-8') as f:
        log_content = f.readlines()
    
    with open(markdown_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write(f"# SINDy Markov Model Log: {os.path.basename(log_file)}\n\n")
        f.write(f"*Generated from {log_file}*\n\n")
        
        # Process log lines
        in_code_block = False
        current_section = None
        
        for line in log_content:
            # Remove timestamp and logger name for cleaner output
            if ' - SINDyMarkovModel - ' in line:
                parts = line.split(' - SINDyMarkovModel - ', 1)
                timestamp = parts[0]
                content = parts[1] if len(parts) > 1 else line
                
                # Format based on content
                if '=' * 20 in content:
                    # Major section header
                    section_text = content.strip()
                    f.write(f"\n## {section_text}\n\n")
                    current_section = section_text
                elif '-' * 20 in content:
                    # Minor section header
                    section_text = content.strip()
                    f.write(f"\n### {section_text}\n\n")
                    current_section = section_text
                elif 'SUCCESS PROBABILITY CALCULATION' in content:
                    f.write(f"\n### {content.strip()}\n\n")
                elif content.startswith('INFO - '):
                    # Regular info message
                    message = content.replace('INFO - ', '', 1).strip()
                    if message.startswith('TRUE MODEL STATE'):
                        f.write(f"\n**{message}**\n\n")
                    else:
                        f.write(f"{message}\n\n")
                elif content.startswith('WARNING - '):
                    # Warning message
                    message = content.replace('WARNING - ', '', 1).strip()
                    f.write(f"> ⚠️ **Warning:** {message}\n\n")
                elif content.startswith('ERROR - '):
                    # Error message
                    message = content.replace('ERROR - ', '', 1).strip()
                    f.write(f"> ❌ **Error:** {message}\n\n")
                elif content.startswith('DEBUG - '):
                    # Debug message - use smaller text
                    message = content.replace('DEBUG - ', '', 1).strip()
                    if 'Transitions from' in message:
                        parts = message.split('Transitions from', 1)
                        f.write(f"<details>\n<summary>Transitions from{parts[1]}</summary>\n\n```\n{message}\n```\n\n</details>\n\n")
                    else:
                        f.write(f"<small>{message}</small>\n\n")
                else:
                    # Other content
                    f.write(f"{line}\n\n")
            else:
                # For lines without the standard format
                f.write(f"{line.strip()}\n\n")
    
    print(f"Markdown log created: {markdown_file}")
    return markdown_file