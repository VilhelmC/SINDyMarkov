# Create a script to set up the initial directory structure

#!/usr/bin/env python3
"""
Initialize project directory structure for SINDy Markov Chain Model.

This script creates the necessary directories and empty __init__.py files
to ensure the project structure works properly as a Python package.
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create the necessary directory structure for the project."""
    # Get the project root directory
    root_dir = Path(__file__).parent
    
    # Create main directories
    directories = [
        'config/custom',
        'models',
        'scripts',
        'logs',
        'results'
    ]
    
    for directory in directories:
        dir_path = root_dir / directory
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Create __init__.py files to make Python recognize directories as packages
    init_files = [
        '__init__.py',
        'models/__init__.py',
        'scripts/__init__.py'
    ]
    
    for init_file in init_files:
        file_path = root_dir / init_file
        if not file_path.exists():
            with open(file_path, 'w') as f:
                pass  # Create empty file
            print(f"Created file: {file_path}")
    
    print("Directory structure initialized successfully.")

if __name__ == "__main__":
    create_directory_structure()