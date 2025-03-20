#!/usr/bin/env python3
"""
Install all improvements to the SINDy Markov Chain Model.

This script copies the fixed files to the appropriate locations and runs tests
to verify that the improvements work correctly.
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
import datetime

def backup_file(file_path):
    """Create a backup of a file with timestamp."""
    if not os.path.exists(file_path):
        print(f"File does not exist, cannot backup: {file_path}")
        return None
    
    # Create backup filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.{timestamp}.bak"
    
    # Copy the file
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    return backup_path

def install_file(source_path, dest_path, backup=True):
    """Install a file, optionally creating a backup of the original."""
    # Ensure destination directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Create backup if requested and file exists
    if backup and os.path.exists(dest_path):
        backup_file(dest_path)
    
    # Copy the file
    shutil.copy2(source_path, dest_path)
    print(f"Installed: {dest_path}")

def run_verification():
    """Run the verification script."""
    script_path = os.path.join(os.path.dirname(__file__), "verify_fixes.py")
    print(f"Running verification script: {script_path}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              check=True, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              universal_newlines=True)
        
        print(f"Verification completed successfully: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Verification failed with error: {e}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main function to install improvements."""
    parser = argparse.ArgumentParser(description="Install SINDy Markov Chain Model improvements")
    parser.add_argument('--no-backup', action='store_true', help="Don't create backups of original files")
    parser.add_argument('--no-verify', action='store_true', help="Don't run verification after installation")
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '..'))
    
    print(f"Installing improvements from {script_dir} to {root_dir}")
    
    # Define the paths of files to be installed
    source_files = {
        'sindy_markov_model.py': os.path.join(script_dir, 'sindy_markov_model.py'),
        'config_loader.py': os.path.join(script_dir, 'config_loader.py'),
        'log_to_markdown.py': os.path.join(script_dir, 'log_to_markdown.py'),
        'verify_fixes.py': os.path.join(script_dir, 'verify_fixes.py')
    }
    
    dest_files = {
        'sindy_markov_model.py': os.path.join(root_dir, 'models', 'sindy_markov_model.py'),
        'config_loader.py': os.path.join(root_dir, 'models', 'config_loader.py'),
        'log_to_markdown.py': os.path.join(root_dir, 'scripts', 'log_to_markdown.py'),
        'verify_fixes.py': os.path.join(root_dir, 'scripts', 'verify_fixes.py')
    }
    
    # Check if source files exist
    missing_files = []
    for name, path in source_files.items():
        if not os.path.exists(path):
            missing_files.append(name)
    
    if missing_files:
        print(f"Error: The following source files are missing: {', '.join(missing_files)}")
        print("Please make sure all files are in the same directory as this script.")
        return False
    
    # Install files
    for name, source_path in source_files.items():
        dest_path = dest_files[name]
        install_file(source_path, dest_path, backup=not args.no_backup)
    
    # Make scripts executable
    if os.name != 'nt':  # Skip on Windows
        os.chmod(dest_files['log_to_markdown.py'], 0o755)
        os.chmod(dest_files['verify_fixes.py'], 0o755)
    
    print("\nAll improvements installed successfully!")
    
    # Run verification if requested
    if not args.no_verify:
        print("\nRunning verification tests...")
        success = run_verification()
        
        if success:
            print("\nVerification passed! The improvements are working correctly.")
        else:
            print("\nVerification failed. Please check the logs for details.")
        
        return success
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)