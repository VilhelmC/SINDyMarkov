"""
Configuration loader for SINDy Markov Chain Model experiments.

This module handles loading, validation, and merging of configuration files
and creates timestamp-based directories for results organization.
"""

import os
import yaml
import numpy as np
from pathlib import Path
import logging
import datetime

# Function registry to map string names to actual functions
FUNCTION_REGISTRY = {
    # Basic functions
    'linear': lambda x: x,
    'constant': lambda x: np.ones_like(x),
    'quadratic': lambda x: x**2,
    'cubic': lambda x: x**3,
    
    # Trigonometric functions
    'sin': lambda x: np.sin(x),
    'cos': lambda x: np.cos(x),
    'tan': lambda x: np.tan(x),
    
    # Hyperbolic functions
    'sinh': lambda x: np.sinh(x),
    'cosh': lambda x: np.cosh(x),
    'tanh': lambda x: np.tanh(x),
    
    # Exponential and logarithmic
    'exp': lambda x: np.exp(x),
    'log': lambda x: np.log(np.abs(x) + 1e-10),  # Avoid log(0)
    
    # Other functions
    'sqrt': lambda x: np.sqrt(np.abs(x)),  # Avoid negative values
    'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
}

def get_function_by_name(func_name):
    """Get a function from the registry by name."""
    if func_name not in FUNCTION_REGISTRY:
        raise ValueError(f"Unknown function name: {func_name}. Available functions: {list(FUNCTION_REGISTRY.keys())}")
    
    return FUNCTION_REGISTRY[func_name]

def load_default_config():
    """Load the default configuration."""
    config_dir = Path(__file__).parent.parent / 'config'
    default_config_path = config_dir / 'default_config.yaml'
    
    if not default_config_path.exists():
        raise FileNotFoundError(f"Default config file not found at {default_config_path}")
    
    with open(default_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def load_custom_config(config_path):
    """Load a custom configuration file."""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Custom config file not found at {path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def merge_configs(default_config, custom_config):
    """Merge custom configuration with default configuration."""
    merged_config = default_config.copy()
    
    def _merge_dict(target, source):
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                _merge_dict(target[key], value)
            else:
                # Replace or add values
                target[key] = value
    
    _merge_dict(merged_config, custom_config)
    
    return merged_config

def validate_config(config):
    """Validate that the configuration has all required fields."""
    required_fields = {
        'experiment': ['name', 'save_path'],
        'model': ['library_functions', 'true_coefficients'],
        'simulation': ['n_trials'],
        'logging': ['log_file']
    }
    
    errors = []
    
    for section, fields in required_fields.items():
        if section not in config:
            errors.append(f"Missing required section: {section}")
            continue
        
        for field in fields:
            if field not in config[section]:
                errors.append(f"Missing required field: {section}.{field}")
    
    # Check that model parameters are valid
    if 'model' in config:
        if 'library_functions' in config['model']:
            if not isinstance(config['model']['library_functions'], list):
                errors.append("model.library_functions must be a list")
            elif len(config['model']['library_functions']) == 0:
                errors.append("model.library_functions cannot be empty")
            else:
                # Check that all functions are valid
                for i, func_dict in enumerate(config['model']['library_functions']):
                    if 'name' not in func_dict:
                        errors.append(f"Missing name for function at index {i}")
                    elif func_dict['name'] not in FUNCTION_REGISTRY:
                        errors.append(f"Unknown function name: {func_dict['name']}")
        
        if 'true_coefficients' in config['model']:
            if not isinstance(config['model']['true_coefficients'], list):
                errors.append("model.true_coefficients must be a list")
            elif 'library_functions' in config['model'] and len(config['model']['true_coefficients']) != len(config['model']['library_functions']):
                errors.append("model.true_coefficients must have the same length as model.library_functions")
    
    # Check simulation parameters
    if 'simulation' in config:
        if 'x_range' in config['simulation'] and not isinstance(config['simulation']['x_range'], list):
            errors.append("simulation.x_range must be a list")
        if 'n_samples_range' in config['simulation'] and not isinstance(config['simulation']['n_samples_range'], list):
            errors.append("simulation.n_samples_range must be a list")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(errors)
        raise ValueError(error_msg)
    
    return True

def prepare_library_functions(config):
    """Convert function name strings to actual function objects."""
    if 'model' not in config or 'library_functions' not in config['model']:
        raise ValueError("Configuration must include model.library_functions")
    
    functions = []
    for func_dict in config['model']['library_functions']:
        func_name = func_dict['name']
        func = get_function_by_name(func_name)
        functions.append(func)
    
    return functions

def create_timestamp_directory(base_path, identifier=None):
    """
    Create a timestamped directory for experiment results.
    
    Parameters:
    -----------
    base_path : str
        Base path for the directory
    identifier : str, optional
        Optional identifier to add to the directory name
        
    Returns:
    --------
    dir_path : str
        Path to the created directory
    """
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory name
    if identifier:
        dir_name = f"{identifier}_{timestamp}"
    else:
        dir_name = timestamp
    
    # Create full path
    dir_path = os.path.join(base_path, dir_name)
    
    # Create directory
    os.makedirs(dir_path, exist_ok=True)
    
    return dir_path

def load_config(custom_config_path=None):
    """
    Load and validate configuration.
    
    Parameters:
    -----------
    custom_config_path : str, optional
        Path to a custom configuration file
        
    Returns:
    --------
    config : dict
        Validated configuration dictionary
    """
    # Load default config
    config = load_default_config()
    
    # If custom config is provided, merge it with the default
    if custom_config_path:
        custom_config = load_custom_config(custom_config_path)
        config = merge_configs(config, custom_config)
    
    # Validate the final config
    validate_config(config)
    
    return config

def setup_experiment_from_config(config):
    """
    Set up an experiment based on configuration with timestamp-based directories.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    experiment_params : dict
        Dictionary containing all parameters needed to run the experiment
    """
    # Extract base paths
    base_results_path = os.path.dirname(config['experiment']['save_path'])
    base_logs_path = os.path.dirname(config['logging']['log_file'])
    
    # Create timestamp directories
    experiment_id = config['experiment'].get('id', config['experiment']['name'].replace(' ', '_').lower())
    results_dir = create_timestamp_directory(base_results_path, experiment_id)
    logs_dir = create_timestamp_directory(base_logs_path, experiment_id)
    
    # Update paths in config
    save_path = os.path.join(results_dir, os.path.basename(config['experiment']['save_path']))
    log_file = os.path.join(logs_dir, os.path.basename(config['logging']['log_file']))
    
    # Setup logging
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    console_level = log_level_map.get(config['logging'].get('console_level', 'INFO'), logging.INFO)
    file_level = log_level_map.get(config['logging'].get('file_level', 'DEBUG'), logging.DEBUG)
    
    # Prepare library functions
    library_functions = prepare_library_functions(config)
    
    # Convert true coefficients to numpy array
    true_coefficients = np.array(config['model']['true_coefficients'])
    
    # Prepare experiment parameters
    experiment_params = {
        'name': config['experiment']['name'],
        'id': experiment_id,
        'description': config['experiment'].get('description', ''),
        'save_path': save_path,
        'results_dir': results_dir,
        'logs_dir': logs_dir,
        'library_functions': library_functions,
        'true_coefficients': true_coefficients,
        'sigma': config['model'].get('sigma', 0.1),
        'threshold': config['model'].get('threshold', 0.05),
        
        # Fixed vs adaptive trials
        'adaptive_trials': config['simulation'].get('adaptive_trials', False),
        'n_trials': config['simulation'].get('n_trials', 50),
        
        # Adaptive trial parameters
        'max_trials': config['simulation'].get('max_trials', 500),
        'min_trials': config['simulation'].get('min_trials', 30),
        'confidence': config['simulation'].get('confidence', 0.95),
        'margin': config['simulation'].get('margin', 0.05),
        'batch_size': config['simulation'].get('batch_size', 10),
        
        # Common parameters
        'x_range': np.array(config['simulation'].get('x_range', [0.5, 1.0, 1.5])),
        'n_samples_range': np.array(config['simulation'].get('n_samples_range', [100, 200, 300])),
        'analyze_coefficients': config['simulation'].get('analyze_coefficients', True),
        'log_file': log_file,
        'console_level': console_level,
        'file_level': file_level,
        'diagnose_transitions': config['logging'].get('diagnose_transitions', False),
        
        # Save original config
        'config': config
    }
    
    return experiment_params