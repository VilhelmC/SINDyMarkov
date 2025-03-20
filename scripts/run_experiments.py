#!/usr/bin/env python3
"""
Run experiments for the SINDy Markov Chain Model.

This script executes experiments to validate the theoretical model
for predicting success probabilities in SINDy algorithm with STLSQ.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import sys
import argparse
import importlib.util
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import centralized logging
from models.logging_config import setup_logging, get_logger
from models.logging_config import bold, green, yellow, red, cyan, header, section
from models.logging_config import bold_green, bold_yellow, bold_red

# Import configuration loader
from models.config_loader import load_config, setup_experiment_from_config

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def discover_and_import_module(module_name, possible_locations=None):
    """
    Dynamically discovers and imports a module by searching in common locations.
    
    Args:
        module_name: Name of the module file without .py extension
        possible_locations: Optional list of directories to search in
        
    Returns:
        The imported module object
    """
    # Default search locations if none provided
    if possible_locations is None:
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        possible_locations = [
            current_dir,                     # Same directory as script
            current_dir.parent,              # Parent directory
            current_dir.parent / "models",   # models/ subdirectory
            Path.cwd(),                      # Current working directory
            Path.cwd() / "models",           # models/ in current working directory
        ]
    
    # Search for the module
    module_filename = f"{module_name}.py"
    found_path = None
    
    for location in possible_locations:
        candidate = Path(location) / module_filename
        if candidate.exists():
            found_path = candidate
            break
    
    if found_path is None:
        raise ImportError(f"Could not find module {module_name} in any of the search locations")
    
    # Import the module
    spec = importlib.util.spec_from_file_location(module_name, found_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    print(f"Successfully imported {module_name} from {found_path}")
    return module

def import_modules():
    """Import necessary modules."""
    modules = {}
    
    # Import core model
    sindy_markov_module = discover_and_import_module("sindy_markov_model")
    modules['model'] = sindy_markov_module.SINDyMarkovModel
    
    # Import analysis module
    analysis_module = discover_and_import_module("markov_analysis")
    modules['analysis'] = analysis_module
    
    # Import simulation module
    simulation_module = discover_and_import_module("markov_simulation")
    modules['simulation'] = simulation_module
    
    return modules

def ensure_directory_exists(file_path):
    """Ensure the directory for a file exists, creating it if necessary."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def run_experiment(config_path=None):
    """
    Run an experiment based on configuration.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to a custom configuration file. If None, use default config.
    """
    start_time = time.time()
    
    # Load configuration
    config = load_config(config_path)
    params = setup_experiment_from_config(config)
    
    # Setup logging
    setup_logging(params['log_file'], params['console_level'], params['file_level'])
    logger = get_logger('SINDyExperiment')
    
    logger.info(header(f"STARTING EXPERIMENT: {params['name']}"))
    logger.info(bold(f"Description: {params['description']}"))
    
    # Import modules
    modules = import_modules()
    SINDyMarkovModel = modules['model']
    analysis = modules['analysis']
    
    # Create model
    model = SINDyMarkovModel(
        params['library_functions'], 
        params['true_coefficients'], 
        params['sigma'], 
        params['threshold'], 
        log_file=params['log_file']
    )
    
    # Run comparison between theory and simulation
    logger.info(section("RUNNING THEORY VS SIMULATION COMPARISON"))
    
    # Log experiment parameters
    logger.info(bold_yellow("Experiment Parameters:"))
    logger.info(f"  Library Functions: {[getattr(f, '__name__', str(f)) for f in params['library_functions']]}")
    logger.info(f"  True Coefficients: {params['true_coefficients']}")
    logger.info(f"  Sigma (noise): {params['sigma']}")
    logger.info(f"  Threshold: {params['threshold']}")
    logger.info(f"  Lambda/Sigma Ratio: {params['threshold']/params['sigma']:.4f}")
    
    if params['adaptive_trials']:
        logger.info(f"  Using adaptive trials (max: {params['max_trials']}, min: {params['min_trials']})")
        logger.info(f"  Confidence: {params['confidence']*100:.0f}%, Margin: {params['margin']*100:.1f}%")
    else:
        logger.info(f"  Fixed trials: {params['n_trials']}")
    
    logger.info(f"  x_range: {params['x_range']}")
    logger.info(f"  n_samples_range: {params['n_samples_range']}")
    
    # Run the comparison
    if params['adaptive_trials']:
        results = analysis.compare_theory_to_simulation(
            model, 
            params['x_range'], 
            params['n_samples_range'],
            adaptive_trials=True,
            max_trials=params['max_trials'],
            min_trials=params['min_trials'],
            confidence=params['confidence'],
            margin=params['margin'],
            batch_size=params['batch_size']
        )
    else:
        results = analysis.compare_theory_to_simulation(
            model, 
            params['x_range'], 
            params['n_samples_range'], 
            n_trials=params['n_trials']
        )
    
    # Save results to CSV
    results_file = f"{params['save_path']}_results.csv"
    results.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")
    
    # Plot and save figures
    fig1 = analysis.plot_comparison(results, x_axis='log_gram_det')
    fig1.savefig(f"{params['save_path']}_log_gram_det.png", dpi=300, bbox_inches='tight')
    
    fig1b = analysis.plot_comparison(results, x_axis='discriminability')
    fig1b.savefig(f"{params['save_path']}_discriminability.png", dpi=300, bbox_inches='tight')
    
    fig2 = analysis.plot_direct_comparison(results)
    fig2.savefig(f"{params['save_path']}_direct_comparison.png", dpi=300, bbox_inches='tight')
    
    # If using adaptive trials, also create a plot showing trials used vs. discrepancy
    if params['adaptive_trials'] and 'trials_used' in results.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(results['trials_used'], results['discrepancy'], s=80, alpha=0.7)
        plt.xlabel('Number of Trials Used')
        plt.ylabel('Discrepancy (|Theoretical - Empirical|)')
        plt.title('Discrepancy vs. Number of Trials Used')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{params['save_path']}_trials_vs_discrepancy.png", dpi=300, bbox_inches='tight')
    
    # Calculate evaluation metrics
    metrics = analysis.evaluate_model(results)
    analysis.print_metrics(metrics)
    
    # Convert numpy types to Python native types for JSON serialization
    def numpy_to_python_types(obj):
        """
        Recursively convert numpy types to standard Python types.
        Compatible with NumPy 2.0+
        
        Parameters:
        -----------
        obj : any
            Object that may contain numpy data types
            
        Returns:
        --------
        converted_obj : any
            Same object with all numpy types converted to Python native types
        """
        # Handle NumPy scalars
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return [numpy_to_python_types(x) for x in obj]
        
        # Handle other containers
        elif isinstance(obj, dict):
            return {numpy_to_python_types(k): numpy_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [numpy_to_python_types(x) for x in obj]
        elif hasattr(obj, 'item') and callable(getattr(obj, 'item')):
            # Handle other numpy scalar types that have an 'item' method
            try:
                return obj.item()
            except:
                return obj
        else:
            # Return unchanged for non-numpy types
            return obj
        
    metrics_json = numpy_to_python_types(metrics)
    
    # Add adaptive trial information if applicable
    if params['adaptive_trials']:
        metrics_json['adaptive_trials'] = {
            'enabled': True,
            'max_trials': params['max_trials'],
            'min_trials': params['min_trials'],
            'confidence': params['confidence'],
            'margin': params['margin'],
            'average_trials_used': results['trials_used'].mean(),
            'min_trials_used': results['trials_used'].min(),
            'max_trials_used': results['trials_used'].max()
        }
    
    metrics_file = f"{params['save_path']}_metrics.json"
    ensure_directory_exists(metrics_file)
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    logger.info(bold_green(f"\nExperiment completed in {execution_time:.2f} seconds"))
    
    return results, metrics

def main():
    """Parse arguments and run experiment."""
    parser = argparse.ArgumentParser(description='Run SINDy Markov Chain Model experiments')
    parser.add_argument('--config', type=str, help='Path to custom configuration file')
    args = parser.parse_args()
    
    run_experiment(args.config)

if __name__ == "__main__":
    main()