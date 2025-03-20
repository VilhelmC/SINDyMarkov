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
from pathlib import Path

# Import centralized logging
from models.logging_config import setup_logging, get_logger
from models.logging_config import bold, green, yellow, red, cyan, header, section
from models.logging_config import bold_green, bold_yellow, bold_red

# Import configuration loader
from models.config_loader import load_config, setup_experiment_from_config

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
    logger.info(f"  n_trials: {params['n_trials']}")
    logger.info(f"  x_range: {params['x_range']}")
    logger.info(f"  n_samples_range: {params['n_samples_range']}")
    
    # Run the comparison
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
    
    # Calculate evaluation metrics
    metrics = analysis.evaluate_model(results)
    analysis.print_metrics(metrics)
    
    # Save metrics to JSON
    import json
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    metrics_json = convert_for_json(metrics)
    with open(f"{params['save_path']}_metrics.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
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