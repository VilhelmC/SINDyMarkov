#!/usr/bin/env python3
"""
Run comprehensive diagnostics on the SINDy Markov Chain Model.

This script performs in-depth analysis of theoretical vs empirical transition probabilities,
coefficient distributions, and tests the independence assumption.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import importlib.util
import json
from pathlib import Path

# Import centralized logging configuration
from models.logging_config import setup_logging, get_logger
from models.logging_config import bold, green, yellow, red, cyan, header, section
from models.logging_config import bold_green, bold_yellow, bold_red

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

# Import necessary modules
def import_modules():
    """Import SINDy model and diagnostics modules."""
    # Import core model
    sindy_markov_module = discover_and_import_module("sindy_markov_model")
    
    # Import diagnostics module
    diagnostics_module = discover_and_import_module("markov_diagnostics")
    
    return {
        'model': sindy_markov_module.SINDyMarkovModel,
        'diagnostics': diagnostics_module
    }

def run_diagnostics(data_range=1.0, n_samples=300, n_trials=100, sigma=0.1, threshold=0.05):
    """
    Run comprehensive diagnostics on the SINDy Markov Chain Model.
    
    Parameters:
    -----------
    data_range : float
        Range of data to sample from
    n_samples : int
        Number of samples to use
    n_trials : int
        Number of trials to run
    sigma : float
        Noise level
    threshold : float
        STLSQ threshold
    """
    # Set up logging
    log_file = 'logs/diagnostics.log'
    setup_logging(log_file)
    logger = get_logger('diagnostics')
    
    # Print colorful header
    logger.info(header(f"RUNNING SINDY MARKOV MODEL DIAGNOSTICS", width=80))
    logger.info(bold("Settings:"))
    logger.info(f"  {yellow('Data Range:')} {data_range}")
    logger.info(f"  {yellow('Number of Samples:')} {n_samples}")
    logger.info(f"  {yellow('Number of Trials:')} {n_trials}")
    logger.info(f"  {yellow('Noise Level (σ):')} {sigma}")
    logger.info(f"  {yellow('Threshold (λ):')} {threshold}")
    logger.info(f"  {yellow('λ/σ Ratio:')} {threshold/sigma:.2f}")
    
    # Create output directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Import modules
    modules = import_modules()
    SINDyMarkovModel = modules['model']
    diagnostics = modules['diagnostics']
    
    # Define simple library functions
    def f1(x): return x          # Term 1: x
    def f2(x): return np.sin(x)  # Term 2: sin(x)
    def f3(x): return np.tanh(x) # Term 3: tanh(x)
    
    library_functions = [f1, f2, f3]
    
    # Define true model: only the first term is present
    true_coefs = np.array([1.0, 0.0, 0.0])
    
    # Create model instance with detailed logging
    model = SINDyMarkovModel(library_functions, true_coefs, sigma, threshold, log_file=log_file)
    
    # Generate sample points
    logger.info(f"\n{green(bold('Generating ' + str(n_samples) + ' sample points in range [-' + str(data_range) + ', ' + str(data_range) + ']'))}")
    x_data = np.random.uniform(-data_range, data_range, n_samples)
    
    # Compute Gram matrix
    logger.info(f"{green(bold('Computing Gram matrix'))}")
    model.compute_gram_matrix(x_data)
    
    # Step 1: Calculate theoretical success probability
    logger.info(header("1. THEORETICAL SUCCESS PROBABILITY ANALYSIS"))
    theoretical_prob = model.calculate_success_probability()
    
    # Step 2: Calculate empirical success probability
    logger.info(header("2. EMPIRICAL SUCCESS PROBABILITY SIMULATION"))
    empirical_prob = model.simulate_stlsq(x_data, n_trials)
    
    logger.info(f"\n{bold_green('Success Probability Summary:')}")
    logger.info(f"  {yellow('Theoretical:')} {theoretical_prob:.4f}")
    logger.info(f"  {yellow('Empirical:')} {empirical_prob:.4f}")
    difference = abs(theoretical_prob - empirical_prob)
    if difference > 0.1:
        logger.info(f"  {red('Difference:')} {difference:.4f} {red('(Significant)')}")
    else:
        logger.info(f"  {yellow('Difference:')} {difference:.4f}")
    
    # Step 3: Run transition probability comparison
    logger.info(header("3. TRANSITION PROBABILITY ANALYSIS"))
    comparison_data = diagnostics.compare_theory_to_simulation_transitions(model, x_data, n_trials)
    
    # Step 4: Analyze coefficient distributions
    logger.info(header("4. COEFFICIENT DISTRIBUTION ANALYSIS"))
    distribution_data = diagnostics.analyze_coefficient_distributions(model, x_data, min(50, n_trials))
    
    # Step 5: Test independence assumption
    logger.info(header("5. INDEPENDENCE ASSUMPTION TESTING"))
    independence_metrics = diagnostics.test_independence_assumption(model, x_data, min(50, n_trials))
    
    # Step 6: Verify coefficient distributions
    logger.info(header("6. COEFFICIENT DISTRIBUTION VERIFICATION"))
    verification_results = diagnostics.verify_coefficient_distributions(model, x_data, min(50, n_trials))
    
    # Compile all diagnostic results
    all_diagnostics = {
        'model_settings': {
            'data_range': data_range,
            'n_samples': n_samples,
            'n_trials': n_trials,
            'sigma': sigma,
            'threshold': threshold,
            'lambda_sigma_ratio': threshold/sigma
        },
        'gram_matrix': {
            'log_determinant': model.log_gram_det
        },
        'success_probability': {
            'theoretical': theoretical_prob,
            'empirical': empirical_prob,
            'difference': difference
        },
        'transitions': comparison_data,
        'distributions': distribution_data,
        'independence': independence_metrics,
        'verification': verification_results
    }
    
    # Print summary of findings
    logger.info(header("DIAGNOSTIC SUMMARY"))
    logger.info(f"{bold_green('Model Parameters:')}")
    logger.info(f"  Data Range: {data_range}, Samples: {n_samples}, Trials: {n_trials}")
    logger.info(f"  Sigma: {sigma}, Threshold: {threshold}, λ/σ Ratio: {threshold/sigma:.4f}")
    logger.info(f"  Log Determinant of Gram Matrix: {model.log_gram_det:.4f}")
    
    logger.info(f"\n{bold_green('Success Probability:')}")
    logger.info(f"  Theoretical: {theoretical_prob:.4f}, Empirical: {empirical_prob:.4f}")
    if difference > 0.1:
        logger.info(f"  {bold_red('Large discrepancy: ' + f'{difference:.4f}')}")
    else:
        logger.info(f"  Difference: {difference:.4f}")
    
    # Look for significant issues
    logger.info(f"\n{bold_green('Significant Issues Identified:')}")
    
    issues_found = False
    
    # Check success probability discrepancy
    if difference > 0.1:
        issues_found = True
        logger.info(f"  {bold_red('- Large discrepancy in success probability (' + f'{difference:.4f}' + ')')}")
    
    # Check for significant transition probability differences
    significant_diffs = [(k, v) for k, v in comparison_data.items() if v['significant']]
    if significant_diffs:
        issues_found = True
        logger.info(f"  {bold_red('- ' + str(len(significant_diffs)) + ' transitions with significant probability differences')}")
        logger.info("    Top 3 most significant:")
        significant_diffs.sort(key=lambda x: abs(x[1]['difference']), reverse=True)
        for i, (key, data) in enumerate(significant_diffs[:3]):
            logger.info(f"      {data['from_state']} → {data['to_state']}: Diff={data['difference']:+.4f}")
    
    # Check for independence assumption violations
    independence_issues = False
    for trans_key, metrics in independence_metrics.items():
        corr_matrix = metrics['correlation_matrix']
        if np.any(np.abs(corr_matrix) > 0.3):
            independence_issues = True
            break
    
    if independence_issues:
        issues_found = True
        logger.info(f"  {bold_red('- Violations of independence assumption in coefficient thresholding')}")
    
    # Distribution verification issues
    try:
        mean_diff = np.max(np.abs(verification_results['theoretical']['mean'] - verification_results['empirical']['mean']))
        if mean_diff > 0.1:
            issues_found = True
            logger.info(f"  {bold_red('- Large difference in coefficient means (max diff: ' + f'{mean_diff:.6f}' + ')')}")
    except Exception as e:
        logger.warning(f"Could not compute mean difference: {e}")
    
    if not issues_found:
        logger.info(f"  {bold_green('No significant issues detected.')}")
    
    # Print possible solutions
    if issues_found:
        logger.info(f"\n{bold_green('Possible Solutions to Consider:')}")
        if difference > 0.1 or independence_issues:
            logger.info("  1. Revise transition probability calculation to account for dependencies")
            logger.info("  2. Consider conditional probabilities instead of independence assumption")
            logger.info("  3. Develop an empirical correction factor for specific transitions")
        if mean_diff > 0.1:
            logger.info("  4. Improve coefficient distribution modeling")
            logger.info("  5. Consider using regularization to stabilize coefficient estimates")
    
    logger.info(f"\n{yellow('See detailed logs in ' + log_file + ' for more information.')}")
    
    # Save diagnostic results
    results_file = 'results/diagnostics_results.json'
    try:
        # Convert NumPy arrays to lists for JSON serialization
        def prepare_for_json(obj):
            if isinstance(obj, dict):
                return {k: prepare_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list) or isinstance(obj, tuple):
                return [prepare_for_json(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.int64) or isinstance(obj, np.int32):
                return int(obj)
            elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
                return float(obj)
            else:
                return obj
        
        # Filter and prepare data for JSON
        json_data = prepare_for_json(all_diagnostics)
        
        # Save to file
        with open(results_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"{green('Diagnostic results saved to ' + results_file)}")
    except Exception as e:
        logger.error(f"{red('Error saving diagnostic results: ' + str(e))}")
    
    return all_diagnostics

if __name__ == "__main__":
    # Parse command line arguments if provided
    import argparse
    parser = argparse.ArgumentParser(description='Run diagnostics on SINDy Markov Chain Model')
    parser.add_argument('--data-range', type=float, default=1.0, help='Range of data to sample from')
    parser.add_argument('--n-samples', type=int, default=300, help='Number of samples to use')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of trials to run')
    parser.add_argument('--sigma', type=float, default=0.1, help='Noise level')
    parser.add_argument('--threshold', type=float, default=0.05, help='STLSQ threshold')
    
    args = parser.parse_args()
    
    run_diagnostics(
        data_range=args.data_range,
        n_samples=args.n_samples,
        n_trials=args.n_trials,
        sigma=args.sigma,
        threshold=args.threshold
    )