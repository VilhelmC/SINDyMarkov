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
from pathlib import Path

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

# Import the SINDy Markov model
sindy_markov_module = discover_and_import_module("sindy_markov_model")
SINDyMarkovModel = sindy_markov_module.SINDyMarkovModel

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
    print(f"Running diagnostics with data_range={data_range}, n_samples={n_samples}, n_trials={n_trials}")
    
    # Create output directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Define simple library functions
    def f1(x): return x          # Term 1: x
    def f2(x): return np.sin(x)  # Term 2: sin(x)
    def f3(x): return np.tanh(x) # Term 3: tanh(x)
    
    library_functions = [f1, f2, f3]
    
    # Define true model: only the first term is present
    true_coefs = np.array([1.0, 0.0, 0.0])
    
    # Create model instance
    model = SINDyMarkovModel(library_functions, true_coefs, sigma, threshold, log_file='logs/diagnostics.log')
    
    # Generate sample points
    x_data = np.random.uniform(-data_range, data_range, n_samples)
    
    # Compute Gram matrix
    model.compute_gram_matrix(x_data)
    
    # Step 1: Calculate theoretical success probability
    theoretical_prob = model.calculate_success_probability()
    
    # Step 2: Calculate empirical success probability
    empirical_prob = model.simulate_stlsq(x_data, n_trials)
    
    print(f"\nSuccess Probability Summary:")
    print(f"  Theoretical: {theoretical_prob:.4f}")
    print(f"  Empirical: {empirical_prob:.4f}")
    print(f"  Difference: {abs(theoretical_prob - empirical_prob):.4f}")
    
    # Step 3: Run transition probability comparison
    print("\nRunning transition probability comparison...")
    comparison_data = model.compare_theory_to_simulation_transitions(x_data, n_trials)
    
    # Step 4: Analyze coefficient distributions
    print("\nAnalyzing coefficient distributions...")
    distribution_data = model.analyze_coefficient_distributions(x_data, min(50, n_trials))
    
    # Step 5: Test independence assumption
    print("\nTesting independence assumption...")
    independence_metrics = model.test_independence_assumption(x_data, min(50, n_trials))
    
    # Step 6: Debug specific transitions with high discrepancy
    print("\nDebugging most problematic transitions...")
    # Identify the true state and initial state
    true_indices = set(model.true_term_indices.tolist())
    all_indices = set(range(model.n_terms))
    
    # Debug direct transition from initial to true state
    debug_results = model.debug_calculate_transition_probability(all_indices, true_indices, samples=100000)
    empirical_results = model.debug_calculate_transition_probability_with_data(all_indices, true_indices, x_data, n_trials=min(100, n_trials))
    
    # Step 7: Verify distribution recalculation
    print("\nVerifying distribution recalculation...")
    verification = model.verify_distribution_recalculation(all_indices, x_data)
    
    # Print summary of findings
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print(f"Data Range: {data_range}, Samples: {n_samples}, Trials: {n_trials}")
    print(f"Sigma: {sigma}, Threshold: {threshold}")
    print(f"Log Determinant of Gram Matrix: {model.log_gram_det:.4f}")
    print(f"Success Probability: Theoretical={theoretical_prob:.4f}, Empirical={empirical_prob:.4f}")
    
    # Look for significant issues
    print("\nSignificant Issues Identified:")
    
    # Check success probability discrepancy
    if abs(theoretical_prob - empirical_prob) > 0.1:
        print(f"- Large discrepancy in success probability ({abs(theoretical_prob - empirical_prob):.4f})")
    
    # Check for significant transition probability differences
    significant_diffs = [(k, v) for k, v in comparison_data.items() if v['significant']]
    if significant_diffs:
        print(f"- {len(significant_diffs)} transitions with significant probability differences")
        print("  Top 3 most significant:")
        significant_diffs.sort(key=lambda x: abs(x[1]['difference']), reverse=True)
        for i, ((from_str, to_str), data) in enumerate(significant_diffs[:3]):
            print(f"    {from_str} -> {to_str}: Diff={data['difference']:+.4f}")
    
    # Check for independence assumption violations
    independence_issues = False
    for trans_key, metrics in independence_metrics.items():
        corr_matrix = metrics['correlation_matrix']
        if np.any(np.abs(corr_matrix) > 0.3):
            independence_issues = True
            break
    
    if independence_issues:
        print("- Violations of independence assumption in coefficient thresholding")
    
    # Print possible solutions
    print("\nPossible Solutions to Consider:")
    if abs(theoretical_prob - empirical_prob) > 0.1:
        print("1. Revise transition probability calculation to account for dependencies")
        print("2. Consider conditional probabilities instead of independence assumption")
        print("3. Develop an empirical correction factor for specific transitions")
    
    print("\nSee detailed logs in logs/diagnostics.log for more information.")

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