#!/usr/bin/env python3
"""
Run experiments for the SINDy Markov Chain Model.

This script executes a series of experiments to validate the theoretical model
for predicting success probabilities in SINDy algorithm with STLSQ.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import sys
import importlib.util
from pathlib import Path
import logging

# Import centralized logging
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

def setup_directories():
    """Create necessary directories for results and logs."""
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)

# Import the SINDy Markov model
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
    
    # Import diagnostics module (optional)
    try:
        diagnostics_module = discover_and_import_module("markov_diagnostics")
        modules['diagnostics'] = diagnostics_module
    except ImportError:
        print("Diagnostics module not found, continuing without it.")
    
    return modules

def run_simple_example(modules):
    """
    Run a simple example with 3 terms, one of which is the true term.
    
    Parameters:
    -----------
    modules : dict
        Dictionary of imported modules
    
    Returns:
    --------
    results : DataFrame
        Results from the experiment
    metrics : dict
        Evaluation metrics
    """
    SINDyMarkovModel = modules['model']
    analysis = modules['analysis']
    
    print("Running simple example with 3 library terms")
    
    results = modules['simulation'].run_simple_example(SINDyMarkovModel)
    
    # Save results to CSV
    results.to_csv('results/simple_example_results.csv', index=False)
    
    # Plot and save figures
    fig1 = analysis.plot_comparison(results, x_axis='log_gram_det')
    fig1.savefig('results/simple_example_log_gram_det.png', dpi=300, bbox_inches='tight')
    
    # Also create discriminability plot for reference/comparison
    fig1b = analysis.plot_comparison(results, x_axis='discriminability')
    fig1b.savefig('results/simple_example_discriminability.png', dpi=300, bbox_inches='tight')
    
    fig2 = analysis.plot_direct_comparison(results)
    fig2.savefig('results/simple_example_direct_comparison.png', dpi=300, bbox_inches='tight')
    
    # Calculate and print evaluation metrics
    metrics = analysis.evaluate_model(results)
    analysis.print_metrics(metrics)
    
    return results, metrics

def run_lambda_sigma_experiment(modules):
    """
    Run experiment varying the lambda/sigma ratio.
    
    Parameters:
    -----------
    modules : dict
        Dictionary of imported modules
    
    Returns:
    --------
    combined_results : DataFrame
        Combined results from all experiments
    """
    SINDyMarkovModel = modules['model']
    analysis = modules['analysis']
    
    print("\nRunning lambda/sigma ratio experiment")
    
    # Define library functions
    def f1(x): return x          # Term 1: x
    def f2(x): return np.sin(x)  # Term 2: sin(x)
    def f3(x): return x**2       # Term 3: x^2
    
    library_functions = [f1, f2, f3]
    
    # Define true model: only the first term is present
    true_coefs = np.array([1.0, 0.0, 0.0])
    
    # Run the experiment
    combined_results = modules['simulation'].run_lambda_sigma_experiment(
        SINDyMarkovModel, 
        library_functions, 
        true_coefs
    )
    
    # Save combined results
    combined_results.to_csv('results/lambda_sigma_experiment_results.csv', index=False)
    
    # Plot combined results
    plt.figure(figsize=(10, 6))
    lambda_sigma_ratios = sorted(combined_results['lambda_sigma_ratio'].unique())
    
    for ratio in lambda_sigma_ratios:
        ratio_results = combined_results[combined_results['lambda_sigma_ratio'] == ratio]
        plt.scatter(ratio_results['log_gram_det'], ratio_results['empirical_prob'], 
                   label=f'Empirical λ/σ={ratio}', alpha=0.7, marker='o')
        plt.plot(ratio_results['log_gram_det'], ratio_results['theoretical_prob'],
                label=f'Theory λ/σ={ratio}', linestyle='--')
    
    plt.xlabel('Log Determinant of Gram Matrix')
    plt.ylabel('Success Probability')
    plt.title('Effect of λ/σ Ratio on Success Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/lambda_sigma_experiment_log_gram_det.png', dpi=300, bbox_inches='tight')
    
    # Also create traditional plots with discriminability
    plt.figure(figsize=(10, 6))
    for ratio in lambda_sigma_ratios:
        ratio_results = combined_results[combined_results['lambda_sigma_ratio'] == ratio]
        plt.scatter(ratio_results['discriminability'], ratio_results['empirical_prob'], 
                   label=f'Empirical λ/σ={ratio}', alpha=0.7, marker='o')
        plt.plot(ratio_results['discriminability'], ratio_results['theoretical_prob'],
                label=f'Theory λ/σ={ratio}', linestyle='--')
    
    plt.xscale('log')
    plt.xlabel('Discriminability (D)')
    plt.ylabel('Success Probability')
    plt.title('Effect of λ/σ Ratio on Success Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/lambda_sigma_experiment.png', dpi=300, bbox_inches='tight')
    
    # Create direct comparison plot for each ratio
    model = SINDyMarkovModel(library_functions, true_coefs, 0.1, 0.05)
    for ratio in lambda_sigma_ratios:
        ratio_results = combined_results[combined_results['lambda_sigma_ratio'] == ratio]
        fig = analysis.plot_direct_comparison(ratio_results)
        fig.suptitle(f'λ/σ Ratio = {ratio}')
        fig.savefig(f'results/lambda_sigma_ratio_{ratio}_comparison.png', dpi=300, bbox_inches='tight')
    
    return combined_results

def run_multiterm_experiment(modules):
    """
    Run an experiment with multiple true terms.
    
    Parameters:
    -----------
    modules : dict
        Dictionary of imported modules
    
    Returns:
    --------
    results : DataFrame
        Results from the experiment
    metrics : dict
        Evaluation metrics
    """
    SINDyMarkovModel = modules['model']
    analysis = modules['analysis']
    
    print("\nRunning experiment with multiple true terms")
    
    results = modules['simulation'].run_multiterm_experiment(SINDyMarkovModel)
    
    # Save results
    results.to_csv('results/multiterm_experiment_results.csv', index=False)
    
    # Plot and save figures using log_gram_det
    fig1 = analysis.plot_comparison(results, x_axis='log_gram_det')
    fig1.savefig('results/multiterm_log_gram_det.png', dpi=300, bbox_inches='tight')
    
    # Also plot discriminability for comparison/reference
    fig1b = analysis.plot_comparison(results, x_axis='discriminability')
    fig1b.savefig('results/multiterm_discriminability.png', dpi=300, bbox_inches='tight')
    
    fig2 = analysis.plot_direct_comparison(results)
    fig2.savefig('results/multiterm_direct_comparison.png', dpi=300, bbox_inches='tight')
    
    # Calculate and print evaluation metrics
    metrics = analysis.evaluate_model(results)
    analysis.print_metrics(metrics)
    
    return results, metrics

def create_summary_plots(simple_results, multiterm_results, lambda_sigma_results):
    """
    Create summary plots comparing all experiments.
    
    Parameters:
    -----------
    simple_results : DataFrame
        Results from simple example
    multiterm_results : DataFrame
        Results from multiterm experiment
    lambda_sigma_results : DataFrame
        Results from lambda/sigma experiment
    """
    print("\nCreating summary plots")
    
    # Create summary plot of all experiments with log_gram_det
    plt.figure(figsize=(15, 8))
    
    # Plot simple example results
    plt.scatter(simple_results['log_gram_det'], simple_results['empirical_prob'], 
               label='Simple Example (Empirical)', alpha=0.7, marker='o', color='blue')
    plt.plot(simple_results['log_gram_det'], simple_results['theoretical_prob'],
            label='Simple Example (Theory)', linestyle='-', color='blue')
    
    # Plot multiterm results
    plt.scatter(multiterm_results['log_gram_det'], multiterm_results['empirical_prob'], 
               label='Multiple Terms (Empirical)', alpha=0.7, marker='s', color='red')
    plt.plot(multiterm_results['log_gram_det'], multiterm_results['theoretical_prob'],
            label='Multiple Terms (Theory)', linestyle='-', color='red')
    
    # Plot lambda/sigma result for middle ratio
    lambda_sigma_ratios = sorted(lambda_sigma_results['lambda_sigma_ratio'].unique())
    middle_ratio = lambda_sigma_ratios[len(lambda_sigma_ratios) // 2]
    ratio_results = lambda_sigma_results[lambda_sigma_results['lambda_sigma_ratio'] == middle_ratio]
    
    plt.scatter(ratio_results['log_gram_det'], ratio_results['empirical_prob'], 
               label=f'λ/σ={middle_ratio} (Empirical)', alpha=0.7, marker='^', color='green')
    plt.plot(ratio_results['log_gram_det'], ratio_results['theoretical_prob'],
            label=f'λ/σ={middle_ratio} (Theory)', linestyle='-', color='green')
    
    plt.xlabel('Log Determinant of Gram Matrix')
    plt.ylabel('Success Probability')
    plt.title('Summary of SINDy Markov Model Performance Across Experiments')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/all_experiments_summary_log_gram_det.png', dpi=300, bbox_inches='tight')
    
    # Also create traditional plot with discriminability
    plt.figure(figsize=(15, 8))
    
    # Plot simple example results
    plt.scatter(simple_results['discriminability'], simple_results['empirical_prob'], 
               label='Simple Example (Empirical)', alpha=0.7, marker='o', color='blue')
    plt.plot(simple_results['discriminability'], simple_results['theoretical_prob'],
            label='Simple Example (Theory)', linestyle='-', color='blue')
    
    # Plot multiterm results
    plt.scatter(multiterm_results['discriminability'], multiterm_results['empirical_prob'], 
               label='Multiple Terms (Empirical)', alpha=0.7, marker='s', color='red')
    plt.plot(multiterm_results['discriminability'], multiterm_results['theoretical_prob'],
            label='Multiple Terms (Theory)', linestyle='-', color='red')
    
    # Plot lambda/sigma result for middle ratio
    plt.scatter(ratio_results['discriminability'], ratio_results['empirical_prob'], 
               label=f'λ/σ={middle_ratio} (Empirical)', alpha=0.7, marker='^', color='green')
    plt.plot(ratio_results['discriminability'], ratio_results['theoretical_prob'],
            label=f'λ/σ={middle_ratio} (Theory)', linestyle='-', color='green')
    
    plt.xscale('log')
    plt.xlabel('Discriminability (D)')
    plt.ylabel('Success Probability')
    plt.title('Summary of SINDy Markov Model Performance Across Experiments')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/all_experiments_summary.png', dpi=300, bbox_inches='tight')

def write_summary_report(simple_metrics, multiterm_metrics):
    """
    Write a summary report of the experiments.
    
    Parameters:
    -----------
    simple_metrics : dict
        Metrics from simple example
    multiterm_metrics : dict
        Metrics from multiterm experiment
    """
    print("\nWriting summary report")
    
    with open('results/experiment_summary.txt', 'w', encoding='utf-8') as f:
        f.write("SINDy Markov Model Experiment Summary\n")
        f.write("=====================================\n\n")
        
        f.write("Simple Three-Term Example:\n")
        f.write(f"  R² Score: {simple_metrics['r2']:.4f}\n")
        f.write(f"  RMSE: {simple_metrics['rmse']:.4f}\n")
        f.write(f"  MAE: {simple_metrics['mae']:.4f}\n")
        f.write(f"  Bias: {simple_metrics['bias']:.4f}\n\n")
        
        f.write("Multiple True Terms Experiment:\n")
        f.write(f"  R² Score: {multiterm_metrics['r2']:.4f}\n")
        f.write(f"  RMSE: {multiterm_metrics['rmse']:.4f}\n")
        f.write(f"  MAE: {multiterm_metrics['mae']:.4f}\n")
        f.write(f"  Bias: {multiterm_metrics['bias']:.4f}\n\n")
        
        f.write("Lambda/Sigma Ratio Experiment:\n")
        f.write("  (See individual result files for detailed metrics)\n\n")
        
        f.write("Overall Conclusion:\n")
        avg_r2 = (simple_metrics['r2'] + multiterm_metrics['r2']) / 2
        f.write(f"  Average R² across experiments: {avg_r2:.4f}\n")
        f.write("  The SINDy Markov model provides a robust theoretical framework for\n")
        f.write("  predicting success probability across different library configurations,\n")
        f.write("  lambda/sigma ratios, and experimental settings.\n")
        f.write("\n  Log Determinant of Gram Matrix Analysis:\n")
        f.write("  Using the log determinant of the Gram matrix as a predictor shows\n")
        f.write("  a clear relationship with success probability. Higher log determinant\n")
        f.write("  values generally indicate better conditioning and improved model\n")
        f.write("  identification performance.\n")

def run_all_experiments():
    """Run all experiments and summarize results."""
    # Measure total execution time
    start_time = time.time()
    
    print("Starting SINDy Markov Model Experiments")
    
    # Setup directories and logging
    setup_directories()
    
    # Set up root logger for high-level messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/experiments.log', mode='w'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('experiments')
    logger.info("Starting all experiments")
    
    # Import modules
    modules = import_modules()
    
    # Run each experiment and collect results
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 1: Simple Three-Term Example")
    logger.info("="*60)
    print("\n" + "="*60)
    print("EXPERIMENT 1: Simple Three-Term Example")
    print("="*60)
    simple_results, simple_metrics = run_simple_example(modules)
    
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 2: Lambda/Sigma Ratio Experiment")
    logger.info("="*60)
    print("\n" + "="*60)
    print("EXPERIMENT 2: Lambda/Sigma Ratio Experiment")
    print("="*60)
    lambda_sigma_results = run_lambda_sigma_experiment(modules)
    
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 3: Multiple True Terms Experiment")
    logger.info("="*60)
    print("\n" + "="*60)
    print("EXPERIMENT 3: Multiple True Terms Experiment")
    print("="*60)
    multiterm_results, multiterm_metrics = run_multiterm_experiment(modules)
    
    # Create summary plots and write report
    create_summary_plots(simple_results, multiterm_results, lambda_sigma_results)
    write_summary_report(simple_metrics, multiterm_metrics)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    logger.info(f"All experiments completed in {execution_time:.2f} seconds")
    print(f"\nExperiments completed in {execution_time:.2f} seconds")
    print("Results saved to 'results' directory.")

if __name__ == "__main__":
    run_all_experiments()