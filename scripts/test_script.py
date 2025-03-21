#!/usr/bin/env python3
"""
Test script for SINDy Markov Chain Analysis.

This script tests the improved SINDy Markov Chain Model with a simple example.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse

# Import necessary modules
from models.sindy_markov_model import SINDyMarkovModel
from models.logger_utils import setup_logging, get_logger, log_to_markdown
from models.simulation_utils import simulate_stlsq, simulate_stlsq_adaptive
from models.library_utils import compute_discriminability, analyze_library_correlations, plot_library_correlations
from models.sindy_markov_analysis import SINDyMarkovAnalysis

from models.logger_utils import suppress_common_warnings
suppress_common_warnings()

def simple_test():
    """Run a simple test with three library terms."""
    print("Running simple test with 3 library terms...")
    
    # Set up logging
    log_file = 'logs/simple_test.log'
    logger = setup_logging(log_file)
    logger.info("Starting simple test")
    
    # Define library functions
    def f1(x): return x           # Term 1: x
    def f2(x): return np.sin(x)   # Term 2: sin(x)
    def f3(x): return np.tanh(x)  # Term 3: tanh(x)
    
    library_functions = [f1, f2, f3]
    
    # Define true model: only the first term is present
    true_coefs = np.array([1.0, 0.0, 0.0])
    
    # Create model
    sigma = 0.1  # Noise level
    threshold = 0.05  # STLSQ threshold
    model = SINDyMarkovModel(library_functions, true_coefs, sigma, threshold, log_file)
    
    # Generate sample points
    x_range = 1.0
    n_samples = 200
    x_data = np.random.uniform(-x_range, x_range, n_samples)
    
    # Compute Gram matrix
    model.compute_gram_matrix(x_data)
    
    # Calculate theoretical success probability
    start_time = time.time()
    theoretical_prob = model.calculate_success_probability()
    theo_time = time.time() - start_time
    logger.info(f"Theoretical success probability: {theoretical_prob:.6f} (calculated in {theo_time:.2f} seconds)")
    
    # Calculate empirical success probability
    start_time = time.time()
    empirical_prob = simulate_stlsq(model, x_data, n_trials=100)
    emp_time = time.time() - start_time
    logger.info(f"Empirical success probability: {empirical_prob:.6f} (calculated in {emp_time:.2f} seconds)")
    
    # Calculate discrepancy
    discrepancy = abs(theoretical_prob - empirical_prob)
    logger.info(f"Discrepancy: {discrepancy:.6f}")
    
    # Create a markdown version of the log
    log_md = os.path.splitext(log_file)[0] + '.md'
    log_to_markdown(log_file, log_md)
    
    print(f"Simple test completed. Results:")
    print(f"  Theoretical success probability: {theoretical_prob:.6f}")
    print(f"  Empirical success probability: {empirical_prob:.6f}")
    print(f"  Discrepancy: {discrepancy:.6f}")
    print(f"See {log_file} and {log_md} for detailed logs")

def advanced_test():
    """Run a comprehensive analysis with the SINDyMarkovAnalysis class."""
    print("Running advanced test with comprehensive analysis...")
    
    # Define library functions
    def f1(x): return x           # Term 1: x
    def f2(x): return x**2        # Term 2: x^2
    def f3(x): return np.sin(x)   # Term 3: sin(x)
    def f4(x): return np.cos(x)   # Term 4: cos(x)
    
    library_functions = [f1, f2, f3, f4]
    
    # Define true model: only the first term is present
    true_coefs = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Create analysis object
    sigma = 0.1  # Noise level
    threshold = 0.05  # STLSQ threshold
    log_file = 'logs/advanced_test.log'
    
    analysis = SINDyMarkovAnalysis(library_functions, true_coefs, sigma, threshold, log_file)
    
    # Prepare data
    x_range = 1.0
    n_samples = 200
    x_data = analysis.prepare_data(x_range, n_samples)
    
    # Analyze success probability
    success_analysis = analysis.analyze_success_probability(x_data, n_trials=50)
    
    # Analyze data range effect
    range_analysis = analysis.analyze_data_range_effect(n_samples=150, x_ranges=[0.1, 0.5, 1.0, 2.0])
    
    # Analyze sample size effect
    sample_analysis = analysis.analyze_sample_size_effect(x_range=1.0, n_samples_list=[50, 100, 200, 300])
    
    # Analyze lambda/sigma effect
    lambda_analysis = analysis.analyze_lambda_sigma_effect(n_trials=30)
    
    # Save results
    analysis.save_results()
    
    print("Advanced test completed. Summary:")
    print(f"  Theoretical success probability: {success_analysis['theoretical']:.6f}")
    print(f"  Empirical success probability: {success_analysis['empirical']:.6f}")
    print(f"  Discrepancy: {success_analysis['discrepancy']:.6f}")
    print("Results saved to results/sindy_markov_analysis_results.json")
    print(f"Figures saved to figures/ directory")
    
    # Create a markdown version of the log
    log_md = os.path.splitext(log_file)[0] + '.md'
    log_to_markdown(log_file, log_md)
    print(f"See {log_file} and {log_md} for detailed logs")

def adaptive_test():
    """Test the adaptive simulation approach."""
    print("Running adaptive trial count test...")
    
    # Set up logging
    log_file = 'logs/adaptive_test.log'
    logger = setup_logging(log_file)
    logger.info("Starting adaptive simulation test")
    
    # Define library functions
    def f1(x): return x           # Term 1: x
    def f2(x): return np.sin(x)   # Term 2: sin(x)
    def f3(x): return np.tanh(x)  # Term 3: tanh(x)
    
    library_functions = [f1, f2, f3]
    
    # Define true model: only the first term is present
    true_coefs = np.array([1.0, 0.0, 0.0])
    
    # Create model
    sigma = 0.1  # Noise level
    threshold = 0.05  # STLSQ threshold
    model = SINDyMarkovModel(library_functions, true_coefs, sigma, threshold, log_file)
    
    # Generate sample points
    x_range = 1.0
    n_samples = 200
    x_data = np.random.uniform(-x_range, x_range, n_samples)
    
    # Compute Gram matrix
    model.compute_gram_matrix(x_data)
    
    # Calculate theoretical success probability
    theoretical_prob = model.calculate_success_probability()
    logger.info(f"Theoretical success probability: {theoretical_prob:.6f}")
    
    # Run adaptive trial simulation
    start_time = time.time()
    empirical_prob, trials_used = simulate_stlsq_adaptive(
        model, 
        x_data, 
        max_trials=500,
        confidence=0.95,
        margin=0.03,
        min_trials=30
    )
    adaptive_time = time.time() - start_time
    
    logger.info(f"Adaptive simulation results:")
    logger.info(f"  Empirical success probability: {empirical_prob:.6f}")
    logger.info(f"  Trials used: {trials_used}")
    logger.info(f"  Simulation time: {adaptive_time:.2f} seconds")
    
    # Calculate discrepancy
    discrepancy = abs(theoretical_prob - empirical_prob)
    logger.info(f"Discrepancy: {discrepancy:.6f}")
    
    # Create a markdown version of the log
    log_md = os.path.splitext(log_file)[0] + '.md'
    log_to_markdown(log_file, log_md)
    
    print(f"Adaptive test completed. Results:")
    print(f"  Theoretical success probability: {theoretical_prob:.6f}")
    print(f"  Empirical success probability: {empirical_prob:.6f}")
    print(f"  Trials used: {trials_used}")
    print(f"  Discrepancy: {discrepancy:.6f}")
    print(f"See {log_file} and {log_md} for detailed logs")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test SINDy Markov Chain Model')
    parser.add_argument('--test', choices=['simple', 'advanced', 'adaptive', 'all'], 
                       default='all', help='Test to run')
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Run selected test
    if args.test == 'simple' or args.test == 'all':
        simple_test()
        print()
        
    if args.test == 'advanced' or args.test == 'all':
        advanced_test()
        print()
        
    if args.test == 'adaptive' or args.test == 'all':
        adaptive_test()
        print()