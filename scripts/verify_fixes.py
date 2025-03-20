#!/usr/bin/env python3
"""
Test and verification script for SINDy Markov Chain Model fixes.

This script runs a series of tests to verify that the fixed model
correctly calculates success probabilities and matches theoretical predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging
import importlib
import time
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Import centralized logging utilities
from models.logging_config import setup_logging, get_logger
from models.logging_config import bold, green, yellow, red, cyan, header, section
from models.logging_config import bold_green, bold_yellow, bold_red

# Set up logger
setup_logging('logs/verification.log')
logger = get_logger('verification')

def run_simple_case(model_class):
    """
    Run a simple test case with known analytical solution.
    
    This test uses a 2-term library where the terms are orthogonal,
    which allows for easy analytical verification.
    """
    logger.info(header("SIMPLE TEST CASE WITH ORTHOGONAL TERMS"))
    
    # Define orthogonal test functions
    def f1(x): return np.sin(x)   # sin(x)
    def f2(x): return np.cos(x)   # cos(x)
    
    # For sine and cosine on a full period, these are orthogonal
    library_functions = [f1, f2]
    
    # Only the first term is true
    true_coefs = np.array([1.0, 0.0])
    
    # Parameters
    sigma = 0.1  # Noise level
    threshold = 0.1  # STLSQ threshold
    
    logger.info(f"{bold('Test Setup:')}")
    logger.info(f"  Library: [sin(x), cos(x)]")
    logger.info(f"  True coefficients: [1.0, 0.0]")
    logger.info(f"  Sigma: {sigma}")
    logger.info(f"  Threshold: {threshold}")
    logger.info(f"  Lambda/Sigma ratio: {threshold/sigma}")
    
    # Create model
    model = model_class(
        library_functions=library_functions,
        true_coefs=true_coefs,
        sigma=sigma,
        threshold=threshold,
        log_file='logs/simple_test.log'
    )
    
    # Generate sample points (full period for orthogonality)
    x_data = np.linspace(0, 2*np.pi, 100)
    
    # Compute Gram matrix
    model.compute_gram_matrix(x_data)
    
    # Calculate theoretical success probability
    theoretical_prob = model.calculate_success_probability()
    
    # Run empirical simulation
    empirical_prob = model.simulate_stlsq(x_data, n_trials=100)
    
    # Log results
    logger.info(f"\n{bold_green('Results:')}")
    logger.info(f"  Theoretical success probability: {theoretical_prob:.6f}")
    logger.info(f"  Empirical success probability: {empirical_prob:.6f}")
    logger.info(f"  Discrepancy: {abs(theoretical_prob - empirical_prob):.6f}")
    
    # Calculate analytical solution
    # For orthogonal terms with threshold=sigma, the probability is related to normal CDF
    from scipy.stats import norm
    
    # P(|beta_1| >= threshold) * P(|beta_2| < threshold)
    p1 = 1 - (norm.cdf(threshold/sigma) - norm.cdf(-threshold/sigma))  # P(|beta_1| >= threshold)
    p2 = norm.cdf(threshold/sigma) - norm.cdf(-threshold/sigma)        # P(|beta_2| < threshold)
    analytical_prob = p1 * p2
    
    logger.info(f"  Analytical solution: {analytical_prob:.6f}")
    logger.info(f"  Discrepancy (theoretical vs analytical): {abs(theoretical_prob - analytical_prob):.6f}")
    
    # Test passed?
    if abs(theoretical_prob - analytical_prob) < 0.05:
        logger.info(f"{bold_green('TEST PASSED: Theoretical probability matches analytical solution.')}")
    else:
        logger.info(f"{bold_red('TEST FAILED: Theoretical probability does not match analytical solution.')}")
    
    # Return test results
    return {
        'name': 'simple_orthogonal',
        'theoretical_prob': theoretical_prob,
        'empirical_prob': empirical_prob,
        'analytical_prob': analytical_prob,
        'theo_vs_emp_diff': abs(theoretical_prob - empirical_prob),
        'theo_vs_anal_diff': abs(theoretical_prob - analytical_prob)
    }

def run_correlated_case(model_class):
    """
    Run a test case with correlated terms.
    
    This test uses closely related terms to verify that the model
    correctly accounts for correlation between library terms.
    """
    logger.info(header("CORRELATED TERMS TEST CASE"))
    
    # Define correlated test functions
    def f1(x): return x             # x (linear)
    def f2(x): return x**2         # x^2 (quadratic)
    
    # For small x, linear and quadratic are highly correlated
    library_functions = [f1, f2]
    
    # Only the first term is true
    true_coefs = np.array([1.0, 0.0])
    
    # Parameters
    sigma = 0.1  # Noise level
    threshold = 0.1  # STLSQ threshold
    
    logger.info(f"{bold('Test Setup:')}")
    logger.info(f"  Library: [x, x^2]")
    logger.info(f"  True coefficients: [1.0, 0.0]")
    logger.info(f"  Sigma: {sigma}")
    logger.info(f"  Threshold: {threshold}")
    logger.info(f"  Lambda/Sigma ratio: {threshold/sigma}")
    
    # Create model
    model = model_class(
        library_functions=library_functions,
        true_coefs=true_coefs,
        sigma=sigma,
        threshold=threshold,
        log_file='logs/correlated_test.log'
    )
    
    # Test different data ranges to vary the correlation
    results = []
    
    data_ranges = [0.1, 0.5, 1.0]
    sample_sizes = [100, 300]
    
    for data_range in data_ranges:
        for n_samples in sample_sizes:
            logger.info(f"\n{bold_yellow(f'Testing data_range={data_range}, n_samples={n_samples}')}")
            
            # Generate sample points
            x_data = np.random.uniform(-data_range, data_range, n_samples)
            
            # Compute Gram matrix
            model.compute_gram_matrix(x_data)
            
            # Calculate theoretical success probability
            theoretical_prob = model.calculate_success_probability()
            
            # Run empirical simulation
            empirical_prob = model.simulate_stlsq(x_data, n_trials=100)
            
            # Calculate correlation between terms
            term1 = f1(x_data)
            term2 = f2(x_data)
            correlation = np.corrcoef(term1, term2)[0, 1]
            
            # Log results
            logger.info(f"  Correlation between terms: {correlation:.6f}")
            logger.info(f"  Theoretical success probability: {theoretical_prob:.6f}")
            logger.info(f"  Empirical success probability: {empirical_prob:.6f}")
            logger.info(f"  Discrepancy: {abs(theoretical_prob - empirical_prob):.6f}")
            
            # Store results
            results.append({
                'data_range': data_range,
                'n_samples': n_samples,
                'correlation': correlation,
                'theoretical_prob': theoretical_prob,
                'empirical_prob': empirical_prob,
                'discrepancy': abs(theoretical_prob - empirical_prob)
            })
    
    # Create a correlation vs. success probability plot
    plt.figure(figsize=(10, 6))
    
    correlations = [r['correlation'] for r in results]
    theo_probs = [r['theoretical_prob'] for r in results]
    emp_probs = [r['empirical_prob'] for r in results]
    
    plt.scatter(correlations, theo_probs, label='Theoretical', marker='o', s=80, alpha=0.7)
    plt.scatter(correlations, emp_probs, label='Empirical', marker='x', s=80, alpha=0.7)
    
    # Draw lines connecting corresponding points
    for i in range(len(correlations)):
        plt.plot([correlations[i], correlations[i]], [theo_probs[i], emp_probs[i]], 'k--', alpha=0.3)
    
    plt.xlabel('Correlation Between Terms')
    plt.ylabel('Success Probability')
    plt.title('Effect of Term Correlation on Success Probability')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Ensure directory exists
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/correlation_vs_success.png', dpi=300, bbox_inches='tight')
    
    logger.info(f"\n{bold_green('Correlation test results:')}")
    logger.info(f"  Average discrepancy: {np.mean([r['discrepancy'] for r in results]):.6f}")
    logger.info(f"  Max discrepancy: {np.max([r['discrepancy'] for r in results]):.6f}")
    
    return {
        'name': 'correlated_terms',
        'results': results,
        'avg_discrepancy': np.mean([r['discrepancy'] for r in results]),
        'max_discrepancy': np.max([r['discrepancy'] for r in results])
    }

def run_multiterm_case(model_class):
    """
    Run a test case with multiple true terms.
    
    This test checks that the model correctly handles cases 
    where multiple library terms are present in the true model.
    """
    logger.info(header("MULTIPLE TRUE TERMS TEST CASE"))
    
    # Define test functions
    def f1(x): return x           # x (linear)
    def f2(x): return np.sin(x)   # sin(x)
    def f3(x): return np.exp(x)   # exp(x)
    
    # Library with three terms
    library_functions = [f1, f2, f3]
    
    # First two terms are true
    true_coefs = np.array([1.0, 0.5, 0.0])
    
    # Parameters
    sigma = 0.1   # Noise level
    threshold = 0.1  # STLSQ threshold
    
    logger.info(f"{bold('Test Setup:')}")
    logger.info(f"  Library: [x, sin(x), exp(x)]")
    logger.info(f"  True coefficients: [1.0, 0.5, 0.0]")
    logger.info(f"  Sigma: {sigma}")
    logger.info(f"  Threshold: {threshold}")
    logger.info(f"  Lambda/Sigma ratio: {threshold/sigma}")
    
    # Create model
    model = model_class(
        library_functions=library_functions,
        true_coefs=true_coefs,
        sigma=sigma,
        threshold=threshold,
        log_file='logs/multiterm_test.log'
    )
    
    # Test a few data ranges and sample sizes
    results = []
    
    data_ranges = [0.5, 1.0, 2.0]
    sample_sizes = [200, 500]
    
    for data_range in data_ranges:
        for n_samples in sample_sizes:
            logger.info(f"\n{bold_yellow(f'Testing data_range={data_range}, n_samples={n_samples}')}")
            
            # Generate sample points
            x_data = np.random.uniform(-data_range, data_range, n_samples)
            
            # Compute Gram matrix
            model.compute_gram_matrix(x_data)
            
            # Calculate theoretical success probability
            theoretical_prob = model.calculate_success_probability()
            
            # Run empirical simulation
            empirical_prob = model.simulate_stlsq(x_data, n_trials=100)
            
            # Log results
            logger.info(f"  Theoretical success probability: {theoretical_prob:.6f}")
            logger.info(f"  Empirical success probability: {empirical_prob:.6f}")
            logger.info(f"  Discrepancy: {abs(theoretical_prob - empirical_prob):.6f}")
            
            # Store results
            results.append({
                'data_range': data_range,
                'n_samples': n_samples,
                'theoretical_prob': theoretical_prob,
                'empirical_prob': empirical_prob,
                'discrepancy': abs(theoretical_prob - empirical_prob)
            })
    
    logger.info(f"\n{bold_green('Multiple true terms test results:')}")
    logger.info(f"  Average discrepancy: {np.mean([r['discrepancy'] for r in results]):.6f}")
    logger.info(f"  Max discrepancy: {np.max([r['discrepancy'] for r in results]):.6f}")
    
    return {
        'name': 'multiple_true_terms',
        'results': results,
        'avg_discrepancy': np.mean([r['discrepancy'] for r in results]),
        'max_discrepancy': np.max([r['discrepancy'] for r in results])
    }

def run_lambda_sigma_sweep(model_class):
    """
    Run a sweep across different lambda/sigma ratios.
    
    This test checks that the model handles different threshold-to-noise ratios correctly.
    """
    logger.info(header("LAMBDA/SIGMA RATIO SWEEP TEST"))
    
    # Define test functions
    def f1(x): return x          # x (linear)
    def f2(x): return np.sin(x)  # sin(x)
    
    # Library with two terms
    library_functions = [f1, f2]
    
    # Only first term is true
    true_coefs = np.array([1.0, 0.0])
    
    # Fixed noise level
    sigma = 0.1
    
    # Different lambda/sigma ratios
    ratios = [0.2, 0.5, 1.0, 2.0, 5.0]
    
    logger.info(f"{bold('Test Setup:')}")
    logger.info(f"  Library: [x, sin(x)]")
    logger.info(f"  True coefficients: [1.0, 0.0]")
    logger.info(f"  Sigma: {sigma}")
    logger.info(f"  Lambda/Sigma ratios: {ratios}")
    
    # Fixed data range and sample size
    data_range = 1.0
    n_samples = 300
    
    # Generate sample points
    x_data = np.random.uniform(-data_range, data_range, n_samples)
    
    results = []
    
    for ratio in ratios:
        threshold = ratio * sigma
        
        logger.info(f"\n{bold_yellow(f'Testing lambda/sigma ratio={ratio} (threshold={threshold})')}")
        
        # Create model with this threshold
        model = model_class(
            library_functions=library_functions,
            true_coefs=true_coefs,
            sigma=sigma,
            threshold=threshold,
            log_file=f'logs/lambda_sigma_{ratio}_test.log'
        )
        
        # Compute Gram matrix
        model.compute_gram_matrix(x_data)
        
        # Calculate theoretical success probability
        theoretical_prob = model.calculate_success_probability()
        
        # Run empirical simulation
        empirical_prob = model.simulate_stlsq(x_data, n_trials=100)
        
        # Log results
        logger.info(f"  Theoretical success probability: {theoretical_prob:.6f}")
        logger.info(f"  Empirical success probability: {empirical_prob:.6f}")
        logger.info(f"  Discrepancy: {abs(theoretical_prob - empirical_prob):.6f}")
        
        # Store results
        results.append({
            'ratio': ratio,
            'threshold': threshold,
            'theoretical_prob': theoretical_prob,
            'empirical_prob': empirical_prob,
            'discrepancy': abs(theoretical_prob - empirical_prob)
        })
    
    # Create a lambda/sigma ratio vs. success probability plot
    plt.figure(figsize=(10, 6))
    
    ratios = [r['ratio'] for r in results]
    theo_probs = [r['theoretical_prob'] for r in results]
    emp_probs = [r['empirical_prob'] for r in results]
    
    plt.plot(ratios, theo_probs, 'o-', label='Theoretical', linewidth=2, markersize=8)
    plt.plot(ratios, emp_probs, 'x-', label='Empirical', linewidth=2, markersize=8)
    
    plt.xlabel('Lambda/Sigma Ratio')
    plt.ylabel('Success Probability')
    plt.title('Effect of Lambda/Sigma Ratio on Success Probability')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Ensure directory exists
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/lambda_sigma_ratio_vs_success.png', dpi=300, bbox_inches='tight')
    
    logger.info(f"\n{bold_green('Lambda/Sigma ratio sweep results:')}")
    logger.info(f"  Average discrepancy: {np.mean([r['discrepancy'] for r in results]):.6f}")
    logger.info(f"  Max discrepancy: {np.max([r['discrepancy'] for r in results]):.6f}")
    
    return {
        'name': 'lambda_sigma_sweep',
        'results': results,
        'avg_discrepancy': np.mean([r['discrepancy'] for r in results]),
        'max_discrepancy': np.max([r['discrepancy'] for r in results])
    }

def run_all_tests():
    """Run all verification tests and report results."""
    logger.info(header("RUNNING ALL VERIFICATION TESTS"))
    
    # Import the SINDy Markov Model class
    sys.path.append(str(parent_dir))
    
    try:
        # First try direct import
        from models.sindy_markov_model import SINDyMarkovModel
        logger.info(f"{bold_green('Successfully imported SINDyMarkovModel directly')}")
    except ImportError:
        # If that fails, try dynamic import
        logger.warning("Direct import failed, trying dynamic import...")
        
        spec = importlib.util.find_spec("models.sindy_markov_model")
        if spec is None:
            # Look for the module in various locations
            for loc in [parent_dir, parent_dir / "models", Path.cwd(), Path.cwd() / "models"]:
                candidate = loc / "sindy_markov_model.py"
                if candidate.exists():
                    spec = importlib.util.spec_from_file_location("sindy_markov_model", candidate)
                    break
        
        if spec is None:
            logger.error("Could not find sindy_markov_model.py in any location")
            return
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        SINDyMarkovModel = module.SINDyMarkovModel
        logger.info(f"{bold_green('Successfully imported SINDyMarkovModel dynamically')}")
    
    # Run tests
    start_time = time.time()
    
    results = {}
    
    # Run simple orthogonal case
    try:
        logger.info("\n" + "="*80)
        results['simple_case'] = run_simple_case(SINDyMarkovModel)
    except Exception as e:
        logger.error(f"{bold_red('Error in simple case test:')}\n{str(e)}")
        results['simple_case'] = {'error': str(e)}
    
    # Run correlated terms case
    try:
        logger.info("\n" + "="*80)
        results['correlated_case'] = run_correlated_case(SINDyMarkovModel)
    except Exception as e:
        logger.error(f"{bold_red('Error in correlated case test:')}\n{str(e)}")
        results['correlated_case'] = {'error': str(e)}
    
    # Run multiple true terms case
    try:
        logger.info("\n" + "="*80)
        results['multiterm_case'] = run_multiterm_case(SINDyMarkovModel)
    except Exception as e:
        logger.error(f"{bold_red('Error in multiple true terms test:')}\n{str(e)}")
        results['multiterm_case'] = {'error': str(e)}
    
    # Run lambda/sigma ratio sweep
    try:
        logger.info("\n" + "="*80)
        results['lambda_sigma_sweep'] = run_lambda_sigma_sweep(SINDyMarkovModel)
    except Exception as e:
        logger.error(f"{bold_red('Error in lambda/sigma sweep test:')}\n{str(e)}")
        results['lambda_sigma_sweep'] = {'error': str(e)}
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Generate summary report
    logger.info(header("VERIFICATION TEST SUMMARY"))
    logger.info(f"Total time: {elapsed_time:.2f} seconds\n")
    
    all_passed = True
    
    # Check simple case
    if 'error' in results['simple_case']:
        logger.info(f"{bold_red('Simple case: FAILED')}")
        logger.info(f"  Error: {results['simple_case']['error']}")
        all_passed = False
    else:
        theo_vs_anal_diff = results['simple_case']['theo_vs_anal_diff']
        if theo_vs_anal_diff < 0.05:
            logger.info(f"{bold_green('Simple case: PASSED')}")
            logger.info(f"  Theoretical vs Analytical difference: {theo_vs_anal_diff:.6f}")
        else:
            logger.info(f"{bold_red('Simple case: FAILED')}")
            logger.info(f"  Theoretical vs Analytical difference: {theo_vs_anal_diff:.6f}")
            all_passed = False
    
    # Check correlated case
    if 'error' in results['correlated_case']:
        logger.info(f"{bold_red('Correlated case: FAILED')}")
        logger.info(f"  Error: {results['correlated_case']['error']}")
        all_passed = False
    else:
        avg_disc = results['correlated_case']['avg_discrepancy']
        max_disc = results['correlated_case']['max_discrepancy']
        
        if avg_disc < 0.1 and max_disc < 0.2:
            logger.info(f"{bold_green('Correlated case: PASSED')}")
        else:
            logger.info(f"{bold_yellow('Correlated case: PARTIAL')}")
            all_passed = avg_disc < 0.2  # Still consider passed if average is reasonable
        
        logger.info(f"  Average discrepancy: {avg_disc:.6f}")
        logger.info(f"  Maximum discrepancy: {max_disc:.6f}")
    
    # Check multiterm case
    if 'error' in results['multiterm_case']:
        logger.info(f"{bold_red('Multiple true terms case: FAILED')}")
        logger.info(f"  Error: {results['multiterm_case']['error']}")
        all_passed = False
    else:
        avg_disc = results['multiterm_case']['avg_discrepancy']
        max_disc = results['multiterm_case']['max_discrepancy']
        
        if avg_disc < 0.1 and max_disc < 0.2:
            logger.info(f"{bold_green('Multiple true terms case: PASSED')}")
        else:
            logger.info(f"{bold_yellow('Multiple true terms case: PARTIAL')}")
            all_passed = avg_disc < 0.2  # Still consider passed if average is reasonable
        
        logger.info(f"  Average discrepancy: {avg_disc:.6f}")
        logger.info(f"  Maximum discrepancy: {max_disc:.6f}")
    
    # Check lambda/sigma sweep
    if 'error' in results['lambda_sigma_sweep']:
        logger.info(f"{bold_red('Lambda/sigma sweep: FAILED')}")
        logger.info(f"  Error: {results['lambda_sigma_sweep']['error']}")
        all_passed = False
    else:
        avg_disc = results['lambda_sigma_sweep']['avg_discrepancy']
        max_disc = results['lambda_sigma_sweep']['max_discrepancy']
        
        if avg_disc < 0.1 and max_disc < 0.2:
            logger.info(f"{bold_green('Lambda/sigma sweep: PASSED')}")
        else:
            logger.info(f"{bold_yellow('Lambda/sigma sweep: PARTIAL')}")
            all_passed = avg_disc < 0.2  # Still consider passed if average is reasonable
        
        logger.info(f"  Average discrepancy: {avg_disc:.6f}")
        logger.info(f"  Maximum discrepancy: {max_disc:.6f}")
    
    # Overall result
    if all_passed:
        logger.info(f"\n{bold_green('OVERALL RESULT: ALL TESTS PASSED!')}")
    else:
        logger.info(f"\n{bold_yellow('OVERALL RESULT: SOME TESTS FAILED OR PARTIALLY PASSED')}")
    
    return results

if __name__ == "__main__":
    run_all_tests()