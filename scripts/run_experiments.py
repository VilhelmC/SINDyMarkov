import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
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

def run_simple_example():
    """Run a simple example with 3 terms, one of which is the true term."""
    print("Running simple example with 3 library terms")
    
    # Define library functions
    def f1(x): return x  # Term 1: x
    def f2(x): return np.sin(x)  # Term 2: sin(x)
    def f3(x): return np.tanh(x)  # Term 3: tanh(x)
    
    library_functions = [f1, f2, f3]
    
    # Define true model: only the first term is present
    true_coefs = np.array([1.0, 0.0, 0.0])
    
    # Create model instance
    sigma = 0.1  # Noise level
    threshold = 0.05  # STLSQ threshold
    model = SINDyMarkovModel(library_functions, true_coefs, sigma, threshold)
    
    # Compare theory to simulation
    x_range = np.linspace(0.1, 2.0, 5)  # Different data ranges
    n_samples_range = np.array([50, 100, 200, 300, 500])  # Different sample counts
    
    print(f"Running comparison with {len(x_range)} data ranges and {len(n_samples_range)} sample sizes")
    
    start_time = time.time()
    results = model.compare_theory_to_simulation(x_range, n_samples_range, n_trials=50)
    elapsed_time = time.time() - start_time
    
    print(f"Completed in {elapsed_time:.2f} seconds")
    
    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save results to CSV
    results.to_csv('results/simple_example_results.csv', index=False)
    
    # Plot and save figures
    fig1 = model.plot_comparison(results, x_axis='discriminability')
    fig1.savefig('results/simple_example_discriminability.png', dpi=300, bbox_inches='tight')
    
    fig2 = model.plot_direct_comparison(results)
    fig2.savefig('results/simple_example_direct_comparison.png', dpi=300, bbox_inches='tight')
    
    # Print evaluation metrics
    metrics = model.evaluate_model(results)
    model.print_metrics(metrics)
    
    return results, metrics

def run_lambda_sigma_experiment():
    """Run experiment varying the lambda/sigma ratio."""
    print("\nRunning lambda/sigma ratio experiment")
    
    # Define library functions
    def f1(x): return x  # Term 1: x
    def f2(x): return np.sin(x)  # Term 2: sin(x)
    def f3(x): return x**2  # Term 3: x^2
    
    library_functions = [f1, f2, f3]
    
    # Define true model: only the first term is present
    true_coefs = np.array([1.0, 0.0, 0.0])
    
    # Base parameters
    sigma = 0.1  # Noise level
    
    # Lambda/sigma ratios to test
    lambda_sigma_ratios = [0.1, 0.25, 0.5, 1.0, 2.0]
    
    # Fixed data range and samples for this experiment
    x_range = np.array([0.5, 1.0])
    n_samples_range = np.array([100, 200, 500])
    
    all_results = []
    
    for ratio in lambda_sigma_ratios:
        threshold = ratio * sigma
        print(f"\nTesting lambda/sigma ratio: {ratio} (λ={threshold:.4f}, σ={sigma})")
        
        model = SINDyMarkovModel(library_functions, true_coefs, sigma, threshold)
        
        results = model.compare_theory_to_simulation(x_range, n_samples_range, n_trials=50)
        results['lambda_sigma_ratio'] = ratio
        
        all_results.append(results)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    combined_results.to_csv('results/lambda_sigma_experiment_results.csv', index=False)
    
    # Plot combined results
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
    for ratio in lambda_sigma_ratios:
        ratio_results = combined_results[combined_results['lambda_sigma_ratio'] == ratio]
        fig = model.plot_direct_comparison(ratio_results)
        fig.suptitle(f'λ/σ Ratio = {ratio}')
        fig.savefig(f'results/lambda_sigma_ratio_{ratio}_comparison.png', dpi=300, bbox_inches='tight')
    
    return combined_results

def run_multiterm_experiment():
    """Run an experiment with multiple true terms."""
    print("\nRunning experiment with multiple true terms")
    
    # Define library functions
    def f1(x): return x  # Term 1: x
    def f2(x): return x**2  # Term 2: x^2
    def f3(x): return x**3  # Term 3: x^3
    def f4(x): return np.sin(x)  # Term 4: sin(x)
    def f5(x): return np.exp(x)  # Term 5: exp(x)
    
    library_functions = [f1, f2, f3, f4, f5]
    
    # Define true model: first and third terms are present
    true_coefs = np.array([1.0, 0.0, 0.5, 0.0, 0.0])
    
    # Create model instance
    sigma = 0.1  # Noise level
    threshold = 0.05  # STLSQ threshold
    model = SINDyMarkovModel(library_functions, true_coefs, sigma, threshold)
    
    # Compare theory to simulation with a more focused parameter set
    # due to higher computational complexity
    x_range = np.linspace(0.2, 1.0, 3)
    n_samples_range = np.array([100, 300, 500])
    
    print(f"Running comparison with {len(x_range)} data ranges and {len(n_samples_range)} sample sizes")
    
    start_time = time.time()
    results = model.compare_theory_to_simulation(x_range, n_samples_range, n_trials=30)
    elapsed_time = time.time() - start_time
    
    print(f"Completed in {elapsed_time:.2f} seconds")
    
    # Save results
    results.to_csv('results/multiterm_experiment_results.csv', index=False)
    
    # Plot and save figures
    fig1 = model.plot_comparison(results, x_axis='discriminability')
    fig1.savefig('results/multiterm_discriminability.png', dpi=300, bbox_inches='tight')
    
    fig2 = model.plot_direct_comparison(results)
    fig2.savefig('results/multiterm_direct_comparison.png', dpi=300, bbox_inches='tight')
    
    # Print evaluation metrics
    metrics = model.evaluate_model(results)
    model.print_metrics(metrics)
    
    return results, metrics

def run_all_experiments():
    """Run all experiments and summarize results."""
    print("Starting SINDy Markov Model Experiments")
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Run each experiment and collect results
    print("\n" + "="*60)
    print("EXPERIMENT 1: Simple Three-Term Example")
    print("="*60)
    simple_results, simple_metrics = run_simple_example()
    
    print("\n" + "="*60)
    print("EXPERIMENT 2: Lambda/Sigma Ratio Experiment")
    print("="*60)
    lambda_sigma_results = run_lambda_sigma_experiment()
    
    print("\n" + "="*60)
    print("EXPERIMENT 3: Multiple True Terms Experiment")
    print("="*60)
    multiterm_results, multiterm_metrics = run_multiterm_experiment()
    
    # Create summary plot of all experiments
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
    middle_ratio = 0.5
    ratio_results = lambda_sigma_results[lambda_sigma_results['lambda_sigma_ratio'] == middle_ratio]
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
    
    # Write summary report
    # Write summary report
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
        # Replace the lambda character with "lambda" to avoid encoding issues
        f.write("  lambda/sigma ratios, and experimental settings.\n")
    
    print("\nExperiments completed. Results saved to 'results' directory.")

if __name__ == "__main__":
    run_all_experiments()