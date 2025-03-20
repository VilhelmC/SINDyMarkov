# Make sure the import is at the top
import numpy as np
import pandas as pd
from itertools import combinations

# Import centralized logging utilities
from models.logging_config import setup_logging, get_logger
from models.logging_config import bold, green, yellow, red, cyan, header, section
from models.logging_config import bold_green, bold_yellow, bold_red

# Setup logger
logger = get_logger('markov_simulation')

def run_lambda_sigma_experiment(model_class, library_functions, true_coefs, sigma=0.1, 
                                lambda_sigma_ratios=None, x_range=None, n_samples_range=None, 
                                n_trials=50, log_file_template='logs/lambda_sigma_ratio_{}.log'):
    """
    Run experiment varying the lambda/sigma ratio.
    
    Parameters:
    -----------
    model_class : class
        SINDyMarkovModel class
    library_functions : list
        List of library functions
    true_coefs : array
        True coefficient values
    sigma : float
        Noise standard deviation
    lambda_sigma_ratios : list
        List of lambda/sigma ratios to test
    x_range : array
        Data ranges to test
    n_samples_range : array
        Sample sizes to test
    n_trials : int
        Number of trials per configuration
    log_file_template : str
        Template for log file path
        
    Returns:
    --------
    combined_results : DataFrame
        Combined results from all experiments
    """
    if lambda_sigma_ratios is None:
        lambda_sigma_ratios = [0.1, 0.25, 0.5, 1.0, 2.0]
    
    if x_range is None:
        x_range = np.array([0.5, 1.0])
    
    if n_samples_range is None:
        n_samples_range = np.array([100, 200, 500])
    
    print(f"\n{bold_green('Running lambda/sigma ratio experiment')}")
    print(f"{bold('Testing ratios:')} {lambda_sigma_ratios}")
    print(f"{bold('Data ranges:')} {x_range}")
    print(f"{bold('Sample sizes:')} {n_samples_range}")
    
    all_results = []
    
    for ratio in lambda_sigma_ratios:
        threshold = ratio * sigma
        print(f"\n{bold_yellow(f'Testing lambda/sigma ratio: {ratio} (λ={threshold:.4f}, σ={sigma})')}")
        
        log_file = log_file_template.format(ratio)
        model = model_class(library_functions, true_coefs, sigma, threshold, log_file=log_file)
        
        # Import analysis functions from markov_analysis
        from markov_analysis import compare_theory_to_simulation
        
        results = compare_theory_to_simulation(model, x_range, n_samples_range, n_trials=n_trials)
        results['lambda_sigma_ratio'] = ratio
        
        all_results.append(results)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    return combined_results

def run_multiterm_experiment(model_class, n_trials=30, log_file='logs/multiterm_experiment.log'):
    """
    Run an experiment with multiple true terms.
    
    Parameters:
    -----------
    model_class : class
        SINDyMarkovModel class
    n_trials : int
        Number of trials per configuration
    log_file : str
        Log file path
        
    Returns:
    --------
    results : DataFrame
        Results from the experiment
    """
    print(f"\n{bold_green('Running experiment with multiple true terms')}")
    
    # Define library functions
    def f1(x): return x          # Term 1: x
    def f2(x): return x**2       # Term 2: x^2
    def f3(x): return x**3       # Term 3: x^3
    def f4(x): return np.sin(x)  # Term 4: sin(x)
    def f5(x): return np.exp(x)  # Term 5: exp(x)
    
    library_functions = [f1, f2, f3, f4, f5]
    
    # Define true model: first and third terms are present
    true_coefs = np.array([1.0, 0.0, 0.5, 0.0, 0.0])
    
    # Create model instance
    sigma = 0.1  # Noise level
    threshold = 0.05  # STLSQ threshold
    model = model_class(library_functions, true_coefs, sigma, threshold, log_file=log_file)
    
    # Compare theory to simulation with a focused parameter set due to higher computational complexity
    x_range = np.linspace(0.2, 1.0, 3)
    n_samples_range = np.array([100, 300, 500])
    
    print(f"{bold(f'Running comparison with {len(x_range)} data ranges and {len(n_samples_range)} sample sizes')}")
    
    # Import analysis functions from markov_analysis
    from markov_analysis import compare_theory_to_simulation
    
    results = compare_theory_to_simulation(model, x_range, n_samples_range, n_trials=n_trials)
    
    return results

def run_simple_example(model_class, n_trials=50, log_file='logs/simple_example.log'):
    """
    Run a simple example with 3 terms, one of which is the true term.
    
    Parameters:
    -----------
    model_class : class
        SINDyMarkovModel class
    n_trials : int
        Number of trials per configuration
    log_file : str
        Log file path
        
    Returns:
    --------
    results : DataFrame
        Results from the experiment
    """
    print(f"{bold_green('Running simple example with 3 library terms')}")
    
    # Define library functions
    def f1(x): return x           # Term 1: x
    def f2(x): return np.sin(x)   # Term 2: sin(x)
    def f3(x): return np.tanh(x)  # Term 3: tanh(x)
    
    library_functions = [f1, f2, f3]
    
    # Define true model: only the first term is present
    true_coefs = np.array([1.0, 0.0, 0.0])
    
    # Create model instance
    sigma = 0.1  # Noise level
    threshold = 0.05  # STLSQ threshold
    model = model_class(library_functions, true_coefs, sigma, threshold, log_file=log_file)
    
    # Compare theory to simulation
    x_range = np.linspace(0.1, 2.0, 5)  # Different data ranges
    n_samples_range = np.array([50, 100, 200, 300, 500])  # Different sample counts
    
    print(f"{bold(f'Running comparison with {len(x_range)} data ranges and {len(n_samples_range)} sample sizes')}")
    
    # Import analysis functions from markov_analysis
    from markov_analysis import compare_theory_to_simulation
    
    results = compare_theory_to_simulation(model, x_range, n_samples_range, n_trials=n_trials)
    
    return results

def generate_true_dynamics(x, true_coefs, library_functions):
    """
    Generate true dynamics from coefficients and library functions.
    
    Parameters:
    -----------
    x : array
        Input values
    true_coefs : array
        True coefficient values
    library_functions : list
        List of library functions
        
    Returns:
    --------
    y : array
        True dynamics values
    """
    y = np.zeros_like(x)
    for i, coef in enumerate(true_coefs):
        if abs(coef) > 1e-10:
            y += coef * library_functions[i](x)
    return y

# In markov_simulation.py, update the run_stlsq_simulation function:

def run_stlsq_simulation(model, x_data, n_trials=100, return_details=False):
    """
    Run STLSQ simulation with detailed tracking of results.
    
    Parameters:
    -----------
    model : SINDyMarkovModel
        Model instance
    x_data : array
        Sample points
    n_trials : int
        Number of trials
    return_details : bool
        Whether to return detailed trajectory information
        
    Returns:
    --------
    success_rate : float
        Empirical success probability
    details : dict, optional
        Detailed trajectory information (if return_details=True)
    """
    # Define true dynamics function
    def true_dynamics(x):
        return generate_true_dynamics(x, model.true_coefs, model.library_functions)
    
    successful_trials = 0
    trial_results = []
    trajectory_counts = {}
    
    for trial in range(n_trials):
        # Generate noise-free dynamics
        y_true = true_dynamics(x_data)
        
        # Add noise
        y_noisy = y_true + np.random.normal(0, model.sigma, size=len(x_data))
        
        # Build library matrix
        theta = np.zeros((len(x_data), model.n_terms))
        for j, func in enumerate(model.library_functions):
            theta[:, j] = func(x_data)
        
        # Run STLSQ with trajectory tracking
        xi, trajectory = model.run_stlsq_with_trajectory(theta, y_noisy)
        
        # Convert trajectory to a string representation for counting
        trajectory_str = ' -> '.join([str(sorted(list(state))) for state in trajectory]) + " -> [STOP]"
        trajectory_counts[trajectory_str] = trajectory_counts.get(trajectory_str, 0) + 1
        
        # Check for success
        true_pattern = np.zeros(model.n_terms)
        true_pattern[model.true_term_indices] = 1
        identified_pattern = np.zeros(model.n_terms)
        identified_pattern[np.abs(xi) > 1e-10] = 1
        
        # Determine if this is a success
        is_success = np.array_equal(true_pattern, identified_pattern)
        if is_success:
            successful_trials += 1
        
        # Store details about this trial
        if return_details:
            trial_results.append({
                'trial': trial + 1,
                'success': is_success,
                'identified_terms': np.where(identified_pattern == 1)[0].tolist(),
                'trajectory': trajectory,
                'coefficients': xi
            })
    
    success_rate = successful_trials / n_trials
    
    if return_details:
        details = {
            'success_rate': success_rate,
            'trials': trial_results,
            'trajectory_counts': trajectory_counts
        }
        return success_rate, details
    else:
        return success_rate