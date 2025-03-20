import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import logging
from collections import defaultdict

# Import centralized logging utilities
from models.logging_config import get_logger, header, section
from models.logging_config import bold, green, yellow, red, cyan
from models.logging_config import bold_green, bold_yellow, bold_red


def compare_theory_to_simulation_transitions(model, x_data, n_trials=100):
    """
    Compare theoretical transition probabilities to empirically observed transitions.
    
    Parameters:
    -----------
    model : SINDyMarkovModel
        Model instance
    x_data : array
        Sample points where to evaluate library functions
    n_trials : int
        Number of simulation trials
        
    Returns:
    --------
    comparison_data : dict
        Dictionary with comparison data
    """
    logger = model.logger
    
    logger.info(header("TRANSITION PROBABILITY COMPARISON"))
    
    # Define true dynamics function
    def true_dynamics(x):
        y = np.zeros_like(x)
        for i, coef in enumerate(model.true_coefs):
            if abs(coef) > 1e-10:
                y += coef * model.library_functions[i](x)
        return y
    
    # Step 1: Calculate theoretical transition probabilities
    all_indices = set(range(model.n_terms))
    true_indices = set(model.true_term_indices.tolist())
    
    # Initial state
    initial_state = all_indices
    
    # Generate all valid states
    valid_states = []
    for r in range(len(true_indices), len(all_indices) + 1):
        for subset in combinations(all_indices, r):
            subset_set = set(subset)
            if true_indices.issubset(subset_set):
                valid_states.append(subset_set)
    
    logger.info(f"{bold('Analyzing')} {len(valid_states)} {bold('valid states and their transitions')}")
    logger.info(f"True indices: {true_indices}")
    logger.info(f"All indices: {all_indices}")
    
    # Calculate theoretical transition probabilities
    theoretical_transitions = {}
    for from_state in valid_states:
        for to_state in valid_states:
            if to_state.issubset(from_state) and to_state != from_state:
                prob = model.calculate_transition_probability(from_state, to_state)
                if prob > 0:
                    key = (str(from_state), str(to_state))
                    theoretical_transitions[key] = prob
    
    # Step 2: Run simulations to get empirical transition counts
    observed_transitions = {}
    
    logger.info(f"{bold('Running')} {n_trials} {bold('STLSQ simulations to track transitions')}")
    
    for trial in range(n_trials):
        # Generate data with noise
        y_true = true_dynamics(x_data)
        y_noisy = y_true + np.random.normal(0, model.sigma, size=len(x_data))
        
        # Build library matrix
        theta = np.zeros((len(x_data), model.n_terms))
        for j, func in enumerate(model.library_functions):
            theta[:, j] = func(x_data)
        
        # Run STLSQ with trajectory tracking
        _, trajectory = model.run_stlsq_with_trajectory(theta, y_noisy)
        
        # Record each transition in the trajectory
        for i in range(len(trajectory) - 1):
            from_state = trajectory[i]
            to_state = trajectory[i + 1]
            key = (str(from_state), str(to_state))
            observed_transitions[key] = observed_transitions.get(key, 0) + 1
    
    # Step 3: Convert counts to frequencies
    # Group by from_state
    from_state_counts = {}
    for (from_str, to_str), count in observed_transitions.items():
        from_state_counts[from_str] = from_state_counts.get(from_str, 0) + count
    
    # Calculate empirical probabilities
    empirical_transitions = {}
    for (from_str, to_str), count in observed_transitions.items():
        total_from_state = from_state_counts[from_str]
        empirical_transitions[(from_str, to_str)] = count / total_from_state
    
    # Step 4: Compare theoretical vs empirical transitions
    logger.info(f"\n{bold('TRANSITION PROBABILITY COMPARISON (Theoretical vs Empirical):')}")
    logger.info("-" * 80)
    
    # Combine all transitions
    all_transitions = set(theoretical_transitions.keys()) | set(empirical_transitions.keys())
    
    # Sort by from_state size (largest first)
    sorted_transitions = sorted(all_transitions, 
                            key=lambda k: (len(eval(k[0])), len(eval(k[1]))),
                            reverse=True)
    
    # Group by from_state
    current_from_state = None
    comparison_data = {}
    
    for key in sorted_transitions:
        from_str, to_str = key
        theoretical = theoretical_transitions.get(key, 0.0)
        empirical = empirical_transitions.get(key, 0.0)
        
        # Start a new section if from_state changes
        if from_str != current_from_state:
            logger.info(f"\n{bold_yellow('Transitions from ' + from_str + ':')}")
            current_from_state = from_str
        
        # Calculate difference and whether it's significant
        diff = empirical - theoretical
        is_significant = abs(diff) > 0.1  # Consider a difference > 10% as significant
        
        # Format the output to highlight significant differences
        diff_str = f"{diff:+.4f}"
        if is_significant:
            diff_str = red(diff_str)
            
        logger.info(f"  → {to_str}: Theoretical: {theoretical:.4f}, Empirical: {empirical:.4f}, " 
                    f"Diff: {diff_str}")
        
        # Store comparison data
        comparison_data[key] = {
            'from_state': from_str,
            'to_state': to_str,
            'theoretical': theoretical,
            'empirical': empirical,
            'difference': diff,
            'significant': is_significant
        }
    
    # Step 5: Summarize significant differences
    significant_diffs = [(k, v) for k, v in comparison_data.items() if v['significant']]
    
    if significant_diffs:
        logger.info(f"\n{bold_red('SIGNIFICANT DIFFERENCES (|Diff| > 0.1):')}")
        logger.info("-" * 80)
        
        significant_diffs.sort(key=lambda x: abs(x[1]['difference']), reverse=True)
        
        for (from_str, to_str), data in significant_diffs:
            diff_str = f"{data['difference']:+.4f}"
            logger.info(f"{from_str} → {to_str}: "
                        f"Theoretical: {data['theoretical']:.4f}, "
                        f"Empirical: {data['empirical']:.4f}, "
                        f"Diff: {red(diff_str)}")
    
    # Step 6: Check for direct transitions to true state
    true_str = str(true_indices)
    direct_to_true = [(k, v) for k, v in comparison_data.items() if k[1] == true_str]
    
    if direct_to_true:
        logger.info(f"\n{bold_yellow('DIRECT TRANSITIONS TO TRUE STATE:')}")
        logger.info("-" * 80)
        
        for (from_str, _), data in direct_to_true:
            diff_str = f"{data['difference']:+.4f}"
            if data['significant']:
                diff_str = red(diff_str)
                
            logger.info(f"{from_str} → {true_str}: "
                        f"Theoretical: {data['theoretical']:.4f}, "
                        f"Empirical: {data['empirical']:.4f}, "
                        f"Diff: {diff_str}")
    
    logger.info("=" * 80 + "\n")
    
    return comparison_data


def analyze_coefficient_distributions(model, x_data, n_trials=50):
    """
    Analyze coefficient distributions at each state during STLSQ algorithm execution.
    
    Parameters:
    -----------
    model : SINDyMarkovModel
        Model instance
    x_data : array
        Sample points where to evaluate library functions
    n_trials : int
        Number of simulation trials
        
    Returns:
    --------
    distribution_data : dict
        Dictionary with empirical and theoretical distribution data for each state
    """
    logger = model.logger
    
    logger.info(header("COEFFICIENT DISTRIBUTION ANALYSIS"))
    
    # Define true dynamics
    def true_dynamics(x):
        y = np.zeros_like(x)
        for i, coef in enumerate(model.true_coefs):
            if abs(coef) > 1e-10:
                y += coef * model.library_functions[i](x)
        return y
    
    # Build library matrix
    theta = np.zeros((len(x_data), model.n_terms))
    for j, func in enumerate(model.library_functions):
        theta[:, j] = func(x_data)
    
    # Track coefficients at each state
    state_coefficients = {}
    state_counts = {}
    
    # Run multiple trials
    for trial in range(n_trials):
        # Generate data with noise
        y_true = true_dynamics(x_data)
        y_noisy = y_true + np.random.normal(0, model.sigma, size=len(x_data))
        
        # Track algorithm trajectory with coefficients
        xi, trajectory, coefs_at_states = run_stlsq_with_coefficients(model, theta, y_noisy)
        
        # Record coefficient values at each state
        for state, coeffs in coefs_at_states.items():
            state_key = str(state)
            if state_key not in state_coefficients:
                state_coefficients[state_key] = []
                state_counts[state_key] = 0
            
            state_coefficients[state_key].append(coeffs)
            state_counts[state_key] += 1
    
    # Analyze coefficient distributions for each state
    distribution_data = {}
    
    for state_key, coeffs_list in state_coefficients.items():
        state = eval(state_key)  # Convert string back to set
        state_tuple = tuple(sorted(state))
        
        # Get theoretical distribution
        try:
            theo_mean, theo_cov = model.get_coef_distribution(state_tuple)
            
            # Calculate empirical statistics
            emp_coeffs = np.array(coeffs_list)
            emp_mean = np.mean(emp_coeffs, axis=0)
            emp_cov = np.cov(emp_coeffs, rowvar=False) if len(emp_coeffs) > 1 else np.zeros((len(state), len(state)))
            
            # Compare distributions
            logger.info(f"\n{bold_yellow('State: ' + state_key + ' (visited ' + str(state_counts[state_key]) + ' times)')}")
            
            # Compare means
            logger.info(f"{bold('Coefficient means:')}")
            for i, idx in enumerate(state_tuple):
                diff = emp_mean[i] - theo_mean[i]
                diff_str = f"{diff:+.6f}"
                if abs(diff) > 0.1 * abs(theo_mean[i]) and abs(theo_mean[i]) > 1e-6:
                    # Highlight significant differences
                    logger.info(f"    Coefficient {idx}: Theoretical: {theo_mean[i]:.6f}, Empirical: {emp_mean[i]:.6f}, Diff: {red(diff_str)}")
                else:
                    logger.info(f"    Coefficient {idx}: Theoretical: {theo_mean[i]:.6f}, Empirical: {emp_mean[i]:.6f}, Diff: {diff_str}")
            
            # Compare standard deviations (from diagonal of covariance matrix)
            logger.info(f"{bold('Coefficient standard deviations:')}")
            for i, idx in enumerate(state_tuple):
                theo_std = np.sqrt(theo_cov[i, i])
                emp_std = np.sqrt(emp_cov[i, i]) if emp_cov.size > 0 else 0
                
                if theo_std > 0:
                    ratio = emp_std/theo_std
                    ratio_str = f"{ratio:.4f}"
                    
                    if abs(ratio - 1.0) > 0.2:  # If ratio differs by more than 20%
                        logger.info(f"    Coefficient {idx}: Theoretical: {theo_std:.6f}, Empirical: {emp_std:.6f}, Ratio: {red(ratio_str)}")
                    else:
                        logger.info(f"    Coefficient {idx}: Theoretical: {theo_std:.6f}, Empirical: {emp_std:.6f}, Ratio: {ratio_str}")
                else:
                    logger.info(f"    Coefficient {idx}: Theoretical: {theo_std:.6f}, Empirical: {emp_std:.6f}, Ratio: N/A")
            
            # Compare correlations if more than one coefficient
            if len(state_tuple) > 1:
                logger.info(f"{bold('Coefficient correlations:')}")
                for i in range(len(state_tuple)):
                    for j in range(i+1, len(state_tuple)):
                        if theo_cov[i,i] > 0 and theo_cov[j,j] > 0:
                            theo_corr = theo_cov[i,j] / np.sqrt(theo_cov[i,i] * theo_cov[j,j])
                        else:
                            theo_corr = 0
                            
                        if emp_cov.size > 0 and emp_cov[i,i] > 0 and emp_cov[j,j] > 0:
                            emp_corr = emp_cov[i,j] / np.sqrt(emp_cov[i,i] * emp_cov[j,j])
                        else:
                            emp_corr = 0
                            
                        diff = emp_corr - theo_corr
                        diff_str = f"{diff:+.4f}"
                        
                        idx_i = state_tuple[i]
                        idx_j = state_tuple[j]
                        
                        if abs(diff) > 0.2:  # If correlation difference is significant
                            logger.info(f"    Terms {idx_i}-{idx_j}: Theoretical: {theo_corr:.4f}, Empirical: {emp_corr:.4f}, Diff: {red(diff_str)}")
                        else:
                            logger.info(f"    Terms {idx_i}-{idx_j}: Theoretical: {theo_corr:.4f}, Empirical: {emp_corr:.4f}, Diff: {diff_str}")
            
            # Store data for this state
            distribution_data[state_key] = {
                'theoretical': {
                    'mean': theo_mean,
                    'cov': theo_cov
                },
                'empirical': {
                    'mean': emp_mean,
                    'cov': emp_cov,
                    'samples': len(emp_coeffs)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing state {state_key}: {str(e)}")
    
    logger.info("="*80 + "\n")
    
    return distribution_data


def run_stlsq_with_coefficients(model, theta, y):
    """
    Run STLSQ algorithm and track coefficients at each state.
    
    Parameters:
    -----------
    model : SINDyMarkovModel
        Model instance
    theta : array
        Library matrix
    y : array
        Target dynamics
        
    Returns:
    --------
    xi : array
        Final coefficients
    trajectory : list
        List of states visited
    coefs_at_states : dict
        Dictionary mapping states to coefficient values
    """
    n_terms = theta.shape[1]
    
    # Initial least squares
    xi = np.linalg.lstsq(theta, y, rcond=None)[0]
    
    # Initialize with all terms active
    small_indices = np.abs(xi) < model.threshold
    active_indices = model.normalize_state(np.where(~small_indices)[0])
    
    # Track trajectory and coefficients at each state
    trajectory = [active_indices.copy()]
    coefs_at_states = {}
    
    # Store coefficients for initial state
    coefs_at_states[frozenset(active_indices)] = xi.copy()
    
    # Iterative thresholding
    max_iterations = 10
    converged = False
    
    for _ in range(max_iterations):
        if converged:
            break
            
        # Apply threshold
        small_indices = np.abs(xi) < model.threshold
        xi[small_indices] = 0
        
        # Update active terms
        active_indices = model.normalize_state(np.where(~small_indices)[0])
        
        # If no active terms left, break
        if len(active_indices) == 0:
            break
        
        # Add the new state to the trajectory if it's different from the last state
        if active_indices != trajectory[-1]:
            trajectory.append(active_indices.copy())
        
        # Recalculate coefficients for active terms
        active_list = list(active_indices)
        theta_active = theta[:, active_list]
        xi_active = np.linalg.lstsq(theta_active, y, rcond=None)[0]
        
        # Update coefficient vector
        xi = np.zeros(n_terms)
        for i, idx in enumerate(active_list):
            xi[idx] = xi_active[i]
        
        # Store coefficients for this state
        coefs_at_states[frozenset(active_indices)] = xi_active.copy()
        
        # Check for convergence
        converged = True
        for idx in active_indices:
            if abs(xi[idx]) < model.threshold:
                converged = False
                break
    
    return xi, trajectory, coefs_at_states


def test_independence_assumption(model, x_data, n_trials=50):
    """
    Test the independence assumption in the sequential thresholding process.
    
    Parameters:
    -----------
    model : SINDyMarkovModel
        Model instance
    x_data : array
        Sample points where to evaluate library functions
    n_trials : int
        Number of simulation trials
        
    Returns:
    --------
    independence_metrics : dict
        Dictionary with independence testing metrics
    """
    logger = model.logger
    
    logger.info(header("TESTING INDEPENDENCE ASSUMPTION"))
    
    # Define true dynamics
    def true_dynamics(x):
        y = np.zeros_like(x)
        for i, coef in enumerate(model.true_coefs):
            if abs(coef) > 1e-10:
                y += coef * model.library_functions[i](x)
        return y
    
    # Build library matrix
    theta = np.zeros((len(x_data), model.n_terms))
    for j, func in enumerate(model.library_functions):
        theta[:, j] = func(x_data)
    
    # Track transitions and decisions
    transition_decisions = {}
    
    for trial in range(n_trials):
        # Generate data with noise
        y_true = true_dynamics(x_data)
        y_noisy = y_true + np.random.normal(0, model.sigma, size=len(x_data))
        
        # Run STLSQ with detailed tracking
        _, trajectory, decisions = run_stlsq_with_thresholding_decisions(model, theta, y_noisy)
        
        # Record decisions for each transition
        for i in range(len(trajectory) - 1):
            from_state = frozenset(trajectory[i])
            to_state = frozenset(trajectory[i + 1])
            
            # Create key for this transition
            transition_key = (from_state, to_state)
            
            if transition_key not in transition_decisions:
                transition_decisions[transition_key] = []
            
            transition_decisions[transition_key].append(decisions[i])
    
    # Analyze independence in thresholding decisions
    independence_metrics = {}
    
    for (from_state, to_state), decision_list in transition_decisions.items():
        if len(decision_list) < 5:  # Skip transitions with too few samples
            continue
        
        # Convert decision list to array for analysis
        decision_array = np.array(decision_list)
        
        # Calculate empirical probabilities
        empirical_prob = len(decision_array) / n_trials
        
        # Calculate joint and conditional probabilities when possible
        if decision_array.shape[1] > 1:
            # For each pair of terms, check if thresholding decisions are correlated
            n_terms_in_state = len(from_state)
            correlation_matrix = np.zeros((n_terms_in_state, n_terms_in_state))
            
            for i in range(n_terms_in_state):
                for j in range(i+1, n_terms_in_state):
                    if i < decision_array.shape[1] and j < decision_array.shape[1]:
                        # Calculate correlation between thresholding decisions
                        corr = np.corrcoef(decision_array[:, i], decision_array[:, j])[0, 1]
                        correlation_matrix[i, j] = corr
                        correlation_matrix[j, i] = corr
            
            # Save correlation matrix for this transition
            independence_metrics[str((from_state, to_state))] = {
                'correlation_matrix': correlation_matrix,
                'n_samples': len(decision_array)
            }
            
            # Report high correlations
            high_correlations = np.abs(correlation_matrix) > 0.3
            if np.any(high_correlations):
                logger.info(f"\n{bold_yellow('High correlation detected in transition ' + str(from_state) + ' → ' + str(to_state) + ':')}")
                for i in range(n_terms_in_state):
                    for j in range(i+1, n_terms_in_state):
                        if high_correlations[i, j]:
                            corr_val = correlation_matrix[i, j]
                            if abs(corr_val) > 0.7:
                                logger.info(f"  Terms {i} and {j}: correlation = {bold_red(f'{corr_val:.4f}')}")
                            else:
                                logger.info(f"  Terms {i} and {j}: correlation = {yellow(f'{corr_val:.4f}')}")
    
    logger.info("="*80 + "\n")
    
    return independence_metrics


def run_stlsq_with_thresholding_decisions(model, theta, y):
    """
    Run STLSQ algorithm and track detailed thresholding decisions.
    
    Parameters:
    -----------
    model : SINDyMarkovModel
        Model instance
    theta : array
        Library matrix
    y : array
        Target dynamics
        
    Returns:
    --------
    xi : array
        Final coefficients
    trajectory : list
        List of states visited
    thresholding_decisions : list
        List of thresholding decisions at each step (1 for kept, 0 for eliminated)
    """
    n_terms = theta.shape[1]
    
    # Initial least squares
    xi = np.linalg.lstsq(theta, y, rcond=None)[0]
    
    # Initialize with all terms active
    small_indices = np.abs(xi) < model.threshold
    active_indices = model.normalize_state(np.where(~small_indices)[0])
    
    # Track trajectory and thresholding decisions
    trajectory = [active_indices.copy()]
    thresholding_decisions = []
    
    # Iterative thresholding
    max_iterations = 10
    converged = False
    
    for _ in range(max_iterations):
        if converged:
            break
            
        # Record thresholding decisions for this step
        decisions = np.zeros(n_terms)
        for i in range(n_terms):
            if i in active_indices:
                decisions[i] = 1 if abs(xi[i]) >= model.threshold else 0
        
        # Apply threshold
        small_indices = np.abs(xi) < model.threshold
        xi[small_indices] = 0
        
        # Update active terms
        new_active_indices = model.normalize_state(np.where(~small_indices)[0])
        
        # If active terms changed, record the transition
        if new_active_indices != active_indices:
            thresholding_decisions.append(decisions)
            active_indices = new_active_indices
            trajectory.append(active_indices.copy())
        
        # If no active terms left, break
        if len(active_indices) == 0:
            break
        
        # Recalculate coefficients for active terms
        active_list = list(active_indices)
        theta_active = theta[:, active_list]
        xi_active = np.linalg.lstsq(theta_active, y, rcond=None)[0]
        
        # Update coefficient vector
        xi = np.zeros(n_terms)
        for i, idx in enumerate(active_list):
            xi[idx] = xi_active[i]
        
        # Check for convergence
        converged = True
        for idx in active_indices:
            if abs(xi[idx]) < model.threshold:
                converged = False
                break
    
    return xi, trajectory, thresholding_decisions


def verify_coefficient_distributions(model, x_data, n_trials=50):
    """
    Verify the coefficient distributions by comparing theoretical predictions with empirical observations.
    
    Parameters:
    -----------
    model : SINDyMarkovModel
        Model instance
    x_data : array
        Sample points
    n_trials : int
        Number of trials
        
    Returns:
    --------
    verification_results : dict
        Dictionary with verification results
    """
    logger = model.logger
    logger.info("\nVerifying coefficient distributions...")
    
    # Define true dynamics
    def true_dynamics(x):
        y = np.zeros_like(x)
        for i, coef in enumerate(model.true_coefs):
            if abs(coef) > 1e-10:
                y += coef * model.library_functions[i](x)
        return y
    
    # All terms active state
    all_terms = set(range(model.n_terms))
    
    # Calculate theoretical distribution
    mean_theo, cov_theo = model.get_coef_distribution(list(all_terms))
    
    # Generate empirical distribution through simulation
    coefficients = []
    
    for trial in range(n_trials):
        # Generate data with noise
        y_true = true_dynamics(x_data)
        y_noisy = y_true + np.random.normal(0, model.sigma, size=len(x_data))
        
        # Build library matrix
        theta = np.zeros((len(x_data), model.n_terms))
        for j, func in enumerate(model.library_functions):
            theta[:, j] = func(x_data)
        
        # Perform least squares
        xi = np.linalg.lstsq(theta, y_noisy, rcond=None)[0]
        coefficients.append(xi)
    
    # Calculate empirical statistics
    coef_array = np.array(coefficients)
    mean_emp = np.mean(coef_array, axis=0)
    cov_emp = np.cov(coef_array, rowvar=False)
    
    # Compare theoretical and empirical statistics
    logger.info("\nCoefficient Distribution Verification (All Terms Active):")
    logger.info("-" * 60)
    
    logger.info("\nMean Vector Comparison:")
    for i in range(model.n_terms):
        logger.info(f"  Term {i}: Theoretical: {mean_theo[i]:.6f}, Empirical: {mean_emp[i]:.6f}, Diff: {mean_emp[i] - mean_theo[i]:.6f}")
    
    logger.info("\nStandard Deviation Comparison:")
    for i in range(model.n_terms):
        theo_std = np.sqrt(cov_theo[i, i])
        emp_std = np.sqrt(cov_emp[i, i])
        logger.info(f"  Term {i}: Theoretical: {theo_std:.6f}, Empirical: {emp_std:.6f}, Ratio: {emp_std/theo_std:.6f}")
    
    logger.info("\nCorrelation Matrix Comparison:")
    # Calculate correlation matrices
    corr_theo = np.zeros_like(cov_theo)
    corr_emp = np.zeros_like(cov_emp)
    
    for i in range(model.n_terms):
        for j in range(model.n_terms):
            if cov_theo[i, i] > 0 and cov_theo[j, j] > 0:
                corr_theo[i, j] = cov_theo[i, j] / np.sqrt(cov_theo[i, i] * cov_theo[j, j])
            if cov_emp[i, i] > 0 and cov_emp[j, j] > 0:
                corr_emp[i, j] = cov_emp[i, j] / np.sqrt(cov_emp[i, i] * cov_emp[j, j])
    
    for i in range(model.n_terms):
        for j in range(i+1, model.n_terms):
            logger.info(f"  Terms {i},{j}: Theoretical: {corr_theo[i, j]:.6f}, Empirical: {corr_emp[i, j]:.6f}, Diff: {corr_emp[i, j] - corr_theo[i, j]:.6f}")
    
    # Return results for further analysis
    verification_results = {
        'theoretical': {
            'mean': mean_theo,
            'cov': cov_theo,
            'corr': corr_theo
        },
        'empirical': {
            'mean': mean_emp,
            'cov': cov_emp,
            'corr': corr_emp,
            'samples': coef_array
        }
    }
    
    return verification_results