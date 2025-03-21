"""
Utilities for running simulations in the SINDy Markov Chain Model.

This module provides functions for simulating the Sequential Thresholded
Least Squares (STLSQ) algorithm and analyzing its performance.
"""

import numpy as np
from scipy import stats
import logging
from itertools import combinations

from models.state_utils import normalize_state

# Get logger
logger = logging.getLogger('SINDyMarkovModel')

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

def run_stlsq(theta, y, threshold, use_regularization=True, reg_param=1e-8):
    """
    Run sequential thresholded least squares algorithm.
    
    Parameters:
    -----------
    theta : array
        Library matrix
    y : array
        Target dynamics
    threshold : float
        Threshold for sparsification
    use_regularization : bool
        Whether to use regularization for numerical stability
    reg_param : float
        Regularization parameter (if use_regularization is True)
            
    Returns:
    --------
    xi : array
        Identified coefficients
    """
    n_terms = theta.shape[1]
    
    # Initial least squares
    if use_regularization:
        # Add small regularization for numerical stability
        gram = theta.T @ theta
        regularized_gram = gram + reg_param * np.eye(gram.shape[0])
        xi = np.linalg.solve(regularized_gram, theta.T @ y)
    else:
        xi = np.linalg.lstsq(theta, y, rcond=None)[0]
    
    # Iterative thresholding
    max_iterations = 10
    for _ in range(max_iterations):
        # Apply threshold
        small_indices = np.abs(xi) < threshold
        xi[small_indices] = 0
        
        # If all coefficients are zero, break
        if np.all(small_indices):
            break
                
        # Get active terms
        active_indices = normalize_state(np.where(~small_indices)[0])
        
        # Recalculate coefficients for active terms
        if active_indices:
            active_list = list(active_indices)
            theta_active = theta[:, active_list]
            
            try:
                if use_regularization:
                    # Add small regularization for numerical stability
                    gram = theta_active.T @ theta_active
                    regularized_gram = gram + reg_param * np.eye(gram.shape[0])
                    xi_active = np.linalg.solve(regularized_gram, theta_active.T @ y)
                else:
                    xi_active = np.linalg.lstsq(theta_active, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                # If standard approach fails, use more aggressive regularization
                logger.warning("LinAlgError in run_stlsq, using more aggressive regularization")
                gram = theta_active.T @ theta_active
                regularized_gram = gram + 1e-4 * np.eye(gram.shape[0])
                xi_active = np.linalg.solve(regularized_gram, theta_active.T @ y)
            
            # Update coefficient vector
            xi = np.zeros(n_terms)
            for i, idx in enumerate(active_list):
                xi[idx] = xi_active[i]
        
        # Check if converged
        if np.all(np.abs(xi) >= threshold) or np.all(xi == 0):
            break
    
    return xi

def run_stlsq_with_trajectory(theta, y, threshold, use_regularization=True, reg_param=1e-8):
    """
    Run sequential thresholded least squares algorithm and track the trajectory.
    
    Parameters:
    -----------
    theta : array
        Library matrix
    y : array
        Target dynamics
    threshold : float
        Threshold for sparsification
    use_regularization : bool
        Whether to use regularization for numerical stability
    reg_param : float
        Regularization parameter (if use_regularization is True)
            
    Returns:
    --------
    xi : array
        Identified coefficients
    trajectory : list
        List of sets representing the states visited during the algorithm
    """
    n_terms = theta.shape[1]
    
    # Initial least squares
    if use_regularization:
        # Add small regularization for numerical stability
        gram = theta.T @ theta
        regularized_gram = gram + reg_param * np.eye(gram.shape[0])
        xi = np.linalg.solve(regularized_gram, theta.T @ y)
    else:
        xi = np.linalg.lstsq(theta, y, rcond=None)[0]
    
    # Initialize with all terms active
    small_indices = np.abs(xi) < threshold
    active_indices = normalize_state(np.where(~small_indices)[0])
    
    # Track trajectory through state space
    trajectory = [active_indices.copy()]
    
    # Iterative thresholding
    max_iterations = 10
    converged = False
    
    for _ in range(max_iterations):
        if converged:
            break
                
        # Apply threshold
        small_indices = np.abs(xi) < threshold
        xi[small_indices] = 0
        
        # Update active terms
        active_indices = normalize_state(np.where(~small_indices)[0])
        
        # If no active terms left, break
        if len(active_indices) == 0:
            break
        
        # Add the new state to the trajectory if it's different from the last state
        if active_indices != trajectory[-1]:
            trajectory.append(active_indices.copy())
        
        # Recalculate coefficients for active terms
        active_list = list(active_indices)
        theta_active = theta[:, active_list]
        
        try:
            if use_regularization:
                # Add small regularization for numerical stability
                gram = theta_active.T @ theta_active
                regularized_gram = gram + reg_param * np.eye(gram.shape[0])
                xi_active = np.linalg.solve(regularized_gram, theta_active.T @ y)
            else:
                xi_active = np.linalg.lstsq(theta_active, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            # If standard approach fails, use more aggressive regularization
            logger.warning("LinAlgError in run_stlsq_with_trajectory, using more aggressive regularization")
            gram = theta_active.T @ theta_active
            regularized_gram = gram + 1e-4 * np.eye(gram.shape[0])
            xi_active = np.linalg.solve(regularized_gram, theta_active.T @ y)
        
        # Update coefficient vector
        xi = np.zeros(n_terms)
        for i, idx in enumerate(active_list):
            xi[idx] = xi_active[i]
        
        # Check for convergence
        converged = True
        for idx in active_indices:
            if abs(xi[idx]) < threshold:
                converged = False
                break
    
    return xi, trajectory

def run_stlsq_with_coefficients(theta, y, threshold, use_regularization=True, reg_param=1e-8):
    """
    Run STLSQ algorithm and track coefficients at each state.
    
    Parameters:
    -----------
    theta : array
        Library matrix
    y : array
        Target dynamics
    threshold : float
        Threshold for sparsification
    use_regularization : bool
        Whether to use regularization for numerical stability
    reg_param : float
        Regularization parameter (if use_regularization is True)
        
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
    if use_regularization:
        # Add small regularization for numerical stability
        gram = theta.T @ theta
        regularized_gram = gram + reg_param * np.eye(gram.shape[0])
        xi = np.linalg.solve(regularized_gram, theta.T @ y)
    else:
        xi = np.linalg.lstsq(theta, y, rcond=None)[0]
    
    # Initialize with all terms active
    small_indices = np.abs(xi) < threshold
    active_indices = normalize_state(np.where(~small_indices)[0])
    
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
        small_indices = np.abs(xi) < threshold
        xi[small_indices] = 0
        
        # Update active terms
        active_indices = normalize_state(np.where(~small_indices)[0])
        
        # If no active terms left, break
        if len(active_indices) == 0:
            break
        
        # Add the new state to the trajectory if it's different from the last state
        if active_indices != trajectory[-1]:
            trajectory.append(active_indices.copy())
        
        # Recalculate coefficients for active terms
        active_list = list(active_indices)
        theta_active = theta[:, active_list]
        
        try:
            if use_regularization:
                # Add small regularization for numerical stability
                gram = theta_active.T @ theta_active
                regularized_gram = gram + reg_param * np.eye(gram.shape[0])
                xi_active = np.linalg.solve(regularized_gram, theta_active.T @ y)
            else:
                xi_active = np.linalg.lstsq(theta_active, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            # If standard approach fails, use more aggressive regularization
            logger.warning("LinAlgError in run_stlsq_with_coefficients, using more aggressive regularization")
            gram = theta_active.T @ theta_active
            regularized_gram = gram + 1e-4 * np.eye(gram.shape[0])
            xi_active = np.linalg.solve(regularized_gram, theta_active.T @ y)
        
        # Update coefficient vector
        xi = np.zeros(n_terms)
        for i, idx in enumerate(active_list):
            xi[idx] = xi_active[i]
        
        # Store coefficients for this state
        coefs_at_states[frozenset(active_indices)] = xi_active.copy()
        
        # Check for convergence
        converged = True
        for idx in active_indices:
            if abs(xi[idx]) < threshold:
                converged = False
                break
    
    return xi, trajectory, coefs_at_states

def simulate_stlsq(model, x_data, n_trials=100, return_details=False):
    """
    Empirically simulate the STLSQ algorithm to estimate success probability.
    
    Parameters:
    -----------
    model : SINDyMarkovModel
        Model instance
    x_data : array
        Sample points where to evaluate library functions
    n_trials : int
        Number of simulation trials
    return_details : bool
        Whether to return detailed trajectory information
            
    Returns:
    --------
    success_rate : float
        Empirical success probability
    details : dict, optional
        Detailed trajectory information (if return_details=True)
    """
    logger.info(f"Running STLSQ simulation with {n_trials} trials...")
    
    # Generate true dynamics using true_coefs
    def true_dynamics(x):
        y = np.zeros_like(x)
        for i, coef in enumerate(model.true_coefs):
            if abs(coef) > 1e-10:
                y += coef * model.library_functions[i](x)
        return y
    
    successful_trials = 0
    trial_results = []
    trajectory_counts = {}
    
    # Build library matrix once (it's the same for all trials)
    theta = np.zeros((len(x_data), model.n_terms))
    for j, func in enumerate(model.library_functions):
        theta[:, j] = func(x_data)
    
    # Generate true dynamics once (it's the same for all trials)
    y_true = true_dynamics(x_data)
    
    for trial in range(n_trials):
        # Add noise
        y_noisy = y_true + np.random.normal(0, model.sigma, size=len(x_data))
        
        # Run STLSQ with trajectory tracking
        xi, trajectory = run_stlsq_with_trajectory(theta, y_noisy, model.threshold)
        
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
    logger.info(f"STLSQ simulation results: {successful_trials}/{n_trials} successful, {success_rate:.4f} success rate")
    
    if return_details:
        details = {
            'success_rate': success_rate,
            'trials': trial_results,
            'trajectory_counts': trajectory_counts
        }
        return success_rate, details
    else:
        return success_rate

def simulate_stlsq_adaptive(model, x_data, max_trials=500, confidence=0.95, margin=0.05, 
                           min_trials=30, batch_size=10, return_details=False):
    """
    Empirically simulate the STLSQ algorithm with adaptive trial count determination.
    
    This method automatically determines how many trials are needed to achieve 
    a certain confidence level for the estimated success probability.
    
    Parameters:
    -----------
    model : SINDyMarkovModel
        Model instance
    x_data : array
        Sample points where to evaluate library functions
    max_trials : int
        Maximum number of trials to run
    confidence : float
        Confidence level (e.g., 0.95 for 95% confidence)
    margin : float
        Maximum margin of error desired
    min_trials : int
        Minimum number of trials to run regardless of convergence
    batch_size : int
        Number of trials to run in each batch
    return_details : bool
        Whether to return detailed trajectory information
            
    Returns:
    --------
    success_rate : float
        Empirical success probability
    trials_used : int
        Number of trials that were actually run
    details : dict, optional
        Detailed trajectory information (if return_details=True)
    """
    logger = logging.getLogger('SINDyMarkovModel')
    logger.info("\nRunning adaptive STLSQ simulation (max %d trials, %d%% confidence, %.1f%% margin)", 
               max_trials, confidence*100, margin*100)
    
    # Generate true dynamics using true_coefs
    def true_dynamics(x):
        y = np.zeros_like(x)
        for i, coef in enumerate(model.true_coefs):
            if abs(coef) > 1e-10:
                y += coef * model.library_functions[i](x)
        return y
    
    successful_trials = 0
    total_trials = 0
    trial_results = []
    trajectory_counts = {}
    
    # Build library matrix once (it's the same for all trials)
    theta = np.zeros((len(x_data), model.n_terms))
    for j, func in enumerate(model.library_functions):
        theta[:, j] = func(x_data)
    
    # Generate true dynamics once (it's the same for all trials)
    y_true = true_dynamics(x_data)
    
    # Calculate initial z-score for desired confidence
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    # Run batches of trials until convergence or max_trials
    convergence_achieved = False
    
    while total_trials < max_trials:
        # Run a batch of trials
        batch_successes = 0
        batch_trials = []
        
        for _ in range(batch_size):
            if total_trials >= max_trials:
                break
                
            # Add noise
            y_noisy = y_true + np.random.normal(0, model.sigma, size=len(x_data))
            
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
                batch_successes += 1
            
            # Store details about this trial
            trial_info = {
                'trial': total_trials + 1,
                'success': is_success,
                'identified_terms': np.where(identified_pattern == 1)[0].tolist(),
                'trajectory': trajectory,
                'coefficients': xi
            }
            
            trial_results.append(trial_info)
            batch_trials.append(trial_info)
            
            total_trials += 1
        
        # Calculate current success rate
        current_success_rate = successful_trials / total_trials
        
        # Calculate margin of error using Wilson score interval
        z2 = z_score ** 2
        factor = z2 / (2 * total_trials)
        
        wilson_p = (current_success_rate + factor) / (1 + 2 * factor)
        wilson_error_margin = z_score * np.sqrt(
            current_success_rate * (1 - current_success_rate) / total_trials + factor / 4
        ) / (1 + 2 * factor)
        
        # Log progress for important milestones
        if total_trials == 20 or total_trials % 100 == 0 or total_trials == max_trials:
            logger.info(f"After {total_trials} trials: Success rate = {current_success_rate:.4f}, Margin of error = ±{wilson_error_margin:.4f} (target: {margin:.4f})")
        
        # Check if we've reached the desired margin of error and minimum trials
        if wilson_error_margin <= margin and total_trials >= min_trials:
            logger.info(f"Convergence achieved! Margin of error {wilson_error_margin:.4f} <= target {margin:.4f}")
            convergence_achieved = True
            break
    
    # Final success rate
    success_rate = successful_trials / total_trials
    
    # Log detailed results with nice formatting
    logger.info("\n--------------------------------------------------------------------------------")
    logger.info(f"ADAPTIVE STLSQ SIMULATION RESULTS ({successful_trials}/{total_trials} successful, {success_rate:.4f})")
    logger.info("--------------------------------------------------------------------------------")
    
    # Report if we didn't converge
    if not convergence_achieved:
        logger.info(f"Warning: Maximum trials reached without achieving target margin of error.")
        logger.info(f"Target margin: {margin:.4f}, Achieved margin: {wilson_error_margin:.4f}")
    
    # Show confidence interval
    lower_bound = max(0, wilson_p - wilson_error_margin)
    upper_bound = min(1, wilson_p + wilson_error_margin)
    logger.info(f"{confidence*100:.0f}% Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}] (width: {upper_bound-lower_bound:.4f})")
    
    # Show sample trials
    num_to_show = min(5, total_trials)
    logger.info(f"\nSample of {num_to_show} trials:")
    for i in range(num_to_show):
        trial = trial_results[i]
        traj_str = ' -> '.join([str(sorted(list(state))) for state in trial['trajectory']])
        status = "SUCCESS" if trial['success'] else "FAILURE"
        logger.info(f"  Trial {trial['trial']}: {status}, Identified terms: {trial['identified_terms']}")
        logger.info(f"    Trajectory: {traj_str} -> [STOP]")
    
    # Analyze trajectories
    logger.info("\nPath Analysis:")
    sorted_trajectories = sorted(trajectory_counts.items(), key=lambda x: x[1], reverse=True)
    num_trajectories = min(5, len(sorted_trajectories))
    logger.info(f"Top {num_trajectories} paths through state space:")
    
    total_path_count = sum(trajectory_counts.values())
    for traj, count in sorted_trajectories[:num_trajectories]:
        percentage = (count / total_path_count) * 100
        logger.info(f"  {traj}: {count} occurrences ({percentage:.1f}% of trials)")
    
    # Analyze failure patterns if there were any failures
    if successful_trials < total_trials:
        failure_trials = [t for t in trial_results if not t['success']]
        term_counts = {}
        for trial in failure_trials:
            terms_str = str(sorted(trial['identified_terms']))
            term_counts[terms_str] = term_counts.get(terms_str, 0) + 1
        
        # Sort and report top failure patterns
        top_failures = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        logger.info("\nMost common failure patterns:")
        for terms_str, count in top_failures[:3]:  # Show top 3
            percentage = (count / len(failure_trials)) * 100
            logger.info(f"  {terms_str}: {count} occurrences ({percentage:.1f}% of failures)")
    
    # Perform coefficient analysis
    run_stlsq_with_coefficient_analysis(model, x_data, min(20, total_trials))
    
    logger.info(f"Trials used: {total_trials} (adaptive)")
    
    if return_details:
        details = {
            'success_rate': success_rate,
            'trials_used': total_trials,
            'margin': wilson_error_margin,
            'trials': trial_results,
            'trajectory_counts': trajectory_counts
        }
        return success_rate, total_trials, details
    else:
        return success_rate, total_trials

def run_stlsq_with_coefficient_analysis(model, x_data, n_trials=50):
    """
    Run STLSQ simulations and analyze coefficient distributions at different states.
    
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
    coef_analysis : dict
        Analysis of coefficient distributions
    """
    logger = logging.getLogger('SINDyMarkovModel')
    logger.info("\nCoefficient Distribution Analysis:")
    
    # Generate true dynamics
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
    
    # Generate true dynamics
    y_true = true_dynamics(x_data)
    
    # Track coefficients at each state
    state_coefficients = {}
    
    # Run trials
    for trial in range(n_trials):
        # Add noise
        y_noisy = y_true + np.random.normal(0, model.sigma, size=len(x_data))
        
        # Run STLSQ with detailed tracking
        _, trajectory, coefs_by_state = run_stlsq_with_coefficients(
            theta, y_noisy, model.threshold
        )
        
        # Record coefficients for each state
        for state, coeffs in coefs_by_state.items():
            if state not in state_coefficients:
                state_coefficients[state] = []
            state_coefficients[state].append(coeffs)
    
    # Analyze coefficient distributions
    coef_analysis = {}
    
    for state, coeffs_list in state_coefficients.items():
        # Convert state from frozenset to list
        state_list = sorted(list(state))
        
        # Only analyze states visited frequently enough
        if len(coeffs_list) <= 3:
            continue
            
        # Get theoretical distribution
        mean_theo, cov_theo = model.get_coef_distribution(state_list)
        
        # Calculate empirical statistics
        coeffs_array = np.array(coeffs_list)
        mean_emp = np.mean(coeffs_array, axis=0)
        
        # Handle case where we only have one sample or the shape is incorrect
        if len(coeffs_array) > 1 and coeffs_array.ndim == 2:
            # Make sure we have more samples than dimensions
            if len(coeffs_array) > coeffs_array.shape[1]:
                cov_emp = np.cov(coeffs_array, rowvar=False)
                # Ensure cov_emp is 2D even for 1D data
                if cov_emp.ndim == 0:
                    cov_emp = np.array([[cov_emp]])
                std_emp = np.sqrt(np.diag(cov_emp))
            else:
                # Not enough samples for proper covariance estimation
                cov_emp = np.eye(len(state_list)) * 1e-10
                std_emp = np.zeros(len(state_list))
        else:
            # Handle edge cases
            cov_emp = np.eye(len(state_list)) * 1e-10
            std_emp = np.zeros(len(state_list))
        
        # Calculate theoretical std
        std_theo = np.sqrt(np.diag(cov_theo))
        
        # Log comparison
        logger.info(f"\nState {state_list} (observed {len(coeffs_list)} times):")
        logger.info("Coefficient means:")
        for i, idx in enumerate(state_list):
            m_theo = mean_theo[i]
            m_emp = mean_emp[i]
            diff = m_emp - m_theo
            
            # Calculate percentage difference
            if abs(m_theo) > 1e-8:  # Avoid division by zero for near-zero values
                pct_diff = (diff / abs(m_theo)) * 100
                logger.info(f"  Coef {idx}: Theoretical: {m_theo:.6f}, Empirical: {m_emp:.6f}, Diff: {diff:.6f} ({pct_diff:.1f}%)")
            else:
                logger.info(f"  Coef {idx}: Theoretical: {m_theo:.6f}, Empirical: {m_emp:.6f}, Diff: {diff:.6f} (inf%)")
        
        logger.info("\nCoefficient standard deviations:")
        for i, idx in enumerate(state_list):
            s_theo = std_theo[i]
            s_emp = std_emp[i] if i < len(std_emp) else 0
            
            # Calculate ratio
            if s_theo > 0:
                ratio = s_emp / s_theo
                logger.info(f"  Coef {idx}: Theoretical: {s_theo:.6f}, Empirical: {s_emp:.6f}, Ratio: {ratio:.4f}")
            else:
                logger.info(f"  Coef {idx}: Theoretical: {s_theo:.6f}, Empirical: {s_emp:.6f}, Ratio: N/A")
        
        # Store analysis results
        coef_analysis[str(state_list)] = {
            'theoretical': {
                'mean': mean_theo,
                'std': std_theo,
                'cov': cov_theo
            },
            'empirical': {
                'mean': mean_emp,
                'std': std_emp,
                'cov': cov_emp,
                'samples': len(coeffs_list)
            }
        }
    
    return coef_analysis

def run_stlsq_with_coefficients(theta, y, threshold):
    """
    Run STLSQ algorithm and track coefficients at each state.
    
    Parameters:
    -----------
    theta : array
        Library matrix
    y : array
        Target dynamics
    threshold : float
        Threshold for sparsification
        
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
    small_indices = np.abs(xi) < threshold
    active_indices = set(np.where(~small_indices)[0].tolist())
    
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
        small_indices = np.abs(xi) < threshold
        xi[small_indices] = 0
        
        # Update active terms
        active_indices = set(np.where(~small_indices)[0].tolist())
        
        # If no active terms left, break
        if len(active_indices) == 0:
            break
        
        # Add the new state to the trajectory if it's different from the last state
        if active_indices != trajectory[-1]:
            trajectory.append(active_indices.copy())
        
        # Recalculate coefficients for active terms
        active_list = list(active_indices)
        theta_active = theta[:, active_list]
        
        try:
            xi_active = np.linalg.lstsq(theta_active, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Use ridge regression if standard approach fails
            ridge_lambda = 1e-6
            gram = theta_active.T @ theta_active
            regularized_gram = gram + ridge_lambda * np.eye(gram.shape[0])
            xi_active = np.linalg.solve(regularized_gram, theta_active.T @ y)
        
        # Update coefficient vector
        xi = np.zeros(n_terms)
        for i, idx in enumerate(active_list):
            xi[idx] = xi_active[i]
        
        # Store coefficients for this state
        coefs_at_states[frozenset(active_indices)] = xi_active.copy()
        
        # Check for convergence
        converged = True
        for idx in active_indices:
            if abs(xi[idx]) < threshold:
                converged = False
                break
    
    return xi, trajectory, coefs_at_states

def analyze_transition_probabilities(model, x_data, n_trials=50):
    """
    Compare theoretical transition probabilities with empirical observations.
    
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
    transition_analysis : dict
        Analysis of transition probabilities
    """
    logger = logging.getLogger('SINDyMarkovModel')
    logger.info("Analyzing transition probabilities in STLSQ simulations")
    
    # Generate valid states
    all_indices = set(range(model.n_terms))
    true_indices = model.normalize_state(model.true_term_indices)
    
    valid_states = []
    for r in range(len(true_indices), len(all_indices) + 1):
        for subset in combinations(all_indices, r):
            subset_set = set(subset)
            if true_indices.issubset(subset_set):
                valid_states.append(subset_set)
    
    # Calculate theoretical transition probabilities
    theoretical_trans = {}
    for from_state in valid_states:
        for to_state in valid_states:
            if to_state.issubset(from_state) and to_state != from_state:
                prob = model.calculate_transition_probability(from_state, to_state)
                key = (str(sorted(list(from_state))), str(sorted(list(to_state))))
                theoretical_trans[key] = prob
    
    # Generate true dynamics
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
    
    # Generate true dynamics
    y_true = true_dynamics(x_data)
    
    # Track observed transitions
    observed_transitions = {}
    from_state_counts = {}
    
    # Run trials
    for trial in range(n_trials):
        # Add noise
        y_noisy = y_true + np.random.normal(0, model.sigma, size=len(x_data))
        
        # Run STLSQ with trajectory tracking
        _, trajectory = model.run_stlsq_with_trajectory(theta, y_noisy)
        
        # Record transitions
        for i in range(len(trajectory) - 1):
            from_state = trajectory[i]
            to_state = trajectory[i + 1]
            
            from_key = str(sorted(list(from_state)))
            to_key = str(sorted(list(to_state)))
            key = (from_key, to_key)
            
            observed_transitions[key] = observed_transitions.get(key, 0) + 1
            from_state_counts[from_key] = from_state_counts.get(from_key, 0) + 1
    
    # Calculate empirical probabilities
    empirical_trans = {}
    for key, count in observed_transitions.items():
        from_key, _ = key
        if from_key in from_state_counts and from_state_counts[from_key] > 0:
            empirical_trans[key] = count / from_state_counts[from_key]
    
    # Compare theoretical vs empirical
    transition_analysis = {}
    
    # Log the comparison
    logger.info("Transition Probability Comparison (Theoretical vs Empirical):")
    
    # Group by from_state for easier reading
    from_states = set(k[0] for k in theoretical_trans.keys()) | set(k[0] for k in empirical_trans.keys())
    
    for from_state in from_states:
        logger.info(f"Transitions from {from_state}:")
        
        # Find all transitions from this state
        from_theoretical = {k[1]: v for k, v in theoretical_trans.items() if k[0] == from_state}
        from_empirical = {k[1]: v for k, v in empirical_trans.items() if k[0] == from_state}
        
        # Combine keys
        to_states = set(from_theoretical.keys()) | set(from_empirical.keys())
        
        for to_state in to_states:
            theo_prob = from_theoretical.get(to_state, 0.0)
            emp_prob = from_empirical.get(to_state, 0.0)
            
            # Calculate discrepancy
            discrepancy = abs(theo_prob - emp_prob)
            
            logger.info(f"  -> {to_state}:")
            logger.info(f"    Theoretical: {theo_prob:.6f}, Empirical: {emp_prob:.6f}, " +
                      f"Discrepancy: {discrepancy:.6f}")
            
            # Store results
            key = (from_state, to_state)
            transition_analysis[key] = {
                'theoretical': theo_prob,
                'empirical': emp_prob,
                'discrepancy': discrepancy,
                'empirical_counts': observed_transitions.get((from_state, to_state), 0)
            }
    
    return transition_analysis