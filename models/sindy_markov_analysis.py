"""
Main module for performing SINDy Markov Chain analysis.

This module provides the complete workflow for analyzing success probabilities
in the SINDy algorithm with Sequential Thresholded Least Squares (STLSQ).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import logging
from itertools import combinations
import json

# Import custom modules
from models.sindy_markov_model import SINDyMarkovModel
from models.logger_utils import setup_logging, get_logger, log_to_markdown
from models.state_utils import normalize_state, generate_valid_states
from models.simulation_utils import simulate_stlsq, simulate_stlsq_adaptive
from models.library_utils import (
    compute_discriminability, 
    compute_gram_discriminability,
    analyze_library_correlations,
    plot_library_correlations,
    calculate_term_significance
)

class SINDyMarkovAnalysis:
    """
    Class for performing comprehensive SINDy Markov Chain analysis.
    """
    
    def __init__(self, library_functions, true_coefs, sigma=0.1, threshold=0.05, 
                 log_file='logs/sindy_markov_analysis.log'):
        """
        Initialize the analysis.
        
        Parameters:
        -----------
        library_functions : list
            List of library functions
        true_coefs : array
            True coefficient values
        sigma : float
            Noise standard deviation
        threshold : float
            STLSQ threshold
        log_file : str
            Path to log file
        """
        # Setup logging
        self.logger = setup_logging(log_file)
        self.logger.info("Initializing SINDy Markov Chain Analysis")
        
        # Create the model
        self.model = SINDyMarkovModel(library_functions, true_coefs, sigma, threshold, log_file)
        
        # Set parameters
        self.library_functions = library_functions
        self.true_coefs = true_coefs
        self.sigma = sigma
        self.threshold = threshold
        self.log_file = log_file
        
        # Initialize results storage
        self.results = {}
        
        # Create output directories
        os.makedirs('results', exist_ok=True)
        os.makedirs('figures', exist_ok=True)
    
    def prepare_data(self, x_range, n_samples):
        """
        Prepare data for analysis.
        
        Parameters:
        -----------
        x_range : float
            Range of data points (-x_range to x_range)
        n_samples : int
            Number of sample points
            
        Returns:
        --------
        x_data : array
            Generated sample points
        """
        self.logger.info(f"Preparing data with range {x_range} and {n_samples} samples")
        
        # Generate sample points
        x_data = np.random.uniform(-x_range, x_range, n_samples)
        
        # Compute the Gram matrix
        self.model.compute_gram_matrix(x_data)
        
        # Compute discriminability
        theta = np.zeros((n_samples, len(self.library_functions)))
        for j, func in enumerate(self.library_functions):
            theta[:, j] = func(x_data)
            
        self.discriminability = compute_discriminability(theta, self.sigma)
        
        # Log the discriminability between true and false terms
        true_indices = self.model.true_term_indices
        false_indices = np.array([i for i in range(len(self.library_functions)) if i not in true_indices])
        
        if len(true_indices) > 0 and len(false_indices) > 0:
            self.logger.info("Discriminability between true and false terms:")
            for i in true_indices:
                for j in false_indices:
                    self.logger.info(f"  D({i},{j}) = {self.discriminability[i, j]:.4f}")
        
        # Calculate library correlations
        corr_df, _ = analyze_library_correlations(x_data, self.library_functions)
        self.correlations = corr_df
        
        # Log high correlations
        high_corr_threshold = 0.7
        for i in range(len(corr_df)):
            for j in range(i+1, len(corr_df)):
                if abs(corr_df.iloc[i, j]) > high_corr_threshold:
                    self.logger.info(f"High correlation detected: terms {i} and {j}: {corr_df.iloc[i, j]:.4f}")
        
        return x_data
    
    def analyze_transition_probabilities(self, x_data, n_trials=30):
        """
        Analyze theoretical vs empirical transition probabilities.
        
        Parameters:
        -----------
        x_data : array
            Sample points
        n_trials : int
            Number of trials for empirical probabilities
            
        Returns:
        --------
        transition_analysis : dict
            Analysis results for transitions
        """
        self.logger.info(f"Analyzing transition probabilities with {n_trials} trials")
        
        # Calculate theoretical transition probabilities
        all_indices = set(range(len(self.library_functions)))
        true_indices = normalize_state(self.model.true_term_indices)
        
        # Generate valid states
        valid_states = generate_valid_states(true_indices, all_indices)
        self.logger.info(f"Analyzing {len(valid_states)} valid states")
        
        # Calculate transitions for a subset of states (to keep computation manageable)
        if len(valid_states) > 10:
            self.logger.info("Too many states, selecting a subset for detailed analysis")
            selected_states = [all_indices, true_indices]  # Initial and target states
            
            # Add some intermediate states
            max_add = min(8, len(valid_states) - 2)
            for state in valid_states:
                if len(selected_states) >= max_add + 2:
                    break
                if state not in [all_indices, true_indices]:
                    selected_states.append(state)
        else:
            selected_states = valid_states
        
        # Calculate theoretical transitions
        theoretical_transitions = {}
        for from_state in selected_states:
            for to_state in selected_states:
                if to_state.issubset(from_state) and to_state != from_state:
                    prob = self.model.calculate_transition_probability(from_state, to_state)
                    key = (str(from_state), str(to_state))
                    theoretical_transitions[key] = prob
                    self.logger.info(f"Transition {from_state} -> {to_state}: theoretical prob = {prob:.6f}")
        
        # Simulate to get empirical transitions
        empirical_transitions = self._empirical_transition_analysis(x_data, selected_states, n_trials)
        
        # Compare theoretical vs empirical
        comparison = {}
        for key in theoretical_transitions:
            theo_prob = theoretical_transitions.get(key, 0.0)
            emp_prob = empirical_transitions.get(key, 0.0)
            discrepancy = abs(theo_prob - emp_prob)
            
            comparison[key] = {
                'theoretical': theo_prob,
                'empirical': emp_prob,
                'discrepancy': discrepancy
            }
            
            self.logger.info(f"Transition {key} - Theoretical: {theo_prob:.4f}, Empirical: {emp_prob:.4f}, " +
                           f"Discrepancy: {discrepancy:.4f}")
        
        # Store results
        transition_analysis = {
            'theoretical': theoretical_transitions,
            'empirical': empirical_transitions,
            'comparison': comparison
        }
        
        return transition_analysis
    
    def _empirical_transition_analysis(self, x_data, states, n_trials):
        """
        Run STLSQ simulations to gather empirical transition statistics.
        
        Parameters:
        -----------
        x_data : array
            Sample points
        states : list
            List of states to analyze
        n_trials : int
            Number of trials
            
        Returns:
        --------
        empirical_transitions : dict
            Dictionary of empirical transition probabilities
        """
        # Generate true dynamics
        def true_dynamics(x):
            y = np.zeros_like(x)
            for i, coef in enumerate(self.true_coefs):
                if abs(coef) > 1e-10:
                    y += coef * self.library_functions[i](x)
            return y
        
        # Build library matrix
        theta = np.zeros((len(x_data), len(self.library_functions)))
        for j, func in enumerate(self.library_functions):
            theta[:, j] = func(x_data)
        
        # Generate true dynamics
        y_true = true_dynamics(x_data)
        
        # Track transitions
        observed_transitions = {}
        from_state_counts = {}
        
        for trial in range(n_trials):
            # Add noise
            y_noisy = y_true + np.random.normal(0, self.sigma, size=len(x_data))
            
            # Run STLSQ with trajectory tracking
            _, trajectory = self.model.run_stlsq_with_trajectory(theta, y_noisy)
            
            # Record transitions
            for i in range(len(trajectory) - 1):
                from_state = trajectory[i]
                to_state = trajectory[i + 1]
                key = (str(from_state), str(to_state))
                observed_transitions[key] = observed_transitions.get(key, 0) + 1
                
                # Track from_state counts
                from_key = str(from_state)
                from_state_counts[from_key] = from_state_counts.get(from_key, 0) + 1
        
        # Calculate empirical probabilities
        empirical_transitions = {}
        for (from_key, to_key), count in observed_transitions.items():
            if from_key in from_state_counts:
                total = from_state_counts[from_key]
                empirical_transitions[(from_key, to_key)] = count / total
        
        return empirical_transitions
    
    def analyze_success_probability(self, x_data, n_trials=100, adaptive=False, 
                                   max_trials=500, confidence=0.95, margin=0.05, min_trials=30):
        """
        Analyze theoretical and empirical success probabilities.
        
        Parameters:
        -----------
        x_data : array
            Sample points
        n_trials : int
            Number of simulation trials (if not using adaptive)
        adaptive : bool
            Whether to use adaptive trial count determination
        max_trials : int
            Maximum number of trials when using adaptive approach
        confidence : float
            Confidence level for adaptive approach
        margin : float
            Maximum margin of error for adaptive approach
        min_trials : int
            Minimum number of trials when using adaptive approach
            
        Returns:
        --------
        success_analysis : dict
            Analysis results for success probabilities
        """
        self.logger.info("Analyzing success probability")
        
        # Calculate theoretical success probability
        theoretical_prob = self.model.calculate_success_probability()
        self.logger.info(f"Theoretical success probability: {theoretical_prob:.6f}")
        
        # Calculate empirical success probability
        start_time = time.time()
        
        if adaptive:
            self.logger.info(f"Running adaptive STLSQ simulation with max {max_trials} trials, " +
                           f"{confidence*100:.0f}% confidence, {margin*100:.1f}% margin")
            
            empirical_prob, trials_used = simulate_stlsq_adaptive(
                self.model, 
                x_data, 
                max_trials=max_trials,
                confidence=confidence,
                margin=margin,
                min_trials=min_trials
            )
            
            self.logger.info(f"Adaptive simulation used {trials_used} trials")
            
            # Calculate margin of error using Wilson score interval
            z_score = stats.norm.ppf((1 + confidence) / 2)
            z2 = z_score ** 2
            factor = z2 / (2 * trials_used)
            
            wilson_p = (empirical_prob + factor) / (1 + 2 * factor)
            wilson_error_margin = z_score * np.sqrt(
                empirical_prob * (1 - empirical_prob) / trials_used + factor / 4
            ) / (1 + 2 * factor)
            
            self.logger.info(f"Empirical success probability: {empirical_prob:.6f} ± {wilson_error_margin:.6f}")
        else:
            self.logger.info(f"Running fixed STLSQ simulation with {n_trials} trials")
            
            empirical_prob = simulate_stlsq(self.model, x_data, n_trials)
            
            # Calculate simple margin of error
            margin_of_error = 1.96 * np.sqrt(empirical_prob * (1 - empirical_prob) / n_trials)
            self.logger.info(f"Empirical success probability: {empirical_prob:.6f} ± {margin_of_error:.6f}")
            
            # For consistency with adaptive approach
            trials_used = n_trials
            wilson_error_margin = margin_of_error
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")
        
        # Compare theoretical and empirical results
        discrepancy = abs(theoretical_prob - empirical_prob)
        self.logger.info(f"Discrepancy between theoretical and empirical: {discrepancy:.6f}")
        
        if discrepancy > 2 * wilson_error_margin:
            self.logger.warning(f"Significant discrepancy detected (> 2x margin of error)")
        
        # Store results
        success_analysis = {
            'theoretical': theoretical_prob,
            'empirical': empirical_prob,
            'discrepancy': discrepancy,
            'trials_used': trials_used,
            'margin_of_error': wilson_error_margin,
            'simulation_time': elapsed_time
        }
        
        # Add to overall results
        self.results['success_probability'] = success_analysis
        
        return success_analysis
    
    def analyze_data_range_effect(self, n_samples=200, x_ranges=None, n_trials=50):
        """
        Analyze the effect of data range on success probability.
        
        Parameters:
        -----------
        n_samples : int
            Number of sample points
        x_ranges : list
            List of data ranges to test
        n_trials : int
            Number of trials per configuration
            
        Returns:
        --------
        range_analysis : dict
            Analysis results for different data ranges
        """
        if x_ranges is None:
            x_ranges = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        self.logger.info(f"Analyzing effect of data range: {x_ranges}")
        
        results = []
        
        for x_range in x_ranges:
            self.logger.info(f"Testing data range: {x_range}")
            
            # Generate data
            x_data = self.prepare_data(x_range, n_samples)
            
            # Calculate log determinant of Gram matrix
            log_gram_det = self.model.log_gram_det
            
            # Calculate discriminability
            avg_discriminability = np.mean(self.discriminability[self.discriminability > 0])
            
            # Calculate theoretical success probability
            theoretical_prob = self.model.calculate_success_probability()
            
            # Calculate empirical success probability
            empirical_prob = simulate_stlsq(self.model, x_data, n_trials)
            
            # Calculate discrepancy
            discrepancy = abs(theoretical_prob - empirical_prob)
            
            # Store results
            results.append({
                'x_range': x_range,
                'log_gram_det': log_gram_det,
                'avg_discriminability': avg_discriminability,
                'theoretical_prob': theoretical_prob,
                'empirical_prob': empirical_prob,
                'discrepancy': discrepancy
            })
            
            self.logger.info(f"Results - Theoretical: {theoretical_prob:.4f}, Empirical: {empirical_prob:.4f}, " +
                           f"Discrepancy: {discrepancy:.4f}")
        
        # Convert to DataFrame
        range_df = pd.DataFrame(results)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot success probability vs data range
        plt.subplot(1, 2, 1)
        plt.plot(range_df['x_range'], range_df['theoretical_prob'], 'ro-', label='Theoretical')
        plt.plot(range_df['x_range'], range_df['empirical_prob'], 'bo-', label='Empirical')
        plt.xlabel('Data Range')
        plt.ylabel('Success Probability')
        plt.title('Success Probability vs Data Range')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot success probability vs log_gram_det
        plt.subplot(1, 2, 2)
        plt.plot(range_df['log_gram_det'], range_df['theoretical_prob'], 'ro-', label='Theoretical')
        plt.plot(range_df['log_gram_det'], range_df['empirical_prob'], 'bo-', label='Empirical')
        plt.xlabel('Log Determinant of Gram Matrix')
        plt.ylabel('Success Probability')
        plt.title('Success Probability vs Log Gram Determinant')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figures/data_range_effect.png', dpi=300, bbox_inches='tight')
        
        # Store results
        range_analysis = {
            'dataframe': range_df,
            'figure_path': 'figures/data_range_effect.png'
        }
        
        # Add to overall results
        self.results['data_range_effect'] = range_analysis
        
        return range_analysis
    
    def analyze_sample_size_effect(self, x_range=1.0, n_samples_list=None, n_trials=50):
        """
        Analyze the effect of sample size on success probability.
        
        Parameters:
        -----------
        x_range : float
            Data range
        n_samples_list : list
            List of sample sizes to test
        n_trials : int
            Number of trials per configuration
            
        Returns:
        --------
        sample_analysis : dict
            Analysis results for different sample sizes
        """
        if n_samples_list is None:
            n_samples_list = [50, 100, 200, 500, 1000]
        
        self.logger.info(f"Analyzing effect of sample size: {n_samples_list}")
        
        results = []
        
        for n_samples in n_samples_list:
            self.logger.info(f"Testing sample size: {n_samples}")
            
            # Generate data
            x_data = self.prepare_data(x_range, n_samples)
            
            # Calculate log determinant of Gram matrix
            log_gram_det = self.model.log_gram_det
            
            # Calculate discriminability
            avg_discriminability = np.mean(self.discriminability[self.discriminability > 0])
            
            # Calculate theoretical success probability
            theoretical_prob = self.model.calculate_success_probability()
            
            # Calculate empirical success probability
            empirical_prob = simulate_stlsq(self.model, x_data, n_trials)
            
            # Calculate discrepancy
            discrepancy = abs(theoretical_prob - empirical_prob)
            
            # Store results
            results.append({
                'n_samples': n_samples,
                'log_gram_det': log_gram_det,
                'avg_discriminability': avg_discriminability,
                'theoretical_prob': theoretical_prob,
                'empirical_prob': empirical_prob,
                'discrepancy': discrepancy
            })
            
            self.logger.info(f"Results - Theoretical: {theoretical_prob:.4f}, Empirical: {empirical_prob:.4f}, " +
                           f"Discrepancy: {discrepancy:.4f}")
        
        # Convert to DataFrame
        sample_df = pd.DataFrame(results)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot success probability vs sample size
        plt.subplot(1, 2, 1)
        plt.plot(sample_df['n_samples'], sample_df['theoretical_prob'], 'ro-', label='Theoretical')
        plt.plot(sample_df['n_samples'], sample_df['empirical_prob'], 'bo-', label='Empirical')
        plt.xlabel('Sample Size')
        plt.ylabel('Success Probability')
        plt.title('Success Probability vs Sample Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot success probability vs log_gram_det
        plt.subplot(1, 2, 2)
        plt.plot(sample_df['log_gram_det'], sample_df['theoretical_prob'], 'ro-', label='Theoretical')
        plt.plot(sample_df['log_gram_det'], sample_df['empirical_prob'], 'bo-', label='Empirical')
        plt.xlabel('Log Determinant of Gram Matrix')
        plt.ylabel('Success Probability')
        plt.title('Success Probability vs Log Gram Determinant')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figures/sample_size_effect.png', dpi=300, bbox_inches='tight')
        
        # Store results
        sample_analysis = {
            'dataframe': sample_df,
            'figure_path': 'figures/sample_size_effect.png'
        }
        
        # Add to overall results
        self.results['sample_size_effect'] = sample_analysis
        
        return sample_analysis
    
    def analyze_lambda_sigma_effect(self, x_range=1.0, n_samples=200, lambda_sigma_ratios=None, n_trials=50):
        """
        Analyze the effect of lambda/sigma ratio on success probability.
        
        Parameters:
        -----------
        x_range : float
            Data range
        n_samples : int
            Number of sample points
        lambda_sigma_ratios : list
            List of lambda/sigma ratios to test
        n_trials : int
            Number of trials per configuration
            
        Returns:
        --------
        lambda_sigma_analysis : dict
            Analysis results for different lambda/sigma ratios
        """
        if lambda_sigma_ratios is None:
            lambda_sigma_ratios = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        
        self.logger.info(f"Analyzing effect of lambda/sigma ratio: {lambda_sigma_ratios}")
        
        # Generate data once (same for all ratios)
        x_data = np.random.uniform(-x_range, x_range, n_samples)
        
        # Build library matrix once
        theta = np.zeros((n_samples, len(self.library_functions)))
        for j, func in enumerate(self.library_functions):
            theta[:, j] = func(x_data)
        
        # Generate true dynamics once
        def true_dynamics(x):
            y = np.zeros_like(x)
            for i, coef in enumerate(self.true_coefs):
                if abs(coef) > 1e-10:
                    y += coef * self.library_functions[i](x)
            return y
        
        y_true = true_dynamics(x_data)
        
        results = []
        
        for ratio in lambda_sigma_ratios:
            self.logger.info(f"Testing lambda/sigma ratio: {ratio}")
            
            # Set new threshold
            threshold = ratio * self.sigma
            
            # Create a new model with this threshold
            model = SINDyMarkovModel(
                self.library_functions,
                self.true_coefs,
                self.sigma,
                threshold,
                self.log_file
            )
            
            # Compute Gram matrix
            model.compute_gram_matrix(x_data)
            
            # Calculate log determinant of Gram matrix
            log_gram_det = model.log_gram_det
            
            # Calculate theoretical success probability
            theoretical_prob = model.calculate_success_probability()
            
            # Calculate empirical success probability
            success_count = 0
            
            for _ in range(n_trials):
                # Add noise
                y_noisy = y_true + np.random.normal(0, self.sigma, size=n_samples)
                
                # Run STLSQ
                xi = model.run_stlsq(theta, y_noisy)
                
                # Check for success
                true_pattern = np.zeros(len(self.library_functions))
                true_pattern[model.true_term_indices] = 1
                identified_pattern = np.zeros(len(self.library_functions))
                identified_pattern[np.abs(xi) > 1e-10] = 1
                
                # Determine if this is a success
                is_success = np.array_equal(true_pattern, identified_pattern)
                if is_success:
                    success_count += 1
            
            empirical_prob = success_count / n_trials
            
            # Calculate discrepancy
            discrepancy = abs(theoretical_prob - empirical_prob)
            
            # Store results
            results.append({
                'lambda_sigma_ratio': ratio,
                'threshold': threshold,
                'log_gram_det': log_gram_det,
                'theoretical_prob': theoretical_prob,
                'empirical_prob': empirical_prob,
                'discrepancy': discrepancy
            })
            
            self.logger.info(f"Results - Theoretical: {theoretical_prob:.4f}, Empirical: {empirical_prob:.4f}, " +
                           f"Discrepancy: {discrepancy:.4f}")
        
        # Convert to DataFrame
        lambda_sigma_df = pd.DataFrame(results)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot success probability vs lambda/sigma ratio
        plt.plot(lambda_sigma_df['lambda_sigma_ratio'], lambda_sigma_df['theoretical_prob'], 'ro-', label='Theoretical')
        plt.plot(lambda_sigma_df['lambda_sigma_ratio'], lambda_sigma_df['empirical_prob'], 'bo-', label='Empirical')
        plt.xlabel('λ/σ Ratio')
        plt.ylabel('Success Probability')
        plt.title('Success Probability vs λ/σ Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figures/lambda_sigma_effect.png', dpi=300, bbox_inches='tight')
        
        # Store results
        lambda_sigma_analysis = {
            'dataframe': lambda_sigma_df,
            'figure_path': 'figures/lambda_sigma_effect.png'
        }
        
        # Add to overall results
        self.results['lambda_sigma_effect'] = lambda_sigma_analysis
        
        return lambda_sigma_analysis
    
    def save_results(self, filename='results/sindy_markov_analysis_results.json'):
        """
        Save all analysis results to a JSON file.
        
        Parameters:
        -----------
        filename : str
            Path to save the results
            
        Returns:
        --------
        saved_path : str
            Path where results were saved
        """
        # Convert results to a serializable format
        serializable_results = {}
        
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_dict = {}
                for k, v in value.items():
                    if isinstance(v, pd.DataFrame):
                        serializable_dict[k] = v.to_dict(orient='records')
                    elif isinstance(v, np.ndarray):
                        serializable_dict[k] = v.tolist()
                    elif isinstance(v, (np.integer, np.floating)):
                        serializable_dict[k] = float(v)
                    else:
                        serializable_dict[k] = v
                serializable_results[key] = serializable_dict
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {filename}")
        
        # Also create a markdown log
        log_md = os.path.splitext(self.log_file)[0] + '.md'
        log_to_markdown(self.log_file, log_md)
        
        return filename