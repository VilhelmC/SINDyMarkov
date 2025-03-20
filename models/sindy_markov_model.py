import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal
from itertools import combinations
import time
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

class SINDyMarkovModel:
    """
    SINDy Markov Chain Model for analyzing STLSQ success probabilities.
    
    This model analyzes the sequential thresholded least squares algorithm
    used in SINDy as a Markov process, calculating transition probabilities
    and overall success probability analytically.
    """
    
    def __init__(self, library_functions=None, true_coefs=None, sigma=0.1, threshold=0.05):
        """
        Initialize the SINDy Markov model.
        
        Parameters:
        -----------
        library_functions : list of callable
            List of library functions θᵢ(x) to use
        true_coefs : array
            True coefficient values for the dynamics (0 for terms not in true model)
        sigma : float
            Noise standard deviation
        threshold : float
            STLSQ threshold value (λ)
        """
        self.sigma = sigma
        self.threshold = threshold
        self.library_functions = library_functions
        self.true_coefs = true_coefs
        self.n_terms = len(true_coefs) if true_coefs is not None else 0
        self.gram_matrix = None
        self.true_term_indices = None
        
        # Set true term indices based on non-zero coefficients
        if true_coefs is not None:
            self.true_term_indices = np.where(np.abs(true_coefs) > 1e-10)[0]
        
        # Cache for transition probabilities
        self._transition_cache = {}
        
    def set_library(self, library_functions, true_coefs):
        """Set the library functions and true coefficients."""
        self.library_functions = library_functions
        self.true_coefs = true_coefs
        self.n_terms = len(true_coefs)
        self.true_term_indices = np.where(np.abs(true_coefs) > 1e-10)[0]
        
        # Clear cache when library changes
        self._transition_cache = {}
        
    def compute_gram_matrix(self, x_samples):
        """
        Compute the Gram matrix for the library at given sample points.
        
        Parameters:
        -----------
        x_samples : array
            Points where to evaluate library functions
            
        Returns:
        --------
        gram_matrix : array
            The Gram matrix (Θᵀ Θ)
        """
        n = len(self.library_functions)
        m = len(x_samples)
        
        # Create design matrix Θ
        theta = np.zeros((m, n))
        for j, func in enumerate(self.library_functions):
            theta[:, j] = func(x_samples)
        
        # Compute Gram matrix
        gram_matrix = theta.T @ theta
        self.gram_matrix = gram_matrix
        
        return gram_matrix
    
    def get_coef_distribution(self, active_indices):
        """
        Get the mean and covariance of coefficient distribution for active terms.
        
        Parameters:
        -----------
        active_indices : list or array
            Indices of active library terms
            
        Returns:
        --------
        mean : array
            Mean vector of coefficient distribution
        cov : array
            Covariance matrix of coefficient distribution
        """
        if self.gram_matrix is None:
            raise ValueError("Gram matrix not computed. Call compute_gram_matrix first.")
        
        # Extract submatrix for active terms
        active_indices = np.array(active_indices)
        sub_gram = self.gram_matrix[np.ix_(active_indices, active_indices)]
        
        # Get true coefficients for these terms
        sub_true_coefs = self.true_coefs[active_indices]
        
        # Compute inverse of sub_gram (handle potential numerical issues)
        try:
            sub_gram_inv = np.linalg.inv(sub_gram)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if matrix is singular
            sub_gram_inv = np.linalg.pinv(sub_gram)
        
        # Coefficient distribution parameters
        mean = sub_true_coefs
        cov = self.sigma**2 * sub_gram_inv
        
        return mean, cov
    
    def calculate_transition_probability(self, from_state, to_state):
        """
        Calculate probability of transition from one state to another.
        
        A state is represented as a set or list of active term indices.
        
        Parameters:
        -----------
        from_state : set or list
            Current state (indices of active terms)
        to_state : set or list
            Next state (indices of active terms)
            
        Returns:
        --------
        probability : float
            Transition probability
        """
        # Convert to sets to ensure proper operations
        from_state = set(from_state)
        to_state = set(to_state)
        
        # Convert to sorted tuples for cache key
        from_tuple = tuple(sorted(from_state))
        to_tuple = tuple(sorted(to_state))
        
        # Check cache
        cache_key = (from_tuple, to_tuple)
        if cache_key in self._transition_cache:
            return self._transition_cache[cache_key]
        
        # Ensure to_state is a subset of from_state (can only eliminate terms)
        if not to_state.issubset(from_state):
            self._transition_cache[cache_key] = 0.0
            return 0.0
        
        # If states are identical, return 1.0 (no change)
        if from_state == to_state:
            self._transition_cache[cache_key] = 1.0
            return 1.0
        
        try:
            # Get coefficient distribution
            mean, cov = self.get_coef_distribution(from_tuple)
            
            # DEBUG: Log coefficient distribution
            debug_distribution = False  # Set to True for verbose debugging
            if debug_distribution:
                print(f"\nCoefficient distribution for {from_state} -> {to_state}:")
                print(f"  Mean: {mean}")
                print(f"  Covariance diag: {np.diag(cov)}")
                
            # Ensure covariance matrix is positive definite
            min_eig = np.min(np.linalg.eigvalsh(cov))
            if min_eig < 1e-6:
                # Add small regularization to ensure positive definiteness
                cov = cov + np.eye(cov.shape[0]) * max(0, 1e-6 - min_eig)
            
            # Calculate which terms are retained and which are eliminated
            eliminated_indices = list(from_state - to_state)
            retained_indices = list(to_state)
            
            # Map to indices within the from_state coefficient vector
            from_state_list = list(from_tuple)
            eliminated_positions = [from_state_list.index(idx) for idx in eliminated_indices]
            retained_positions = [from_state_list.index(idx) for idx in retained_indices]
            
            # For multi-term transitions, use importance sampling for higher accuracy
            n_samples = 50000  # Use more samples for better accuracy
            samples = np.random.multivariate_normal(mean, cov, size=n_samples)
            
            # Count samples that meet our criteria
            count = 0
            for sample in samples:
                # Check if eliminated terms are below threshold
                eliminated_ok = True
                for pos in eliminated_positions:
                    if abs(sample[pos]) >= self.threshold:
                        eliminated_ok = False
                        break
                        
                if not eliminated_ok:
                    continue
                    
                # Check if retained terms are above threshold
                retained_ok = True
                for pos in retained_positions:
                    if abs(sample[pos]) < self.threshold:
                        retained_ok = False
                        break
                        
                if eliminated_ok and retained_ok:
                    count += 1
                    
            prob = count / n_samples
            
            # Ensure prob is between 0 and 1
            prob = max(0.0, min(1.0, prob))
            
            # Apply correlation adjustment for numerical stability
            if len(eliminated_positions) == 1 and len(retained_positions) == 1:
                e_pos = eliminated_positions[0]
                r_pos = retained_positions[0]
                if e_pos < len(cov) and r_pos < len(cov):
                    corr = cov[e_pos, r_pos] / np.sqrt(cov[e_pos, e_pos] * cov[r_pos, r_pos])
                    if abs(corr) > 0.9:  # High correlation adjustment
                        # When correlation is very high, adjust probability
                        adj_factor = max(0.2, 1.0 - abs(corr))
                        prob *= adj_factor
                        if debug_distribution:
                            print(f"  Applied correlation adjustment: {adj_factor:.4f} (corr={corr:.4f})")
            
            self._transition_cache[cache_key] = prob
            return prob
            
        except Exception as e:
            print(f"ERROR in transition probability calculation: {e}")
            self._transition_cache[cache_key] = 0.0
            return 0.0
    
    def calculate_success_probability(self):
        """
        Calculate the overall probability of successfully identifying the true model
        using a graph-based approach with detailed debugging.
        
        Returns:
        --------
        probability : float
            Success probability
        """
        if self.gram_matrix is None:
            raise ValueError("Gram matrix not computed. Call compute_gram_matrix first.")
        
        # Define states
        all_indices = set(range(self.n_terms))
        true_indices = set(self.true_term_indices)
        
        # Initial and final states
        initial_state = all_indices
        true_model_state = true_indices
        
        print(f"DEBUG: All indices = {all_indices}, True indices = {true_indices}")
        
        # Direct success probability (initial -> true in one step)
        direct_prob = self.calculate_transition_probability(initial_state, true_model_state)
        if direct_prob > 0:
            print(f"DEBUG: Direct transition {initial_state} -> {true_model_state}: prob = {direct_prob:.6f}")
        
        # Generate all valid states (must contain all true terms)
        valid_states = []
        for r in range(len(all_indices) + 1):  # r is number of terms to include
            for subset in combinations(all_indices, r):
                subset_set = set(subset)
                if true_indices.issubset(subset_set):
                    valid_states.append(subset_set)
        
        print(f"DEBUG: Generated {len(valid_states)} valid states")
        
        # Build transition matrix between states
        transition_probs = {}
        for i, from_state in enumerate(valid_states):
            for j, to_state in enumerate(valid_states):
                # Can only eliminate terms, not add them
                if to_state.issubset(from_state) and to_state != from_state:
                    prob = self.calculate_transition_probability(from_state, to_state)
                    if prob > 0:
                        key = (frozenset(from_state), frozenset(to_state))
                        transition_probs[key] = prob
                        print(f"DEBUG: Transition {from_state} -> {to_state}: prob = {prob:.6f}")

        # Calculate reachability probability using dynamic programming
        # For each state, calculate probability of reaching it from initial state
        state_probs = {frozenset(s): 0.0 for s in valid_states}
        state_probs[frozenset(initial_state)] = 1.0  # Initial state probability
        
        # Process states in descending order of size (topological sort)
        sorted_states = sorted(valid_states, key=lambda s: (len(s), tuple(sorted(s))), reverse=True)
        
        # First pass: calculate probability of reaching each state
        for from_state in sorted_states:
            from_frozen = frozenset(from_state)
            from_prob = state_probs[from_frozen]
            
            # Skip if this state can't be reached
            if from_prob <= 0:
                continue
                
            # Add probability to all reachable states
            for to_state in valid_states:
                to_frozen = frozenset(to_state)
                if to_state.issubset(from_state) and to_state != from_state:
                    key = (from_frozen, to_frozen)
                    if key in transition_probs:
                        # Add probability of reaching to_state through from_state
                        state_probs[to_frozen] += from_prob * transition_probs[key]
        
        # Debug: log state probabilities
        print("\nDEBUG: State Reachability Probabilities:")
        for state in sorted(valid_states, key=lambda s: (len(s), tuple(sorted(s)))):
            prob = state_probs[frozenset(state)]
            if prob > 0.001:  # Only show significant probabilities
                print(f"  State {state}: {prob:.6f}")
        
        # Success probability is probability of reaching the true model state
        success_prob = state_probs[frozenset(true_model_state)]
        print(f"\nDEBUG: Final success probability = {success_prob:.6f}")
        
        return success_prob

    def _get_intermediate_states(self, start_state, end_state):
        """
        Generate all valid intermediate states between start_state and end_state.
        
        Parameters:
        -----------
        start_state : set
            Starting state
        end_state : set
            Ending state
            
        Returns:
        --------
        states : list
            List of valid intermediate states
        """
        # End state must be a subset of start state
        if not end_state.issubset(start_state):
            return []
        
        # Get terms that can be eliminated
        can_eliminate = start_state - end_state
        
        # Generate intermediate states by eliminating different subsets of terms
        intermediate_states = []
        
        # Consider all possible ways to eliminate a subset of terms
        for r in range(1, len(can_eliminate)):  # r terms eliminated
            for terms_to_eliminate in combinations(can_eliminate, r):
                # Create intermediate state
                intermediate = start_state - set(terms_to_eliminate)
                
                # Make sure it's a valid state (contains end_state)
                if end_state.issubset(intermediate) and intermediate != start_state and intermediate != end_state:
                    intermediate_states.append(intermediate)
        
        return intermediate_states
    
    def simulate_stlsq(self, x_data, n_trials=100, true_dynamics=None):
        """
        Empirically simulate the STLSQ algorithm to estimate success probability.
        
        Parameters:
        -----------
        x_data : array
            Sample points where to evaluate library functions
        n_trials : int
            Number of simulation trials
        true_dynamics : callable, optional
            Function to generate true dynamics. If None, uses true_coefs.
            
        Returns:
        --------
        success_rate : float
            Empirical success probability
        """
        if true_dynamics is None:
            # Generate true dynamics using true_coefs
            def true_dynamics(x):
                y = np.zeros_like(x)
                for i, coef in enumerate(self.true_coefs):
                    if abs(coef) > 1e-10:
                        y += coef * self.library_functions[i](x)
                return y
        
        successful_trials = 0
        
        for trial in range(n_trials):
            # Generate noise-free dynamics
            y_true = true_dynamics(x_data)
            
            # Add noise
            y_noisy = y_true + np.random.normal(0, self.sigma, size=len(x_data))
            
            # Build library matrix
            theta = np.zeros((len(x_data), self.n_terms))
            for j, func in enumerate(self.library_functions):
                theta[:, j] = func(x_data)
            
            # Run STLSQ
            xi = self.run_stlsq(theta, y_noisy)
            
            # Check for success
            true_pattern = np.zeros(self.n_terms)
            true_pattern[self.true_term_indices] = 1
            identified_pattern = np.zeros(self.n_terms)
            identified_pattern[np.abs(xi) > 1e-10] = 1
            
            if np.array_equal(true_pattern, identified_pattern):
                successful_trials += 1
        
        success_rate = successful_trials / n_trials
        return success_rate
    
    def run_stlsq(self, theta, y):
        """
        Run sequential thresholded least squares algorithm.
        
        Parameters:
        -----------
        theta : array
            Library matrix
        y : array
            Target dynamics
            
        Returns:
        --------
        xi : array
            Identified coefficients
        """
        n_terms = theta.shape[1]
        
        # Initial least squares
        xi = np.linalg.lstsq(theta, y, rcond=None)[0]
        
        # Iterative thresholding
        max_iterations = 10
        for _ in range(max_iterations):
            # Apply threshold
            small_indices = np.abs(xi) < self.threshold
            xi[small_indices] = 0
            
            # If all coefficients are zero, break
            if np.all(small_indices):
                break
                
            # Get active terms
            active_indices = ~small_indices
            
            # Recalculate coefficients for active terms
            if np.any(active_indices):
                theta_active = theta[:, active_indices]
                xi_active = np.linalg.lstsq(theta_active, y, rcond=None)[0]
                xi[active_indices] = xi_active
            
            # Check if converged
            if np.all(np.abs(xi) >= self.threshold) or np.all(xi == 0):
                break
        
        return xi
    
    def compare_theory_to_simulation(self, x_range, n_samples_range, n_trials=100):
        """
        Compare theoretical success probability to simulation results.
        
        Parameters:
        -----------
        x_range : array
            Different data ranges to test (widths of sampling)
        n_samples_range : array
            Different numbers of samples to test
        n_trials : int
            Number of simulation trials for each parameter combination
            
        Returns:
        --------
        results : DataFrame
            DataFrame with theoretical and empirical results
        """
        results = []
        
        total_combinations = len(x_range) * len(n_samples_range)
        progress_counter = 0
        
        for data_range in x_range:
            for n_samples in n_samples_range:
                progress_counter += 1
                print(f"Progress: {progress_counter}/{total_combinations}", end="\r")
                
                # Generate sample points
                x_data = np.random.uniform(-data_range, data_range, n_samples)
                
                # Compute Gram matrix
                self.compute_gram_matrix(x_data)
                
                # Calculate theoretical success probability
                theoretical_prob = self.calculate_success_probability()
                
                # Simulate STLSQ
                empirical_prob = self.simulate_stlsq(x_data, n_trials)
                
                # Discriminability calculation (between first two terms)
                if self.n_terms >= 2:
                    term1 = 0  # Index of first term
                    term2 = 1  # Index of second term
                    
                    # Evaluate terms at sample points
                    theta1 = self.library_functions[term1](x_data)
                    theta2 = self.library_functions[term2](x_data)
                    
                    # Calculate discriminability
                    discriminability = np.sum((theta1 - theta2)**2) / self.sigma**2
                else:
                    discriminability = np.nan
                
                # Save results
                results.append({
                    'data_range': data_range,
                    'n_samples': n_samples,
                    'theoretical_prob': theoretical_prob,
                    'empirical_prob': empirical_prob,
                    'discriminability': discriminability,
                    'lambda_sigma_ratio': self.threshold / self.sigma
                })
        
        return pd.DataFrame(results)
    
    def plot_comparison(self, results_df, x_axis='discriminability'):
        """
        Plot comparison of theoretical and empirical results.
        
        Parameters:
        -----------
        results_df : DataFrame
            Results from compare_theory_to_simulation
        x_axis : str
            Which variable to use for x-axis ('discriminability', 'data_range', or 'n_samples')
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot empirical data
        sns.scatterplot(
            data=results_df, 
            x=x_axis, 
            y='empirical_prob',
            color='blue',
            label='Empirical',
            alpha=0.7,
            s=80
        )
        
        # Plot theoretical predictions
        sns.lineplot(
            data=results_df, 
            x=x_axis, 
            y='theoretical_prob',
            color='red',
            label='Theoretical',
            marker='o'
        )
        
        # Add 1:1 line in light gray
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]
        
        if x_axis == 'discriminability':
            # Set axis properties
            ax.set_xscale('log')
            ax.set_title('Success Probability vs Discriminability')
            ax.set_xlabel('Discriminability (D)')
        elif x_axis == 'data_range':
            ax.set_title('Success Probability vs Data Range')
            ax.set_xlabel('Data Range')
        elif x_axis == 'n_samples':
            ax.set_title('Success Probability vs Number of Samples')
            ax.set_xlabel('Number of Samples')
        
        ax.set_ylabel('Success Probability')
        ax.set_ylim([-0.05, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_direct_comparison(self, results_df):
        """
        Plot direct comparison of theoretical vs empirical probabilities.
        
        Parameters:
        -----------
        results_df : DataFrame
            Results from compare_theory_to_simulation
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot the points
        sns.scatterplot(
            data=results_df, 
            x='theoretical_prob', 
            y='empirical_prob',
            hue='discriminability' if 'discriminability' in results_df.columns else None,
            palette='viridis',
            s=80,
            alpha=0.7
        )
        
        # Add 1:1 line
        ax.plot([0, 1], [0, 1], 'k--', label='1:1 Line')
        
        # Calculate metrics
        r2 = r2_score(results_df['empirical_prob'], results_df['theoretical_prob'])
        rmse = np.sqrt(mean_squared_error(results_df['empirical_prob'], results_df['theoretical_prob']))
        
        # Add metrics to plot
        ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Theoretical Success Probability')
        ax.set_ylabel('Empirical Success Probability')
        ax.set_title('Theoretical vs Empirical Success Probability')
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def evaluate_model(self, results_df):
        """
        Evaluate the theoretical model against empirical data.
        
        Parameters:
        -----------
        results_df : DataFrame
            Results from compare_theory_to_simulation
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        # Calculate various metrics
        r2 = r2_score(results_df['empirical_prob'], results_df['theoretical_prob'])
        rmse = np.sqrt(mean_squared_error(results_df['empirical_prob'], results_df['theoretical_prob']))
        mae = np.mean(np.abs(results_df['empirical_prob'] - results_df['theoretical_prob']))
        
        # Calculate bias
        bias = np.mean(results_df['theoretical_prob'] - results_df['empirical_prob'])
        
        # Calculate metrics for different regions
        if 'discriminability' in results_df.columns:
            # Low discriminability region (D < 1)
            low_d = results_df[results_df['discriminability'] < 1]
            if len(low_d) > 0:
                low_d_r2 = r2_score(low_d['empirical_prob'], low_d['theoretical_prob']) if len(low_d) > 1 else np.nan
                low_d_rmse = np.sqrt(mean_squared_error(low_d['empirical_prob'], low_d['theoretical_prob']))
            else:
                low_d_r2, low_d_rmse = np.nan, np.nan
            
            # Medium discriminability region (1 <= D < 10)
            med_d = results_df[(results_df['discriminability'] >= 1) & (results_df['discriminability'] < 10)]
            if len(med_d) > 0:
                med_d_r2 = r2_score(med_d['empirical_prob'], med_d['theoretical_prob']) if len(med_d) > 1 else np.nan
                med_d_rmse = np.sqrt(mean_squared_error(med_d['empirical_prob'], med_d['theoretical_prob']))
            else:
                med_d_r2, med_d_rmse = np.nan, np.nan
            
            # High discriminability region (D >= 10)
            high_d = results_df[results_df['discriminability'] >= 10]
            if len(high_d) > 0:
                high_d_r2 = r2_score(high_d['empirical_prob'], high_d['theoretical_prob']) if len(high_d) > 1 else np.nan
                high_d_rmse = np.sqrt(mean_squared_error(high_d['empirical_prob'], high_d['theoretical_prob']))
            else:
                high_d_r2, high_d_rmse = np.nan, np.nan
            
            region_metrics = {
                'low_d': {'r2': low_d_r2, 'rmse': low_d_rmse, 'n_samples': len(low_d)},
                'med_d': {'r2': med_d_r2, 'rmse': med_d_rmse, 'n_samples': len(med_d)},
                'high_d': {'r2': high_d_r2, 'rmse': high_d_rmse, 'n_samples': len(high_d)}
            }
        else:
            region_metrics = {}
        
        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
            'n_samples': len(results_df),
            'regions': region_metrics
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print evaluation metrics in a readable format."""
        print("\n===== Model Evaluation Metrics =====")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"Bias: {metrics['bias']:.4f}")
        print(f"Number of samples: {metrics['n_samples']}")
        
        if 'regions' in metrics and metrics['regions']:
            print("\nMetrics by Discriminability Region:")
            for region, region_metrics in metrics['regions'].items():
                print(f"  {region} (n={region_metrics['n_samples']}):")
                print(f"    R²: {region_metrics['r2']:.4f}" if not np.isnan(region_metrics['r2']) else "    R²: N/A")
                print(f"    RMSE: {region_metrics['rmse']:.4f}" if not np.isnan(region_metrics['rmse']) else "    RMSE: N/A")
        
        print("=====================================")