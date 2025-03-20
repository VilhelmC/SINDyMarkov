import numpy as np
import logging
import os
from itertools import combinations
from scipy.stats import multivariate_normal

class SINDyMarkovModel:
    """
    SINDy Markov Chain Model for analyzing STLSQ success probabilities.
    
    This model analyzes the sequential thresholded least squares algorithm
    used in SINDy as a Markov process, calculating transition probabilities
    and overall success probability analytically.
    """
    
    def __init__(self, library_functions=None, true_coefs=None, sigma=0.1, threshold=0.05, log_file='sindy_model.log'):
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
        log_file : str
            File path for logging output
        """
        # Set up logging
        self._setup_logging(log_file)
        
        self.logger = logging.getLogger('SINDyMarkovModel')
        self.logger.info("Initializing SINDy Markov Model")
        
        self.sigma = sigma
        self.threshold = threshold
        self.library_functions = library_functions
        self.true_coefs = true_coefs
        self.n_terms = len(true_coefs) if true_coefs is not None else 0
        self.gram_matrix = None
        self.true_term_indices = None
        self.log_gram_det = None
        
        # Set true term indices based on non-zero coefficients
        if true_coefs is not None:
            self.true_term_indices = np.where(np.abs(true_coefs) > 1e-10)[0]
            self.logger.info(f"True term indices: {self.true_term_indices}")
        
        # Cache for transition probabilities
        self._transition_cache = {}
    
    def _setup_logging(self, log_file):
        """Set up logging configuration."""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()  # Also output to console
            ]
        )
    
    def set_library(self, library_functions, true_coefs):
        """Set the library functions and true coefficients."""
        self.logger.info("Setting new library functions and coefficients")
        self.library_functions = library_functions
        self.true_coefs = true_coefs
        self.n_terms = len(true_coefs)
        self.true_term_indices = np.where(np.abs(true_coefs) > 1e-10)[0]
        self.logger.info(f"New true term indices: {self.true_term_indices}")
        
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
        self.logger.info(f"Computing Gram matrix with {len(x_samples)} sample points")
        n = len(self.library_functions)
        m = len(x_samples)
        
        # Create design matrix Θ
        theta = np.zeros((m, n))
        for j, func in enumerate(self.library_functions):
            theta[:, j] = func(x_samples)
        
        # Compute Gram matrix
        gram_matrix = theta.T @ theta
        self.gram_matrix = gram_matrix
        
        # Compute log determinant of Gram matrix
        try:
            self.log_gram_det = np.log(np.linalg.det(gram_matrix))
            self.logger.info(f"Log determinant of Gram matrix: {self.log_gram_det:.4f}")
        except np.linalg.LinAlgError:
            self.log_gram_det = float('-inf')
            self.logger.warning("Failed to compute determinant of Gram matrix (singular matrix)")
        
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
            self.logger.error("Gram matrix not computed. Call compute_gram_matrix first.")
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
            self.logger.warning("Singular sub-Gram matrix detected, using pseudoinverse")
            # Use pseudoinverse if matrix is singular
            sub_gram_inv = np.linalg.pinv(sub_gram)
        
        # Coefficient distribution parameters
        mean = sub_true_coefs
        cov = self.sigma**2 * sub_gram_inv
        
        return mean, cov
    
    @staticmethod
    def normalize_state(state):
        """Convert all elements in a state to standard Python integers."""
        return set(int(idx) for idx in state)

    def calculate_transition_probability(self, from_state, to_state, samples=100000):
        """
        Calculate the probability of transitioning from one state to another.
        
        Parameters:
        -----------
        from_state : set or list
            Current state (indices of active terms)
        to_state : set or list
            Next state (indices of active terms)
        samples : int
            Number of samples for Monte Carlo estimation
                
        Returns:
        --------
        probability : float
            Transition probability
        """
        # Convert to normalized sets to ensure proper operations
        from_state = self.normalize_state(from_state)
        to_state = self.normalize_state(to_state)
        
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
            
            # If there are significant numerical issues, use regularization
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
            
            # Estimate using Monte Carlo
            mc_samples = np.random.multivariate_normal(mean, cov, size=samples)
            
            # Count samples that meet both criteria
            count = 0
            for sample in mc_samples:
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
                    
            # Calculate probability
            prob = count / samples
            
            # Ensure prob is between 0 and 1
            prob = max(0.0, min(1.0, prob))
            
            self._transition_cache[cache_key] = prob
            return prob
            
        except Exception as e:
            self.logger.error(f"Error calculating transition probability: {str(e)}")
            self._transition_cache[cache_key] = 0.0
            return 0.0
    
    def calculate_success_probability(self):
        """
        Calculate the overall probability of successfully identifying the true model
        using a directed acyclic graph (DAG) approach with comprehensive state and transition enumeration.
        
        Returns:
        --------
        probability : float
            Success probability
        """
        if self.gram_matrix is None:
            self.logger.error("Gram matrix not computed. Call compute_gram_matrix first.")
            raise ValueError("Gram matrix not computed. Call compute_gram_matrix first.")
        
        # First ensure all indices are Python integers to avoid type mismatches
        self.true_term_indices = np.array([int(idx) for idx in self.true_term_indices])
        
        # Define states
        all_indices = set(range(self.n_terms))
        true_indices = self.normalize_state(self.true_term_indices)
        
        # Initial and final states
        initial_state = all_indices
        true_model_state = true_indices
        
        self.logger.info("\n" + "="*80)
        self.logger.info("BEGINNING SUCCESS PROBABILITY CALCULATION")
        self.logger.info("="*80)
        self.logger.info(f"All indices = {all_indices}")
        self.logger.info(f"True indices = {true_indices}")
        self.logger.info(f"Target state (true model) = {true_model_state}")
        
        # Generate ALL possible states that contain the true indices
        valid_states = []
        for r in range(len(true_indices), len(all_indices) + 1):
            for subset in combinations(all_indices, r):
                subset_set = set(subset)
                if true_indices.issubset(subset_set):
                    valid_states.append(subset_set)
        
        self.logger.info(f"Generated {len(valid_states)} valid states:")
        for i, state in enumerate(valid_states):
            self.logger.debug(f"  State {i+1}: {state}")
        
        # Calculate all possible transitions between states
        transition_probs = {}
        state_transitions = {frozenset(s): [] for s in valid_states}  # Store outgoing transitions for each state
        
        # Calculate all transitions between valid states
        for from_state in valid_states:
            from_frozen = frozenset(from_state)
            from_state_log = f"Transitions from {from_state}:"
            total_outgoing = 0.0
            
            # Calculate transitions to all subsets of this state that contain true indices
            for to_state in valid_states:
                # Can only eliminate terms, not add them, and must go to a different state
                if to_state.issubset(from_state) and to_state != from_state:
                    prob = self.calculate_transition_probability(from_state, to_state)
                    total_outgoing += prob
                    
                    if prob > 0:
                        key = (from_frozen, frozenset(to_state))
                        transition_probs[key] = prob
                        state_transitions[from_frozen].append((frozenset(to_state), prob))
                        from_state_log += f"\n    -> {to_state}: {prob:.6f}"
            
            # Calculate stopping probability (probability that this is the final state)
            stopping_prob = 1.0 - total_outgoing
            from_state_log += f"\n    -> [STOP]: {stopping_prob:.6f}"
            
            # Highlight if this is the true model state
            if from_state == true_model_state:
                self.logger.info(f"\n>> TRUE MODEL STATE TRANSITIONS <<")
                self.logger.info(from_state_log)
                self.logger.info(f">> TRUE MODEL STATE STOPPING PROBABILITY: {stopping_prob:.6f} <<\n")
            else:
                self.logger.debug(from_state_log)
        
        # Calculate reachability probabilities using dynamic programming on the DAG
        # For each state, calculate the probability of reaching that state from the initial state
        reachability_probs = {frozenset(s): 0.0 for s in valid_states}
        reachability_probs[frozenset(initial_state)] = 1.0  # Initial state has 100% reachability
        
        # Process states in topological order (largest to smallest)
        sorted_states = sorted(valid_states, key=lambda s: (len(s), tuple(sorted(s))), reverse=True)
        
        for from_state in sorted_states:
            from_frozen = frozenset(from_state)
            from_prob = reachability_probs[from_frozen]
            
            # Skip if this state can't be reached
            if from_prob <= 0:
                continue
            
            # Propagate probability to successor states
            for to_frozen, trans_prob in state_transitions[from_frozen]:
                reachability_probs[to_frozen] += from_prob * trans_prob
        
        # Calculate success probabilities for each potential trajectory
        success_probabilities = {}
        for state in valid_states:
            state_frozen = frozenset(state)
            state_str = str(state)  # Use string representation as dictionary key
            reach_prob = reachability_probs[state_frozen]
            
            # Calculate stopping probability for this state
            outgoing_prob = sum(prob for (from_s, _), prob in transition_probs.items() if from_s == state_frozen)
            stopping_prob = 1.0 - outgoing_prob
            
            # If this state has a non-zero probability of being reached and stopped at
            if reach_prob > 0 and stopping_prob > 0:
                # Success means: reached this state AND stopped at this state AND this is the true state
                is_true_state = (state == true_model_state)
                success_prob = reach_prob * stopping_prob * (1.0 if is_true_state else 0.0)
                
                if success_prob > 0:
                    success_probabilities[state_str] = {
                        'reach_prob': reach_prob,
                        'stopping_prob': stopping_prob,
                        'success_prob': success_prob
                    }
        
        # Total success probability is the sum over all success paths
        total_success_prob = sum(info['success_prob'] for info in success_probabilities.values())
        
        # Log reachability probabilities for all states
        self.logger.info("\nState Reachability Probabilities:")
        for state in sorted_states:
            state_frozen = frozenset(state)
            prob = reachability_probs[state_frozen]
            if prob > 0.001:  # Only show significant probabilities
                if state == true_model_state:
                    self.logger.info(f"  >> TRUE MODEL STATE {state}: {prob:.6f} <<")
                else:
                    self.logger.info(f"  State {state}: {prob:.6f}")
        
        # Calculate success probability directly for the true state
        true_state_frozen = frozenset(true_model_state)
        true_state_reach_prob = reachability_probs[true_state_frozen]
        true_state_outgoing_prob = sum(prob for (from_s, _), prob in transition_probs.items() 
                                   if from_s == true_state_frozen)
        true_state_stopping_prob = 1.0 - true_state_outgoing_prob
        direct_success_prob = true_state_reach_prob * true_state_stopping_prob
        
        self.logger.info("\n" + "-"*80)
        self.logger.info("SUCCESS PROBABILITY CALCULATION")
        self.logger.info("-"*80)
        self.logger.info(f"Method 1 - Direct Calculation:")
        self.logger.info(f"  Probability of reaching true state:    {true_state_reach_prob:.6f}")
        self.logger.info(f"  Probability of stopping at true state: {true_state_stopping_prob:.6f}")
        self.logger.info(f"  Success probability = {true_state_reach_prob:.6f} × {true_state_stopping_prob:.6f} = {direct_success_prob:.6f}")
        
        # Check that the two methods give the same result (they should)
        if abs(total_success_prob - direct_success_prob) > 1e-10:
            self.logger.warning(f"WARNING: Different success probability calculations don't match!")
            self.logger.warning(f"Method 1 (direct): {direct_success_prob}")
            self.logger.warning(f"Method 2 (summation): {total_success_prob}")
        
        self.logger.info("="*80 + "\n")
        
        return direct_success_prob

    def run_stlsq_with_trajectory(self, theta, y):
        """
        Run sequential thresholded least squares algorithm and track the trajectory through state space.
        
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
        trajectory : list
            List of sets representing the states visited during the algorithm
        """
        n_terms = theta.shape[1]
        
        # Initial least squares
        xi = np.linalg.lstsq(theta, y, rcond=None)[0]
        
        # Initialize with all terms active
        small_indices = np.abs(xi) < self.threshold
        active_indices = self.normalize_state(np.where(~small_indices)[0])
        
        # Track trajectory through state space
        trajectory = [active_indices.copy()]
        
        # Iterative thresholding
        max_iterations = 10
        converged = False
        
        for _ in range(max_iterations):
            if converged:
                break
                
            # Apply threshold
            small_indices = np.abs(xi) < self.threshold
            xi[small_indices] = 0
            
            # Update active terms
            active_indices = self.normalize_state(np.where(~small_indices)[0])
            
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
            
            # Check for convergence
            converged = True
            for idx in active_indices:
                if abs(xi[idx]) < self.threshold:
                    converged = False
                    break
        
        return xi, trajectory
    
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
            active_indices = self.normalize_state(np.where(~small_indices)[0])
            
            # Recalculate coefficients for active terms
            if active_indices:
                active_list = list(active_indices)
                theta_active = theta[:, active_list]
                xi_active = np.linalg.lstsq(theta_active, y, rcond=None)[0]
                
                # Update coefficient vector
                xi = np.zeros(n_terms)
                for i, idx in enumerate(active_list):
                    xi[idx] = xi_active[i]
            
            # Check if converged
            if np.all(np.abs(xi) >= self.threshold) or np.all(xi == 0):
                break
        
        return xi
    
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
        self.logger.info(f"\nRunning STLSQ simulation with {n_trials} trials...")
        
        if true_dynamics is None:
            # Generate true dynamics using true_coefs
            def true_dynamics(x):
                y = np.zeros_like(x)
                for i, coef in enumerate(self.true_coefs):
                    if abs(coef) > 1e-10:
                        y += coef * self.library_functions[i](x)
                return y
        
        successful_trials = 0
        trial_results = []
        
        # Track paths through state space
        trajectory_counts = {}
        
        for trial in range(n_trials):
            # Generate noise-free dynamics
            y_true = true_dynamics(x_data)
            
            # Add noise
            y_noisy = y_true + np.random.normal(0, self.sigma, size=len(x_data))
            
            # Build library matrix
            theta = np.zeros((len(x_data), self.n_terms))
            for j, func in enumerate(self.library_functions):
                theta[:, j] = func(x_data)
            
            # Run STLSQ with trajectory tracking
            xi, trajectory = self.run_stlsq_with_trajectory(theta, y_noisy)
            
            # Convert trajectory to a string representation for counting
            trajectory_str = ' -> '.join([str(sorted(list(state))) for state in trajectory])
            trajectory_counts[trajectory_str] = trajectory_counts.get(trajectory_str, 0) + 1
            
            # Check for success
            true_pattern = np.zeros(self.n_terms)
            true_pattern[self.true_term_indices] = 1
            identified_pattern = np.zeros(self.n_terms)
            identified_pattern[np.abs(xi) > 1e-10] = 1
            
            # Determine if this is a success
            is_success = np.array_equal(true_pattern, identified_pattern)
            if is_success:
                successful_trials += 1
            
            # Store details about this trial
            trial_results.append({
                'trial': trial + 1,
                'success': is_success,
                'identified_terms': np.where(identified_pattern == 1)[0].tolist(),
                'trajectory': trajectory
            })
        
        success_rate = successful_trials / n_trials
        
        # Log detailed simulation results
        self.logger.info(f"\nSTLSQ Simulation Results ({successful_trials}/{n_trials} successful, {success_rate:.4f})")
        
        # Log detailed results for the first few trials
        num_to_show = min(5, n_trials)
        self.logger.info(f"\nSample of {num_to_show} trials:")
        
        for i in range(num_to_show):
            trial = trial_results[i]
            traj_str = ' -> '.join([str(sorted(list(state))) for state in trial['trajectory']])
            self.logger.info(f"  Trial {trial['trial']}: {'SUCCESS' if trial['success'] else 'FAILURE'}, "
                        f"Identified terms: {trial['identified_terms']}")
            self.logger.info(f"    Trajectory: {traj_str}")
        
        # Analyze trajectories
        self.logger.info("\nTrajectory Analysis:")
        
        # Sort trajectories by frequency
        sorted_trajectories = sorted(trajectory_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate what percentage of trials each trajectory represents
        total_trials = sum(trajectory_counts.values())
        
        # Report the most common trajectories
        num_trajectories = min(5, len(sorted_trajectories))
        self.logger.info(f"Top {num_trajectories} trajectories:")
        
        for traj, count in sorted_trajectories[:num_trajectories]:
            percentage = (count / total_trials) * 100
            self.logger.info(f"  {traj}: {count} occurrences ({percentage:.1f}% of trials)")
        
        # If there were failures, analyze what went wrong
        if successful_trials < n_trials:
            # Count identified terms in failures
            failure_trials = [t for t in trial_results if not t['success']]
            term_counts = {}
            for trial in failure_trials:
                terms_str = str(sorted(trial['identified_terms']))
                term_counts[terms_str] = term_counts.get(terms_str, 0) + 1
            
            # Sort by frequency
            top_failures = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Log the most common failure patterns
            self.logger.info("\nMost common failure patterns:")
            for terms_str, count in top_failures[:3]:  # Show top 3 patterns
                percentage = (count / len(failure_trials)) * 100
                self.logger.info(f"  {terms_str}: {count} occurrences "
                            f"({percentage:.1f}% of failures)")
        
        return success_rate