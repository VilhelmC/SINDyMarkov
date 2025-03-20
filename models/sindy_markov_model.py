import numpy as np
from itertools import combinations
from scipy.stats import multivariate_normal, norm

# Import centralized logging utilities at the module level
from models.logging_config import setup_logging, get_logger
from models.logging_config import bold, green, yellow, red, cyan, header, section
from models.logging_config import bold_green, bold_yellow, bold_red

class SINDyMarkovModel:
    """
    SINDy Markov Chain Model for analyzing STLSQ success probabilities.
    
    This model analyzes the sequential thresholded least squares algorithm
    used in SINDy as a Markov process, calculating transition probabilities
    and overall success probability analytically.
    """
    
    def __init__(self, library_functions=None, true_coefs=None, sigma=0.1, threshold=0.05, log_file='logs/sindy_model.log'):
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
        setup_logging(log_file)
        self.logger = get_logger('SINDyMarkovModel')
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
    
    # Removed _setup_logging as it's now handled by the centralized system
    
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
        """
        Convert all elements in a state to standard Python integers.
        
        Parameters:
        -----------
        state : set, list, or array
            State containing indices
            
        Returns:
        --------
        normalized_state : set
            Set containing standard Python integers
        """
        if state is None:
            return set()
        
        # Ensure we're working with an iterable
        try:
            return set(int(idx) for idx in state)
        except TypeError:
            # Handle case of a single value
            return {int(state)}

    def calculate_transition_probability(self, from_state, to_state, samples=100000, diagnose=False):
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
        diagnose : bool
            Whether to log diagnostic information
                    
        Returns:
        --------
        probability : float
            Transition probability
        """
        # Convert to normalized sets to ensure proper operations
        from_state = self.normalize_state(from_state)
        to_state = self.normalize_state(to_state)
        
        # Convert to sorted tuples for cache key (standard Python types)
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
                if diagnose:
                    self.logger.debug(f"Applied regularization to covariance matrix, min eigenvalue was {min_eig:.2e}")
            
            # Calculate which terms are retained and which are eliminated
            eliminated_indices = list(from_state - to_state)
            retained_indices = list(to_state)
            
            # Map to indices within the from_state coefficient vector
            from_state_list = list(from_tuple)
            eliminated_positions = [from_state_list.index(idx) for idx in eliminated_indices]
            retained_positions = [from_state_list.index(idx) for idx in retained_indices]
            
            if diagnose:
                self.logger.debug(f"Transition {from_state} -> {to_state}:")
                self.logger.debug(f"  Coefficient mean: {mean}")
                self.logger.debug(f"  Eliminated indices: {eliminated_indices} (positions: {eliminated_positions})")
                self.logger.debug(f"  Retained indices: {retained_indices} (positions: {retained_positions})")
                
                # Calculate theoretical probabilities for each term independently
                self.logger.debug("  Independent term probabilities:")
                for i, idx in enumerate(from_state_list):
                    pos = i  # Position in the coefficient vector
                    term_mean = mean[pos]
                    term_std = np.sqrt(cov[pos, pos])
                    
                    # Calculate probability this term is below threshold
                    below_prob = norm.cdf((self.threshold - term_mean) / term_std) - norm.cdf((-self.threshold - term_mean) / term_std)
                    
                    # Whether this term should be eliminated or retained
                    should_eliminate = idx in eliminated_indices
                    
                    if should_eliminate:
                        self.logger.debug(f"    Term {idx}: P(|coef| < {self.threshold}) = {below_prob:.6f} (should be eliminated)")
                    else:
                        self.logger.debug(f"    Term {idx}: P(|coef| ≥ {self.threshold}) = {1-below_prob:.6f} (should be retained)")
            
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
            
            # Add to cache
            self._transition_cache[cache_key] = prob
            
            # Log discrepancy between independent probabilities and joint probability if significant
            if diagnose and abs(prob) > 1e-10:
                # Calculate independent probability estimate (assumes independence)
                indep_prob = 1.0
                for i, idx in enumerate(from_state_list):
                    pos = i  # Position in the coefficient vector
                    term_mean = mean[pos]
                    term_std = np.sqrt(cov[pos, pos])
                    
                    # Calculate probability this term is below or above threshold as needed
                    if idx in eliminated_indices:
                        # Term should be eliminated, so we want |coef| < threshold
                        term_prob = norm.cdf((self.threshold - term_mean) / term_std) - norm.cdf((-self.threshold - term_mean) / term_std)
                    else:
                        # Term should be retained, so we want |coef| ≥ threshold
                        term_prob = 1.0 - (norm.cdf((self.threshold - term_mean) / term_std) - norm.cdf((-self.threshold - term_mean) / term_std))
                    
                    indep_prob *= term_prob
                
                # Log the discrepancy
                discrepancy = abs(prob - indep_prob)
                if discrepancy > 0.1:  # More than 10% difference
                    self.logger.debug(f"  Independence assumption discrepancy: {discrepancy:.4f}")
                    self.logger.debug(f"    Joint probability: {prob:.6f}")
                    self.logger.debug(f"    Independent product: {indep_prob:.6f}")
                    
                    # Analyze correlation structure
                    corr_matrix = np.zeros_like(cov)
                    for i in range(cov.shape[0]):
                        for j in range(cov.shape[0]):
                            if cov[i,i] > 0 and cov[j,j] > 0:
                                corr_matrix[i,j] = cov[i,j] / np.sqrt(cov[i,i] * cov[j,j])
                            else:
                                corr_matrix[i,j] = 0
                    
                    # Check for strong correlations
                    for i in range(corr_matrix.shape[0]):
                        for j in range(i+1, corr_matrix.shape[0]):
                            if abs(corr_matrix[i,j]) > 0.5:  # Strong correlation
                                term_i = from_state_list[i]
                                term_j = from_state_list[j]
                                self.logger.debug(f"    Strong correlation ({corr_matrix[i,j]:.4f}) between terms {term_i} and {term_j}")
            
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
        
        self.logger.info(header("BEGINNING SUCCESS PROBABILITY CALCULATION"))
        self.logger.info(f"{bold('All indices')} = {all_indices}")
        self.logger.info(f"{bold('True indices')} = {true_indices}")
        self.logger.info(f"{bold('Target state (true model)')} = {true_model_state}")
        
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
                    prob = self.calculate_transition_probability(from_state, to_state, diagnose=True)
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
                self.logger.info(f"\n{bold_yellow('TRUE MODEL STATE TRANSITIONS')}")
                self.logger.info(from_state_log)
                self.logger.info(f"{bold_yellow(f'TRUE MODEL STATE STOPPING PROBABILITY: {stopping_prob:.6f}')}\n")
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
        self.logger.info(f"\n{bold('State Reachability Probabilities:')}")
        for state in sorted_states:
            state_frozen = frozenset(state)
            prob = reachability_probs[state_frozen]
            if prob > 0.001:  # Only show significant probabilities
                if state == true_model_state:
                    self.logger.info(f"  {bold_green(f'TRUE MODEL STATE {state}: {prob:.6f}')}")
                else:
                    self.logger.info(f"  State {state}: {prob:.6f}")
        
        # Calculate success probability directly for the true state
        true_state_frozen = frozenset(true_model_state)
        true_state_reach_prob = reachability_probs[true_state_frozen]
        true_state_outgoing_prob = sum(prob for (from_s, _), prob in transition_probs.items() 
                                   if from_s == true_state_frozen)
        true_state_stopping_prob = 1.0 - true_state_outgoing_prob
        direct_success_prob = true_state_reach_prob * true_state_stopping_prob
        
        self.logger.info(section("SUCCESS PROBABILITY CALCULATION"))
        self.logger.info(f"{bold('Method 1 - Direct Calculation:')}")
        self.logger.info(f"  {bold('Probability of reaching true state:')}    {true_state_reach_prob:.6f}")
        self.logger.info(f"  {bold('Probability of stopping at true state:')} {true_state_stopping_prob:.6f}")
        self.logger.info(f"  {bold_green(f'Success probability = {true_state_reach_prob:.6f} × {true_state_stopping_prob:.6f} = {direct_success_prob:.6f}')}")
        
        # Check that the two methods give the same result (they should)
        if abs(total_success_prob - direct_success_prob) > 1e-10:
            self.logger.warning(f"{bold_yellow('WARNING: Different success probability calculations do not match!')}")
            self.logger.warning(f"{yellow(f'Method 1 (direct): {direct_success_prob}')}")
            self.logger.warning(f"{yellow(f'Method 2 (summation): {total_success_prob}')}")
        
        self.logger.info(header("END OF SUCCESS PROBABILITY CALCULATION", color_func=green))
        
        return direct_success_prob

    def run_stlsq_with_trajectory(self, theta, y):
        """
        Run sequential thresholded least squares algorithm and track the trajectory through state space.
        Uses robust least squares methods to handle ill-conditioned matrices.
        
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
        
        # Initial least squares with regularization for numerical stability
        try:
            # Try standard least squares first
            xi = np.linalg.lstsq(theta, y, rcond=None)[0]
            
            # If coefficients are unreasonably large, try ridge regression
            if np.any(np.abs(xi) > 1e4):
                # Add small regularization
                ridge_lambda = 1e-6
                gram = theta.T @ theta
                regularized_gram = gram + ridge_lambda * np.eye(gram.shape[0])
                xi = np.linalg.solve(regularized_gram, theta.T @ y)
        except np.linalg.LinAlgError:
            # If standard least squares fails, use ridge regression
            ridge_lambda = 1e-4
            gram = theta.T @ theta
            regularized_gram = gram + ridge_lambda * np.eye(gram.shape[0])
            xi = np.linalg.solve(regularized_gram, theta.T @ y)
        
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
            
            try:
                # Try standard least squares first
                xi_active = np.linalg.lstsq(theta_active, y, rcond=None)[0]
                
                # If coefficients are unreasonably large, try ridge regression
                if np.any(np.abs(xi_active) > 1e4):
                    # Add small regularization
                    ridge_lambda = 1e-6
                    gram = theta_active.T @ theta_active
                    regularized_gram = gram + ridge_lambda * np.eye(gram.shape[0])
                    xi_active = np.linalg.solve(regularized_gram, theta_active.T @ y)
            except np.linalg.LinAlgError:
                # If standard least squares fails, use ridge regression
                ridge_lambda = 1e-4
                gram = theta_active.T @ theta_active
                regularized_gram = gram + ridge_lambda * np.eye(gram.shape[0])
                xi_active = np.linalg.solve(regularized_gram, theta_active.T @ y)
            
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
        Uses robust least squares methods to handle ill-conditioned matrices.
        
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
        
        # Initial least squares with regularization for numerical stability
        try:
            # Try standard least squares first
            xi = np.linalg.lstsq(theta, y, rcond=None)[0]
            
            # If coefficients are unreasonably large, try ridge regression
            if np.any(np.abs(xi) > 1e4):
                # Add small regularization
                ridge_lambda = 1e-6
                gram = theta.T @ theta
                regularized_gram = gram + ridge_lambda * np.eye(gram.shape[0])
                xi = np.linalg.solve(regularized_gram, theta.T @ y)
        except np.linalg.LinAlgError:
            # If standard least squares fails, use ridge regression
            ridge_lambda = 1e-4
            gram = theta.T @ theta
            regularized_gram = gram + ridge_lambda * np.eye(gram.shape[0])
            xi = np.linalg.solve(regularized_gram, theta.T @ y)
        
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
                
                try:
                    # Try standard least squares first
                    xi_active = np.linalg.lstsq(theta_active, y, rcond=None)[0]
                    
                    # If coefficients are unreasonably large, try ridge regression
                    if np.any(np.abs(xi_active) > 1e4):
                        # Add small regularization
                        ridge_lambda = 1e-6
                        gram = theta_active.T @ theta_active
                        regularized_gram = gram + ridge_lambda * np.eye(gram.shape[0])
                        xi_active = np.linalg.solve(regularized_gram, theta_active.T @ y)
                except np.linalg.LinAlgError:
                    # If standard least squares fails, use ridge regression
                    ridge_lambda = 1e-4
                    gram = theta_active.T @ theta_active
                    regularized_gram = gram + ridge_lambda * np.eye(gram.shape[0])
                    xi_active = np.linalg.solve(regularized_gram, theta_active.T @ y)
                
                # Update coefficient vector
                xi = np.zeros(n_terms)
                for i, idx in enumerate(active_list):
                    xi[idx] = xi_active[i]
            
            # Check if converged
            if np.all(np.abs(xi) >= self.threshold) or np.all(xi == 0):
                break
        
        return xi
    
    def analyze_transition_probabilities(self, x_data, n_trials=50):
        """
        Analyze theoretical vs empirical transition probabilities.
        
        Parameters:
        -----------
        x_data : array
            Sample points where to evaluate library functions
        n_trials : int
            Number of trials for empirical observation
            
        Returns:
        --------
        analysis_results : dict
            Dictionary with comparison results
        """
        self.logger.info(section("TRANSITION PROBABILITY ANALYSIS"))
        self.logger.info(f"Comparing theoretical and empirical transition probabilities using {n_trials} trials")
        
        # Get theoretical transition probabilities
        all_indices = set(range(self.n_terms))
        true_indices = self.normalize_state(self.true_term_indices)
        
        # Generate all valid states
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
                    prob = self.calculate_transition_probability(from_state, to_state)
                    key = (str(from_state), str(to_state))
                    theoretical_trans[key] = prob
        
        # Prepare to run trials
        if true_dynamics is None:
            # Generate true dynamics using true_coefs
            def true_dynamics(x):
                y = np.zeros_like(x)
                for i, coef in enumerate(self.true_coefs):
                    if abs(coef) > 1e-10:
                        y += coef * self.library_functions[i](x)
                return y
        
        # Build library matrix
        theta = np.zeros((len(x_data), self.n_terms))
        for j, func in enumerate(self.library_functions):
            theta[:, j] = func(x_data)
            
        # Run trials to collect empirical transition probabilities
        empirical_counts = {}
        total_from_state = {}
        
        for trial in range(n_trials):
            # Generate data with noise
            y_true = true_dynamics(x_data)
            y_noisy = y_true + np.random.normal(0, self.sigma, size=len(x_data))
            
            # Run STLSQ with trajectory tracking
            _, trajectory = self.run_stlsq_with_trajectory(theta, y_noisy)
            
            # Record transitions
            for i in range(len(trajectory) - 1):
                from_state = trajectory[i]
                to_state = trajectory[i + 1]
                
                # Create transition key
                key = (str(from_state), str(to_state))
                empirical_counts[key] = empirical_counts.get(key, 0) + 1
                
                # Track from_state counts for probability calculation
                from_key = str(from_state)
                total_from_state[from_key] = total_from_state.get(from_key, 0) + 1
        
        # Calculate empirical probabilities
        empirical_trans = {}
        for (from_key, to_key), count in empirical_counts.items():
            total = total_from_state[from_key]
            empirical_trans[(from_key, to_key)] = count / total
        
        # Also calculate stopping probabilities
        for from_state in valid_states:
            from_key = str(from_state)
            if from_key in total_from_state:
                total = total_from_state[from_key]
                exits = sum(count for (f, _), count in empirical_counts.items() 
                            if f == from_key)
                stopping_prob = 1.0 - (exits / total)
                
                # Only add if we observed this state
                if total > 0:
                    empirical_trans[(from_key, "[STOP]")] = stopping_prob
        
        # Compare theoretical and empirical
        self.logger.info(bold("\nTransition Probability Comparison:"))
        
        # Combine all observed transitions
        all_trans = set(theoretical_trans.keys()) | set(empirical_trans.keys())
        if len(all_trans) == 0:
            self.logger.info("No transitions observed or predicted")
            return {}
        
        # Group by from_state and sort by state size (largest first)
        from_states = set(f for f, _ in all_trans)
        sorted_from = sorted(from_states, key=lambda s: (-len(eval(s)), s))
        
        analysis_results = {}
        
        for from_state in sorted_from:
            self.logger.info(bold_yellow(f"\nTransitions from {from_state}:"))
            
            # Get all transitions from this state
            to_states = set(t for f, t in all_trans if f == from_state)
            sorted_to = sorted(to_states, key=lambda s: (s == "[STOP]", -len(eval(s)) if s != "[STOP]" else 0, s))
            
            for to_state in sorted_to:
                key = (from_state, to_state)
                theo_prob = theoretical_trans.get(key, 0.0)
                emp_prob = empirical_trans.get(key, 0.0)
                
                # Calculate discrepancy
                discrepancy = abs(theo_prob - emp_prob)
                
                # Format for display, highlighting significant discrepancies
                if discrepancy > 0.1:  # More than 10% difference
                    disc_str = red(f"{discrepancy:.4f}")
                else:
                    disc_str = f"{discrepancy:.4f}"
                
                self.logger.info(f"  → {to_state}:")
                self.logger.info(f"    Theoretical: {theo_prob:.6f}")
                self.logger.info(f"    Empirical:   {emp_prob:.6f}")
                self.logger.info(f"    Discrepancy: {disc_str}")
                
                # Store results
                analysis_results[key] = {
                    'from_state': from_state,
                    'to_state': to_state,
                    'theoretical': theo_prob,
                    'empirical': emp_prob,
                    'discrepancy': discrepancy
                }
        
        # Highlight critical discrepancies
        significant_diffs = [(k, v) for k, v in analysis_results.items() 
                            if v['discrepancy'] > 0.1]
        
        if significant_diffs:
            self.logger.info(bold_red("\nSignificant Discrepancies (>0.1):"))
            
            for (from_state, to_state), data in significant_diffs:
                self.logger.info(f"  {from_state} → {to_state}: " +
                            f"Theoretical {data['theoretical']:.4f} vs " +
                            f"Empirical {data['empirical']:.4f} " +
                            f"(Diff: {data['discrepancy']:.4f})")
        
        return analysis_results
    
    def simulate_stlsq(self, x_data, n_trials=100, true_dynamics=None, analyze_coefficients=True):
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
        analyze_coefficients : bool
            Whether to analyze coefficient distributions
            
        Returns:
        --------
        success_rate : float
            Empirical success probability
        """
        self.logger.info(f"\n{bold_green(f'Running STLSQ simulation with {n_trials} trials...')}")
        
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
        
        # Build library matrix once (it's the same for all trials)
        theta = np.zeros((len(x_data), self.n_terms))
        for j, func in enumerate(self.library_functions):
            theta[:, j] = func(x_data)
        
        # Generate true dynamics once (it's the same for all trials)
        y_true = true_dynamics(x_data)
        
        # Store noisy samples for coefficient analysis
        y_noisy_samples = []
        
        for trial in range(n_trials):
            # Add noise
            y_noisy = y_true + np.random.normal(0, self.sigma, size=len(x_data))
            
            # Store noisy sample for later analysis
            y_noisy_samples.append(y_noisy.copy())
            
            # Run STLSQ with trajectory tracking
            xi, trajectory = self.run_stlsq_with_trajectory(theta, y_noisy)
            
            # Convert trajectory to a string representation for counting
            trajectory_str = ' -> '.join([str(sorted(list(state))) for state in trajectory]) + " -> [STOP]"
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
        self.logger.info(section(f"STLSQ SIMULATION RESULTS ({successful_trials}/{n_trials} successful, {success_rate:.4f})"))
        
        # Log detailed results for the first few trials
        num_to_show = min(5, n_trials)
        self.logger.info(f"\n{bold('Sample of')} {num_to_show} {bold('trials:')}")
        
        for i in range(num_to_show):
            trial = trial_results[i]
            traj_str = ' -> '.join([str(sorted(list(state))) for state in trial['trajectory']])
            if trial['success']:
                status_str = green('SUCCESS')
            else:
                status_str = red('FAILURE')
            self.logger.info(f"  Trial {bold(str(trial['trial']))}: {status_str}, "
                        f"Identified terms: {trial['identified_terms']}")
            self.logger.info(f"    Trajectory: {traj_str} -> [STOP]")
        
        # Analyze trajectories
        self.logger.info(f"\n{bold('Path Analysis:')}")
        
        # Sort trajectories by frequency
        sorted_trajectories = sorted(trajectory_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate what percentage of trials each trajectory represents
        total_trials = sum(trajectory_counts.values())
        
        # Report the most common trajectories
        num_trajectories = min(5, len(sorted_trajectories))
        self.logger.info(f"{bold(f'Top {num_trajectories} paths through state space:')}")
        
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
            self.logger.info(f"\n{bold_red('Most common failure patterns:')}")
            for terms_str, count in top_failures[:3]:  # Show top 3 patterns
                percentage = (count / len(failure_trials)) * 100
                self.logger.info(f"  {yellow(terms_str)}: {bold(str(count))} occurrences "
                            f"({percentage:.1f}% of failures)")
        
        # Run transition probability analysis
        self.analyze_transition_probabilities(x_data, n_trials=min(50, total_trials))

        # Run coefficient analysis if requested
        if analyze_coefficients and total_trials > 0:
            self.analyze_coefficient_distributions(theta, y_noisy_samples[:min(20, len(y_noisy_samples))],
                                                n_samples=min(20, total_trials))
        
        return success_rate

    def analyze_coefficient_distributions(self, theta, y_noisy_batch, n_samples=10):
        """
        Analyze coefficient distributions from empirical data vs theoretical predictions.
        
        Parameters:
        -----------
        theta : array
            Library matrix for a single dataset
        y_noisy_batch : list of arrays
            Multiple noise realizations of the target dynamics
        n_samples : int
            Number of samples to analyze (for performance)
            
        Returns:
        --------
        analysis_results : dict
            Dictionary with analysis results
        """
        # Dictionary to store coefficient values by state
        empirical_coeffs_by_state = {}
        
        # Process a subset of the noisy samples
        for i in range(min(n_samples, len(y_noisy_batch))):
            y_noisy = y_noisy_batch[i]
            
            # Run STLSQ and track coefficients at each state
            _, trajectory, coeffs_by_state = self._run_stlsq_with_coefficient_tracking(theta, y_noisy)
            
            # Record coefficient values for each state
            for state, coeffs in coeffs_by_state.items():
                state_key = str(sorted(list(state)))
                if state_key not in empirical_coeffs_by_state:
                    empirical_coeffs_by_state[state_key] = []
                
                # Check for extremely large coefficients that might be numerical artifacts
                # Only include if all coefficients are reasonably sized
                if np.all(np.abs(coeffs) < 1e6):  # Reject unrealistically large values
                    empirical_coeffs_by_state[state_key].append(coeffs)
        
        # Now compare with theoretical distributions
        analysis_results = {}
        
        for state_key, emp_coeffs_list in empirical_coeffs_by_state.items():
            # Skip states with too few observations
            if len(emp_coeffs_list) < 3:
                continue
                
            # Convert to numpy array for statistics
            emp_coeffs = np.array(emp_coeffs_list)
            
            # Convert state key to actual state indices
            try:
                state_indices = eval(state_key)
            except:
                # If state_key is not a valid Python expression, try parsing it differently
                state_indices = [int(s.strip()) for s in state_key.strip('[]{}').split(',') if s.strip()]
            
            # Get theoretical distribution for this state
            try:
                theo_mean, theo_cov = self.get_coef_distribution(state_indices)
                
                # Calculate empirical statistics with outlier filtering
                # Compute median instead of mean for robustness against outliers
                emp_median = np.median(emp_coeffs, axis=0)
                # Use median absolute deviation for robust std estimation
                emp_mad = np.median(np.abs(emp_coeffs - emp_median), axis=0) * 1.4826  # Scale factor for normal distribution
                
                # Store comparison
                analysis_results[state_key] = {
                    'theoretical': {
                        'mean': theo_mean,
                        'std': np.sqrt(np.diag(theo_cov))
                    },
                    'empirical': {
                        'mean': emp_median,  # Using median for robustness
                        'std': emp_mad,      # Using MAD for robustness
                        'samples': len(emp_coeffs)
                    },
                    'discrepancy': {
                        'mean_diff': np.abs(theo_mean - emp_median),
                        'std_ratio': np.sqrt(np.diag(theo_cov)) / (emp_mad + 1e-10)  # Add small epsilon to avoid division by zero
                    }
                }
            except Exception as e:
                self.logger.warning(f"Error analyzing state {state_key}: {str(e)}")
        
        # Log the analysis results
        self.logger.info(bold_green("\nCoefficient Distribution Analysis:"))
        
        for state_key, result in analysis_results.items():
            theo = result['theoretical']
            emp = result['empirical']
            disc = result['discrepancy']
            
            self.logger.info(bold_yellow(f"\nState {state_key} (observed {emp['samples']} times):"))
            
            # Log coefficient means
            self.logger.info(bold("Coefficient means:"))
            for i, (t_mean, e_mean, diff) in enumerate(zip(theo['mean'], emp['mean'], disc['mean_diff'])):
                # Calculate relative difference as percentage
                if abs(t_mean) > 1e-10:
                    rel_diff_pct = (diff / abs(t_mean)) * 100
                else:
                    # If theoretical mean is close to zero, use absolute difference
                    rel_diff_pct = float('inf') if diff > 1e-10 else 0.0
                
                # Highlight significant differences
                if rel_diff_pct > 20 and diff > 1e-3:  # More than 20% difference and absolute diff > 0.001
                    diff_str = red(f"{diff:.6f} ({rel_diff_pct:.1f}%)")
                else:
                    diff_str = f"{diff:.6f} ({rel_diff_pct:.1f}%)"
                    
                self.logger.info(f"  Coef {i}: Theoretical: {t_mean:.6f}, Empirical: {e_mean:.6f}, Diff: {diff_str}")
            
            # Log coefficient standard deviations
            self.logger.info(bold("\nCoefficient standard deviations:"))
            for i, (t_std, e_std, ratio) in enumerate(zip(theo['std'], emp['std'], disc['std_ratio'])):
                # Highlight significant differences
                if (ratio < 0.5 or ratio > 2.0) and t_std > 1e-6:  # More than 2x difference and theoretical std not tiny
                    ratio_str = red(f"{ratio:.4f}")
                else:
                    ratio_str = f"{ratio:.4f}"
                    
                self.logger.info(f"  Coef {i}: Theoretical: {t_std:.6f}, Empirical: {e_std:.6f}, Ratio: {ratio_str}")
        
        return analysis_results


    def _run_stlsq_with_coefficient_tracking(self, theta, y):
        """
        Run STLSQ algorithm and track coefficients at each state.
        Uses robust least squares methods to handle ill-conditioned matrices.
        
        Parameters:
        -----------
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
        coeffs_by_state : dict
            Dictionary mapping states to coefficient values
        """
        n_terms = theta.shape[1]
        
        # Initial least squares with regularization for numerical stability
        try:
            # Try standard least squares first
            xi = np.linalg.lstsq(theta, y, rcond=None)[0]
            
            # If coefficients are unreasonably large, try ridge regression
            if np.any(np.abs(xi) > 1e4):
                # Add small regularization
                ridge_lambda = 1e-6
                gram = theta.T @ theta
                regularized_gram = gram + ridge_lambda * np.eye(gram.shape[0])
                xi = np.linalg.solve(regularized_gram, theta.T @ y)
        except np.linalg.LinAlgError:
            # If standard least squares fails, use ridge regression
            ridge_lambda = 1e-4
            gram = theta.T @ theta
            regularized_gram = gram + ridge_lambda * np.eye(gram.shape[0])
            xi = np.linalg.solve(regularized_gram, theta.T @ y)
        
        # Initialize with all terms active
        small_indices = np.abs(xi) < self.threshold
        active_indices = self.normalize_state(np.where(~small_indices)[0])
        
        # Track trajectory and coefficients at each state
        trajectory = [active_indices.copy()]
        coeffs_by_state = {}
        
        # Store coefficients for initial state
        state_key = frozenset(active_indices)
        coeffs_by_state[state_key] = xi.copy()
        
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
            
            try:
                # Try standard least squares first
                xi_active = np.linalg.lstsq(theta_active, y, rcond=None)[0]
                
                # If coefficients are unreasonably large, try ridge regression
                if np.any(np.abs(xi_active) > 1e4):
                    # Add small regularization
                    ridge_lambda = 1e-6
                    gram = theta_active.T @ theta_active
                    regularized_gram = gram + ridge_lambda * np.eye(gram.shape[0])
                    xi_active = np.linalg.solve(regularized_gram, theta_active.T @ y)
            except np.linalg.LinAlgError:
                # If standard least squares fails, use ridge regression
                ridge_lambda = 1e-4
                gram = theta_active.T @ theta_active
                regularized_gram = gram + ridge_lambda * np.eye(gram.shape[0])
                xi_active = np.linalg.solve(regularized_gram, theta_active.T @ y)
            
            # Update coefficient vector
            xi = np.zeros(n_terms)
            for i, idx in enumerate(active_list):
                xi[idx] = xi_active[i]
            
            # Store coefficients for this state
            state_key = frozenset(active_indices)
            coeffs_by_state[state_key] = xi_active.copy()
            
            # Check for convergence
            converged = True
            for idx in active_indices:
                if abs(xi[idx]) < self.threshold:
                    converged = False
                    break
        
        return xi, trajectory, coeffs_by_state

    def simulate_stlsq_adaptive(self, x_data, max_trials=500, confidence=0.95, margin=0.05, 
                            min_trials=30, batch_size=10, true_dynamics=None, analyze_coefficients=True):
        """
        Empirically simulate the STLSQ algorithm with adaptive trial count determination.
        
        This method automatically determines how many trials are needed to achieve 
        a certain confidence level for the estimated success probability.
        
        Parameters:
        -----------
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
        true_dynamics : callable, optional
            Function to generate true dynamics. If None, uses true_coefs.
        analyze_coefficients : bool
            Whether to analyze coefficient distributions
            
        Returns:
        --------
        success_rate : float
            Empirical success probability
        trials_used : int
            Number of trials that were actually run
        """
        from scipy import stats
        
        self.logger.info(f"\n{bold_green(f'Running adaptive STLSQ simulation (max {max_trials} trials, {confidence*100:.0f}% confidence, {margin*100:.1f}% margin)')}")
        
        if true_dynamics is None:
            # Generate true dynamics using true_coefs
            def true_dynamics(x):
                y = np.zeros_like(x)
                for i, coef in enumerate(self.true_coefs):
                    if abs(coef) > 1e-10:
                        y += coef * self.library_functions[i](x)
                return y
        
        successful_trials = 0
        total_trials = 0
        trial_results = []
        
        # Track paths through state space
        trajectory_counts = {}
        
        # Build library matrix once (it's the same for all trials)
        theta = np.zeros((len(x_data), self.n_terms))
        for j, func in enumerate(self.library_functions):
            theta[:, j] = func(x_data)
        
        # Generate true dynamics once (it's the same for all trials)
        y_true = true_dynamics(x_data)
        
        # Store noisy samples for coefficient analysis
        y_noisy_samples = []
        
        # Calculate initial z-score for desired confidence
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Store batch results for logging (reduced to only log significant changes)
        last_reported_rate = None
        convergence_achieved = False
        
        # Run batches of trials until convergence or max_trials
        while total_trials < max_trials:
            # Run a batch of trials
            batch_successes = 0
            batch_trials = []
            
            for _ in range(batch_size):
                if total_trials >= max_trials:
                    break
                    
                # Add noise
                y_noisy = y_true + np.random.normal(0, self.sigma, size=len(x_data))
                y_noisy_samples.append(y_noisy.copy())
                
                # Run STLSQ with trajectory tracking
                xi, trajectory = self.run_stlsq_with_trajectory(theta, y_noisy)
                
                # Convert trajectory to a string representation for counting
                trajectory_str = ' -> '.join([str(sorted(list(state))) for state in trajectory]) + " -> [STOP]"
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
                    batch_successes += 1
                
                # Store details about this trial
                trial_info = {
                    'trial': total_trials + 1,
                    'success': is_success,
                    'identified_terms': np.where(identified_pattern == 1)[0].tolist(),
                    'trajectory': trajectory
                }
                
                trial_results.append(trial_info)
                batch_trials.append(trial_info)
                
                total_trials += 1
            
            # Calculate current success rate
            current_success_rate = successful_trials / total_trials
            
            # Calculate margin of error for the current number of trials
            # Using Wilson score interval for better accuracy with small samples and extreme probabilities
            # Source: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
            
            z2 = z_score ** 2
            factor = z2 / (2 * total_trials)
            
            wilson_p = (current_success_rate + factor) / (1 + 2 * factor)
            wilson_error_margin = z_score * np.sqrt(current_success_rate * (1 - current_success_rate) / total_trials + factor / 4) / (1 + 2 * factor)
            
            # Only log progress at important milestones to reduce noise
            should_log = (
                total_trials % (5 * batch_size) == 0 or  # Every 5 batches
                total_trials == max_trials or            # At max trials
                total_trials == min_trials or            # At min trials
                (wilson_error_margin <= margin and total_trials >= min_trials) or  # At convergence
                last_reported_rate is None or            # First batch
                abs(current_success_rate - (last_reported_rate or 0)) >= 0.05  # Rate changed by at least 5%
            )
            
            if should_log:
                self.logger.info(f"After {total_trials} trials: Success rate = {current_success_rate:.4f}, " +
                                f"Margin of error = ±{wilson_error_margin:.4f} (target: {margin:.4f})")
                
                # Only show sample trials on first report
                if last_reported_rate is None:
                    # Display recent trial results - just a single trial from the first batch
                    for i, trial in enumerate(batch_trials):
                        if i == 0:  # Just show the first trial
                            if trial['success']:
                                status_str = green('SUCCESS')
                            else:
                                status_str = red('FAILURE')
                            self.logger.debug(f"  Sample trial: {status_str}, " +
                                            f"Identified terms: {trial['identified_terms']}")
                
                last_reported_rate = current_success_rate
            
            # Check if we've reached the desired margin of error and minimum trials
            if wilson_error_margin <= margin and total_trials >= min_trials:
                self.logger.info(f"{bold_green('Convergence achieved!')} Margin of error {wilson_error_margin:.4f} <= target {margin:.4f}")
                convergence_achieved = True
                break
        
        # Final success rate
        success_rate = successful_trials / total_trials
        
        # Log detailed simulation results
        self.logger.info(section(f"ADAPTIVE STLSQ SIMULATION RESULTS ({successful_trials}/{total_trials} successful, {success_rate:.4f})"))
        
        # Construct confidence interval
        z2 = z_score ** 2
        factor = z2 / (2 * total_trials)
        wilson_p = (success_rate + factor) / (1 + 2 * factor)
        wilson_error_margin = z_score * np.sqrt(success_rate * (1 - success_rate) / total_trials + factor / 4) / (1 + 2 * factor)
        
        lower_bound = max(0, wilson_p - wilson_error_margin)
        upper_bound = min(1, wilson_p + wilson_error_margin)
        
        # Report convergence status
        if not convergence_achieved:
            self.logger.info(f"{bold_yellow('Warning: Maximum trials reached without achieving target margin of error.')}")
            self.logger.info(f"Target margin: {margin:.4f}, Achieved margin: {wilson_error_margin:.4f}")
        
        self.logger.info(f"{confidence*100:.0f}% Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}] " +
                        f"(width: {upper_bound-lower_bound:.4f})")
        
        # Log detailed results for the first few trials
        num_to_show = min(5, total_trials)
        self.logger.info(f"\n{bold('Sample of')} {num_to_show} {bold('trials:')}")
        
        for i in range(num_to_show):
            trial = trial_results[i]
            traj_str = ' -> '.join([str(sorted(list(state))) for state in trial['trajectory']])
            if trial['success']:
                status_str = green('SUCCESS')
            else:
                status_str = red('FAILURE')
            self.logger.info(f"  Trial {bold(str(trial['trial']))}: {status_str}, " +
                            f"Identified terms: {trial['identified_terms']}")
            self.logger.info(f"    Trajectory: {traj_str} -> [STOP]")
        
        # Analyze trajectories
        self.logger.info(f"\n{bold('Path Analysis:')}")
        
        # Sort trajectories by frequency
        sorted_trajectories = sorted(trajectory_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate what percentage of trials each trajectory represents
        total_trajectories = sum(trajectory_counts.values())
        
        # Report the most common trajectories (limit to top 5 or fewer)
        num_trajectories = min(5, len(sorted_trajectories))
        self.logger.info(f"{bold(f'Top {num_trajectories} paths through state space:')}")
        
        for traj, count in sorted_trajectories[:num_trajectories]:
            percentage = (count / total_trajectories) * 100
            self.logger.info(f"  {traj}: {count} occurrences ({percentage:.1f}% of trials)")
        
        # If there were failures, analyze what went wrong
        if successful_trials < total_trials:
            # Count identified terms in failures
            failure_trials = [t for t in trial_results if not t['success']]
            term_counts = {}
            for trial in failure_trials:
                terms_str = str(sorted(trial['identified_terms']))
                term_counts[terms_str] = term_counts.get(terms_str, 0) + 1
            
            # Sort by frequency
            top_failures = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Log the most common failure patterns (limit to top 3 or fewer)
            fail_patterns_to_show = min(3, len(top_failures))
            if fail_patterns_to_show > 0:
                self.logger.info(f"\n{bold_red('Most common failure patterns:')}")
                for terms_str, count in top_failures[:fail_patterns_to_show]:  # Show top patterns
                    percentage = (count / len(failure_trials)) * 100
                    self.logger.info(f"  {yellow(terms_str)}: {bold(str(count))} occurrences " +
                                    f"({percentage:.1f}% of failures)")
        
        # Run coefficient analysis if requested - limit to a smaller sample for memory efficiency
        if analyze_coefficients and total_trials > 0:
            self.analyze_coefficient_distributions(theta, y_noisy_samples[:min(20, len(y_noisy_samples))],
                                                n_samples=min(20, total_trials))
        
        return success_rate, total_trials