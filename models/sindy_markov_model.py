import numpy as np
from itertools import combinations
from scipy.stats import multivariate_normal, norm
import logging

# Import centralized logging utilities
from models.logger_utils import setup_logging, get_logger

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
        self.logger = setup_logging(log_file)
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
        
        # Cache for coefficient distributions
        self._coef_dist_cache = {}
    
    def set_library(self, library_functions, true_coefs):
        """Set the library functions and true coefficients."""
        self.logger.info("Setting new library functions and coefficients")
        self.library_functions = library_functions
        self.true_coefs = true_coefs
        self.n_terms = len(true_coefs)
        self.true_term_indices = np.where(np.abs(true_coefs) > 1e-10)[0]
        self.logger.info(f"New true term indices: {self.true_term_indices}")
        
        # Clear caches when library changes
        self._transition_cache = {}
        self._coef_dist_cache = {}
    
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
            # Use more stable method for log determinant
            _, logdet = np.linalg.slogdet(gram_matrix)
            self.log_gram_det = logdet
            self.logger.info(f"Log determinant of Gram matrix: {self.log_gram_det:.4f}")
        except np.linalg.LinAlgError:
            self.log_gram_det = float('-inf')
            self.logger.warning("Failed to compute determinant of Gram matrix (singular matrix)")
        
        # Store the design matrix for later use
        self.theta = theta
        
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
        
        # Convert to tuple for cache key
        if not isinstance(active_indices, tuple):
            active_indices_tuple = tuple(sorted(map(int, active_indices)))
        else:
            active_indices_tuple = active_indices
                
        # Check cache
        if active_indices_tuple in self._coef_dist_cache:
            return self._coef_dist_cache[active_indices_tuple]
        
        # Extract submatrix for active terms
        active_indices_array = np.array(active_indices_tuple)
        sub_gram = self.gram_matrix[np.ix_(active_indices_array, active_indices_array)]
        
        # Get true coefficients for these terms
        sub_true_coefs = self.true_coefs[active_indices_array]
        
        # Compute inverse of sub_gram with improved numerical stability
        try:
            # Check condition number to assess numerical stability
            cond_num = np.linalg.cond(sub_gram)
            
            if cond_num > 1e10:
                # Use regularized inversion for ill-conditioned matrices
                self.logger.debug(f"High condition number detected: {cond_num:.2e}, using regularized inversion")
                # Apply Tikhonov regularization
                reg_param = 1e-8 * np.trace(sub_gram) / len(active_indices_tuple)
                sub_gram_reg = sub_gram + reg_param * np.eye(len(active_indices_tuple))
                sub_gram_inv = np.linalg.inv(sub_gram_reg)
            else:
                # Use standard inversion for well-conditioned matrices
                sub_gram_inv = np.linalg.inv(sub_gram)
                
        except np.linalg.LinAlgError:
            self.logger.warning("Singular sub-Gram matrix detected, using pseudoinverse")
            # Use pseudoinverse if matrix is singular
            sub_gram_inv = np.linalg.pinv(sub_gram)
        
        # Coefficient distribution parameters
        mean = sub_true_coefs
        cov = self.sigma**2 * sub_gram_inv
        
        # Ensure covariance matrix is symmetric and positive definite
        cov = (cov + cov.T) / 2  # Make perfectly symmetric
        
        # Add small regularization if needed to ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(cov))
        if min_eig < 1e-10:
            self.logger.debug(f"Adding regularization to ensure positive definite covariance (min eigenvalue: {min_eig:.2e})")
            cov += np.eye(cov.shape[0]) * max(0, 1e-10 - min_eig) * 1.1  # Small buffer
        
        # Store in cache using tuple as key
        self._coef_dist_cache[active_indices_tuple] = (mean, cov)
        
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

    def calculate_transition_probability(self, from_state, to_state, diagnose=False):
        """
        Calculate the probability of transitioning from one state to another analytically.
        
        Parameters:
        -----------
        from_state : set or list
            Current state (indices of active terms)
        to_state : set or list
            Next state (indices of active terms)
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
            # Get coefficient distribution for the from_state
            mean, cov = self.get_coef_distribution(from_tuple)
            
            # Calculate which terms are retained and which are eliminated
            eliminated_indices = list(from_state - to_state)
            retained_indices = list(to_state)
            
            # Map to indices within the from_state coefficient vector
            from_state_list = list(from_tuple)
            eliminated_positions = [from_state_list.index(idx) for idx in eliminated_indices]
            retained_positions = [from_state_list.index(idx) for idx in retained_indices]
            
            # To calculate the joint probability analytically, we need to:
            # 1. Calculate P(|β_e| < λ for all eliminated terms)
            # 2. Calculate P(|β_r| ≥ λ for all retained terms)
            # 3. Calculate the joint probability considering correlations
            
            # For eliminated terms: P(|β_e| < λ)
            # We can represent this as a probability within a hypercube in the eliminated terms' space
            
            # Extract the relevant submatrices for eliminated terms
            mean_e = mean[eliminated_positions]
            cov_e = cov[np.ix_(eliminated_positions, eliminated_positions)]
            
            # If there's only one eliminated term, we can use the simple CDF
            if len(eliminated_positions) == 1:
                term_mean = mean_e[0]
                term_std = np.sqrt(cov_e[0, 0])
                # Probability of |coef| < threshold
                elim_prob = norm.cdf((self.threshold - term_mean) / term_std) - norm.cdf((-self.threshold - term_mean) / term_std)
                
                if diagnose:
                    self.logger.debug(f"Single term elimination probability for term {from_state_list[eliminated_positions[0]]}: {elim_prob:.6f}")
            else:
                # For multiple eliminated terms, we need to account for their correlation
                # This requires numerical integration over a multivariate normal in a hypercube
                # Since this is computationally intensive, we'll use Monte Carlo for this case
                from scipy.stats import multivariate_normal
                
                # Setup the multivariate normal distribution
                mvn = multivariate_normal(mean=mean_e, cov=cov_e)
                
                # Define the integration region (hypercube where all |β_e| < λ)
                # We'll use Monte Carlo integration
                mc_samples = 100000
                samples = mvn.rvs(size=mc_samples)
                
                # Count samples where all eliminated terms are below threshold
                count = 0
                for sample in samples:
                    if np.all(np.abs(sample) < self.threshold):
                        count += 1
                
                elim_prob = count / mc_samples
                
                if diagnose:
                    self.logger.debug(f"Multiple term elimination probability for terms {[from_state_list[i] for i in eliminated_positions]}: {elim_prob:.6f}")
            
            # For retained terms: P(|β_r| ≥ λ for all r)
            # Extract the relevant submatrices for retained terms
            mean_r = mean[retained_positions]
            cov_r = cov[np.ix_(retained_positions, retained_positions)]
            
            # Calculate the probability for each retained term being above threshold
            # Start with the assumption of independence (we'll correct for correlation later)
            indep_retained_probs = []
            for i, pos in enumerate(retained_positions):
                term_mean = mean_r[i]
                term_std = np.sqrt(cov_r[i, i])
                # P(|coef| ≥ threshold)
                term_above_prob = 1.0 - (norm.cdf((self.threshold - term_mean) / term_std) - 
                                        norm.cdf((-self.threshold - term_mean) / term_std))
                indep_retained_probs.append(term_above_prob)
                
                if diagnose:
                    self.logger.debug(f"Term {from_state_list[pos]} P(|coef| ≥ {self.threshold}) = {term_above_prob:.6f}")
            
            # If there are no correlations or only one retained term, independent probabilities are accurate
            if len(retained_positions) <= 1 or np.allclose(cov_r - np.diag(np.diag(cov_r)), 0):
                retained_prob = np.prod(indep_retained_probs)
            else:
                # For correlated retained terms, we need to use Monte Carlo
                # Setup the multivariate normal distribution
                mvn = multivariate_normal(mean=mean_r, cov=cov_r)
                
                # Define the integration region (where all |β_r| ≥ λ)
                mc_samples = 100000
                samples = mvn.rvs(size=mc_samples)
                
                # Count samples where all retained terms are above threshold
                count = 0
                for sample in samples:
                    if np.all(np.abs(sample) >= self.threshold):
                        count += 1
                
                retained_prob = count / mc_samples
                
                if diagnose:
                    self.logger.debug(f"Correlated retained probability: {retained_prob:.6f}")
                    self.logger.debug(f"Independent product would be: {np.prod(indep_retained_probs):.6f}")
            
            # Now for the joint probability, we need to account for correlation between eliminated and retained terms
            # The full joint probability would require integration over a complex region
            
            # If all eliminated and retained terms are uncorrelated, we can multiply the probabilities
            if np.allclose(cov - np.diag(np.diag(cov)), 0):
                joint_prob = elim_prob * retained_prob
                
                if diagnose:
                    self.logger.debug(f"Terms uncorrelated, joint probability = {elim_prob:.6f} * {retained_prob:.6f} = {joint_prob:.6f}")
            else:
                # If there are correlations between eliminated and retained terms,
                # we need to use a full Monte Carlo integration over the joint distribution
                
                # Setup the full multivariate normal distribution
                mvn = multivariate_normal(mean=mean, cov=cov)
                
                # Monte Carlo integration
                mc_samples = 100000
                samples = mvn.rvs(size=mc_samples)
                
                # Count samples meeting both criteria
                count = 0
                for sample in samples:
                    eliminated_ok = np.all(np.abs(sample[eliminated_positions]) < self.threshold)
                    retained_ok = np.all(np.abs(sample[retained_positions]) >= self.threshold)
                    
                    if eliminated_ok and retained_ok:
                        count += 1
                
                joint_prob = count / mc_samples
                
                if diagnose:
                    self.logger.debug(f"Correlated terms, joint probability = {joint_prob:.6f}")
                    self.logger.debug(f"Simple product would be: {elim_prob:.6f} * {retained_prob:.6f} = {elim_prob * retained_prob:.6f}")
            
            # Ensure prob is between 0 and 1
            prob = max(0.0, min(1.0, joint_prob))
            
            # Add to cache
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
        
        self.logger.info("\n================================================================================")
        self.logger.info("BEGINNING SUCCESS PROBABILITY CALCULATION")
        self.logger.info("================================================================================")
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
            self.logger.info(f"  State {i+1}: {state}")
        
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
                    
                    if prob > 0.0001:  # Only show significant transitions
                        key = (from_frozen, frozenset(to_state))
                        transition_probs[key] = prob
                        state_transitions[from_frozen].append((frozenset(to_state), prob))
                        from_state_log += f"\n    -> {to_state}: {prob:.6f}"
            
            # Calculate stopping probability (probability that this is the final state)
            stopping_prob = 1.0 - total_outgoing
            from_state_log += f"\n    -> [STOP]: {stopping_prob:.6f}"
            
            # Highlight if this is the true model state
            if from_state == true_model_state:
                self.logger.info("\nTRUE MODEL STATE TRANSITIONS")
                self.logger.info(from_state_log)
                self.logger.info(f"TRUE MODEL STATE STOPPING PROBABILITY: {stopping_prob:.6f}\n")
            else:
                self.logger.info(from_state_log)
        
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
        
        # Log reachability probabilities for all states
        self.logger.info("\nState Reachability Probabilities:")
        for state in sorted_states:
            state_frozen = frozenset(state)
            prob = reachability_probs[state_frozen]
            if prob > 0.001:  # Only show significant probabilities
                if state == true_model_state:
                    self.logger.info(f"  TRUE MODEL STATE {state}: {prob:.6f}")
                else:
                    self.logger.info(f"  State {state}: {prob:.6f}")
        
        # Calculate success probability directly for the true state
        true_state_frozen = frozenset(true_model_state)
        true_state_reach_prob = reachability_probs[true_state_frozen]
        true_state_outgoing_prob = sum(prob for (from_s, _), prob in transition_probs.items() 
                                if from_s == true_state_frozen)
        true_state_stopping_prob = 1.0 - true_state_outgoing_prob
        direct_success_prob = true_state_reach_prob * true_state_stopping_prob
        
        self.logger.info("\n--------------------------------------------------------------------------------")
        self.logger.info("SUCCESS PROBABILITY CALCULATION")
        self.logger.info("--------------------------------------------------------------------------------")
        self.logger.info("Method 1 - Direct Calculation:")
        self.logger.info(f"  Probability of reaching true state:    {true_state_reach_prob:.6f}")
        self.logger.info(f"  Probability of stopping at true state: {true_state_stopping_prob:.6f}")
        self.logger.info(f"  Success probability = {true_state_reach_prob:.6f} × {true_state_stopping_prob:.6f} = {direct_success_prob:.6f}")
        
        self.logger.info("\n================================================================================")
        self.logger.info("END OF SUCCESS PROBABILITY CALCULATION")
        self.logger.info("================================================================================")
        
        return direct_success_prob

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
                
                try:
                    xi_active = np.linalg.lstsq(theta_active, y, rcond=None)[0]
                except np.linalg.LinAlgError:
                    # If standard least squares fails, use ridge regression
                    ridge_lambda = 1e-8
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
    
    def run_stlsq_with_trajectory(self, theta, y):
        """
        Run sequential thresholded least squares algorithm and track the trajectory.
        
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
            
            try:
                xi_active = np.linalg.lstsq(theta_active, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                # If standard least squares fails, use ridge regression
                ridge_lambda = 1e-8
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
        self.logger.info(f"Running STLSQ simulation with {n_trials} trials...")
        
        if true_dynamics is None:
            # Generate true dynamics using true_coefs
            def true_dynamics(x):
                y = np.zeros_like(x)
                for i, coef in enumerate(self.true_coefs):
                    if abs(coef) > 1e-10:
                        y += coef * self.library_functions[i](x)
                return y
        
        successful_trials = 0
        
        # Build library matrix once (it's the same for all trials)
        theta = np.zeros((len(x_data), self.n_terms))
        for j, func in enumerate(self.library_functions):
            theta[:, j] = func(x_data)
        
        # Generate true dynamics once (it's the same for all trials)
        y_true = true_dynamics(x_data)
        
        for _ in range(n_trials):
            # Add noise
            y_noisy = y_true + np.random.normal(0, self.sigma, size=len(x_data))
            
            # Run STLSQ
            xi = self.run_stlsq(theta, y_noisy)
            
            # Check for success
            true_pattern = np.zeros(self.n_terms)
            true_pattern[self.true_term_indices] = 1
            identified_pattern = np.zeros(self.n_terms)
            identified_pattern[np.abs(xi) > 1e-10] = 1
            
            # Determine if this is a success
            is_success = np.array_equal(true_pattern, identified_pattern)
            if is_success:
                successful_trials += 1
        
        success_rate = successful_trials / n_trials
        self.logger.info(f"STLSQ simulation results: {successful_trials}/{n_trials} successful, {success_rate:.4f} success rate")
        
        return success_rate