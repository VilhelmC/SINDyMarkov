"""
Utilities for handling state operations in the SINDy Markov Chain Model.

This module provides functions for managing and analyzing the state space
of the SINDy algorithm with Sequential Thresholded Least Squares (STLSQ).
"""

import numpy as np
from itertools import combinations

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

def generate_valid_states(true_indices, all_indices):
    """
    Generate all valid states containing the true indices.
    
    Parameters:
    -----------
    true_indices : set or list
        Indices of true terms that must be included in all valid states
    all_indices : set or list
        Indices of all available terms
        
    Returns:
    --------
    valid_states : list
        List of valid states as sets
    """
    true_indices = normalize_state(true_indices)
    all_indices = normalize_state(all_indices)
    
    valid_states = []
    
    for r in range(len(true_indices), len(all_indices) + 1):
        for subset in combinations(all_indices, r):
            subset_set = set(subset)
            if true_indices.issubset(subset_set):
                valid_states.append(subset_set)
    
    return valid_states

def state_to_string(state):
    """
    Convert a state to a sorted string representation.
    
    Parameters:
    -----------
    state : set, list, or array
        State containing indices
        
    Returns:
    --------
    state_str : str
        String representation of the state
    """
    normalized = normalize_state(state)
    return str(sorted(normalized))

def string_to_state(state_str):
    """
    Convert a string representation back to a state.
    
    Parameters:
    -----------
    state_str : str
        String representation of a state
        
    Returns:
    --------
    state : set
        Set containing the state indices
    """
    # Handle different string formats
    if state_str.startswith('{') and state_str.endswith('}'):
        # Handle {0, 1, 2} format
        content = state_str[1:-1]
        if content:
            return set(int(idx.strip()) for idx in content.split(','))
        else:
            return set()
    elif state_str.startswith('[') and state_str.endswith(']'):
        # Handle [0, 1, 2] format
        content = state_str[1:-1]
        if content:
            return set(int(idx.strip()) for idx in content.split(','))
        else:
            return set()
    else:
        raise ValueError(f"Invalid state string format: {state_str}")

def get_direct_transitions(state):
    """
    Get all possible direct transitions from a state by removing one term at a time.
    
    Parameters:
    -----------
    state : set
        Current state
        
    Returns:
    --------
    transitions : list
        List of possible next states
    """
    normalized = normalize_state(state)
    transitions = []
    
    for term in normalized:
        next_state = normalized.copy()
        next_state.remove(term)
        transitions.append(next_state)
    
    return transitions

def state_to_pattern(state, n_terms):
    """
    Convert a state to a binary pattern.
    
    Parameters:
    -----------
    state : set
        Set of active term indices
    n_terms : int
        Total number of terms
        
    Returns:
    --------
    pattern : array
        Binary array with 1s at active indices
    """
    pattern = np.zeros(n_terms)
    normalized = normalize_state(state)
    pattern[list(normalized)] = 1
    return pattern

def pattern_to_state(pattern):
    """
    Convert a binary pattern to a state.
    
    Parameters:
    -----------
    pattern : array
        Binary array with 1s at active indices
        
    Returns:
    --------
    state : set
        Set of active term indices
    """
    return set(np.where(pattern > 0)[0])

def is_state_equal(state1, state2):
    """
    Check if two states are equal, handling type conversions.
    
    Parameters:
    -----------
    state1 : set, list, or array
        First state
    state2 : set, list, or array
        Second state
        
    Returns:
    --------
    is_equal : bool
        True if states are equal, False otherwise
    """
    normalized1 = normalize_state(state1)
    normalized2 = normalize_state(state2)
    return normalized1 == normalized2

def get_intermediate_states(start_state, end_state):
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
    # Normalize both states
    start_state = normalize_state(start_state)
    end_state = normalize_state(end_state)
    
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

def get_all_paths(from_state, to_state):
    """
    Find all possible paths from from_state to to_state.
    
    Parameters:
    -----------
    from_state : set
        Starting state
    to_state : set
        Target state
        
    Returns:
    --------
    paths : list
        List of paths, where each path is a list of states
    """
    # Normalize states
    from_state = normalize_state(from_state)
    to_state = normalize_state(to_state)
    
    # to_state must be a subset of from_state
    if not to_state.issubset(from_state):
        return []
    
    # If states are the same, return a single path
    if from_state == to_state:
        return [[from_state]]
    
    # DFS to find all paths
    def dfs(current, target, path, all_paths):
        # If we've reached the target, add this path
        if current == target:
            all_paths.append(path.copy())
            return
        
        # Try eliminating one term at a time
        for term in current - target:
            next_state = current.copy()
            next_state.remove(term)
            
            # Avoid cycles
            if next_state not in path:
                path.append(next_state)
                dfs(next_state, target, path, all_paths)
                path.pop()  # Backtrack
    
    all_paths = []
    dfs(from_state, to_state, [from_state], all_paths)
    
    return all_paths

def calculate_state_entropy(probs):
    """
    Calculate the entropy of a state distribution.
    
    Parameters:
    -----------
    probs : dict
        Dictionary mapping states to probabilities
        
    Returns:
    --------
    entropy : float
        Entropy of the distribution
    """
    entropy = 0.0
    for prob in probs.values():
        if prob > 0:
            entropy -= prob * np.log2(prob)
    
    return entropy