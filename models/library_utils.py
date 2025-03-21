"""
Utilities for analyzing and manipulating function libraries in the SINDy framework.

This module provides functions for computing and analyzing properties of library
functions, including correlation and discriminability.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from itertools import combinations

# Get logger
logger = logging.getLogger('SINDyMarkovModel')

def compute_discriminability(theta, sigma=1.0):
    """
    Compute discriminability matrix between library terms.
    
    The discriminability between terms i and j is defined as:
    D_ij = (1/σ²) * Σ[θ_i(x_k) - θ_j(x_k)]²
    
    Parameters:
    -----------
    theta : array
        Design matrix where columns are library terms evaluated at sample points
    sigma : float
        Noise standard deviation
        
    Returns:
    --------
    D : array
        Discriminability matrix
    """
    n_terms = theta.shape[1]
    D = np.zeros((n_terms, n_terms))
    
    for i in range(n_terms):
        for j in range(i+1, n_terms):
            # Compute squared difference between terms
            diff_squared = np.sum((theta[:, i] - theta[:, j])**2) / (sigma**2)
            D[i, j] = diff_squared
            D[j, i] = diff_squared  # Symmetric
    
    return D

def compute_gram_discriminability(G, sigma=1.0):
    """
    Compute discriminability matrix from Gram matrix.
    
    The discriminability using the Gram matrix is:
    D_ij = (1/σ²) * (G_ii + G_jj - 2*G_ij)
    
    Parameters:
    -----------
    G : array
        Gram matrix (θᵀθ)
    sigma : float
        Noise standard deviation
        
    Returns:
    --------
    D : array
        Discriminability matrix
    """
    n_terms = G.shape[0]
    D = np.zeros((n_terms, n_terms))
    
    for i in range(n_terms):
        for j in range(i+1, n_terms):
            # Compute discriminability from Gram matrix
            disc = (G[i, i] + G[j, j] - 2*G[i, j]) / (sigma**2)
            D[i, j] = disc
            D[j, i] = disc  # Symmetric
    
    return D

def critical_discriminability(threshold_sigma_ratio):
    """
    Calculate critical discriminability for a given threshold/sigma ratio.
    
    The critical discriminability D* represents the discriminability value where
    the success probability equals 0.5 for a given λ/σ ratio.
    
    Parameters:
    -----------
    threshold_sigma_ratio : float
        Ratio of threshold to noise (λ/σ)
        
    Returns:
    --------
    D_star : float
        Critical discriminability
    """
    # Approximation based on analytical derivation
    lambda_sigma = threshold_sigma_ratio
    D_star = 4 * lambda_sigma**2
    
    return D_star

def analyze_library_correlations(x_data, library_functions, names=None):
    """
    Analyze correlations between library terms.
    
    Parameters:
    -----------
    x_data : array
        Sample points where to evaluate library functions
    library_functions : list
        List of library functions
    names : list, optional
        Names for the library functions
        
    Returns:
    --------
    corr_matrix : array
        Correlation matrix
    term_values : array
        Library terms evaluated at sample points
    """
    n_terms = len(library_functions)
    
    # Evaluate library terms at sample points
    term_values = np.zeros((len(x_data), n_terms))
    for j, func in enumerate(library_functions):
        term_values[:, j] = func(x_data)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(term_values.T)
    
    # Create correlation DataFrame if names are provided
    if names is not None:
        corr_df = pd.DataFrame(corr_matrix, index=names, columns=names)
    else:
        corr_df = pd.DataFrame(corr_matrix)
    
    return corr_df, term_values

def plot_library_correlations(corr_df, figsize=(10, 8), cmap='coolwarm', save_path=None):
    """
    Plot correlation matrix as a heatmap.
    
    Parameters:
    -----------
    corr_df : DataFrame
        Correlation matrix
    figsize : tuple
        Figure size
    cmap : str
        Colormap for the heatmap
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    cax = ax.matshow(corr_df, cmap=cmap, vmin=-1, vmax=1)
    fig.colorbar(cax)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha='left')
    ax.set_yticklabels(corr_df.index)
    
    # Add correlation values to cells
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}", 
                    ha='center', va='center', 
                    color='white' if abs(corr_df.iloc[i, j]) > 0.5 else 'black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def find_problematic_library_terms(corr_df, d_matrix=None, threshold=0.9):
    """
    Find highly correlated or low-discriminability term pairs.
    
    Parameters:
    -----------
    corr_df : DataFrame
        Correlation matrix
    d_matrix : array, optional
        Discriminability matrix
    threshold : float
        Correlation threshold for considering terms as problematic
        
    Returns:
    --------
    problematic_pairs : list
        List of problematic term pairs with their correlation values
    """
    problematic_pairs = []
    
    # Find highly correlated pairs
    for i in range(len(corr_df.index)):
        for j in range(i+1, len(corr_df.columns)):
            corr = abs(corr_df.iloc[i, j])
            if corr > threshold:
                term_i = corr_df.index[i]
                term_j = corr_df.columns[j]
                
                pair_info = {
                    'term1': term_i,
                    'term2': term_j,
                    'correlation': corr,
                }
                
                # Add discriminability if available
                if d_matrix is not None:
                    pair_info['discriminability'] = d_matrix[i, j]
                
                problematic_pairs.append(pair_info)
    
    # Sort by correlation in descending order
    problematic_pairs.sort(key=lambda x: x['correlation'], reverse=True)
    
    return problematic_pairs

def calculate_term_significance(true_coefs, sigma, gram_matrix):
    """
    Calculate significance metrics for library terms.
    
    Parameters:
    -----------
    true_coefs : array
        True coefficient values
    sigma : float
        Noise standard deviation
    gram_matrix : array
        Gram matrix (θᵀθ)
        
    Returns:
    --------
    significance : dict
        Dictionary with significance metrics for each term
    """
    n_terms = len(true_coefs)
    significance = {}
    
    for i in range(n_terms):
        if abs(true_coefs[i]) < 1e-10:
            # Skip terms with zero coefficients
            continue
            
        # Calculate signal-to-noise ratio
        variance = sigma**2 * gram_matrix[i, i]**(-1)
        snr = abs(true_coefs[i]) / np.sqrt(variance)
        
        # Calculate probability of being detected
        detect_prob = 1 - 2 * (0.5 - 0.5 * np.exp(-snr**2 / 2))
        
        significance[i] = {
            'coefficient': true_coefs[i],
            'signal_to_noise': snr,
            'detection_probability': detect_prob
        }
    
    return significance

def generate_orthogonal_library(n_terms, n_samples=1000):
    """
    Generate an orthogonal library for testing.
    
    Parameters:
    -----------
    n_terms : int
        Number of library terms
    n_samples : int
        Number of sample points
        
    Returns:
    --------
    functions : list
        List of orthogonal library functions
    x_range : array
        Sample points
    """
    # Generate sample points
    x_range = np.linspace(-1, 1, n_samples)
    
    # Create orthogonal library functions (e.g., Legendre polynomials)
    functions = []
    
    # Function 0: constant term
    functions.append(lambda x: np.ones_like(x))
    
    # Function 1: linear term
    functions.append(lambda x: x)
    
    # Function 2: quadratic term (orthogonalized)
    functions.append(lambda x: 1.5 * x**2 - 0.5)
    
    # Function 3: cubic term (orthogonalized)
    functions.append(lambda x: 2.5 * x**3 - 1.5 * x)
    
    # Add more orthogonal functions as needed
    for i in range(4, n_terms):
        # Use simple trigonometric basis
        if i % 2 == 0:
            k = i // 2
            functions.append(lambda x, k=k: np.cos(k * np.pi * x))
        else:
            k = i // 2
            functions.append(lambda x, k=k: np.sin(k * np.pi * x))
    
    return functions[:n_terms], x_range