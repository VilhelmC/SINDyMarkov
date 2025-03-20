#!/usr/bin/env python3
"""
Utility script to analyze SINDy Markov Chain log files.

This script parses log files and provides insights into model performance,
transition probabilities, and areas that might need attention.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import defaultdict
import numpy as np

def parse_log_file(log_file):
    """Parse a log file and extract key information."""
    print(f"Analyzing log file: {log_file}")
    
    # Store information by category
    data = {
        'transitions': [],
        'state_probs': [],
        'log_gram_det': [],
        'theoretical_probs': [],
        'empirical_probs': []
    }
    
    current_section = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract log gram determinant
            match = re.search(r'Log determinant of Gram matrix: ([\-0-9\.]+)', line)
            if match:
                data['log_gram_det'].append(float(match.group(1)))
                
            # Extract transition probabilities
            match = re.search(r'Transition (\{.*?\}) -> (\{.*?\}): prob = ([0-9\.]+)', line)
            if match:
                from_state = eval(match.group(1).replace('{', '[').replace('}', ']'))
                to_state = eval(match.group(2).replace('{', '[').replace('}', ']'))
                prob = float(match.group(3))
                data['transitions'].append({
                    'from_state': from_state,
                    'to_state': to_state,
                    'prob': prob,
                    'from_size': len(from_state),
                    'to_size': len(to_state),
                    'terms_removed': len(from_state) - len(to_state)
                })
            
            # Check for state probability section
            if "State Reachability Probabilities:" in line:
                current_section = 'state_probs'
                continue
                
            # Extract state probabilities
            if current_section == 'state_probs' and '  State' in line:
                match = re.search(r'State (\{.*?\}): ([0-9\.]+)', line)
                if match:
                    state = eval(match.group(1).replace('{', '[').replace('}', ']'))
                    prob = float(match.group(2))
                    data['state_probs'].append({
                        'state': state,
                        'prob': prob,
                        'size': len(state)
                    })
            
            # Check if we're leaving the state probs section
            if current_section == 'state_probs' and not line.strip().startswith('  State'):
                current_section = None
            
            # Extract theoretical and empirical probabilities
            match = re.search(r'Results - Theoretical: ([0-9\.]+), Empirical: ([0-9\.]+)', line)
            if match:
                theoretical = float(match.group(1))
                empirical = float(match.group(2))
                data['theoretical_probs'].append(theoretical)
                data['empirical_probs'].append(empirical)
    
    return data

def analyze_transitions(transitions_data):
    """Analyze transition probability data."""
    if not transitions_data:
        print("No transition data found.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(transitions_data)
    
    # Summary statistics
    print(f"\nTransition Analysis ({len(df)} transitions):")
    print(f"Average transition probability: {df['prob'].mean():.4f}")
    print(f"Median transition probability: {df['prob'].median():.4f}")
    print(f"Min transition probability: {df['prob'].min():.4f}")
    print(f"Max transition probability: {df['prob'].max():.4f}")
    
    # Group by number of terms removed
    terms_removed_stats = df.groupby('terms_removed')['prob'].agg(['mean', 'median', 'count'])
    print("\nTransition probability by terms removed:")
    print(terms_removed_stats)
    
    # Group by from_size
    from_size_stats = df.groupby('from_size')['prob'].agg(['mean', 'median', 'count'])
    print("\nTransition probability by from_state size:")
    print(from_size_stats)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot transition probabilities by terms removed
    plt.subplot(1, 2, 1)
    sns.boxplot(x='terms_removed', y='prob', data=df)
    plt.title('Transition Probability by Terms Removed')
    plt.xlabel('Number of Terms Removed')
    plt.ylabel('Transition Probability')
    
    # Plot transition probabilities by from_size
    plt.subplot(1, 2, 2)
    sns.boxplot(x='from_size', y='prob', data=df)
    plt.title('Transition Probability by From State Size')
    plt.xlabel('From State Size')
    plt.ylabel('Transition Probability')
    
    plt.tight_layout()
    plt.savefig('results/transition_analysis.png', dpi=300, bbox_inches='tight')
    
    return df

def analyze_state_probabilities(state_probs_data):
    """Analyze state probability data."""
    if not state_probs_data:
        print("No state probability data found.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(state_probs_data)
    
    # Summary statistics
    print(f"\nState Probability Analysis ({len(df)} states):")
    print(f"Average state probability: {df['prob'].mean():.4f}")
    print(f"Median state probability: {df['prob'].median():.4f}")
    
    # Group by state size
    size_stats = df.groupby('size')['prob'].agg(['mean', 'median', 'count', 'sum'])
    print("\nState probability by state size:")
    print(size_stats)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='size', y='prob', data=df)
    plt.title('State Probability by State Size')
    plt.xlabel('State Size')
    plt.ylabel('Probability')
    plt.savefig('results/state_prob_analysis.png', dpi=300, bbox_inches='tight')
    
    return df

def analyze_log_gram_det(log_gram_det_data, theoretical_probs, empirical_probs):
    """Analyze log gram determinant data."""
    if not log_gram_det_data or not theoretical_probs or not empirical_probs:
        print("Insufficient data for log gram determinant analysis.")
        return
    
    # Ensure all arrays are the same length
    min_len = min(len(log_gram_det_data), len(theoretical_probs), len(empirical_probs))
    log_gram_det_data = log_gram_det_data[:min_len]
    theoretical_probs = theoretical_probs[:min_len]
    empirical_probs = empirical_probs[:min_len]
    
    # Create DataFrame
    df = pd.DataFrame({
        'log_gram_det': log_gram_det_data,
        'theoretical_prob': theoretical_probs,
        'empirical_prob': empirical_probs,
        'diff': np.abs(np.array(theoretical_probs) - np.array(empirical_probs))
    })
    
    # Summary statistics
    print(f"\nLog Gram Determinant Analysis ({len(df)} samples):")
    print(f"Range: {df['log_gram_det'].min():.4f} to {df['log_gram_det'].max():.4f}")
    print(f"Mean: {df['log_gram_det'].mean():.4f}")
    print(f"Correlation with theoretical prob: {df['log_gram_det'].corr(df['theoretical_prob']):.4f}")
    print(f"Correlation with empirical prob: {df['log_gram_det'].corr(df['empirical_prob']):.4f}")
    
    # Create plots
    plt.figure(figsize=(15, 5))
    
    # Plot log_gram_det vs theoretical probability
    plt.subplot(1, 3, 1)
    plt.scatter(df['log_gram_det'], df['theoretical_prob'])
    plt.title('Log Gram Det vs Theoretical Prob')
    plt.xlabel('Log Gram Determinant')
    plt.ylabel('Theoretical Probability')
    
    # Plot log_gram_det vs empirical probability
    plt.subplot(1, 3, 2)
    plt.scatter(df['log_gram_det'], df['empirical_prob'])
    plt.title('Log Gram Det vs Empirical Prob')
    plt.xlabel('Log Gram Determinant')
    plt.ylabel('Empirical Probability')
    
    # Plot log_gram_det vs difference between empirical and theoretical
    plt.subplot(1, 3, 3)
    plt.scatter(df['log_gram_det'], df['diff'])
    plt.title('Log Gram Det vs |Theory-Empirical|')
    plt.xlabel('Log Gram Determinant')
    plt.ylabel('|Theoretical - Empirical|')
    
    plt.tight_layout()
    plt.savefig('results/log_gram_det_analysis.png', dpi=300, bbox_inches='tight')
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Analyze SINDy Markov Chain log files')
    parser.add_argument('--log-file', type=str, help='Path to log file to analyze')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory containing log files')
    args = parser.parse_args()
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    if args.log_file:
        # Analyze a specific log file
        data = parse_log_file(args.log_file)
        analyze_transitions(data['transitions'])
        analyze_state_probabilities(data['state_probs'])
        analyze_log_gram_det(data['log_gram_det'], data['theoretical_probs'], data['empirical_probs'])
    else:
        # Analyze all log files in the directory
        log_dir = args.log_dir
        if not os.path.exists(log_dir):
            print(f"Log directory {log_dir} not found.")
            return
            
        log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.log')]
        
        if not log_files:
            print(f"No log files found in {log_dir}.")
            return
            
        all_transitions = []
        all_state_probs = []
        all_log_gram_det = []
        all_theoretical_probs = []
        all_empirical_probs = []
        
        for log_file in log_files:
            data = parse_log_file(log_file)
            all_transitions.extend(data['transitions'])
            all_state_probs.extend(data['state_probs'])
            all_log_gram_det.extend(data['log_gram_det'])
            all_theoretical_probs.extend(data['theoretical_probs'])
            all_empirical_probs.extend(data['empirical_probs'])
        
        analyze_transitions(all_transitions)
        analyze_state_probabilities(all_state_probs)
        analyze_log_gram_det(all_log_gram_det, all_theoretical_probs, all_empirical_probs)

if __name__ == "__main__":
    main()