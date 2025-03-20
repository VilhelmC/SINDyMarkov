import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import os

# Set up visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Check if results exist
def load_results(filepath):
    """Load results from CSV file if it exists."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print(f"Warning: {filepath} not found.")
        return None

# Load all experiment results
simple_results = load_results('results/simple_example_results.csv')
lambda_sigma_results = load_results('results/lambda_sigma_experiment_results.csv')
multiterm_results = load_results('results/multiterm_experiment_results.csv')

# Function to analyze individual experiment results
def analyze_experiment(results, title):
    """Analyze and visualize results from a single experiment."""
    if results is None:
        print(f"No data available for: {title}")
        return
    
    print(f"\n{title}")
    print("="*len(title))
    
    # Basic statistics
    print(f"Number of data points: {len(results)}")
    if 'discriminability' in results.columns:
        print(f"Discriminability range: {results['discriminability'].min():.2f} to {results['discriminability'].max():.2f}")
    
    # Calculate performance metrics
    r2 = r2_score(results['empirical_prob'], results['theoretical_prob'])
    rmse = np.sqrt(mean_squared_error(results['empirical_prob'], results['theoretical_prob']))
    bias = np.mean(results['theoretical_prob'] - results['empirical_prob'])
    
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Bias: {bias:.4f}")
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot 1: Discriminability vs Success Probability
    if 'discriminability' in results.columns:
        ax1.scatter(results['discriminability'], results['empirical_prob'], 
                   label='Empirical', alpha=0.7, s=80, color='blue')
        
        # Sort for smoother line
        sorted_idx = results['discriminability'].argsort()
        ax1.plot(results['discriminability'].iloc[sorted_idx], 
                results['theoretical_prob'].iloc[sorted_idx], 
                label='Theoretical', linewidth=2, color='red')
        
        ax1.set_xscale('log')
        ax1.set_xlabel('Discriminability (D)')
        ax1.set_ylabel('Success Probability')
        ax1.set_title('Success Probability vs Discriminability')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "Discriminability data not available", 
                ha='center', va='center', fontsize=14)
        ax1.axis('off')
    
    # Plot 2: Direct Comparison
    ax2.scatter(results['theoretical_prob'], results['empirical_prob'], 
               alpha=0.7, s=80)
    
    # Add 1:1 line
    min_val = min(results['theoretical_prob'].min(), results['empirical_prob'].min())
    max_val = max(results['theoretical_prob'].max(), results['empirical_prob'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
    
    # Add metrics to plot
    ax2.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nBias = {bias:.4f}',
            transform=ax2.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Theoretical Success Probability')
    ax2.set_ylabel('Empirical Success Probability')
    ax2.set_title('Theoretical vs Empirical Comparison')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis based on experiment parameters
    if 'n_samples' in results.columns and 'data_range' in results.columns:
        plt.figure(figsize=(12, 8))
        pivot = results.pivot_table(
            index='data_range',
            columns='n_samples',
            values=['empirical_prob', 'theoretical_prob']
        )
        
        # Calculate residuals
        residuals = results['empirical_prob'] - results['theoretical_prob']
        
        # Plot residuals vs discriminability
        plt.figure(figsize=(10, 6))
        if 'discriminability' in results.columns:
            plt.scatter(results['discriminability'], residuals, alpha=0.7, s=70)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xscale('log')
            plt.xlabel('Discriminability (D)')
            plt.ylabel('Residual (Empirical - Theoretical)')
            plt.title('Residual Analysis')
            plt.grid(True, alpha=0.3)
            plt.show()

# Function to analyze lambda/sigma ratio experiment
def analyze_lambda_sigma_experiment(results):
    """Analyze the effect of lambda/sigma ratio on model performance."""
    if results is None:
        print("No lambda/sigma experiment data available.")
        return
    
    print("\nLambda/Sigma Ratio Experiment Analysis")
    print("======================================")
    
    # Get unique ratios
    ratios = sorted(results['lambda_sigma_ratio'].unique())
    print(f"Lambda/Sigma ratios tested: {ratios}")
    
    # Calculate metrics for each ratio
    metrics_by_ratio = []
    for ratio in ratios:
        ratio_results = results[results['lambda_sigma_ratio'] == ratio]
        r2 = r2_score(ratio_results['empirical_prob'], ratio_results['theoretical_prob'])
        rmse = np.sqrt(mean_squared_error(ratio_results['empirical_prob'], ratio_results['theoretical_prob']))
        bias = np.mean(ratio_results['theoretical_prob'] - ratio_results['empirical_prob'])
        
        metrics_by_ratio.append({
            'lambda_sigma_ratio': ratio,
            'r2': r2,
            'rmse': rmse,
            'bias': bias,
            'n_samples': len(ratio_results)
        })
    
    metrics_df = pd.DataFrame(metrics_by_ratio)
    print(metrics_df)
    
    # Plot metrics vs ratio
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1.plot(metrics_df['lambda_sigma_ratio'], metrics_df['r2'], marker='o', linewidth=2)
    ax1.set_xlabel('λ/σ Ratio')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² vs λ/σ Ratio')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(metrics_df['lambda_sigma_ratio'], metrics_df['rmse'], marker='o', linewidth=2, color='red')
    ax2.set_xlabel('λ/σ Ratio')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE vs λ/σ Ratio')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(metrics_df['lambda_sigma_ratio'], metrics_df['bias'], marker='o', linewidth=2, color='green')
    ax3.set_xlabel('λ/σ Ratio')
    ax3.set_ylabel('Bias')
    ax3.set_title('Bias vs λ/σ Ratio')
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot success probability curves for different ratios
    plt.figure(figsize=(12, 8))
    
    for ratio in ratios:
        ratio_results = results[results['lambda_sigma_ratio'] == ratio]
        # Sort for smoother line
        sorted_idx = ratio_results['discriminability'].argsort()
        
        plt.scatter(ratio_results['discriminability'], ratio_results['empirical_prob'], 
                   label=f'Empirical λ/σ={ratio}', alpha=0.5, s=60)
        plt.plot(ratio_results['discriminability'].iloc[sorted_idx], 
                ratio_results['theoretical_prob'].iloc[sorted_idx], 
                label=f'Theory λ/σ={ratio}', linestyle='--')
    
    plt.xscale('log')
    plt.xlabel('Discriminability (D)')
    plt.ylabel('Success Probability')
    plt.title('Effect of λ/σ Ratio on Success Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot phase diagram with color based on ratio
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(results['discriminability'], results['empirical_prob'], 
                         c=results['lambda_sigma_ratio'], cmap='viridis', 
                         s=80, alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('λ/σ Ratio')
    
    # Plot theoretical curves for each ratio
    for ratio in ratios:
        ratio_results = results[results['lambda_sigma_ratio'] == ratio]
        # Sort for smoother line
        sorted_idx = ratio_results['discriminability'].argsort()
        plt.plot(ratio_results['discriminability'].iloc[sorted_idx], 
                ratio_results['theoretical_prob'].iloc[sorted_idx], 
                'k--', alpha=0.5)
    
    plt.xscale('log')
    plt.xlabel('Discriminability (D)')
    plt.ylabel('Success Probability')
    plt.title('Phase Diagram with λ/σ Ratio')
    plt.grid(True, alpha=0.3)
    plt.show()

# Function to compare all experiments
def compare_all_experiments():
    """Compare results across all experiments."""
    if simple_results is None or multiterm_results is None:
        print("Not enough data to compare experiments.")
        return
    
    print("\nComparison Across Experiments")
    print("============================")
    
    # Combine results from different experiments
    simple_results_copy = simple_results.copy()
    simple_results_copy['experiment'] = 'Simple (1 true term)'
    
    multiterm_results_copy = multiterm_results.copy()
    multiterm_results_copy['experiment'] = 'Multiple (2 true terms)'
    
    combined = pd.concat([simple_results_copy, multiterm_results_copy], ignore_index=True)
    
    # Summary metrics
    metrics_by_exp = []
    for exp in combined['experiment'].unique():
        exp_results = combined[combined['experiment'] == exp]
        r2 = r2_score(exp_results['empirical_prob'], exp_results['theoretical_prob'])
        rmse = np.sqrt(mean_squared_error(exp_results['empirical_prob'], exp_results['theoretical_prob']))
        
        metrics_by_exp.append({
            'experiment': exp,
            'r2': r2,
            'rmse': rmse,
            'n_samples': len(exp_results)
        })
    
    metrics_df = pd.DataFrame(metrics_by_exp)
    print(metrics_df)
    
    # Plot combined results
    plt.figure(figsize=(12, 8))
    
    for exp in combined['experiment'].unique():
        exp_results = combined[combined['experiment'] == exp]
        
        plt.scatter(exp_results['discriminability'], exp_results['empirical_prob'], 
                   label=f'{exp} (Empirical)', alpha=0.7, s=70)
        
        # Sort for smoother line
        sorted_idx = exp_results['discriminability'].argsort()
        plt.plot(exp_results['discriminability'].iloc[sorted_idx], 
                exp_results['theoretical_prob'].iloc[sorted_idx], 
                label=f'{exp} (Theory)')
    
    plt.xscale('log')
    plt.xlabel('Discriminability (D)')
    plt.ylabel('Success Probability')
    plt.title('Success Probability vs Discriminability Across Experiments')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Direct comparison plot
    plt.figure(figsize=(10, 8))
    
    for exp in combined['experiment'].unique():
        exp_results = combined[combined['experiment'] == exp]
        
        plt.scatter(exp_results['theoretical_prob'], exp_results['empirical_prob'], 
                   label=exp, alpha=0.7, s=70)
    
    # Add 1:1 line
    plt.plot([0, 1], [0, 1], 'k--', label='1:1 Line')
    
    plt.xlabel('Theoretical Success Probability')
    plt.ylabel('Empirical Success Probability')
    plt.title('Theory vs Empirical Across Experiments')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Run the analyses if the data exists
if simple_results is not None:
    analyze_experiment(simple_results, "Simple Three-Term Example")

if multiterm_results is not None:
    analyze_experiment(multiterm_results, "Multiple True Terms Experiment")

if lambda_sigma_results is not None:
    analyze_lambda_sigma_experiment(lambda_sigma_results)

# Compare all experiments
if simple_results is not None and multiterm_results is not None:
    compare_all_experiments()