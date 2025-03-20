import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

# Import centralized logging utilities
from models.logging_config import get_logger, header, section
from models.logging_config import bold, green, yellow, red, cyan
from models.logging_config import bold_green, bold_yellow, bold_red

def compare_theory_to_simulation(model, x_range, n_samples_range, n_trials=100):
    """
    Compare theoretical success probability to simulation results.
    
    Parameters:
    -----------
    model : SINDyMarkovModel
        Model instance to use for comparison
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
    logger = model.logger
    
    # Create a colorful header for the experiment
    logger.info(header("STARTING THEORY VS SIMULATION COMPARISON"))
    logger.info(bold("Experiment Setup:"))
    logger.info(f"  {bold('Testing')} {len(x_range)} {bold('data ranges:')} {x_range}")
    logger.info(f"  {bold('Testing')} {len(n_samples_range)} {bold('sample sizes:')} {n_samples_range}")
    logger.info(f"  {bold('Running')} {n_trials} {bold('trials per configuration')}")
    logger.info(f"  {bold('Lambda/Sigma Ratio:')} {model.threshold/model.sigma:.4f}")
    logger.info("-"*80)
    
    results = []
    
    total_combinations = len(x_range) * len(n_samples_range)
    progress_counter = 0
    
    for data_range in x_range:
        for n_samples in n_samples_range:
            progress_counter += 1
            logger.info(bold_yellow(f"\n{'='*60}"))
            logger.info(bold_yellow(f"CONFIGURATION {progress_counter}/{total_combinations}"))
            logger.info(bold_yellow(f"Data Range: {data_range}, Samples: {n_samples}"))
            logger.info(bold_yellow(f"{'='*60}"))
            
            # Generate sample points
            x_data = np.random.uniform(-data_range, data_range, n_samples)
            
            # Compute Gram matrix
            model.compute_gram_matrix(x_data)
            
            # Calculate theoretical success probability
            theoretical_prob = model.calculate_success_probability()
            
            # Simulate STLSQ
            empirical_prob = model.simulate_stlsq(x_data, n_trials)
            
            # Calculate discriminability
            discriminabilities = []
            if model.n_terms >= 2:
                # Calculate discriminability between each pair of true and false terms
                true_indices = model.normalize_state(model.true_term_indices)
                false_indices = model.normalize_state(set(range(model.n_terms)) - true_indices)
                
                # If there are both true and false terms
                if true_indices and false_indices:
                    for true_idx in true_indices:
                        for false_idx in false_indices:
                            # Evaluate terms at sample points
                            theta_true = model.library_functions[true_idx](x_data)
                            theta_false = model.library_functions[false_idx](x_data)
                            
                            # Calculate discriminability
                            disc = np.sum((theta_true - theta_false)**2) / model.sigma**2
                            discriminabilities.append((true_idx, false_idx, disc))
            
            # Use average discriminability if available, otherwise NaN
            if discriminabilities:
                avg_discriminability = np.mean([d[2] for d in discriminabilities])
            else:
                avg_discriminability = np.nan
            
            # Save results
            results.append({
                'data_range': data_range,
                'n_samples': n_samples,
                'theoretical_prob': theoretical_prob,
                'empirical_prob': empirical_prob,
                'discriminability': avg_discriminability,
                'lambda_sigma_ratio': model.threshold / model.sigma,
                'log_gram_det': model.log_gram_det,
                'discrepancy': abs(theoretical_prob - empirical_prob)
            })
            
            # Log summary of this configuration
            logger.info(section("CONFIGURATION SUMMARY", color_func=bold_green))
            logger.info(f"{bold('Data Range:')} {data_range}, {bold('Samples:')} {n_samples}")
            logger.info(f"{bold('Log Determinant of Gram Matrix:')} {model.log_gram_det:.4f}")
            
            # Log discriminability information
            if discriminabilities:
                logger.info(f"{bold('Discriminability Details:')}")
                for true_idx, false_idx, disc in discriminabilities:
                    logger.info(f"  Term {true_idx}(true) vs Term {false_idx}(false): {disc:.4f}")
                logger.info(f"{bold('Average Discriminability:')} {avg_discriminability:.4f}")
            else:
                logger.info(f"{bold('Discriminability:')} {avg_discriminability:.4f}")
            
            # Log success probabilities with highlighting for discrepancies
            logger.info(f"{bold('Theoretical Success Probability:')} {theoretical_prob:.4f}")
            logger.info(f"{bold('Empirical Success Probability:')} {empirical_prob:.4f}")
            
            discrepancy = abs(theoretical_prob - empirical_prob)
            if discrepancy > 0.1:
                logger.info(f"{bold_red('Large Discrepancy:')} {discrepancy:.4f}")
            else:
                logger.info(f"{bold('Difference:')} {discrepancy:.4f}")
            
            logger.info("-"*80 + "\n\n")
    
    logger.info(header("THEORY VS SIMULATION COMPARISON COMPLETE"))
    
    return pd.DataFrame(results)import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

def compare_theory_to_simulation(model, x_range, n_samples_range, n_trials=100):
    """
    Compare theoretical success probability to simulation results.
    
    Parameters:
    -----------
    model : SINDyMarkovModel
        Model instance to use for comparison
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
    logger = model.logger
    
    # Import enhanced logger utilities if available
    try:
        from enhanced_logger import cyan, green, yellow, red, bold, header
        use_enhanced_logger = True
    except ImportError:
        use_enhanced_logger = False
        
        # Define simple functions as fallback
        def cyan(text, bright=False): return text
        def green(text, bright=False): return text
        def yellow(text, bright=False): return text
        def red(text, bright=False): return text
        def bold(text): return text
        def header(text, width=80, char='='): return char*width
    
    # Create a colorful header for the experiment
    logger.info("\n\n" + "="*80)
    logger.info(cyan(bold("STARTING THEORY VS SIMULATION COMPARISON"), True))
    logger.info("="*80)
    logger.info(bold("Experiment Setup:"))
    logger.info(f"  {bold('Testing')} {len(x_range)} {bold('data ranges:')} {x_range}")
    logger.info(f"  {bold('Testing')} {len(n_samples_range)} {bold('sample sizes:')} {n_samples_range}")
    logger.info(f"  {bold('Running')} {n_trials} {bold('trials per configuration')}")
    logger.info(f"  {bold('Lambda/Sigma Ratio:')} {model.threshold/model.sigma:.4f}")
    logger.info("-"*80)
    
    results = []
    
    total_combinations = len(x_range) * len(n_samples_range)
    progress_counter = 0
    
    for data_range in x_range:
        for n_samples in n_samples_range:
            progress_counter += 1
            logger.info(f"\n{yellow('='*60, True)}")
            logger.info(yellow(bold(f"CONFIGURATION {progress_counter}/{total_combinations}"), True))
            logger.info(yellow(bold(f"Data Range: {data_range}, Samples: {n_samples}"), True))
            logger.info(yellow('='*60, True))
            
            # Generate sample points
            x_data = np.random.uniform(-data_range, data_range, n_samples)
            
            # Compute Gram matrix
            model.compute_gram_matrix(x_data)
            
            # Calculate theoretical success probability
            theoretical_prob = model.calculate_success_probability()
            
            # Simulate STLSQ
            empirical_prob = model.simulate_stlsq(x_data, n_trials)
            
            # Calculate discriminability
            discriminabilities = []
            if model.n_terms >= 2:
                # Calculate discriminability between each pair of true and false terms
                true_indices = model.normalize_state(model.true_term_indices)
                false_indices = model.normalize_state(set(range(model.n_terms)) - true_indices)
                
                # If there are both true and false terms
                if true_indices and false_indices:
                    for true_idx in true_indices:
                        for false_idx in false_indices:
                            # Evaluate terms at sample points
                            theta_true = model.library_functions[true_idx](x_data)
                            theta_false = model.library_functions[false_idx](x_data)
                            
                            # Calculate discriminability
                            disc = np.sum((theta_true - theta_false)**2) / model.sigma**2
                            discriminabilities.append((true_idx, false_idx, disc))
            
            # Use average discriminability if available, otherwise NaN
            if discriminabilities:
                avg_discriminability = np.mean([d[2] for d in discriminabilities])
            else:
                avg_discriminability = np.nan
            
            # Save results
            results.append({
                'data_range': data_range,
                'n_samples': n_samples,
                'theoretical_prob': theoretical_prob,
                'empirical_prob': empirical_prob,
                'discriminability': avg_discriminability,
                'lambda_sigma_ratio': model.threshold / model.sigma,
                'log_gram_det': model.log_gram_det,
                'discrepancy': abs(theoretical_prob - empirical_prob)
            })
            
            # Log summary of this configuration
            logger.info("\n" + "-"*80)
            logger.info(green(bold("CONFIGURATION SUMMARY")))
            logger.info("-"*80)
            logger.info(f"{bold('Data Range:')} {data_range}, {bold('Samples:')} {n_samples}")
            logger.info(f"{bold('Log Determinant of Gram Matrix:')} {model.log_gram_det:.4f}")
            
            # Log discriminability information
            if discriminabilities:
                logger.info(f"{bold('Discriminability Details:')}")
                for true_idx, false_idx, disc in discriminabilities:
                    logger.info(f"  Term {true_idx}(true) vs Term {false_idx}(false): {disc:.4f}")
                logger.info(f"{bold('Average Discriminability:')} {avg_discriminability:.4f}")
            else:
                logger.info(f"{bold('Discriminability:')} {avg_discriminability:.4f}")
            
            # Log success probabilities with highlighting for discrepancies
            logger.info(f"{bold('Theoretical Success Probability:')} {theoretical_prob:.4f}")
            logger.info(f"{bold('Empirical Success Probability:')} {empirical_prob:.4f}")
            
            discrepancy = abs(theoretical_prob - empirical_prob)
            if discrepancy > 0.1:
                logger.info(f"{bold(red('Large Discrepancy:'))} {discrepancy:.4f}")
            else:
                logger.info(f"{bold('Difference:')} {discrepancy:.4f}")
            
            logger.info("-"*80 + "\n\n")
    
    logger.info("="*80)
    logger.info(cyan(bold("THEORY VS SIMULATION COMPARISON COMPLETE")))
    logger.info("="*80 + "\n\n")
    
    return pd.DataFrame(results)

def plot_comparison(results_df, x_axis='log_gram_det'):
    """
    Plot comparison of theoretical and empirical results.
    
    Parameters:
    -----------
    results_df : DataFrame
        Results from compare_theory_to_simulation
    x_axis : str
        Which variable to use for x-axis ('log_gram_det', 'discriminability', 'data_range', or 'n_samples')
        
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
    
    # Set axis properties based on x_axis choice
    if x_axis == 'log_gram_det':
        ax.set_title('Success Probability vs Log Determinant of Gram Matrix')
        ax.set_xlabel('Log Determinant of Gram Matrix')
    elif x_axis == 'discriminability':
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

def plot_direct_comparison(results_df):
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
    scatter = sns.scatterplot(
        data=results_df, 
        x='theoretical_prob', 
        y='empirical_prob',
        hue='log_gram_det',
        palette='viridis',
        s=80,
        alpha=0.7
    )
    
    # Add colorbar label if using log_gram_det for color
    if 'log_gram_det' in results_df.columns:
        norm = plt.Normalize(results_df['log_gram_det'].min(), results_df['log_gram_det'].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Log Determinant of Gram Matrix')
    
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

def evaluate_model(results_df):
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
    # Calculate overall metrics
    r2 = r2_score(results_df['empirical_prob'], results_df['theoretical_prob'])
    rmse = np.sqrt(mean_squared_error(results_df['empirical_prob'], results_df['theoretical_prob']))
    mae = np.mean(np.abs(results_df['empirical_prob'] - results_df['theoretical_prob']))
    bias = np.mean(results_df['theoretical_prob'] - results_df['empirical_prob'])
    
    # Divide results into regions based on log_gram_det
    log_gram_regions = {}
    if 'log_gram_det' in results_df.columns:
        # Define regions
        min_val = results_df['log_gram_det'].min()
        max_val = results_df['log_gram_det'].max()
        
        # Divide into three regions: low, medium, high
        third = (max_val - min_val) / 3
        low_thresh = min_val + third
        high_thresh = max_val - third
        
        # Create masks for each region
        low_mask = results_df['log_gram_det'] <= low_thresh
        medium_mask = (results_df['log_gram_det'] > low_thresh) & (results_df['log_gram_det'] < high_thresh)
        high_mask = results_df['log_gram_det'] >= high_thresh
        
        # Calculate metrics for each region
        regions = {
            'low': {
                'range': f"≤ {low_thresh:.2f}",
                'n_samples': low_mask.sum(),
                'r2': r2_score(results_df.loc[low_mask, 'empirical_prob'], 
                              results_df.loc[low_mask, 'theoretical_prob']) if low_mask.sum() > 1 else np.nan,
                'rmse': np.sqrt(mean_squared_error(results_df.loc[low_mask, 'empirical_prob'], 
                                               results_df.loc[low_mask, 'theoretical_prob'])) if low_mask.sum() > 1 else np.nan,
                'bias': np.mean(results_df.loc[low_mask, 'theoretical_prob'] - 
                               results_df.loc[low_mask, 'empirical_prob']) if low_mask.sum() > 0 else np.nan
            },
            'medium': {
                'range': f"{low_thresh:.2f} - {high_thresh:.2f}",
                'n_samples': medium_mask.sum(),
                'r2': r2_score(results_df.loc[medium_mask, 'empirical_prob'], 
                              results_df.loc[medium_mask, 'theoretical_prob']) if medium_mask.sum() > 1 else np.nan,
                'rmse': np.sqrt(mean_squared_error(results_df.loc[medium_mask, 'empirical_prob'], 
                                               results_df.loc[medium_mask, 'theoretical_prob'])) if medium_mask.sum() > 1 else np.nan,
                'bias': np.mean(results_df.loc[medium_mask, 'theoretical_prob'] - 
                               results_df.loc[medium_mask, 'empirical_prob']) if medium_mask.sum() > 0 else np.nan
            },
            'high': {
                'range': f"≥ {high_thresh:.2f}",
                'n_samples': high_mask.sum(),
                'r2': r2_score(results_df.loc[high_mask, 'empirical_prob'], 
                              results_df.loc[high_mask, 'theoretical_prob']) if high_mask.sum() > 1 else np.nan,
                'rmse': np.sqrt(mean_squared_error(results_df.loc[high_mask, 'empirical_prob'], 
                                               results_df.loc[high_mask, 'theoretical_prob'])) if high_mask.sum() > 1 else np.nan,
                'bias': np.mean(results_df.loc[high_mask, 'theoretical_prob'] - 
                               results_df.loc[high_mask, 'empirical_prob']) if high_mask.sum() > 0 else np.nan
            }
        }
        log_gram_regions = regions
    
    # Divide results into regions based on discriminability
    discriminability_regions = {}
    if 'discriminability' in results_df.columns:
        # Define regions based on typical interpretations
        low_mask = results_df['discriminability'] < 1.0
        medium_mask = (results_df['discriminability'] >= 1.0) & (results_df['discriminability'] < 10.0)
        high_mask = results_df['discriminability'] >= 10.0
        
        # Calculate metrics for each region
        regions = {
            'low': {
                'range': "< 1.0",
                'n_samples': low_mask.sum(),
                'r2': r2_score(results_df.loc[low_mask, 'empirical_prob'], 
                              results_df.loc[low_mask, 'theoretical_prob']) if low_mask.sum() > 1 else np.nan,
                'rmse': np.sqrt(mean_squared_error(results_df.loc[low_mask, 'empirical_prob'], 
                                               results_df.loc[low_mask, 'theoretical_prob'])) if low_mask.sum() > 1 else np.nan
            },
            'medium': {
                'range': "1.0 - 10.0",
                'n_samples': medium_mask.sum(),
                'r2': r2_score(results_df.loc[medium_mask, 'empirical_prob'], 
                              results_df.loc[medium_mask, 'theoretical_prob']) if medium_mask.sum() > 1 else np.nan,
                'rmse': np.sqrt(mean_squared_error(results_df.loc[medium_mask, 'empirical_prob'], 
                                               results_df.loc[medium_mask, 'theoretical_prob'])) if medium_mask.sum() > 1 else np.nan
            },
            'high': {
                'range': "> 10.0",
                'n_samples': high_mask.sum(),
                'r2': r2_score(results_df.loc[high_mask, 'empirical_prob'], 
                              results_df.loc[high_mask, 'theoretical_prob']) if high_mask.sum() > 1 else np.nan,
                'rmse': np.sqrt(mean_squared_error(results_df.loc[high_mask, 'empirical_prob'], 
                                               results_df.loc[high_mask, 'theoretical_prob'])) if high_mask.sum() > 1 else np.nan
            }
        }
        discriminability_regions = regions
    
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'n_samples': len(results_df),
        'regions': {
            'log_gram_det': log_gram_regions,
            'discriminability': discriminability_regions
        }
    }
    
    return metrics

def print_metrics(metrics):
    """Print evaluation metrics in a readable format."""
    print("\n===== Model Evaluation Metrics =====")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Bias: {metrics['bias']:.4f}")
    print(f"Number of samples: {metrics['n_samples']}")
    
    if 'regions' in metrics and metrics['regions']:
        if 'log_gram_det' in metrics['regions']:
            print("\nMetrics by Log Gram Determinant Region:")
            for region_name, region_data in metrics['regions']['log_gram_det'].items():
                print(f"  {region_name.capitalize()} ({region_data['range']}, n={region_data['n_samples']}):")
                print(f"    R²: {region_data['r2']:.4f}" if not np.isnan(region_data['r2']) else "    R²: N/A")
                print(f"    RMSE: {region_data['rmse']:.4f}" if not np.isnan(region_data['rmse']) else "    RMSE: N/A")
                print(f"    Bias: {region_data['bias']:.4f}" if not np.isnan(region_data['bias']) else "    Bias: N/A")
        
        if 'discriminability' in metrics['regions']:
            print("\nMetrics by Discriminability Region:")
            regions = [('low', 'Low (D < 1)'), ('medium', 'Medium (1 ≤ D < 10)'), ('high', 'High (D ≥ 10)')]
            for region_key, region_label in regions:
                region_data = metrics['regions']['discriminability'][region_key]
                print(f"  {region_label} (n={region_data['n_samples']}):")
                print(f"    R²: {region_data['r2']:.4f}" if not np.isnan(region_data['r2']) else "    R²: N/A")
                print(f"    RMSE: {region_data['rmse']:.4f}" if not np.isnan(region_data['rmse']) else "    RMSE: N/A")
    
    print("=====================================\n")