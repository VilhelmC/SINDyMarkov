#!/usr/bin/env python3
"""
Convert SINDy Markov Chain log files to markdown reports.

This script parses log files and generates nicely formatted markdown reports
with embedded visualizations.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import glob
import json

# Function to strip ANSI color codes
def strip_ansi_codes(text):
    """Remove ANSI color codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def extract_experiment_metadata(log_content):
    """Extract experiment metadata from log file."""
    metadata = {
        'name': 'Unknown Experiment',
        'description': 'No description available',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Extract experiment name
    match = re.search(r'STARTING EXPERIMENT: (.+)', log_content)
    if match:
        metadata['name'] = match.group(1).strip()
    
    # Extract description
    match = re.search(r'Description: (.+)', log_content)
    if match:
        metadata['description'] = match.group(1).strip()
    
    # Extract parameters
    params = {}
    
    # Extract library functions
    match = re.search(r'Library Functions: (.+)', log_content)
    if match:
        params['library_functions'] = match.group(1).strip()
    
    # Extract coefficients
    match = re.search(r'True Coefficients: (.+)', log_content)
    if match:
        params['true_coefficients'] = match.group(1).strip()
    
    # Extract noise and threshold
    match = re.search(r'Sigma \(noise\): ([0-9\.]+)', log_content)
    if match:
        params['sigma'] = float(match.group(1))
    
    match = re.search(r'Threshold: ([0-9\.]+)', log_content)
    if match:
        params['threshold'] = float(match.group(1))
    
    match = re.search(r'Lambda/Sigma Ratio: ([0-9\.]+)', log_content)
    if match:
        params['lambda_sigma_ratio'] = float(match.group(1))
    
    # Add parameters to metadata
    metadata['parameters'] = params
    
    return metadata

def extract_success_probability_calculation(log_content):
    """Extract success probability calculation details from log file."""
    # Find the section
    pattern = r'BEGINNING SUCCESS PROBABILITY CALCULATION(.*?)END OF SUCCESS PROBABILITY CALCULATION'
    match = re.search(pattern, log_content, re.DOTALL)
    
    if not match:
        return {
            'found': False,
            'message': 'Success probability calculation section not found in log file.'
        }
    
    section = match.group(1)
    
    # Extract key information
    result = {
        'found': True,
        'valid_states': [],
        'reachability': {},
        'transitions': {},
        'theoretical_prob': None
    }
    
    # Extract valid states count
    states_match = re.search(r'Generated (\d+) valid states:', section)
    if states_match:
        result['states_count'] = int(states_match.group(1))
    
    # Extract individual states
    state_pattern = r'State \d+: \{([\d, ]+)\}'
    for state_match in re.finditer(state_pattern, section):
        state_str = state_match.group(1)
        state = [int(s.strip()) for s in state_str.split(',') if s.strip()]
        result['valid_states'].append(state)
    
    # Extract reachability probabilities
    reach_pattern = r'State \{([\d, ]+)\}: ([0-9\.]+)'
    for reach_match in re.finditer(reach_pattern, section):
        state_str = reach_match.group(1)
        prob = float(reach_match.group(2))
        state = tuple(int(s.strip()) for s in state_str.split(',') if s.strip())
        result['reachability'][state] = prob
    
    # Extract transition probabilities
    trans_pattern = r'Transitions from \{([\d, ]+)\}:(.*?)(?=Transitions from|\[STOP\]|TRUE MODEL STATE STOPPING PROBABILITY)'
    for trans_match in re.finditer(trans_pattern, section, re.DOTALL):
        from_state_str = trans_match.group(1)
        transitions_text = trans_match.group(2)
        
        from_state = tuple(int(s.strip()) for s in from_state_str.split(',') if s.strip())
        transitions = {}
        
        # Extract individual transitions
        to_pattern = r'-> \{([\d, ]+)\}: ([0-9\.]+)'
        for to_match in re.finditer(to_pattern, transitions_text):
            to_state_str = to_match.group(1)
            prob = float(to_match.group(2))
            to_state = tuple(int(s.strip()) for s in to_state_str.split(',') if s.strip())
            transitions[to_state] = prob
        
        # Extract stopping probability
        stop_match = re.search(r'-> \[STOP\]: ([0-9\.]+)', transitions_text)
        if stop_match:
            transitions['stop'] = float(stop_match.group(1))
        
        result['transitions'][from_state] = transitions
    
    # Extract theoretical success probability
    prob_match = re.search(r'Success probability = [0-9\.]+ × [0-9\.]+ = ([0-9\.]+)', section)
    if prob_match:
        result['theoretical_prob'] = float(prob_match.group(1))
    
    return result

def extract_empirical_results(log_content):
    """Extract empirical simulation results from log file."""
    result = {
        'found': False,
        'configs': []
    }
    
    # Find all configuration sections
    config_pattern = r'CONFIGURATION (\d+)/(\d+)(.*?)(?=CONFIGURATION|--------------------------------------------------------------------------------\n\n)'
    for config_match in re.finditer(config_pattern, log_content, re.DOTALL):
        config_num = int(config_match.group(1))
        total_configs = int(config_match.group(2))
        config_content = config_match.group(3)
        
        config_data = {
            'number': config_num,
            'total': total_configs
        }
        
        # Extract data range and samples
        params_match = re.search(r'Data Range: ([0-9\.]+), Samples: (\d+)', config_content)
        if params_match:
            config_data['data_range'] = float(params_match.group(1))
            config_data['n_samples'] = int(params_match.group(2))
        
        # Extract log gram determinant
        log_gram_match = re.search(r'Log Determinant of Gram Matrix: ([0-9\.\-]+)', config_content)
        if log_gram_match:
            config_data['log_gram_det'] = float(log_gram_match.group(1))
        
        # Extract discriminability
        disc_match = re.search(r'Average Discriminability: ([0-9\.]+)', config_content)
        if disc_match:
            config_data['discriminability'] = float(disc_match.group(1))
        
        # Extract theoretical probability
        theo_match = re.search(r'Theoretical Success Probability: ([0-9\.]+)', config_content)
        if theo_match:
            config_data['theoretical_prob'] = float(theo_match.group(1))
        
        # Extract empirical probability
        emp_match = re.search(r'Empirical Success Probability: ([0-9\.]+) \(from (\d+) trials\)', config_content)
        if emp_match:
            config_data['empirical_prob'] = float(emp_match.group(1))
            config_data['trials_used'] = int(emp_match.group(2))
        
        # Extract discrepancy
        disc_match = re.search(r'Difference: ([0-9\.]+)', config_content)
        large_disc_match = re.search(r'Large Discrepancy: ([0-9\.]+)', config_content)
        
        if large_disc_match:
            config_data['discrepancy'] = float(large_disc_match.group(1))
            config_data['large_discrepancy'] = True
        elif disc_match:
            config_data['discrepancy'] = float(disc_match.group(1))
            config_data['large_discrepancy'] = False
        
        result['configs'].append(config_data)
    
    if result['configs']:
        result['found'] = True
    
    return result

def extract_coefficient_analysis(log_content):
    """Extract coefficient distribution analysis from log file."""
    # Find coefficient analysis section
    pattern = r'Coefficient Distribution Analysis:(.*?)(?=================)'
    match = re.search(pattern, log_content, re.DOTALL)
    
    if not match:
        return {
            'found': False,
            'message': 'Coefficient distribution analysis not found in log file.'
        }
    
    section = match.group(1)
    
    # Extract state analyses
    result = {
        'found': True,
        'states': []
    }
    
    # Extract individual state analyses
    state_pattern = r'State \[([\d, ]+)\] \(observed (\d+) times\):(.*?)(?=State \[|$)'
    for state_match in re.finditer(state_pattern, section, re.DOTALL):
        state_str = state_match.group(1)
        obs_count = int(state_match.group(2))
        state_content = state_match.group(3)
        
        state_data = {
            'state': [int(s.strip()) for s in state_str.split(',') if s.strip()],
            'obs_count': obs_count,
            'coefficients': []
        }
        
        # Extract coefficient data
        coef_pattern = r'Coef (\d+): Theoretical: ([0-9\.\-]+), Empirical: ([0-9\.\-]+), Diff: ([0-9\.\-]+) \(([0-9\.\-inf]+)%\)'
        for coef_match in re.finditer(coef_pattern, state_content):
            coef_idx = int(coef_match.group(1))
            theoretical = float(coef_match.group(2))
            empirical = float(coef_match.group(3))
            diff = float(coef_match.group(4))
            
            # Handle inf% case
            pct_str = coef_match.group(5)
            if pct_str == 'inf':
                pct = float('inf')
            else:
                pct = float(pct_str)
            
            state_data['coefficients'].append({
                'index': coef_idx,
                'theoretical': theoretical,
                'empirical': empirical,
                'diff': diff,
                'diff_percent': pct
            })
        
        # Extract standard deviation data
        std_pattern = r'Coef (\d+): Theoretical: ([0-9\.\-]+), Empirical: ([0-9\.\-]+), Ratio: ([0-9\.\-NA]+)'
        for std_match in re.finditer(std_pattern, state_content):
            coef_idx = int(std_match.group(1))
            theoretical = float(std_match.group(2))
            empirical = float(std_match.group(3))
            
            # Handle N/A case
            ratio_str = std_match.group(4)
            if ratio_str == 'N/A':
                ratio = None
            else:
                ratio = float(ratio_str)
            
            # Find the corresponding coefficient
            for coef in state_data['coefficients']:
                if coef['index'] == coef_idx:
                    coef['theo_std'] = theoretical
                    coef['emp_std'] = empirical
                    coef['std_ratio'] = ratio
                    break
        
        result['states'].append(state_data)
    
    return result

def create_report(log_file, output_dir=None):
    """
    Create a markdown report from a log file.
    
    Parameters:
    -----------
    log_file : str
        Path to the log file
    output_dir : str
        Path to output directory for the report
    
    Returns:
    --------
    report_path : str
        Path to the generated report
    """
    # Read log file
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        log_content = f.read()
    
    # Strip ANSI color codes for easier parsing
    clean_content = strip_ansi_codes(log_content)
    
    # Extract information
    metadata = extract_experiment_metadata(clean_content)
    success_calc = extract_success_probability_calculation(clean_content)
    empirical_results = extract_empirical_results(clean_content)
    coef_analysis = extract_coefficient_analysis(clean_content)
    
    # Create report filename
    experiment_name = metadata['name'].replace(' ', '_').lower()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if output_dir is None:
        # Use the directory of the log file
        output_dir = os.path.dirname(log_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    report_filename = f"{experiment_name}_{timestamp}.md"
    report_path = os.path.join(output_dir, report_filename)
    
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create visualizations if empirical results are available
    success_vs_log_gram_path = None
    direct_comparison_path = None
    
    if empirical_results['found'] and len(empirical_results['configs']) > 0:
        # Create DataFrame from config results
        df = pd.DataFrame(empirical_results['configs'])
        
        if 'log_gram_det' in df.columns and 'theoretical_prob' in df.columns and 'empirical_prob' in df.columns:
            # Plot success probability vs log gram determinant
            plt.figure(figsize=(10, 6))
            plt.scatter(df['log_gram_det'], df['empirical_prob'], label='Empirical', s=80, alpha=0.7)
            plt.plot(df['log_gram_det'], df['theoretical_prob'], 'r-', label='Theoretical')
            plt.xlabel('Log Determinant of Gram Matrix')
            plt.ylabel('Success Probability')
            plt.title('Success Probability vs Log Determinant of Gram Matrix')
            plt.legend()
            plt.grid(alpha=0.3)
            
            success_vs_log_gram_path = os.path.join(viz_dir, f"{experiment_name}_success_vs_log_gram.png")
            plt.savefig(success_vs_log_gram_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot direct comparison
            plt.figure(figsize=(8, 8))
            plt.scatter(df['theoretical_prob'], df['empirical_prob'], s=80, alpha=0.7)
            
            # Add 1:1 line
            max_val = max(df['theoretical_prob'].max(), df['empirical_prob'].max())
            min_val = min(df['theoretical_prob'].min(), df['empirical_prob'].min())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
            
            plt.xlabel('Theoretical Success Probability')
            plt.ylabel('Empirical Success Probability')
            plt.title('Theoretical vs Empirical Success Probability')
            plt.grid(alpha=0.3)
            plt.legend()
            
            direct_comparison_path = os.path.join(viz_dir, f"{experiment_name}_direct_comparison.png")
            plt.savefig(direct_comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # If discriminability is available, create a discriminability plot
            if 'discriminability' in df.columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(df['discriminability'], df['empirical_prob'], label='Empirical', s=80, alpha=0.7)
                
                # Sort for better line plot
                df_sorted = df.sort_values('discriminability').copy()
                plt.plot(df_sorted['discriminability'], df_sorted['theoretical_prob'], 'r-', label='Theoretical')
                
                plt.xscale('log')
                plt.xlabel('Discriminability')
                plt.ylabel('Success Probability')
                plt.title('Success Probability vs Discriminability')
                plt.legend()
                plt.grid(alpha=0.3)
                
                disc_plot_path = os.path.join(viz_dir, f"{experiment_name}_success_vs_discriminability.png")
                plt.savefig(disc_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
    
    # Now generate the markdown report
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"# {metadata['name']} Report\n\n")
        f.write(f"*Generated on: {metadata['date']}*\n\n")
        f.write(f"## Experiment Description\n\n")
        f.write(f"{metadata['description']}\n\n")
        
        # Parameters
        f.write(f"## Experiment Parameters\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        
        for param, value in metadata['parameters'].items():
            f.write(f"| {param} | {value} |\n")
        
        f.write("\n")
        
        # Success probability calculation
        f.write(f"## Theoretical Success Probability Analysis\n\n")
        
        if success_calc['found']:
            f.write(f"### Valid States\n\n")
            f.write(f"Number of valid states: {success_calc.get('states_count', len(success_calc['valid_states']))}\n\n")
            
            if len(success_calc['valid_states']) <= 10:  # Only show all states if there are 10 or fewer
                f.write("States:\n\n")
                for i, state in enumerate(success_calc['valid_states']):
                    f.write(f"- State {i+1}: {state}\n")
                f.write("\n")
            
            f.write(f"### State Reachability Probabilities\n\n")
            
            # Sort by probability (highest first)
            sorted_reach = sorted(success_calc['reachability'].items(), key=lambda x: x[1], reverse=True)
            
            f.write("| State | Probability |\n")
            f.write("|-------|-------------|\n")
            
            for state, prob in sorted_reach[:10]:  # Show top 10
                f.write(f"| {state} | {prob:.6f} |\n")
            
            if len(sorted_reach) > 10:
                f.write("| ... | ... |\n")
            
            f.write("\n")
            
            f.write(f"### Transition Probabilities\n\n")
            
            f.write("Select transition probabilities (only showing non-zero probabilities):\n\n")
            
            # Find the true model state if available
            true_state = None
            max_reach_prob = 0
            
            for state, prob in success_calc['reachability'].items():
                if prob > max_reach_prob:
                    max_reach_prob = prob
                    true_state = state
            
            # First show transitions from the initial state
            initial_states = [s for s in success_calc['transitions'].keys() 
                             if len(s) == len(success_calc['valid_states'][0])]
            
            if initial_states:
                initial_state = initial_states[0]
                f.write(f"**From initial state {initial_state}:**\n\n")
                
                if initial_state in success_calc['transitions']:
                    trans = success_calc['transitions'][initial_state]
                    
                    for to_state, prob in trans.items():
                        if prob > 0:
                            if to_state == 'stop':
                                f.write(f"- To [STOP]: {prob:.6f}\n")
                            else:
                                f.write(f"- To {to_state}: {prob:.6f}\n")
                
                f.write("\n")
            
            # Then show transitions from the true model state if identified
            if true_state and true_state in success_calc['transitions']:
                f.write(f"**From true model state {true_state}:**\n\n")
                
                trans = success_calc['transitions'][true_state]
                
                for to_state, prob in trans.items():
                    if prob > 0:
                        if to_state == 'stop':
                            f.write(f"- To [STOP]: {prob:.6f}\n")
                        else:
                            f.write(f"- To {to_state}: {prob:.6f}\n")
                
                f.write("\n")
            
            f.write(f"### Theoretical Success Probability\n\n")
            f.write(f"**Final Success Probability: {success_calc['theoretical_prob']:.6f}**\n\n")
        else:
            f.write(f"{success_calc.get('message', 'Success probability calculation not found in log file.')}\n\n")
        
        # Empirical results
        f.write(f"## Empirical Simulation Results\n\n")
        
        if empirical_results['found']:
            # Create a summary table
            f.write("### Summary of Configurations\n\n")
            
            f.write("| Config | Data Range | Samples | Log Gram Det | Theoretical | Empirical | Trials | Discrepancy |\n")
            f.write("|--------|------------|---------|--------------|-------------|-----------|--------|-------------|\n")
            
            for config in empirical_results['configs']:
                config_num = config['number']
                data_range = config.get('data_range', 'N/A')
                n_samples = config.get('n_samples', 'N/A')
                log_gram_det = f"{config.get('log_gram_det', 'N/A'):.4f}" if 'log_gram_det' in config else 'N/A'
                theo_prob = f"{config.get('theoretical_prob', 'N/A'):.4f}" if 'theoretical_prob' in config else 'N/A'
                emp_prob = f"{config.get('empirical_prob', 'N/A'):.4f}" if 'empirical_prob' in config else 'N/A'
                trials = config.get('trials_used', 'N/A')
                
                discrepancy = config.get('discrepancy', 'N/A')
                if discrepancy != 'N/A':
                    if config.get('large_discrepancy', False):
                        discrepancy = f"**{discrepancy:.4f}**"
                    else:
                        discrepancy = f"{discrepancy:.4f}"
                
                f.write(f"| {config_num} | {data_range} | {n_samples} | {log_gram_det} | {theo_prob} | {emp_prob} | {trials} | {discrepancy} |\n")
            
            f.write("\n")
            
            # Add visualizations
            f.write("### Visualizations\n\n")
            
            if success_vs_log_gram_path:
                rel_path = os.path.relpath(success_vs_log_gram_path, output_dir)
                f.write(f"![Success vs Log Gram Det]({rel_path})\n\n")
            
            if direct_comparison_path:
                rel_path = os.path.relpath(direct_comparison_path, output_dir)
                f.write(f"![Direct Comparison]({rel_path})\n\n")
            
            disc_plot_path = os.path.join(viz_dir, f"{experiment_name}_success_vs_discriminability.png")
            if os.path.exists(disc_plot_path):
                rel_path = os.path.relpath(disc_plot_path, output_dir)
                f.write(f"![Success vs Discriminability]({rel_path})\n\n")
        else:
            f.write("No empirical simulation results found in log file.\n\n")
        
        # Coefficient Analysis
        f.write("## Coefficient Distribution Analysis\n\n")
        
        if coef_analysis['found'] and coef_analysis['states']:
            for i, state_data in enumerate(coef_analysis['states']):
                state = state_data['state']
                obs_count = state_data['obs_count']
                
                f.write(f"### State {state} (observed {obs_count} times)\n\n")
                
                f.write("#### Coefficient Means\n\n")
                f.write("| Coefficient | Theoretical | Empirical | Difference | % Difference |\n")
                f.write("|------------|-------------|-----------|------------|-------------|\n")
                
                for coef in state_data['coefficients']:
                    idx = coef['index']
                    theo = f"{coef['theoretical']:.6f}"
                    emp = f"{coef['empirical']:.6f}"
                    diff = f"{coef['diff']:.6f}"
                    
                    if coef['diff_percent'] == float('inf'):
                        diff_pct = "∞"
                    else:
                        diff_pct = f"{coef['diff_percent']:.1f}%"
                    
                    # Highlight large differences
                    if coef['diff_percent'] > 10 and abs(coef['diff']) > 0.01:
                        diff = f"**{diff}**"
                        diff_pct = f"**{diff_pct}**"
                    
                    f.write(f"| {idx} | {theo} | {emp} | {diff} | {diff_pct} |\n")
                
                f.write("\n")
                
                f.write("#### Standard Deviations\n\n")
                f.write("| Coefficient | Theoretical | Empirical | Ratio |\n")
                f.write("|------------|-------------|-----------|-------|\n")
                
                for coef in state_data['coefficients']:
                    if 'theo_std' in coef and 'emp_std' in coef:
                        idx = coef['index']
                        theo_std = f"{coef['theo_std']:.6f}"
                        emp_std = f"{coef['emp_std']:.6f}"
                        
                        if coef['std_ratio'] is None:
                            ratio = "N/A"
                        else:
                            ratio = f"{coef['std_ratio']:.4f}"
                            
                            # Highlight large ratios
                            if coef['std_ratio'] < 0.5 or coef['std_ratio'] > 2.0:
                                ratio = f"**{ratio}**"
                        
                        f.write(f"| {idx} | {theo_std} | {emp_std} | {ratio} |\n")
                
                f.write("\n")
        else:
            f.write(coef_analysis.get('message', 'No coefficient analysis found in log file.') + "\n\n")
    
    return report_path

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Convert SINDy log files to markdown reports')
    parser.add_argument('--log-file', type=str, help='Path to specific log file to convert')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory containing log files to convert')
    parser.add_argument('--output-dir', type=str, default='results/reports', help='Output directory for reports')
    parser.add_argument('--all', action='store_true', help='Process all log files in the directory')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.log_file:
        # Process a single log file
        if not os.path.exists(args.log_file):
            print(f"Log file not found: {args.log_file}")
            return
        
        report_path = create_report(args.log_file, args.output_dir)
        print(f"Report generated: {report_path}")
    elif args.all:
        # Process all log files in the directory
        if not os.path.exists(args.log_dir):
            print(f"Log directory not found: {args.log_dir}")
            return
        
        log_files = glob.glob(os.path.join(args.log_dir, '*.log'))
        if not log_files:
            print(f"No log files found in {args.log_dir}")
            return
        
        for log_file in log_files:
            try:
                report_path = create_report(log_file, args.output_dir)
                print(f"Report generated: {report_path}")
            except Exception as e:
                print(f"Error processing {log_file}: {str(e)}")
    else:
        print("Please specify either --log-file or --all")

if __name__ == "__main__":
    main()