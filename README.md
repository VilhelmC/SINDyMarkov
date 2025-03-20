# SINDy Markov Chain Model

This project implements an analytical Markov chain model for predicting success probabilities in the Sparse Identification of Nonlinear Dynamics (SINDy) algorithm with Sequential Thresholded Least Squares (STLSQ).

## Overview

The SINDy Markov Chain Model represents the STLSQ algorithm as a Markov process and calculates the probability of correctly identifying the true model under noise. The key features include:

- Analytical calculation of transition probabilities between states
- Computation of overall success probability using dynamic programming
- Integration of the Gram matrix to account for term correlations
- Comparison with empirical simulations for validation

## Project Structure

The project is organized into several modules:

```
sindy_markov_model/
├── models/
│   ├── sindy_markov_model.py      # Core model functionality
│   ├── markov_analysis.py         # Analysis and visualization functions
│   ├── markov_simulation.py       # Simulation and testing functions
│   ├── markov_diagnostics.py      # Debugging and diagnostic tools
│   └── state_utils.py             # State handling utilities
├── scripts/
│   ├── run_experiments.py         # Main script to run experiments
│   ├── run_diagnostics.py         # Run detailed diagnostics
│   └── analyze_results.py         # Analyze experiment results
├── logs/                          # Log files from experiments
├── results/                       # Experimental results and figures
└── docs/
    └── theoretical_derivation.md  # Theoretical foundation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/username/sindy-markov-model.git
cd sindy-markov-model

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Experiments

To run the standard set of experiments:

```bash
python scripts/run_experiments.py
```

This will execute three experiments:
1. Simple three-term example with one true term
2. Lambda/sigma ratio experiment to analyze the effect of threshold values
3. Multiple true terms experiment

Results are saved to the `results/` directory.

### Diagnostics

For in-depth model diagnostics:

```bash
python scripts/run_diagnostics.py
```

### Analyzing Results

To analyze previously generated results:

```bash
python scripts/analyze_results.py
```

## Core Components

### SINDy Markov Model

The core model implements the Markov chain framework for STLSQ success probability prediction:

```python
from models.sindy_markov_model import SINDyMarkovModel

# Define library functions
def f1(x): return x           # Term 1: x
def f2(x): return np.sin(x)   # Term 2: sin(x)
def f3(x): return np.tanh(x)  # Term 3: tanh(x)

library_functions = [f1, f2, f3]
true_coefs = np.array([1.0, 0.0, 0.0])  # Only first term is true

# Create model
model = SINDyMarkovModel(library_functions, true_coefs, 
                         sigma=0.1, threshold=0.05)

# Generate sample points
x_data = np.random.uniform(-1.0, 1.0, 100)

# Compute Gram matrix
model.compute_gram_matrix(x_data)

# Calculate success probability
success_prob = model.calculate_success_probability()
print(f"Theoretical success probability: {success_prob:.4f}")

# Run empirical simulation
empirical_prob = model.simulate_stlsq(x_data, n_trials=100)
print(f"Empirical success probability: {empirical_prob:.4f}")
```

### Analysis and Simulation

The analysis and simulation modules provide functions for evaluating the model:

```python
from models.markov_analysis import compare_theory_to_simulation
from models.markov_analysis import plot_comparison, evaluate_model

# Compare theory to simulation
results = compare_theory_to_simulation(model, x_range, n_samples_range)

# Visualize results
fig = plot_comparison(results, x_axis='log_gram_det')
fig.savefig('comparison.png')

# Evaluate model performance
metrics = evaluate_model(results)
```

## Theoretical Foundation

The theoretical foundation is described in detail in the `docs/theoretical_derivation.md` file. This covers:

- Formulation of the STLSQ algorithm as a Markov process
- Calculation of coefficient distributions for each state
- Calculation of transition probabilities between states
- Computation of overall success probability

## Results

The model shows strong agreement between theoretical predictions and empirical simulations across various parameter settings. The key findings include:

1. The log determinant of the Gram matrix is a strong predictor of success probability
2. There exists an optimal lambda/sigma ratio for each problem setting
3. The model accurately accounts for term correlations in the transition probabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{sindy_markov_model,
  title={A Markov Chain Model for SINDy Algorithm Success Probability},
  author={Author, A.},
  journal={Journal of Computational Physics},
  year={2025},
  publisher={Elsevier}
}
```