# Improved SINDy Markov Chain Model

This project implements an enhanced analytical Markov chain model for predicting success probabilities in the Sparse Identification of Nonlinear Dynamics (SINDy) algorithm with Sequential Thresholded Least Squares (STLSQ).

## Overview

The SINDy Markov Chain Model represents the STLSQ algorithm as a Markov process and calculates the probability of correctly identifying the true model under noise. The key improvements in this implementation include:

- Enhanced numerical stability for coefficient distribution calculations
- Improved transition probability estimation with better Monte Carlo sampling
- Comprehensive analysis of factors affecting success probability
- Adaptive trial count determination for efficient empirical validation
- Detailed logging and visualization capabilities

## Project Structure

The project is organized into several modules:

```
sindy_markov_model/
├── models/
│   ├── sindy_markov_model.py    # Core model implementation
│   ├── state_utils.py           # State handling utilities
│   ├── simulation_utils.py      # Simulation utilities
│   ├── library_utils.py         # Library function analysis
│   ├── logger_utils.py          # Logging utilities
│   └── sindy_markov_analysis.py # Comprehensive analysis module
├── scripts/
│   └── test_script.py           # Test script for the model
├── notebooks/
│   └── sindy_markov_analysis.ipynb # Interactive analysis notebook
├── logs/                        # Log files
├── results/                     # Analysis results
├── figures/                     # Generated figures
└── README.md                    # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/username/improved-sindy-markov-model.git
cd improved-sindy-markov-model

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Test Script

To run the test script with all tests:

```bash
python scripts/test_script.py
```

To run a specific test:

```bash
python scripts/test_script.py --test simple
python scripts/test_script.py --test advanced
python scripts/test_script.py --test adaptive
```

### Using the Jupyter Notebook

Launch Jupyter and open the notebook:

```bash
jupyter notebook notebooks/sindy_markov_analysis.ipynb
```

### Using the Core Model

```python
from models.sindy_markov_model import SINDyMarkovModel
import numpy as np

# Define library functions
def f1(x): return x           # Term 1: x
def f2(x): return np.sin(x)   # Term 2: sin(x)
def f3(x): return np.tanh(x)  # Term 3: tanh(x)

library_functions = [f1, f2, f3]

# Define true model: only the first term is present
true_coefs = np.array([1.0, 0.0, 0.0])

# Create model instance
sigma = 0.1  # Noise level
threshold = 0.05  # STLSQ threshold
model = SINDyMarkovModel(library_functions, true_coefs, sigma, threshold)

# Generate sample points
x_data = np.random.uniform(-1.0, 1.0, 200)

# Compute Gram matrix
model.compute_gram_matrix(x_data)

# Calculate success probability
theoretical_prob = model.calculate_success_probability()
print(f"Theoretical success probability: {theoretical_prob:.4f}")

# Run simulation to validate
from models.simulation_utils import simulate_stlsq
empirical_prob = simulate_stlsq(model, x_data, n_trials=100)
print(f"Empirical success probability: {empirical_prob:.4f}")
```

### Using the Analysis Module

```python
from models.sindy_markov_analysis import SINDyMarkovAnalysis
import numpy as np

# Define library functions and true coefficients
def f1(x): return x
def f2(x): return x**2
def f3(x): return np.sin(x)
def f4(x): return np.cos(x)

library_functions = [f1, f2, f3, f4]
true_coefs = np.array([1.0, 0.5, 0.0, 0.0])  # First and second terms are active

# Create analysis object
sigma = 0.1
threshold = 0.05
analysis = SINDyMarkovAnalysis(library_functions, true_coefs, sigma, threshold)

# Prepare data
x_data = analysis.prepare_data(x_range=1.0, n_samples=200)

# Analyze success probability
success_analysis = analysis.analyze_success_probability(x_data, n_trials=50)

# Analyze effects of different parameters
range_analysis = analysis.analyze_data_range_effect()
sample_analysis = analysis.analyze_sample_size_effect()
lambda_analysis = analysis.analyze_lambda_sigma_effect()

# Save results
analysis.save_results()
```

## Key Components

### 1. SINDy Markov Model (`sindy_markov_model.py`)

The core model calculates theoretical success probabilities by:
- Computing the Gram matrix from evaluated library functions
- Calculating coefficient distributions for each state
- Computing transition probabilities between states
- Using dynamic programming to find the overall success probability

Improvements:
- Enhanced numerical stability for coefficient distribution
- Better handling of ill-conditioned matrices
- More accurate transition probability calculation

### 2. Simulation Utilities (`simulation_utils.py`)

Provides functions for empirical validation through simulation:
- Standard STLSQ simulation with fixed trial count
- Adaptive STLSQ simulation with automatic trial count determination
- Detailed trajectory tracking and analysis

### 3. Library Utilities (`library_utils.py`)

Analyzes properties of the library functions:
- Computes discriminability between terms
- Analyzes correlations between library terms
- Visualizes library properties

### 4. State Utilities (`state_utils.py`) 

Handles operations on states in the Markov process:
- Normalizes state representations
- Generates valid states
- Finds paths between states

### 5. Logger Utilities (`logger_utils.py`)

Provides comprehensive logging functionality:
- Console and file logging with different levels
- Color-coded console output
- Conversion of log files to Markdown for better readability

### 6. Comprehensive Analysis (`sindy_markov_analysis.py`)

Performs in-depth analysis of SINDy success factors:
- Analyzes success probability with detailed diagnostics
- Studies effect of data range on success probability
- Investigates impact of sample size on performance
- Examines influence of λ/σ ratio on identification success

## Results and Visualizations

The analysis provides various visualizations to understand the factors affecting success probability:

1. **Success Probability vs Data Range**: Shows how the width of the sampling region affects success probability.

2. **Success Probability vs Sample Size**: Illustrates the effect of having more or fewer samples.

3. **Success Probability vs λ/σ Ratio**: Demonstrates the impact of the thresholding parameter relative to noise.

4. **Success Probability vs Log Gram Determinant**: Reveals the relationship between the Gram matrix's condition and success probability.

5. **Library Term Correlations**: Visualizes correlations between library terms that can affect identification success.

6. **Discriminability Matrix**: Shows the discriminability between different library terms, which directly impacts the ability to correctly identify active terms.

## Advanced Features

### Adaptive Trial Count Determination

The adaptive simulation approach automatically determines how many trials are needed to achieve a desired level of confidence in the empirical success probability. It:

1. Runs trials in batches
2. Calculates the margin of error after each batch
3. Stops when the desired confidence level and margin of error are reached

This results in more efficient simulations, using fewer trials when success probability is extreme (close to 0 or 1) and more trials when it's close to 0.5.

### Log to Markdown Conversion

The `log_to_markdown` function converts log files to nicely formatted Markdown documents, making it easier to review analysis results.

## Theoretical Foundation

The theoretical foundation of the model is explained in the original document `docs/theoretical_derivation.md`. Key concepts include:

- Representing STLSQ as a Markov process
- Computing coefficient distributions from the Gram matrix
- Calculating transition probabilities between states
- Using discriminability as a key metric for success probability

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{sindy_markov_model,
  title={An Improved Markov Chain Model for SINDy Algorithm Success Probability},
  author={Author, A.},
  journal={Journal of Computational Physics},
  year={2025},
  publisher={Elsevier}
}
```