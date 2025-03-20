# Theoretical Derivation of SINDy Markov Chain Model

This document provides a rigorous mathematical derivation of the SINDy Markov Chain Model for predicting success probabilities in the Sparse Identification of Nonlinear Dynamics (SINDy) algorithm with Sequential Thresholded Least Squares (STLSQ).

## 1. Problem Formulation

We consider the problem of identifying a sparse model for dynamical systems using the SINDy framework. Given:

- A vector of measurements $y \in \mathbb{R}^m$
- A library of candidate functions $\Theta(x) = [\theta_1(x), \theta_2(x), \ldots, \theta_n(x)]$ producing a design matrix $\Theta \in \mathbb{R}^{m \times n}$
- True underlying dynamics $y = \Theta\beta^0 + \varepsilon$ where $\beta^0$ are the true coefficients
- Measurement noise $\varepsilon \sim \mathcal{N}(0, \sigma^2I)$
- Sparsification threshold $\lambda$

The goal is to calculate the probability that STLSQ correctly identifies the true model.

## 2. STLSQ Algorithm as a Markov Process

The STLSQ algorithm can be represented as a Markov process, where:

1. States represent the set of active library terms.
2. Transitions occur when terms are eliminated during thresholding.
3. The initial state contains all library terms.
4. The target state contains only the true terms (no false positives, no false negatives).

### State Space Definition

Let $S = \{1, 2, \ldots, n\}$ be the indices of all library terms, and $T \subset S$ be the indices of true terms (where $\beta^0_i \neq 0$).

The state space consists of all subsets of $S$ that contain all elements of $T$. This reduces the state space from $2^n$ to $2^{n-|T|}$ states.

### Transition Probabilities

The transition probability from state $S_i$ to state $S_j$ is defined as:

$P(S_i \to S_j) = P(|\hat{\beta}_k| < \lambda \text{ for all } k \in S_i \setminus S_j \text{ AND } |\hat{\beta}_k| \geq \lambda \text{ for all } k \in S_j)$

where $\hat{\beta}$ is the vector of coefficient estimates for the active terms in state $S_i$.

## 3. Coefficient Distribution

### 3.1 Full Library Distribution

For the full library (initial state), the least squares coefficient estimates have the distribution:

$\hat{\beta} = ({\Theta}^T \Theta)^{-1}{\Theta}^T y = ({\Theta}^T \Theta)^{-1}{\Theta}^T (\Theta\beta^0 + \varepsilon) = \beta^0 + ({\Theta}^T \Theta)^{-1}{\Theta}^T \varepsilon$

Since $({\Theta}^T \Theta)^{-1}{\Theta}^T \varepsilon$ is a linear transformation of Gaussian noise, we have:

$\hat{\beta} \sim \mathcal{N}(\beta^0, \sigma^2 ({\Theta}^T \Theta)^{-1})$

The Gram matrix $G = {\Theta}^T \Theta$ captures the correlation structure between library terms at the specific sampling points and directly influences the distribution of coefficient estimates.

### 3.2 Reduced Library Distribution

For a state $S_i \subset S$ with active terms, the coefficient distribution is:

$\hat{\beta}_{S_i} \sim \mathcal{N}(\beta^0_{S_i}, \sigma^2 ({\Theta_{S_i}}^T \Theta_{S_i})^{-1})$

Where:
- $\beta^0_{S_i}$ is the subvector of true coefficients for terms in $S_i$
- $\Theta_{S_i}$ is the submatrix of $\Theta$ with columns corresponding to terms in $S_i$
- $({\Theta_{S_i}}^T \Theta_{S_i})^{-1}$ is the inverse of the submatrix of the Gram matrix

## 4. Transition Probability Calculation

For a transition from state $S_i$ to state $S_j$ (where $S_j \subset S_i$), we need to calculate:

$P(S_i \to S_j) = \int_R f(\hat{\beta}_{S_i}) d\hat{\beta}_{S_i}$

Where:
- $f(\hat{\beta}_{S_i})$ is the PDF of the multivariate normal distribution $\mathcal{N}(\beta^0_{S_i}, \sigma^2 ({\Theta_{S_i}}^T \Theta_{S_i})^{-1})$
- $R$ is the region defined by $|\hat{\beta}_k| < \lambda$ for all $k \in S_i \setminus S_j$ and $|\hat{\beta}_k| \geq \lambda$ for all $k \in S_j$

### 4.1 Special Cases

For single term elimination, the probability involves integration over regions of the multivariate normal distribution.

**Case 1: Single term in $S_i$**
For a single coefficient $\hat{\beta} \sim \mathcal{N}(\beta^0, \sigma^2/g)$ where $g$ is the diagonal element of the Gram matrix:

$P(|\hat{\beta}| \geq \lambda) = 1 - \Phi\left(\frac{\lambda-\beta^0}{\sigma/\sqrt{g}}\right) + \Phi\left(\frac{-\lambda-\beta^0}{\sigma/\sqrt{g}}\right)$

Where $\Phi$ is the standard normal CDF.

**Case 2: Two terms in $S_i$**
For two coefficients with correlation $\rho$, the probability can be calculated using the bivariate normal CDF:

$P(|\hat{\beta}_1| \geq \lambda, |\hat{\beta}_2| < \lambda) = \int_{|\beta_1| \geq \lambda} \int_{|\beta_2| < \lambda} f(\beta_1, \beta_2) d\beta_2 d\beta_1$

Where $f(\beta_1, \beta_2)$ is the bivariate normal density with correlation $\rho$.

### 4.2 General Case

For the general case with multiple terms, we express the joint probability as:

$P(S_i \to S_j) = P\left(\bigcap_{k \in S_i \setminus S_j} |\hat{\beta}_k| < \lambda \cap \bigcap_{l \in S_j} |\hat{\beta}_l| \geq \lambda\right)$

This can be evaluated using:
1. Orthogonal transformations of the multivariate normal
2. Sequential conditioning approach
3. Numerical integration for higher dimensions

## 5. Success Probability Calculation

The overall success probability is calculated by summing the probabilities of all paths from the initial state (full library) to the target state (true model only):

$P(\text{success}) = \sum_{\text{all paths }p} P(\text{follow path }p)$

For each path $p = (S_0, S_1, \ldots, S_{T})$, where $S_0$ is the initial state and $S_{T}$ is the target state:

$P(\text{follow path }p) = \prod_{i=0}^{k-1} P(S_i \to S_{i+1})$

Using dynamic programming, we can efficiently calculate this as:

$P(\text{success from }S_i) = \sum_{S_j \in \text{Next}(S_i)} P(S_i \to S_j) \cdot P(\text{success from }S_j)$

Where Next($S_i$) is the set of all possible next states from $S_i$, and $P(\text{success from }S_{T}) = 1$.

## 6. Relationship to Discriminability

The discriminability between terms $i$ and $j$ is defined as:

$D_{ij} = \sum_{k=1}^m \frac{(\theta_i(x_k) - \theta_j(x_k))^2}{\sigma^2}$

This can be expressed in terms of the Gram matrix as:

$D_{ij} = \frac{G_{ii} + G_{jj} - 2G_{ij}}{\sigma^2}$

Where $G_{ij}$ are elements of the Gram matrix.

Higher discriminability leads to lower correlation between coefficient estimates, making it easier to correctly identify the true model. The critical discriminability value $D^*$ represents the threshold where the success probability equals 0.5.

## 7. Effect of λ/σ Ratio

The λ/σ ratio directly affects the success probability by changing the thresholding behavior:

- Higher λ/σ: More aggressive sparsification, increased risk of false negatives
- Lower λ/σ: More conservative sparsification, increased risk of false positives

The optimal λ/σ ratio balances these risks and depends on the structure of the problem.

## 8. Computational Considerations

For practical implementation, we use:

1. Exact analytical formulas for simple cases (1-2 terms)
2. Approximation techniques for larger state spaces
3. Numerical integration for high-dimensional cases
4. Dynamic programming to avoid recalculating success probabilities

## Conclusion

This Markov chain model provides a rigorous mathematical framework for predicting SINDy STLSQ success probabilities. By accounting for the full correlation structure of library terms through the Gram matrix, we can accurately model how the algorithm identifies the correct terms under noise and finite sampling.