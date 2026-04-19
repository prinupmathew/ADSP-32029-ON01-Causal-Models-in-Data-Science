# Week 03: IPW and Doubly Robust ATE

This module implements two causal estimators for the Average Treatment Effect (ATE):

- Inverse Propensity Weighting (IPW)
- Doubly Robust (DR) estimation

Both functions take a DataFrame with a treatment column, an outcome column, and any number of covariates. You pass a Patsy formula string that describes the covariates used in the models.

## Key ideas

### Propensity score
The propensity score is the probability of receiving treatment given covariates:

$$
 e(X) = P(T=1 \mid X)
$$

If we can model $e(X)$ well, we can reweight observations to create a pseudo-population where treatment is independent of confounders.

### IPW estimator
IPW estimates the ATE by reweighting outcomes with inverse propensity scores:

$$
\text{ATE} = E\left[\frac{T - e(X)}{e(X)(1-e(X))} Y\right]
$$

Intuition:

- Treated units with low $e(X)$ get large weight.
- Control units with high $e(X)$ get large weight.
- The reweighting balances confounders between groups.

### Doubly robust estimator
DR combines two ideas:

- a propensity score model $e(X)$
- outcome models $\mu_1(X) = E[Y \mid T=1, X]$ and $\mu_0(X) = E[Y \mid T=0, X]$

The estimator is:

$$
\text{ATE} = E\left[\frac{T(Y-\mu_1(X))}{e(X)} + \mu_1(X)\right]
 - E\left[\frac{(1-T)(Y-\mu_0(X))}{1-e(X)} + \mu_0(X)\right]
$$

It is called doubly robust because it is consistent if either the propensity model or the outcome model is correctly specified (not necessarily both).

## Functions

### ipw(df, ps_formula, T, Y) -> float

- Builds a design matrix from `ps_formula` using Patsy.
- Fits a logistic regression for the propensity score.
- Computes the IPW ATE.

### doubly_robust(df, formula, T, Y) -> float

- Builds a design matrix from `formula` using Patsy.
- Fits a logistic regression for the propensity score.
- Fits two linear regression models for outcomes (treated and control).
- Computes the DR ATE.

## Example

```python
import numpy as np
import pandas as pd
from week03 import ipw, doubly_robust

np.random.seed(42)
n = 1000
x = np.random.normal(0, 1, n)
prob_t = 1 / (1 + np.exp(-(0.5 * x)))
t = np.random.binomial(1, prob_t, n)
y = 2 * t + x + np.random.normal(0, 0.5, n)

df = pd.DataFrame({"x": x, "t": t, "y": y})

ate_ipw = ipw(df, "x", "t", "y")
ate_dr = doubly_robust(df, "x", "t", "y")

print("IPW ATE:", ate_ipw)
print("DR ATE:", ate_dr)
```

Both estimates should be close to 2.0 in expectation.

## Categorical covariates

Use Patsy syntax for categorical variables:

```python
ate_ipw = ipw(df, "C(group)", "t", "y")
ate_dr = doubly_robust(df, "C(group)", "t", "y")
```

## Files

- Implementation: [Assignments/Assignment 3/week03.py](Assignments/Assignment%203/week03.py)
- Test script: [Assignments/Assignment 3/test_week03.py](Assignments/Assignment%203/test_week03.py)

## How to test

From the Assignment 3 folder:

```bash
python test_week03.py
```

Expected behavior:

- Both IPW and DR estimates are near 2.0 for the provided synthetic data.
