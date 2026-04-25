# Week 04: Meta-Learners and Double ML for CATE

This module implements four approaches for estimating heterogeneous treatment effects (CATE):

- S-learner (binary treatment)
- T-learner (binary treatment)
- X-learner (binary treatment)
- Double ML for continuous treatment

All functions take a training DataFrame and a test DataFrame, then return a copy of the test DataFrame with a new column named `cate`.

## Data format

Each DataFrame must contain:

- Covariates: one or more columns (passed as `X`)
- Treatment: column named by `T` (binary for S/T/X; continuous for Double ML)
- Outcome: column named by `y`

## Functions

### 1) s_learner_discrete(train, test, X, T, y)

**Idea**

Fit a single model that predicts outcome from both covariates and treatment:

- Fit: $\mu(X, T)$
- Estimate CATE: $\mu(X, T=1) - \mu(X, T=0)$

**How it works**

- Train a single `LGBMRegressor` on `X + [T]`.
- Create two copies of the test covariates, one with `T=1` and one with `T=0`.
- The difference in predictions is the CATE.

**Strengths**

- Simple and fast.
- Works well when treatment effects are smooth and the model learns the interaction between X and T.

**Potential pitfalls**

- If the model underfits interactions, CATE can be biased toward a constant effect.

**Example usage**

```python
from week04 import s_learner_discrete

result = s_learner_discrete(train, test, ["x1", "x2"], "t", "y")
print(result[["x1", "x2", "cate"]].head())
```

**Business example**

- **Personalized marketing email**: Predict how much an email campaign changes user spend based on customer features. The S-learner estimates the uplift per user by simulating `T=1` vs `T=0` in the model.

---

### 2) t_learner_discrete(train, test, X, T, y)

**Idea**

Fit separate models for treated and control outcomes:

- Fit $\mu_1(X)$ on treated units
- Fit $\mu_0(X)$ on control units
- Estimate CATE: $\mu_1(X) - \mu_0(X)$

**How it works**

- Train two `LGBMRegressor` models, one for `T=1` and one for `T=0`.
- Predict both on the test covariates and take the difference.

**Strengths**

- Flexible when treatment and control response surfaces differ.
- Often works well with non-linear models.

**Potential pitfalls**

- Needs sufficient data in both groups; imbalanced treatment can hurt one of the models.

**Example usage**

```python
from week04 import t_learner_discrete

result = t_learner_discrete(train, test, ["x1", "x2"], "t", "y")
print(result[["x1", "x2", "cate"]].head())
```

**Business example**

- **Subscription retention offer**: Train separate models for users who received a discount and those who did not, then estimate who benefits most from the offer.

---

### 3) x_learner_discrete(train, test, X, T, y)

**Idea**

Use a two-stage procedure that is effective under treatment imbalance:

1. Fit outcome models like T-learner
2. Compute pseudo-treatment effects:
   - For control: $\hat{\tau}_0 = \mu_1(X) - Y$
   - For treated: $\hat{\tau}_1 = Y - \mu_0(X)$
3. Fit models for $\tau_0(X)$ and $\tau_1(X)$
4. Combine using propensity score $e(X)$:
   - $\tau(X) = e(X) \cdot \tau_0(X) + (1 - e(X)) \cdot \tau_1(X)$

**How it works**

- First-stage outcome models: two `LGBMRegressor` models.
- Second-stage effect models: two `LGBMRegressor` models on pseudo-outcomes.
- Propensity score: `LogisticRegression(penalty=None)` on covariates.

**Strengths**

- Strong performance when treatment is imbalanced.
- Explicitly uses propensity to weight the two effect models.

**Potential pitfalls**

- More moving parts; sensitive to poor propensity estimates.

**Example usage**

```python
from week04 import x_learner_discrete

result = x_learner_discrete(train, test, ["x1", "x2"], "t", "y")
print(result[["x1", "x2", "cate"]].head())
```

**Business example**

- **Fraud intervention**: Only a small share of users get an intervention. X-learner can better estimate uplift for the rare treated group.

---

### 4) double_ml_cate(train, test, X, T, y)

**Idea**

Estimate CATE for continuous treatment using Double ML (partialling-out):

- Fit models for $E[T | X]$ and $E[Y | X]$ using cross-fitting
- Residualize:
  - $T_{res} = T - \hat{E}[T|X]$
  - $Y_{res} = Y - \hat{E}[Y|X]$
- Transform outcome: $Y^* = Y_{res} / T_{res}$
- Fit $\tau(X)$ on $X$ using weights $w = T_{res}^2$

**How it works**

- Uses 2-fold cross-fitting to reduce bias in residuals.
- Fits the final CATE model on transformed outcomes with sample weights.

**Strengths**

- Works for continuous treatments.
- Reduces bias from nuisance model overfitting.

**Potential pitfalls**

- If $T_{res}$ is near zero, the transformation can be unstable (handled with a small epsilon).

**Example usage**

```python
from week04 import double_ml_cate

result = double_ml_cate(train, test, ["x1", "x2"], "t", "y")
print(result[["x1", "x2", "cate"]].head())
```

**Business example**

- **Pricing or marketing intensity**: Treatment is continuous (discount level, ad spend). Double ML estimates how response to treatment varies by customer features.

---

## How to test

Run the test script from the Assignment 4 folder:

```bash
python test_week04.py
```

The test script validates:

- S/T/X learners recover a mean CATE close to 2.0 on simple synthetic data
- T/X learners track the heterogeneous CATE pattern on a known synthetic example
- Double ML recovers a mean CATE close to 1.0 for continuous treatment data

## Practical guidance

- Always verify overlap: both treated and control groups must be non-empty for S/T/X learners.
- For real-world data, include important confounders in `X` and ensure stable treatment probabilities.
- Consider calibration checks (e.g., comparing average predicted CATE to known experimental lift).

## Files

- Implementation: Assignments/Assignment 4/week04.py
- Tests: Assignments/Assignment 4/test_week04.py
