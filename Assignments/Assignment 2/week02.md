# Week 02: ATE inference helpers

This module implements basic statistical inference for the Average Treatment Effect (ATE) from randomized controlled trial (RCT) data with columns I, T, and Y.

## What it does

- `calculate_ate_ci(data, alpha=0.05)` returns a 3-tuple: (ATE estimate, CI lower, CI upper).
- `calculate_ate_pvalue(data)` returns a 3-tuple: (ATE estimate, t statistic, two-sided p-value).

Both functions:

- split the data into treatment (T=1) and control (T=0) groups
- compute the ATE as $\bar{Y}_{T=1} - \bar{Y}_{T=0}$
- compute the standard error as $\sqrt{\mathrm{Var}(Y_{T=1})/n_1 + \mathrm{Var}(Y_{T=0})/n_0}$
- use the standard normal distribution for the CI and p-value

## Files

- Implementation: [Assignments/Assignment 2/week02.py](Assignments/Assignment%202/week02.py)
- Test script: [Assignments/Assignment 2/test_week02.py](Assignments/Assignment%202/test_week02.py)

## How to test

Run the test script from the Assignment 2 folder:

```bash
python test_week02.py
```

Expected behavior:

- For positive-effect data, the CI should exclude 0 and the p-value should be small.
- For no-effect data, the CI should include 0 and the p-value should be large.
