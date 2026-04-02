from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _split_groups(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
	required_cols = {"T", "Y"}
	missing = required_cols - set(df.columns)
	if missing:
		missing_list = ", ".join(sorted(missing))
		raise ValueError(f"Missing required columns: {missing_list}")

	treat = df.loc[df["T"] == 1, "Y"].to_numpy(dtype=float)
	control = df.loc[df["T"] == 0, "Y"].to_numpy(dtype=float)
	if treat.size == 0 or control.size == 0:
		raise ValueError("Both treatment and control groups must be non-empty.")
	return treat, control


def _welch_stats(treat: np.ndarray, control: np.ndarray) -> Tuple[float, float, float]:
	n_t = treat.size
	n_c = control.size
	mean_t = float(np.mean(treat))
	mean_c = float(np.mean(control))
	var_t = float(np.var(treat, ddof=1)) if n_t > 1 else 0.0
	var_c = float(np.var(control, ddof=1)) if n_c > 1 else 0.0

	ate = mean_t - mean_c
	se = float(np.sqrt(var_t / n_t + var_c / n_c))

	if se == 0.0:
		return ate, se, float("inf")

	num = (var_t / n_t + var_c / n_c) ** 2
	denom = (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
	dfree = float("inf") if denom == 0.0 else float(num / denom)
	return ate, se, dfree


def calculate_ate_ci(data: pd.DataFrame, alpha: float = 0.05) -> Tuple[float, float, float]:
	"""Return (ATE, CI lower, CI upper) using a normal critical value."""
	treat, control = _split_groups(data)
	ate, se, _ = _welch_stats(treat, control)

	if se == 0.0:
		return ate, ate, ate

	crit = float(stats.norm.ppf(1 - alpha / 2))
	lower = ate - crit * se
	upper = ate + crit * se
	return ate, lower, upper


def calculate_ate_pvalue(data: pd.DataFrame) -> Tuple[float, float, float]:
	"""Return (ATE, t statistic, two-sided p-value) using the normal CDF."""
	treat, control = _split_groups(data)
	ate, se, _ = _welch_stats(treat, control)

	if se == 0.0:
		if ate == 0.0:
			return ate, 0.0, 1.0
		return ate, float("inf"), 0.0

	t_stat = ate / se
	pval = 2 * (1 - stats.norm.cdf(abs(t_stat)))
	return ate, float(t_stat), float(pval)
