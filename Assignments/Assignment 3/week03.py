from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.linear_model import LinearRegression, LogisticRegression


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
	missing = set(columns) - set(df.columns)
	if missing:
		missing_list = ", ".join(sorted(missing))
		raise ValueError(f"Missing required columns: {missing_list}")


def _design_matrix(df: pd.DataFrame, formula: str) -> pd.DataFrame:
	return dmatrix(formula, df, return_type="dataframe")


def _fit_propensity_model(X: pd.DataFrame, t: np.ndarray) -> LogisticRegression:
	base_model = LogisticRegression(
		solver="lbfgs",
		max_iter=1000,
		fit_intercept=False,
	)
	if base_model.penalty is None:
		model = base_model
	else:
		model = LogisticRegression(
			solver="lbfgs",
			C=1e6,
			max_iter=1000,
			fit_intercept=False,
		)
	model.fit(X, t)
	return model


def ipw(df: pd.DataFrame, ps_formula: str, T: str, Y: str) -> float:
	"""Estimate ATE using inverse propensity weighting."""
	_require_columns(df, [T, Y])

	X = _design_matrix(df, ps_formula)
	t = df[T].to_numpy(dtype=float)
	y = df[Y].to_numpy(dtype=float)

	model = _fit_propensity_model(X, t)
	ps = model.predict_proba(X)[:, 1]
	ps = np.clip(ps, 1e-6, 1 - 1e-6)

	weights = (t - ps) / (ps * (1 - ps))
	ate = float(np.mean(weights * y))
	return ate


def doubly_robust(df: pd.DataFrame, formula: str, T: str, Y: str) -> float:
	"""Estimate ATE using doubly robust estimation."""
	_require_columns(df, [T, Y])

	X = _design_matrix(df, formula)
	t = df[T].to_numpy(dtype=float)
	y = df[Y].to_numpy(dtype=float)

	model = _fit_propensity_model(X, t)
	ps = model.predict_proba(X)[:, 1]
	ps = np.clip(ps, 1e-6, 1 - 1e-6)

	mask_treat = t == 1
	mask_control = t == 0
	if not mask_treat.any() or not mask_control.any():
		raise ValueError("Both treatment and control groups must be non-empty.")

	model_treat = LinearRegression(fit_intercept=False)
	model_control = LinearRegression(fit_intercept=False)
	model_treat.fit(X.loc[mask_treat], y[mask_treat])
	model_control.fit(X.loc[mask_control], y[mask_control])

	mu1 = model_treat.predict(X)
	mu0 = model_control.predict(X)

	term_treat = t * (y - mu1) / ps + mu1
	term_control = (1 - t) * (y - mu0) / (1 - ps) + mu0
	ate = float(np.mean(term_treat - term_control))
	return ate
