from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
	missing = set(columns) - set(df.columns)
	if missing:
		missing_list = ", ".join(sorted(missing))
		raise ValueError(f"Missing required columns: {missing_list}")


def _as_list(values: Iterable[str]) -> List[str]:
	return list(values)


def _fit_propensity(train: pd.DataFrame, X: List[str], T: str) -> LogisticRegression:
	model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
	try:
		model.fit(train[X], train[T])
	except ValueError:
		model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000)
		model.fit(train[X], train[T])
	return model


def s_learner_discrete(
	train: pd.DataFrame,
	test: pd.DataFrame,
	X: list[str],
	T: str,
	y: str,
) -> pd.DataFrame:
	_require_columns(train, [*X, T, y])
	_require_columns(test, [*X, T, y])

	features = _as_list(X) + [T]
	model = LGBMRegressor()
	model.fit(train[features], train[y])

	base = test[_as_list(X)].copy()
	test_treat = base.copy()
	test_treat[T] = 1
	test_control = base.copy()
	test_control[T] = 0

	mu1 = model.predict(test_treat[features])
	mu0 = model.predict(test_control[features])

	result = test.copy()
	result["cate"] = mu1 - mu0
	return result


def t_learner_discrete(
	train: pd.DataFrame,
	test: pd.DataFrame,
	X: list[str],
	T: str,
	y: str,
) -> pd.DataFrame:
	_require_columns(train, [*X, T, y])
	_require_columns(test, [*X, T, y])

	mask_treat = train[T] == 1
	mask_control = train[T] == 0
	if not mask_treat.any() or not mask_control.any():
		raise ValueError("Both treatment and control groups must be non-empty.")

	model_treat = LGBMRegressor()
	model_control = LGBMRegressor()
	model_treat.fit(train.loc[mask_treat, X], train.loc[mask_treat, y])
	model_control.fit(train.loc[mask_control, X], train.loc[mask_control, y])

	mu1 = model_treat.predict(test[X])
	mu0 = model_control.predict(test[X])

	result = test.copy()
	result["cate"] = mu1 - mu0
	return result


def x_learner_discrete(
	train: pd.DataFrame,
	test: pd.DataFrame,
	X: list[str],
	T: str,
	y: str,
) -> pd.DataFrame:
	_require_columns(train, [*X, T, y])
	_require_columns(test, [*X, T, y])

	mask_treat = train[T] == 1
	mask_control = train[T] == 0
	if not mask_treat.any() or not mask_control.any():
		raise ValueError("Both treatment and control groups must be non-empty.")

	model_treat = LGBMRegressor()
	model_control = LGBMRegressor()
	model_treat.fit(train.loc[mask_treat, X], train.loc[mask_treat, y])
	model_control.fit(train.loc[mask_control, X], train.loc[mask_control, y])

	mu1_on_control = model_treat.predict(train.loc[mask_control, X])
	mu0_on_treat = model_control.predict(train.loc[mask_treat, X])

	tau0 = mu1_on_control - train.loc[mask_control, y].to_numpy()
	tau1 = train.loc[mask_treat, y].to_numpy() - mu0_on_treat

	tau0_model = LGBMRegressor()
	tau1_model = LGBMRegressor()
	tau0_model.fit(train.loc[mask_control, X], tau0)
	tau1_model.fit(train.loc[mask_treat, X], tau1)

	propensity_model = _fit_propensity(train, X, T)
	e_test = propensity_model.predict_proba(test[X])[:, 1]

	tau0_hat = tau0_model.predict(test[X])
	tau1_hat = tau1_model.predict(test[X])

	result = test.copy()
	result["cate"] = e_test * tau0_hat + (1 - e_test) * tau1_hat
	return result


def double_ml_cate(
	train: pd.DataFrame,
	test: pd.DataFrame,
	X: list[str],
	T: str,
	y: str,
) -> pd.DataFrame:
	_require_columns(train, [*X, T, y])
	_require_columns(test, [*X, T, y])

	X_train = train[X]
	T_train = train[T].to_numpy()
	Y_train = train[y].to_numpy()

	t_hat = np.zeros(len(train))
	y_hat = np.zeros(len(train))

	# Cross-fit nuisance models to reduce bias in residuals.
	kf = KFold(n_splits=2, shuffle=True, random_state=123)
	for train_idx, hold_idx in kf.split(X_train):
		model_t = LGBMRegressor()
		model_y = LGBMRegressor()
		X_tr = X_train.iloc[train_idx]
		X_hold = X_train.iloc[hold_idx]

		model_t.fit(X_tr, T_train[train_idx])
		model_y.fit(X_tr, Y_train[train_idx])

		t_hat[hold_idx] = model_t.predict(X_hold)
		y_hat[hold_idx] = model_y.predict(X_hold)

	t_res = T_train - t_hat
	y_res = Y_train - y_hat

	eps = 1e-6
	t_res_safe = np.where(np.abs(t_res) < eps, eps, t_res)
	y_star = y_res / t_res_safe
	weights = t_res**2

	tau_model = LGBMRegressor()
	tau_model.fit(X_train, y_star, sample_weight=weights)

	result = test.copy()
	result["cate"] = tau_model.predict(test[X])
	return result