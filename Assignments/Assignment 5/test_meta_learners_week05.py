from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd


_HERE = os.path.dirname(__file__)
_WEEK04_DIR = os.path.abspath(os.path.join(_HERE, "..", "Assignment 4"))
if _WEEK04_DIR not in sys.path:
	sys.path.insert(0, _WEEK04_DIR)

from week04 import (
	double_ml_cate,
	s_learner_discrete,
	t_learner_discrete,
	x_learner_discrete,
)


def simple_data():
	"""Generate simple data with known treatment effect"""
	np.random.seed(42)
	n = 1000

	# Covariates
	x1 = np.random.normal(0, 1, n)
	x2 = np.random.normal(0, 1, n)

	# Treatment assignment (confounded)
	prob_t = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
	t = np.random.binomial(1, prob_t, n)

	# Outcome with constant treatment effect = 2.0
	y = 2.0 * t + x1 + 0.5 * x2 + np.random.normal(0, 0.5, n)

	df = pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})

	# Split into train/test
	train = df.iloc[:800].copy()
	test = df.iloc[800:].copy()

	return train, test


def continuous_treatment_data():
	"""Generate data with continuous treatment"""
	np.random.seed(789)
	n = 1000

	# Covariates
	x1 = np.random.normal(0, 1, n)
	x2 = np.random.normal(0, 1, n)

	# Continuous treatment
	t = 10 + x1 + 2 * x2 + np.random.normal(0, 1, n)

	# Outcome: linear effect of treatment
	y = t + x1 + 0.5 * x2 + np.random.normal(0, 0.5, n)

	df = pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})

	train = df.iloc[:800].copy()
	test = df.iloc[800:].copy()

	return train, test


def test_s_learner_returns_dataframe():
	train, test = simple_data()

	result = s_learner_discrete(train, test, ["x1", "x2"], "t", "y")

	assert isinstance(result, pd.DataFrame)


def test_s_learner_has_cate_column():
	train, test = simple_data()

	result = s_learner_discrete(train, test, ["x1", "x2"], "t", "y")

	assert "cate" in result.columns
	assert len(result) == len(test)


def test_s_learner_constant_effect():
	train, test = simple_data()

	result = s_learner_discrete(train, test, ["x1", "x2"], "t", "y")

	true_effect = 2.0
	estimated_ate = result["cate"].mean()
	assert abs(estimated_ate - true_effect) < 0.75


def test_s_learner_return_numeric_cate():
	train, test = simple_data()

	result = s_learner_discrete(train, test, ["x1", "x2"], "t", "y")

	assert pd.api.types.is_numeric_dtype(result["cate"])


def test_s_learner_no_nan_values():
	train, test = simple_data()

	result = s_learner_discrete(train, test, ["x1", "x2"], "t", "y")

	assert not result["cate"].isna().any()


def test_t_learner_returns_dataframe():
	train, test = simple_data()

	result = t_learner_discrete(train, test, ["x1", "x2"], "t", "y")

	assert isinstance(result, pd.DataFrame)


def test_x_learner_returns_dataframe():
	train, test = simple_data()

	result = x_learner_discrete(train, test, ["x1", "x2"], "t", "y")

	assert isinstance(result, pd.DataFrame)


def test_double_ml_returns_dataframe():
	train, test = simple_data()

	result = double_ml_cate(train, test, ["x1", "x2"], "t", "y")

	assert isinstance(result, pd.DataFrame)


def test_double_ml_continuous_treatment():
	train, test = continuous_treatment_data()

	result = double_ml_cate(train, test, ["x1", "x2"], "t", "y")

	assert isinstance(result, pd.DataFrame)
	assert pd.api.types.is_numeric_dtype(result["cate"])
	assert not result["cate"].isna().any()
