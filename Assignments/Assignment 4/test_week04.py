from __future__ import annotations

import importlib.util
import os
from typing import Tuple

import numpy as np
import pandas as pd


def _load_week04() -> Tuple[object, str]:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    local_week04 = os.path.join(repo_root, "week04.py")
    if os.path.exists(local_week04):
        week04_path = local_week04
    else:
        week04_path = os.path.join(
            repo_root, "Assignments", "Assignment 4", "week04.py"
        )
    spec = importlib.util.spec_from_file_location("week04", week04_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load week04.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, week04_path


def simple_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate simple data with known treatment effect."""
    np.random.seed(42)
    n = 1000

    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    # Treatment assignment (confounded)
    prob_t = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    t = np.random.binomial(1, prob_t, n)

    # Outcome with constant treatment effect = 2.0
    y = 2.0 * t + x1 + 0.5 * x2 + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})

    train = df.iloc[:800].copy()
    test = df.iloc[800:].copy()

    return train, test


def heterogeneous_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate data with heterogeneous treatment effect."""
    np.random.seed(123)
    n = 1500

    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    # Treatment assignment (confounded)
    prob_t = 1 / (1 + np.exp(-(0.4 * x1)))
    t = np.random.binomial(1, prob_t, n)

    # Outcome with heterogeneous effect: effect depends on x1
    # CATE(x1) = 1 + 0.5*x1
    te = 1.0 + 0.5 * x1
    y = te * t + x1 + 0.3 * x2 + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})

    train = df.iloc[:1200].copy()
    test = df.iloc[1200:].copy()

    return train, test


def continuous_treatment_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate data with continuous treatment."""
    np.random.seed(789)
    n = 1000

    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    # Continuous treatment assignment (confounded)
    t = 10 + x1 + 2 * x2 + np.random.normal(0, 1, n)

    # Outcome with linear effect of treatment
    y = t + x1 + 0.5 * x2 + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})

    train = df.iloc[:800].copy()
    test = df.iloc[800:].copy()

    return train, test


def _assert_close(name: str, estimate: float, target: float, tol: float) -> None:
    diff = abs(estimate - target)
    print(f"{name}: estimate={estimate:.4f}, target={target:.4f}, diff={diff:.4f}")
    assert diff <= tol


def _assert_cate_frame(name: str, df: pd.DataFrame, expected_len: int) -> None:
    assert "cate" in df.columns, f"{name}: missing cate column"
    assert len(df) == expected_len, f"{name}: unexpected row count"
    assert np.isfinite(df["cate"]).all(), f"{name}: non-finite cate values"


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    week04, week04_path = _load_week04()
    print(f"Loaded: {week04_path}")

    train, test = simple_data()

    s_out = week04.s_learner_discrete(train, test, ["x1", "x2"], "t", "y")
    _assert_cate_frame("S-learner", s_out, len(test))
    _assert_close("S-learner mean CATE", float(s_out["cate"].mean()), 2.0, 0.6)

    t_out = week04.t_learner_discrete(train, test, ["x1", "x2"], "t", "y")
    _assert_cate_frame("T-learner", t_out, len(test))
    _assert_close("T-learner mean CATE", float(t_out["cate"].mean()), 2.0, 0.6)

    x_out = week04.x_learner_discrete(train, test, ["x1", "x2"], "t", "y")
    _assert_cate_frame("X-learner", x_out, len(test))
    _assert_close("X-learner mean CATE", float(x_out["cate"].mean()), 2.0, 0.6)

    train_h, test_h = heterogeneous_data()
    true_cate = 1.0 + 0.5 * test_h["x1"].to_numpy()

    t_out_h = week04.t_learner_discrete(train_h, test_h, ["x1", "x2"], "t", "y")
    x_out_h = week04.x_learner_discrete(train_h, test_h, ["x1", "x2"], "t", "y")

    t_corr = _correlation(t_out_h["cate"].to_numpy(), true_cate)
    x_corr = _correlation(x_out_h["cate"].to_numpy(), true_cate)

    print(f"T-learner corr with true CATE: {t_corr:.4f}")
    print(f"X-learner corr with true CATE: {x_corr:.4f}")

    assert t_corr > 0.3
    assert x_corr > 0.3

    train_c, test_c = continuous_treatment_data()
    dml_out = week04.double_ml_cate(train_c, test_c, ["x1", "x2"], "t", "y")
    _assert_cate_frame("Double ML", dml_out, len(test_c))
    _assert_close("Double ML mean CATE", float(dml_out["cate"].mean()), 1.0, 0.4)

    print("All tests passed.")


if __name__ == "__main__":
    main()
