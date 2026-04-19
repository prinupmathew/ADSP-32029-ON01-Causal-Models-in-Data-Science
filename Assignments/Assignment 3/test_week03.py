from __future__ import annotations

import importlib.util
import os
from typing import Tuple

import numpy as np
import pandas as pd


def _load_week03() -> Tuple[object, str]:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    local_week03 = os.path.join(repo_root, "week03.py")
    if os.path.exists(local_week03):
        week03_path = local_week03
    else:
        week03_path = os.path.join(
            repo_root, "Assignments", "Assignment 3", "week03.py"
        )
    spec = importlib.util.spec_from_file_location("week03", week03_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load week03.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, week03_path


def _make_positive_effect_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 1000
    x = np.random.normal(0, 1, n)
    prob_t = 1 / (1 + np.exp(-(0.5 * x)))
    t = np.random.binomial(1, prob_t, n)
    y = 2 * t + x + np.random.normal(0, 0.5, n)
    return pd.DataFrame({"x": x, "t": t, "y": y})


def _make_categorical_data() -> pd.DataFrame:
    np.random.seed(101)
    n = 1000
    group = np.random.choice(["A", "B", "C"], n)
    group_effect = {"A": 0, "B": 1, "C": 2}
    x_numeric = np.array([group_effect[g] for g in group])

    prob_t = 1 / (1 + np.exp(-(0.5 * x_numeric)))
    t = np.random.binomial(1, prob_t, n)
    y = 2.0 * t + x_numeric + np.random.normal(0, 0.5, n)
    return pd.DataFrame({"group": group, "t": t, "y": y})


def _assert_close(name: str, estimate: float, target: float, tol: float) -> None:
    diff = abs(estimate - target)
    print(f"{name}: estimate={estimate:.4f}, target={target:.4f}, diff={diff:.4f}")
    assert diff <= tol


def main() -> None:
    week03, week03_path = _load_week03()
    print(f"Loaded: {week03_path}")

    df_pos = _make_positive_effect_data()
    ipw_pos = week03.ipw(df_pos, "x", "t", "y")
    dr_pos = week03.doubly_robust(df_pos, "x", "t", "y")

    _assert_close("IPW positive", ipw_pos, 2.0, 0.3)
    _assert_close("DR positive", dr_pos, 2.0, 0.2)

    df_cat = _make_categorical_data()
    ipw_cat = week03.ipw(df_cat, "C(group)", "t", "y")
    dr_cat = week03.doubly_robust(df_cat, "C(group)", "t", "y")

    _assert_close("IPW categorical", ipw_cat, 2.0, 0.3)
    _assert_close("DR categorical", dr_cat, 2.0, 0.2)

    print("All tests passed.")


if __name__ == "__main__":
    main()
