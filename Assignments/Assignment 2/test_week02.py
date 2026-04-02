from __future__ import annotations

import importlib.util
import os
from typing import Tuple

import numpy as np
import pandas as pd


def _load_week02() -> Tuple[object, str]:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    local_week02 = os.path.join(repo_root, "week02.py")
    if os.path.exists(local_week02):
        week02_path = local_week02
    else:
        week02_path = os.path.join(
            repo_root, "Assignments", "Assignment 2", "week02.py"
        )
    spec = importlib.util.spec_from_file_location("week02", week02_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load week02.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, week02_path


def _make_positive_effect_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "I": range(n),
        "T": np.random.binomial(1, 0.5, n),
    })
    df["Y"] = np.where(
        df["T"] == 1,
        np.random.normal(10, 2, n),
        np.random.normal(8, 2, n),
    )
    return df


def _make_no_effect_data() -> pd.DataFrame:
    np.random.seed(123)
    n = 500
    df = pd.DataFrame({
        "I": range(n),
        "T": np.random.binomial(1, 0.5, n),
    })
    df["Y"] = np.random.normal(5, 3, n)
    return df


def main() -> None:
    week02, week02_path = _load_week02()
    print(f"Loaded: {week02_path}")

    pos_df = _make_positive_effect_data()
    pos_ate, pos_lower, pos_upper = week02.calculate_ate_ci(pos_df)
    pos_ate2, pos_t, pos_p = week02.calculate_ate_pvalue(pos_df)

    print("Positive effect CI:", (pos_ate, pos_lower, pos_upper))
    print("Positive effect p-value:", (pos_ate2, pos_t, pos_p))

    assert abs(pos_ate - pos_ate2) < 1e-8
    assert pos_lower < pos_ate < pos_upper
    assert pos_lower > 0 or pos_upper < 0
    assert pos_p < 0.05

    no_df = _make_no_effect_data()
    no_ate, no_lower, no_upper = week02.calculate_ate_ci(no_df)
    no_ate2, no_t, no_p = week02.calculate_ate_pvalue(no_df)

    print("No effect CI:", (no_ate, no_lower, no_upper))
    print("No effect p-value:", (no_ate2, no_t, no_p))

    assert abs(no_ate - no_ate2) < 1e-8
    assert no_lower <= 0 <= no_upper
    assert no_p > 0.05

    print("All tests passed.")


if __name__ == "__main__":
    main()
