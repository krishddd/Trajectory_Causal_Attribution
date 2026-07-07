"""Tests for the statistical estimators."""

from agent_replay.stats import bootstrap_diff_interval, mean, wilson_interval


def test_wilson_bounds_within_unit_interval():
    point, low, high = wilson_interval(3, 10)
    assert point == 0.3
    assert 0.0 <= low <= point <= high <= 1.0


def test_wilson_empty_n():
    point, low, high = wilson_interval(0, 0)
    assert (point, low, high) == (0.0, 0.0, 1.0)


def test_wilson_all_success_upper_near_one():
    point, low, high = wilson_interval(10, 10)
    assert point == 1.0
    # The Wilson upper bound approaches, but does not exceed, 1.0.
    assert 0.7 < high <= 1.0
    assert low < 1.0


def test_wilson_narrows_with_more_data():
    _, low_small, high_small = wilson_interval(5, 10)
    _, low_big, high_big = wilson_interval(500, 1000)
    assert (high_big - low_big) < (high_small - low_small)


def test_bootstrap_diff_point_estimate():
    kept = [True] * 10  # 100% failure
    ablated = [False] * 7 + [True] * 3  # 30% failure
    point, low, high = bootstrap_diff_interval(kept, ablated, iterations=500, seed=1)
    assert abs(point - 0.7) < 1e-9
    assert low <= point <= high


def test_bootstrap_diff_excludes_zero_for_strong_effect():
    kept = [True] * 20
    ablated = [False] * 20
    _, low, high = bootstrap_diff_interval(kept, ablated, iterations=500, seed=2)
    assert low > 0.0  # strong, unambiguous effect


def test_bootstrap_diff_brackets_zero_for_no_effect():
    kept = [True] * 10
    ablated = [True] * 10
    point, low, high = bootstrap_diff_interval(kept, ablated, iterations=500, seed=3)
    assert point == 0.0
    assert low <= 0.0 <= high


def test_mean():
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([]) == 0.0
