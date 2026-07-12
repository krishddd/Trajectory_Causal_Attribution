"""Tests for the attribution scorer.

The central correctness claim of the library: counterfactual step-ablation with
the Point-of-Commitment Rule localises the *known* culprit step of the mock
agent, and does not fall for the butterfly-effect confound that inflates early
steps' raw scores.
"""

from _demo_agent import buggy_agent, verifier
from agent_replay.ablation import AblationEngine
from agent_replay.attribution import (
    attribute,
    contrastive_attribution,
    point_of_commitment,
    shapley_attribution,
)


def _engine(recording):
    return AblationEngine(buggy_agent, recording, verifier)


def test_point_of_commitment_localises_culprit(recording, fail_step):
    engine = _engine(recording)
    steps = contrastive_attribution(engine, rollouts=120)
    poc = point_of_commitment(steps)
    assert poc == fail_step


def test_steps_after_culprit_have_zero_attribution(recording, fail_step):
    engine = _engine(recording)
    steps = contrastive_attribution(engine, rollouts=120)
    for s in steps:
        if s.index > fail_step:
            assert abs(s.attribution) < 1e-9
            assert not s.ci.excludes_zero()


def test_culprit_attribution_is_positive_and_significant(recording, fail_step):
    engine = _engine(recording)
    steps = contrastive_attribution(engine, rollouts=120)
    culprit = steps[fail_step]
    assert culprit.attribution > 0.0
    assert culprit.ci.excludes_zero()


def test_attribute_end_to_end_contrastive(recording, fail_step):
    result = attribute(recording, buggy_agent, verifier, rollouts=120, method="contrastive")
    assert result.failed
    assert result.point_of_commitment == fail_step
    assert result.culprit_index == fail_step
    assert result.culprit.kind == "tool"


def test_attribute_shapley_credits_culprit(recording, fail_step):
    result = attribute(
        recording, buggy_agent, verifier, rollouts=40, method="shapley", permutation_pairs=10
    )
    # The culprit should carry the largest Shapley share.
    best = max(result.steps, key=lambda s: s.shapley)
    assert best.index == fail_step
    assert best.shapley > 0


def test_shapley_efficiency_axiom(recording):
    engine = _engine(recording)
    steps = shapley_attribution(engine, rollouts=40, permutation_pairs=12)
    total = sum(s.shapley for s in steps)
    # sum(phi) == v(full) - v(empty) == 1 - baseline_fail_rate, which is in [0, 1].
    assert 0.0 <= total <= 1.0 + 1e-6


def test_attribute_both_attaches_shapley(recording, fail_step):
    result = attribute(
        recording, buggy_agent, verifier, rollouts=40, method="both", permutation_pairs=8
    )
    assert result.point_of_commitment == fail_step
    for s in result.steps:
        assert s.shapley is not None


def test_attribute_with_repair(recording, fail_step):
    result = attribute(
        recording, buggy_agent, verifier, rollouts=60, method="contrastive", repair=True
    )
    assert result.repair is not None
    assert result.repair.step_index == fail_step
    assert result.repair.valid
    assert result.repair.p_fail_after < 0.5
