"""Property-based tests (hypothesis) on the plan algebra and cassette matcher.

These fuzz the two core invariants of the replay engine:

1. **Factual fidelity** — replaying the factual plan reproduces the recording.
2. **No cross-contamination** — when an upstream resample changes the control
   flow, a held step is *never* served another step's recorded output; it is only
   served the cassette when the very same operation (idempotency key) recurs,
   otherwise it re-runs live. This is what makes branching agents sound.

plus the pure plan-algebra decision precedence.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from agent_replay.recorder import record
from agent_replay.replayer import ReplayPlan, replay


def router(ctx, task="t"):
    """A branching agent: step 0 picks a route; downstream ops depend on it.

    Follows the contract — the context each step consumes (``r``, ``x``) is passed
    as ``inputs``, so the idempotency key reflects it and key-matching is sound.
    """
    r = ctx.tool("route", produce=lambda: "a" if ctx.rng.random() < 0.5 else "b")
    x = ctx.tool(f"fetch_{r}", produce=lambda: f"data_{r}", route=r)
    y = ctx.llm("combine", produce=lambda: f"{r}|{x}", r=r, x=x)
    return {"r": r, "x": x, "y": y}


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=40, deadline=None)
def test_factual_replay_is_faithful(seed):
    traj = record(router, {}, session_id="p", seed=seed)
    out = replay(router, traj, ReplayPlan.factual(len(traj)), seed=seed)
    assert out == traj.result


@given(
    rec_seed=st.integers(min_value=0, max_value=5_000),
    rep_seed=st.integers(min_value=0, max_value=20_000),
)
@settings(max_examples=150, deadline=None)
def test_no_cross_contamination_under_resample(rec_seed, rep_seed):
    # Record, then hold {1,2} while resampling step 0 (the route). If the route
    # flips, the downstream fetch/combine ops change identity: they must never be
    # served the recorded route's output. Every rollout must stay internally
    # consistent with the route it actually took.
    traj = record(router, {}, session_id="p", seed=rec_seed)
    out = replay(router, traj, ReplayPlan(held={1, 2}), seed=rep_seed)
    assert out["x"] == f"data_{out['r']}"
    assert out["y"] == f"{out['r']}|{out['x']}"


@given(
    rec_seed=st.integers(min_value=0, max_value=5_000),
    coalition=st.sets(st.integers(min_value=0, max_value=2), max_size=3),
    rep_seed=st.integers(min_value=0, max_value=20_000),
)
@settings(max_examples=150, deadline=None)
def test_no_contamination_for_any_coalition(rec_seed, coalition, rep_seed):
    # The Shapley value function holds arbitrary (non-contiguous) coalitions.
    # For every coalition, no held step may leak another op's recorded output.
    traj = record(router, {}, session_id="p", seed=rec_seed)
    out = replay(router, traj, ReplayPlan.coalition(coalition), seed=rep_seed)
    assert out["x"] == f"data_{out['r']}"
    assert out["y"] == f"{out['r']}|{out['x']}"


@given(
    i=st.integers(min_value=0, max_value=6),
    n=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=50)
def test_ablate_from_holds_exact_prefix(i, n):
    plan = ReplayPlan.ablate_from(i, n)
    for idx in range(max(i, n) + 2):
        expected = "hold" if idx < i else "resample"
        assert plan.decision(idx) == expected


@given(
    forced=st.sets(st.integers(0, 5), max_size=6),
    removed=st.sets(st.integers(0, 5), max_size=6),
    observed=st.sets(st.integers(0, 5), max_size=6),
    held=st.sets(st.integers(0, 5), max_size=6),
    idx=st.integers(0, 5),
)
@settings(max_examples=200)
def test_decision_precedence(forced, removed, observed, held, idx):
    # Precedence must be force > remove > mock_observe > hold > resample.
    plan = ReplayPlan(
        held=held,
        forced={i: "f" for i in forced},
        removed=removed,
        observed={i: "o" for i in observed},
    )
    d = plan.decision(idx)
    if idx in forced:
        assert d == "force"
    elif idx in removed:
        assert d == "remove"
    elif idx in observed:
        assert d == "mock_observe"
    elif idx in held:
        assert d == "hold"
    else:
        assert d == "resample"
