"""End-to-end integration test over the full public API.

Records a custom agent through a Session, stores it, reloads it, attributes the
failure, and generates both report formats — the exact path a user follows.
"""

from agent_replay import Session, attribute


def make_agent(fail_at=2, n=5, p=0.7):
    def agent(ctx, task="e2e"):
        state = {"trace": [], "ok": True}
        for i in range(n):

            def produce(step=i):
                if step == fail_at:
                    return "BAD" if ctx.rng.random() < p else "GOOD"
                return "GOOD"

            act = ctx.tool(f"step_{i}", produce=produce)
            state["trace"].append(act)
            if act == "BAD":
                state["ok"] = False
        return state

    return agent


def verifier(result):
    return 1.0 if result["ok"] else 0.0


def _record_failing(session, agent):
    for seed in range(1, 100):
        traj = session.record(
            agent, {"task": "e2e"}, session_id=f"e2e-{seed}", seed=seed, verifier=verifier
        )
        if traj.outcome_score == 0.0:
            return traj
    raise AssertionError("could not produce a failing recording")


def test_full_pipeline(tmp_path):
    fail_at = 2
    agent = make_agent(fail_at=fail_at)
    db = str(tmp_path / "e2e.sqlite")

    with Session(db) as session:
        traj = _record_failing(session, agent)
        sid = traj.session_id

        # Reload from disk to prove persistence.
        reloaded = session.load(sid)
        assert len(reloaded) == 5
        assert reloaded.outcome_score == 0.0

        result = attribute(reloaded, agent, verifier, rollouts=100, method="both", repair=True)

    assert result.failed
    assert result.point_of_commitment == fail_at
    assert result.culprit_index == fail_at
    assert result.repair is not None and result.repair.valid

    json_path = str(tmp_path / "e2e.json")
    html_path = str(tmp_path / "e2e.html")
    result.to_json(json_path)
    result.to_html(html_path)
    assert (tmp_path / "e2e.json").exists()
    assert (tmp_path / "e2e.html").exists()


def test_session_lists_recordings(tmp_path):
    db = str(tmp_path / "list.sqlite")
    agent = make_agent()
    with Session(db) as session:
        session.record(agent, {"task": "a"}, session_id="one", seed=1, verifier=verifier)
        session.record(agent, {"task": "b"}, session_id="two", seed=2, verifier=verifier)
        assert set(session.sessions()) == {"one", "two"}
