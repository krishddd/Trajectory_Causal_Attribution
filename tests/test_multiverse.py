"""The Multiverse: fork / resume / diff / branches."""

import asyncio

from _demo_agent import buggy_agent, verifier
from agent_replay.multiverse import afork, diff, fork, resume
from agent_replay.store import CheckpointStore


def test_fork_do_intervention_rescues(recording, fail_step):
    # Force the culprit step to "OK" from the fork point: the child run passes.
    child = fork(buggy_agent, recording, fail_step, do="OK", seed=1, verifier=verifier)
    assert child.outcome_score == 1.0
    assert child.meta["parent_session"] == recording.session_id
    assert child.meta["fork_step"] == fail_step
    assert child.meta["intervention"] == "do"
    # Prefix is shared verbatim with the parent.
    for i in range(fail_step):
        assert child.steps[i].output == recording.steps[i].output
    # The forced step carries the injected action.
    assert child.steps[fail_step].output == "OK"


def test_fork_prefix_matches_parent(recording, fail_step):
    child = fork(buggy_agent, recording, fail_step, do="OK", seed=3)
    assert len(child) == len(recording)  # same shape for this linear agent
    assert child.steps[:fail_step] == recording.steps[:fail_step] or all(
        child.steps[i].output == recording.steps[i].output for i in range(fail_step)
    )


def test_fork_remove(recording, fail_step):
    child = fork(buggy_agent, recording, fail_step, remove=True, seed=1)
    assert child.meta["intervention"] == "remove"
    # Removed step yields the empty sentinel -> not "BAD".
    assert child.steps[fail_step].output != "BAD"


def test_diff_localises_divergence(recording, fail_step):
    child = fork(buggy_agent, recording, fail_step, do="OK", seed=1)
    d = diff(recording, child)
    # They agree on the prefix and first diverge at the forced step.
    assert d["first_divergence"] == fail_step
    assert d["n_diff"] >= 1
    assert d["steps"][0]["same"] is True


def test_diff_identical_is_zero(recording):
    d = diff(recording, recording)
    assert d["first_divergence"] is None
    assert d["n_diff"] == 0


def test_branches_persisted_and_listed(recording, fail_step, tmp_path):
    db = str(tmp_path / "mv.sqlite")
    child_a = fork(
        buggy_agent, recording, fail_step, do="OK", seed=1, session_id="child-a", verifier=verifier
    )
    child_b = fork(
        buggy_agent, recording, fail_step, do="OK", seed=2, session_id="child-b", verifier=verifier
    )
    with CheckpointStore(db) as store:
        store.save_trajectory(recording)
        store.save_trajectory(child_a)
        store.save_trajectory(child_b)
        branches = store.branches(recording.session_id)
        assert set(branches) == {"child-a", "child-b"}
        assert store.branches("child-a") == []
        # Reloaded child keeps its parent link.
        reloaded = store.load_trajectory("child-a")
        assert reloaded.meta["parent_session"] == recording.session_id


def test_resume_reproduces_completed_run(recording):
    child = resume(buggy_agent, recording, seed=0, verifier=verifier)
    assert child.meta["fork_step"] == len(recording)
    # A completed run makes no extra calls, so resume reproduces it.
    assert [s.output for s in child.steps] == [s.output for s in recording.steps]


def test_fork_storage_dedup(recording, fail_step, tmp_path):
    # The shared prefix dedupes through the CAS blob store.
    db = str(tmp_path / "dedup.sqlite")
    child = fork(buggy_agent, recording, fail_step, do="OK", seed=1, session_id="c")
    with CheckpointStore(db) as store:
        store.save_trajectory(recording)
        before = store._conn.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
        store.save_trajectory(child)
        after = store._conn.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
    # The child adds only a few new blobs (forced action + continuation), not a
    # full copy of the shared prefix.
    assert after - before < len(recording)


def test_cli_fork_branches_diff(tmp_path, capsys):
    from agent_replay.cli import main

    db = str(tmp_path / "cli_mv.sqlite")
    AGENT = "_demo_agent:buggy_agent"
    VERIFIER = "_demo_agent:verifier"
    main(
        [
            "record",
            "--db",
            db,
            "--session",
            "root",
            "--agent",
            AGENT,
            "--verifier",
            VERIFIER,
            "--seed",
            "1",
        ]
    )
    capsys.readouterr()
    # Fork forcing step 3 to "OK".
    rc = main(
        [
            "fork",
            "--db",
            db,
            "--session",
            "root",
            "--agent",
            AGENT,
            "--verifier",
            VERIFIER,
            "--at-step",
            "3",
            "--do",
            '"OK"',
            "--out-session",
            "fixed",
            "--seed",
            "1",
        ]
    )
    assert rc == 0
    assert "Forked 'root'" in capsys.readouterr().out
    # Branches lists it.
    main(["branches", "--db", db, "--session", "root"])
    assert "fixed" in capsys.readouterr().out
    # Diff shows divergence at step 3.
    rc = main(["diff", "--db", db, "--a", "root", "--b", "fixed"])
    assert rc == 0
    assert "step 3" in capsys.readouterr().out


def test_async_fork():
    async def async_agent(ctx, task="t", fail_at=1, n=3):
        trace = []
        ok = True
        for i in range(n):

            async def produce(step=i):
                return "bad" if (step == fail_at and ctx.rng.random() < 0.9) else "ok"

            act = await ctx.tool(f"s{i}", produce=produce)
            trace.append(act)
            if act == "bad":
                ok = False
        return {"trace": trace, "ok": ok}

    def verify(r):
        return 1.0 if r["ok"] else 0.0

    from agent_replay.recorder import record

    traj = None
    for seed in range(50):
        t = record(async_agent, {}, session_id=f"a{seed}", seed=seed, verifier=verify)
        if t.outcome_score == 0.0:
            traj = t
            break
    assert traj is not None
    child = asyncio.run(afork(async_agent, traj, 1, do="ok", seed=1, verifier=verify))
    assert child.outcome_score == 1.0
    assert child.steps[1].output == "ok"
