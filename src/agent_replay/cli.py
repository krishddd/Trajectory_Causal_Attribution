"""Command-line interface: ``agent-replay record | replay | attribute | report``.

The CLI operates on a SQLite checkpoint store. Because attribution must re-run
the agent under counterfactual plans, the ``record``/``replay``/``attribute``
commands need a handle to the user's agent (and verifier), supplied as
``module:function`` entrypoint specifications.

Example
-------
    agent-replay record   --db demo.sqlite --session demo \\
        --agent agent_replay.mock_agent:buggy_agent \\
        --verifier agent_replay.mock_agent:verifier --seed 1
    agent-replay attribute --db demo.sqlite --session demo \\
        --agent agent_replay.mock_agent:buggy_agent \\
        --verifier agent_replay.mock_agent:verifier \\
        --rollouts 60 --method both --repair --out report
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any, Callable, Optional

from .attribution import attribute
from .replayer import ReplayPlan, replay
from .session import Session
from .store import CheckpointStore
from .types import AttributionResult


def _load_entrypoint(spec: str) -> Callable[..., Any]:
    """Resolve a ``module:function`` (or ``module.function``) spec to a callable."""
    if ":" in spec:
        module_name, attr = spec.split(":", 1)
    elif "." in spec:
        module_name, attr = spec.rsplit(".", 1)
    else:
        raise ValueError(f"invalid entrypoint '{spec}', expected 'module:function'")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    return fn


def _parse_task(task_str: Optional[str]) -> dict:
    if not task_str:
        return {}
    return json.loads(task_str)


def _print_summary(result: AttributionResult) -> None:
    print("=" * 60)
    print("TRAJECTORY CAUSAL ATTRIBUTION REPORT")
    print("=" * 60)
    print(f"Session:      {result.session_id}")
    print(f"Total steps:  {result.total_steps}")
    outcome = "FAILED" if result.failed else "PASSED"
    print(f"Outcome:      {outcome} (verifier score: {result.outcome_score:.3f})")
    print(f"Method:       {result.method}  ({result.rollouts} rollouts/step)")
    print(f"Mode:         {result.mode}")
    print("-" * 60)
    if result.point_of_commitment is not None:
        label = "Save-Point" if result.mode == "credit" else "Point-of-Commitment"
        print(f"{label}: step {result.point_of_commitment}")
    if result.culprit is not None:
        c = result.culprit
        score = c.shapley if c.shapley is not None else c.attribution
        if result.mode == "credit":
            print(
                f"[RESULT] Success most secured by step {c.index} "
                f"({c.kind} {c.name}); re-decision risk {score:.3f}. "
                f"CI [{c.ci.low:.3f}, {c.ci.high:.3f}]"
            )
        else:
            print(
                f"[RESULT] Failure attributed to step {c.index} "
                f"({c.kind} {c.name}) with score {score:.3f}. "
                f"CI [{c.ci.low:.3f}, {c.ci.high:.3f}]"
            )
    else:
        print("[RESULT] No step reached the attribution significance threshold.")
    if result.repair is not None:
        r = result.repair
        status = "valid" if r.valid else "no valid repair"
        print(
            f"[REPAIR] step {r.step_index}: {r.original_action!r} -> {r.repaired_action!r} "
            f"({status}, minimality {r.minimality:.3f}, P(fail)->{r.p_fail_after:.3f})"
        )
    print("=" * 60)


def cmd_record(args: argparse.Namespace) -> int:
    agent_fn = _load_entrypoint(args.agent)
    verifier = _load_entrypoint(args.verifier) if args.verifier else None
    with Session(args.db) as sess:
        traj = sess.record(
            agent_fn,
            _parse_task(args.task),
            session_id=args.session,
            seed=args.seed,
            verifier=verifier,
        )
    score = "n/a" if traj.outcome_score is None else f"{traj.outcome_score:.3f}"
    print(f"Recorded session '{traj.session_id}': {len(traj)} steps, outcome score {score}")
    print(f"Root hash: {traj.root_hash}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    agent_fn = _load_entrypoint(args.agent)
    verifier = _load_entrypoint(args.verifier) if args.verifier else None
    with CheckpointStore(args.db) as store:
        traj = store.load_trajectory(args.session)
    result = replay(agent_fn, traj, ReplayPlan.factual(len(traj)), seed=traj.seed)
    print(f"Deterministic replay of '{args.session}' complete.")
    if verifier is not None:
        print(f"Replayed outcome score: {float(verifier(result)):.3f}")
    print(f"Result: {result}")
    return 0


def cmd_attribute(args: argparse.Namespace) -> int:
    from .errors import SuccessfulRunError

    agent_fn = _load_entrypoint(args.agent)
    verifier = _load_entrypoint(args.verifier)
    with CheckpointStore(args.db) as store:
        traj = store.load_trajectory(args.session)
        try:
            result = attribute(
                traj,
                agent_fn,
                verifier,
                rollouts=args.rollouts,
                method=args.method,
                permutation_pairs=args.permutation_pairs,
                repair=args.repair,
                fail_threshold=args.fail_threshold,
                base_seed=args.base_seed,
                on_success=args.on_success,
            )
        except SuccessfulRunError as exc:
            print(f"Nothing to attribute: {exc}")
            print("Re-run with --on-success credit to analyse which step secured success.")
            return 1
        store.save_attribution(result)
        explanation = result.explain(traj)
    _print_summary(result)
    if not args.no_explain:
        print("-" * 60)
        print(explanation.to_text())
    if args.out:
        json_path = f"{args.out}.json"
        html_path = f"{args.out}.html"
        result.to_json(json_path, explanation=explanation)
        result.to_html(html_path, explanation=explanation)
        print(f"Wrote {json_path} and {html_path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    with open(args.json, encoding="utf-8") as fh:
        data = json.load(fh)
    result = _result_from_dict(data)
    explanation = None
    if data.get("explanation"):
        from .explain import Explanation

        explanation = Explanation.from_dict(data["explanation"])
    out = args.out or (args.json.rsplit(".", 1)[0] + ".html")
    result.to_html(out, explanation=explanation)
    print(f"Wrote {out}")
    return 0


def _result_from_dict(data: dict) -> AttributionResult:
    """Rebuild an AttributionResult from a previously written JSON report."""
    from .types import ConfidenceInterval, Repair, StepAttribution

    def ci(d: Optional[dict]) -> Optional[ConfidenceInterval]:
        if not d:
            return None
        return ConfidenceInterval(**d)

    steps = []
    for s in data["steps"]:
        steps.append(
            StepAttribution(
                index=s["index"],
                name=s["name"],
                kind=s["kind"],
                p_fail_kept=s["p_fail_kept"],
                p_fail_ablated=s["p_fail_ablated"],
                attribution=s["attribution"],
                ci=ci(s["ci"]),
                shapley=s.get("shapley"),
                shapley_ci=ci(s.get("shapley_ci")),
                resamplable=s.get("resamplable", True),
                screened=s.get("screened", False),
            )
        )
    repair = None
    if data.get("repair"):
        repair = Repair(**data["repair"])
    return AttributionResult(
        session_id=data["session_id"],
        total_steps=data["total_steps"],
        outcome_score=data["outcome_score"],
        failed=data["failed"],
        method=data["method"],
        rollouts=data["rollouts"],
        steps=steps,
        point_of_commitment=data.get("point_of_commitment"),
        culprit_index=data.get("culprit_index"),
        mode=data.get("mode", "failure"),
        repair=repair,
        meta=data.get("meta", {}),
    )


def cmd_list(args: argparse.Namespace) -> int:
    with CheckpointStore(args.db) as store:
        sessions = store.list_sessions()
        if not sessions:
            print("(no sessions)")
            return 0
        for sid in sessions:
            traj = store.load_trajectory(sid)
            score = "n/a" if traj.outcome_score is None else f"{traj.outcome_score:.3f}"
            print(f"{sid}\t{len(traj)} steps\toutcome {score}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent-replay",
        description="Record agent trajectories and attribute failures via counterfactual "
        "step-ablation replay.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    common_agent = argparse.ArgumentParser(add_help=False)
    common_agent.add_argument("--db", required=True, help="SQLite checkpoint store path")
    common_agent.add_argument("--session", required=True, help="session id")
    common_agent.add_argument("--agent", required=True, help="agent entrypoint as module:function")

    pr = sub.add_parser("record", parents=[common_agent], help="record a factual agent run")
    pr.add_argument("--verifier", help="verifier entrypoint module:function")
    pr.add_argument("--task", help="JSON dict of task kwargs")
    pr.add_argument("--seed", type=int, default=0)
    pr.set_defaults(func=cmd_record)

    pl = sub.add_parser("list", help="list recorded sessions in a store")
    pl.add_argument("--db", required=True, help="SQLite checkpoint store path")
    pl.set_defaults(func=cmd_list)

    prp = sub.add_parser("replay", parents=[common_agent], help="deterministically replay a run")
    prp.add_argument("--verifier", help="verifier entrypoint module:function")
    prp.set_defaults(func=cmd_replay)

    pa = sub.add_parser("attribute", parents=[common_agent], help="attribute failure to a step")
    pa.add_argument("--verifier", required=True, help="verifier entrypoint module:function")
    pa.add_argument("--rollouts", type=int, default=50)
    pa.add_argument("--method", choices=["contrastive", "shapley", "both"], default="contrastive")
    pa.add_argument("--permutation-pairs", type=int, default=8, dest="permutation_pairs")
    pa.add_argument("--repair", action="store_true", help="also search for a minimal repair")
    pa.add_argument(
        "--fail-threshold",
        type=float,
        default=0.5,
        dest="fail_threshold",
        help="outcome score below this counts as failure (default 0.5)",
    )
    pa.add_argument(
        "--base-seed",
        type=int,
        default=1_000,
        dest="base_seed",
        help="base seed for the rollout seed stream",
    )
    pa.add_argument(
        "--on-success",
        choices=["error", "credit"],
        default="error",
        dest="on_success",
        help="behaviour when the run passed: error (default) or credit analysis",
    )
    pa.add_argument("--out", help="output report basename (writes .json and .html)")
    pa.add_argument(
        "--no-explain",
        action="store_true",
        dest="no_explain",
        help="suppress the plain-language explanation output",
    )
    pa.set_defaults(func=cmd_attribute)

    prep = sub.add_parser("report", help="regenerate an HTML report from a JSON report")
    prep.add_argument("--json", required=True, help="path to a JSON attribution report")
    prep.add_argument("--out", help="output HTML path")
    prep.set_defaults(func=cmd_report)

    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
