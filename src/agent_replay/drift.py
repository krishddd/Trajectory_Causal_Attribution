"""Per-step drift / entropy-of-autonomy curve (Multiverse deck, slide 2).

Slide 2 of *Architecting the Agent Multiverse* frames the core long-horizon risk
as **alignment decay**: over a long autonomous run an agent's internal state
drifts, and the failure is *silent* — the outcome still looks plausible while the
reasoning has already gone off the rails. Attribution (`attribute`) localises the
decisive step *after* a run has failed; this module instead charts a run's health
*as it unfolds*, giving two complementary per-step signals:

* **entropy of autonomy** — for each step, hold the factual prefix ``< i`` and
  resample everything from ``i`` onward ``rollouts`` times: the resulting
  ``P(success)`` is how *recoverable* the run still is at that juncture, and its
  binary entropy ``H`` is how *open* its fate remains. Early steps carry high
  entropy (much could still change); once the run commits to its outcome the
  entropy collapses to zero. Needs only the outcome ``verifier`` — always
  available. The step where entropy collapses is the empirical point of
  commitment, cross-checking :func:`attribute`.

* **alignment / health decay** — an *intermediate-state* signal the outcome
  verifier cannot see. Supply a ``state_scorer(step) -> float in [0, 1]`` (higher
  = healthier) and the curve records each step's health plus its **drift**: how
  far the state has fallen below the healthiest point reached so far. A run whose
  health decays while its outcome entropy is still high is drifting *silently* —
  degrading internally before the outcome reflects it. This is the hook the gap
  analysis called out as the one missing piece.

The whole thing reuses :class:`~agent_replay.ablation.AblationEngine` wholesale;
it is a scorer plus a self-contained SVG report, no new estimation machinery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, List, Optional

from .ablation import AblationEngine
from .stats import binary_entropy
from .types import Step, Trajectory

# A state scorer maps one recorded step to a health/alignment score in [0, 1],
# where higher is healthier. It sees the intermediate state the outcome verifier
# never does (the step's action/observation), so it can detect silent drift.
StateScorer = Callable[[Step], float]


@dataclass
class DriftPoint:
    """Health and outcome-openness of the run at one step."""

    index: int
    name: str
    kind: str
    p_success: float  # P(success) if the run is resampled from this step onward
    entropy: float  # binary entropy of that outcome distribution (bits, 0..1)
    health: Optional[float] = None  # intermediate-state score (None if no state_scorer)
    drift: float = 0.0  # decline in health below the healthiest step so far (0 if no scorer)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftResult:
    """The per-step drift/entropy curve for a run, plus its summary statistics."""

    session_id: str
    points: List[DriftPoint] = field(default_factory=list)
    entropy_auc: float = 0.0  # mean outcome entropy across steps (autonomy still open)
    commitment_index: Optional[int] = None  # last step whose fate is still open (entropy > eps)
    health_available: bool = False
    total_drift: float = 0.0  # largest health decline below the run's healthiest point
    drift_onset_index: Optional[int] = None  # step with the largest single-step health drop
    decayed: bool = False  # health ended materially below where it began / peaked
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["points"] = [p.to_dict() for p in self.points]
        return d

    def to_text(self) -> str:
        lines = [
            f"Drift curve for '{self.session_id}':",
            f"  entropy of autonomy (mean) {self.entropy_auc:.2f}; "
            + (
                f"outcome commits after step {self.commitment_index}."
                if self.commitment_index is not None
                else "outcome already committed at the first step."
            ),
        ]
        if self.health_available:
            lines.append(
                f"  alignment health: total drift {self.total_drift:.2f}"
                + (
                    f", onset at step {self.drift_onset_index}"
                    if self.drift_onset_index is not None
                    else ""
                )
                + (" -- DECAYED" if self.decayed else " -- stable")
            )
        if self.warning:
            lines.append(f"  ! {self.warning}")
        health_cols = "  health   drift" if self.health_available else ""
        header = "  step  entropy  P(success)" + health_cols
        lines.append(header)
        for p in self.points:
            row = f"    {p.index:>2}   {p.entropy:6.2f}    {p.p_success:7.2f}"
            if self.health_available and p.health is not None:
                row += f"   {p.health:6.2f}  {p.drift:6.2f}"
            lines.append(row)
        return "\n".join(lines)

    def to_html(self, path: str) -> str:
        """Write a standalone SVG drift-curve report and return the path."""
        html = render_drift_html(self)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return path


def _rate_success(fails: List[bool]) -> float:
    if not fails:
        return 0.0
    return sum(1 for f in fails if not f) / len(fails)


def drift(
    trajectory: Trajectory,
    agent_fn: Callable[..., Any],
    verifier: Callable[[Any], float],
    *,
    state_scorer: Optional[StateScorer] = None,
    rollouts: int = 20,
    fail_threshold: float = 0.5,
    entropy_eps: float = 0.05,
    drift_threshold: float = 0.2,
    base_seed: int = 3_000,
    pass_context: bool = True,
    cache=None,
) -> DriftResult:
    """Chart a run's health and outcome-openness step by step.

    For each step ``i`` the run is resampled from ``i`` onward ``rollouts`` times
    to estimate ``P(success)`` and its binary entropy — the "entropy of autonomy"
    that always works from the outcome ``verifier`` alone. If a ``state_scorer`` is
    supplied it is applied to each recorded step to add the *alignment health*
    overlay and its per-step **drift** (decline below the healthiest step so far),
    surfacing silent degradation the outcome verifier cannot see.

    ``commitment_index`` is the last step whose outcome entropy still exceeds
    ``entropy_eps`` (beyond it the fate is sealed); a run is flagged ``decayed``
    when its health falls more than ``drift_threshold`` below its peak.
    """
    engine = AblationEngine(
        agent_fn,
        trajectory,
        verifier,
        fail_threshold=fail_threshold,
        base_seed=base_seed,
        pass_context=pass_context,
        cache=cache,
    )

    points: List[DriftPoint] = []
    best_health = float("-inf")
    prev_health: Optional[float] = None
    total_drift = 0.0
    max_step_drop = 0.0
    drift_onset_index: Optional[int] = None
    entropy_sum = 0.0
    commitment_index: Optional[int] = None

    for step in trajectory.steps:
        p_success = _rate_success(engine.ablate_from(step.index, rollouts))
        ent = binary_entropy(p_success)
        entropy_sum += ent
        if ent > entropy_eps:
            commitment_index = step.index

        health: Optional[float] = None
        step_drift = 0.0
        if state_scorer is not None:
            health = float(state_scorer(step))
            best_health = max(best_health, health)
            step_drift = max(0.0, best_health - health)
            total_drift = max(total_drift, step_drift)
            drop = 0.0 if prev_health is None else max(0.0, prev_health - health)
            if drop > max_step_drop:
                max_step_drop = drop
                drift_onset_index = step.index
            prev_health = health

        points.append(
            DriftPoint(
                index=step.index,
                name=step.name,
                kind=step.kind.value,
                p_success=p_success,
                entropy=ent,
                health=health,
                drift=step_drift,
            )
        )

    health_available = state_scorer is not None and bool(points)
    decayed = health_available and total_drift >= drift_threshold
    entropy_auc = entropy_sum / len(points) if points else 0.0

    warning = None
    if decayed and commitment_index is not None:
        # Health drifted down while the outcome was still recoverable somewhere:
        # the classic silent-failure signature the deck warns about.
        onset = drift_onset_index if drift_onset_index is not None else 0
        if onset <= commitment_index:
            warning = (
                "Silent alignment decay: internal health drifted down "
                f"(from step {onset}) while the outcome still looked recoverable -- "
                "the run degraded before its result reflected it."
            )
    if warning is None and decayed:
        warning = (
            "Alignment health decayed over the run; inspect the drift onset for the "
            "point at which the state started deteriorating."
        )

    return DriftResult(
        session_id=trajectory.session_id,
        points=points,
        entropy_auc=entropy_auc,
        commitment_index=commitment_index,
        health_available=health_available,
        total_drift=total_drift,
        drift_onset_index=drift_onset_index,
        decayed=decayed,
        warning=warning,
    )


def _polyline(values: List[float], width: float, height: float, pad: float) -> str:
    """Map ``values`` (each in [0, 1]) to an SVG polyline points string."""
    n = len(values)
    if n == 0:
        return ""
    span = width - 2 * pad
    inner = height - 2 * pad
    if n == 1:
        x = pad + span / 2
        y = pad + (1.0 - values[0]) * inner
        return f"{x:.1f},{y:.1f}"
    pts = []
    for i, v in enumerate(values):
        x = pad + span * i / (n - 1)
        y = pad + (1.0 - max(0.0, min(1.0, v))) * inner
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def render_drift_html(result: DriftResult) -> str:
    """Render the drift/entropy curve as a self-contained SVG HTML page."""
    import html as _html

    width, height, pad = 720.0, 260.0, 36.0
    entropy_line = _polyline([p.entropy for p in result.points], width, height, pad)
    success_line = _polyline([p.p_success for p in result.points], width, height, pad)
    series = [
        ("Entropy of autonomy", "#8e44ad", entropy_line),
        ("P(success) from step", "#2980b9", success_line),
    ]
    if result.health_available:
        health_line = _polyline(
            [p.health if p.health is not None else 0.0 for p in result.points], width, height, pad
        )
        series.append(("Alignment health", "#27ae60", health_line))

    polylines = "".join(
        f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{pts}"/>'
        for _, color, pts in series
        if pts
    )
    legend = " &nbsp; ".join(
        f'<span style="color:{color}">&#9632;</span> {_html.escape(label)}'
        for label, color, _ in series
    )
    # Commitment marker.
    marker = ""
    if result.commitment_index is not None and len(result.points) > 1:
        n = len(result.points)
        x = pad + (width - 2 * pad) * result.commitment_index / (n - 1)
        marker = (
            f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{height - pad}" '
            f'stroke="#c0392b" stroke-width="1.5" stroke-dasharray="4 3"/>'
            f'<text x="{x + 4:.1f}" y="{pad + 12}" fill="#c0392b" font-size="11">commits</text>'
        )

    rows = []
    for p in result.points:
        health = f"{p.health:.2f}" if p.health is not None else "&ndash;"
        drift = f"{p.drift:.2f}" if result.health_available else "&ndash;"
        rows.append(
            f"<tr><td>{p.index}</td><td><code>{_html.escape(p.kind)}</code></td>"
            f"<td>{_html.escape(p.name)}</td><td>{p.entropy:.2f}</td>"
            f"<td>{p.p_success:.2f}</td><td>{health}</td><td>{drift}</td></tr>"
        )
    warning = (
        f'<p class="warn">&#9888; {_html.escape(result.warning)}</p>' if result.warning else ""
    )
    commit = (
        f"outcome commits after step {result.commitment_index}"
        if result.commitment_index is not None
        else "outcome committed at the first step"
    )
    decay = (
        f" &middot; total health drift {result.total_drift:.2f}"
        f" ({'decayed' if result.decayed else 'stable'})"
        if result.health_available
        else ""
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Drift curve &mdash; {_html.escape(result.session_id)}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #222; }}
 h1 {{ font-size: 18px; }}
 .sub {{ color: #555; margin-bottom: 1rem; }}
 .warn {{ background: #fff3cd; border: 1px solid #ffe08a; padding: .6rem .8rem;
   border-radius: 6px; }}
 svg {{ background: #fafafa; border: 1px solid #e2e2e2; border-radius: 8px; }}
 table {{ border-collapse: collapse; margin-top: 1rem; font-size: 13px; }}
 th, td {{ border: 1px solid #e2e2e2; padding: 4px 10px; text-align: right; }}
 th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
 .legend {{ margin: .6rem 0; font-size: 13px; }}
</style></head><body>
<h1>Entropy-of-autonomy &amp; alignment-drift curve</h1>
<p class="sub">Session <b>{_html.escape(result.session_id)}</b> &middot; mean entropy
 {result.entropy_auc:.2f} &middot; {commit}{decay}</p>
{warning}
<div class="legend">{legend}</div>
<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" preserveAspectRatio="xMidYMid meet">
 <rect x="{pad}" y="{pad}" width="{width - 2 * pad}" height="{height - 2 * pad}"
   fill="none" stroke="#e2e2e2"/>
 {marker}{polylines}
</svg>
<table>
 <tr><th>step</th><th>kind</th><th>name</th><th>entropy</th><th>P(success)</th>
   <th>health</th><th>drift</th></tr>
 {"".join(rows)}
</table>
</body></html>"""
