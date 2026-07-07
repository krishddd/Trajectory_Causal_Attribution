"""Failure-attribution report rendering (standalone HTML).

The JSON form is produced directly from :meth:`AttributionResult.to_dict`; this
module renders the human-facing HTML: a summary banner, a per-step attribution
table with confidence intervals, a lightweight timeline visualisation, and the
validated minimal repair — echoing the diagnostic log format in the paper.

The HTML is self-contained (inline CSS, no external assets) so a report can be
opened directly or emailed as a single file.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import AttributionResult, StepAttribution


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _bar(value: float, max_abs: float) -> str:
    """A horizontal bar cell whose width encodes ``|value| / max_abs``."""
    pct = 0.0 if max_abs <= 0 else min(100.0, abs(value) / max_abs * 100.0)
    color = "#c0392b" if value > 0 else "#7f8c8d"
    return (
        f'<div class="barwrap"><div class="bar" style="width:{pct:.1f}%;'
        f'background:{color}"></div><span class="barval">{_fmt(value)}</span></div>'
    )


def _step_row(s: "StepAttribution", is_culprit: bool, is_poc: bool, max_abs: float) -> str:
    flags = []
    if is_culprit:
        flags.append('<span class="tag culprit">culprit</span>')
    if is_poc:
        flags.append('<span class="tag poc">point-of-commitment</span>')
    flag_html = " ".join(flags)
    shap = _fmt(s.shapley) if s.shapley is not None else "&ndash;"
    ci = s.ci
    row_cls = "culprit-row" if is_culprit else ""
    return f"""
      <tr class="{row_cls}">
        <td class="num">{s.index}</td>
        <td><code>{html.escape(s.kind)}</code></td>
        <td>{html.escape(s.name)} {flag_html}</td>
        <td class="num">{_fmt(s.p_fail_ablated)}</td>
        <td>{_bar(s.attribution, max_abs)}</td>
        <td class="num">[{_fmt(ci.low)}, {_fmt(ci.high)}]</td>
        <td class="num">{shap}</td>
      </tr>"""


def render_html(result: "AttributionResult") -> str:
    """Render ``result`` to a complete, standalone HTML document string."""
    steps = result.steps
    max_abs = max((abs(s.attribution) for s in steps), default=1.0) or 1.0

    rows = "\n".join(
        _step_row(
            s,
            is_culprit=(s.index == result.culprit_index),
            is_poc=(s.index == result.point_of_commitment),
            max_abs=max_abs,
        )
        for s in steps
    )

    outcome_cls = "fail" if result.failed else "pass"
    outcome_txt = "FAILED" if result.failed else "PASSED"

    culprit = result.culprit
    if culprit is not None:
        score = culprit.shapley if culprit.shapley is not None else culprit.attribution
        verdict = (
            f"Failure attributed to <b>step {culprit.index}</b> "
            f"(<code>{html.escape(culprit.kind)}:{html.escape(culprit.name)}</code>) "
            f"with score <b>{_fmt(score)}</b>. "
            f"95% CI [{_fmt(culprit.ci.low)}, {_fmt(culprit.ci.high)}]."
        )
    else:
        verdict = "No single step reached the significance threshold for attribution."

    repair_html = _render_repair(result)

    # Timeline: a row of nodes coloured by responsibility.
    nodes = []
    for s in steps:
        cls = "node"
        if s.index == result.culprit_index:
            cls += " node-culprit"
        elif s.index == result.point_of_commitment:
            cls += " node-poc"
        nodes.append(
            f'<div class="{cls}" title="step {s.index}: {html.escape(s.name)} '
            f'(attr {_fmt(s.attribution)})"><span>{s.index}</span></div>'
        )
        if s.index != steps[-1].index:
            nodes.append('<div class="edge"></div>')
    timeline = "".join(nodes)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>agent-replay &mdash; Failure Attribution Report</title>
<style>
  :root {{ --bg:#0f1419; --card:#1a2029; --ink:#e6e9ef; --muted:#8b95a5; --accent:#c0392b; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:var(--bg); color:var(--ink); line-height:1.5; }}
  .wrap {{ max-width:960px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:24px; }}
  .card {{ background:var(--card); border:1px solid #232b36; border-radius:10px;
          padding:20px 22px; margin-bottom:20px; }}
  .banner {{ display:flex; align-items:center; gap:16px; }}
  .pill {{ font-weight:700; padding:6px 14px; border-radius:999px; font-size:13px; letter-spacing:.5px; }}
  .pill.fail {{ background:#3a1618; color:#ff6b6b; border:1px solid #5a2226; }}
  .pill.pass {{ background:#123020; color:#4ade80; border:1px solid #1e4a33; }}
  .kv {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin-top:8px; }}
  .kv div span {{ display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.5px; }}
  .kv div b {{ font-size:16px; }}
  .verdict {{ font-size:15px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #232b36; }}
  th {{ color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.5px; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  code {{ background:#0c1116; padding:1px 5px; border-radius:4px; font-size:12px; }}
  .culprit-row {{ background:#241416; }}
  .tag {{ font-size:10px; padding:1px 6px; border-radius:999px; margin-left:6px; }}
  .tag.culprit {{ background:var(--accent); color:#fff; }}
  .tag.poc {{ background:#2c3e50; color:#8ecae6; }}
  .barwrap {{ position:relative; background:#0c1116; border-radius:4px; height:18px; min-width:120px; }}
  .bar {{ height:100%; border-radius:4px; }}
  .barval {{ position:absolute; right:6px; top:0; font-size:11px; line-height:18px; }}
  .timeline {{ display:flex; align-items:center; flex-wrap:wrap; gap:2px; padding:8px 0; }}
  .node {{ width:34px; height:34px; border-radius:50%; background:#232b36; display:flex;
          align-items:center; justify-content:center; font-size:12px; border:2px solid #2c3542; }}
  .node-culprit {{ background:var(--accent); border-color:#e74c3c; color:#fff; font-weight:700; }}
  .node-poc {{ border-color:#8ecae6; }}
  .edge {{ width:14px; height:2px; background:#2c3542; }}
  .repair code {{ display:block; padding:8px 10px; margin-top:6px; white-space:pre-wrap; }}
  footer {{ color:var(--muted); font-size:12px; text-align:center; margin-top:24px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Trajectory Causal Attribution Report</h1>
  <div class="sub">session <code>{html.escape(result.session_id)}</code>
    &middot; method <code>{html.escape(result.method)}</code>
    &middot; {result.rollouts} rollouts/step</div>

  <div class="card banner">
    <span class="pill {outcome_cls}">{outcome_txt}</span>
    <div class="kv">
      <div><span>Total steps</span><b>{result.total_steps}</b></div>
      <div><span>Outcome score</span><b>{_fmt(result.outcome_score)}</b></div>
      <div><span>Point of commitment</span><b>{result.point_of_commitment if result.point_of_commitment is not None else "&ndash;"}</b></div>
      <div><span>Culprit step</span><b>{result.culprit_index if result.culprit_index is not None else "&ndash;"}</b></div>
    </div>
  </div>

  <div class="card">
    <p class="verdict">{verdict}</p>
    <div class="timeline">{timeline}</div>
  </div>

  <div class="card">
    <table>
      <thead>
        <tr>
          <th>#</th><th>Kind</th><th>Step</th>
          <th class="num">P(fail&nbsp;|&nbsp;ablated)</th>
          <th>Attribution&nbsp;= P(fail|kept)&minus;P(fail|ablated)</th>
          <th class="num">95% CI</th>
          <th class="num">Shapley</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>

  {repair_html}

  <footer>Generated by agent-replay &mdash; counterfactual step-ablation attribution.</footer>
</div>
</body>
</html>"""


def _render_repair(result: "AttributionResult") -> str:
    r = result.repair
    if r is None:
        return ""
    status = "valid" if r.valid else "no valid repair found"
    return f"""
  <div class="card repair">
    <h1 style="font-size:16px">Minimal Counterfactual Repair</h1>
    <div class="kv">
      <div><span>Step</span><b>{r.step_index}</b></div>
      <div><span>P(fail) after</span><b>{_fmt(r.p_fail_after)}</b></div>
      <div><span>Minimality</span><b>{_fmt(r.minimality)}</b></div>
      <div><span>Status</span><b>{status}</b></div>
    </div>
    <p style="color:var(--muted);font-size:12px;margin-bottom:2px">Original action</p>
    <code class="repair">{html.escape(str(r.original_action))}</code>
    <p style="color:var(--muted);font-size:12px;margin:10px 0 2px">Repaired action</p>
    <code class="repair">{html.escape(str(r.repaired_action))}</code>
  </div>"""
