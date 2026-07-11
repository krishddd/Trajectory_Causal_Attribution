"""The Multiverse Console: a zero-dependency web browser for recorded runs.

A tiny ``http.server`` UI (stdlib only) that turns a checkpoint store into the
"AgentOps Multiverse" console from the research deck (slide 15): list sessions,
step through a trajectory's frozen state (each step's inputs/outputs), and walk
the branch graph produced by :func:`agent_replay.fork` (parent ↔ children), with
outcome badges throughout.

The HTML is produced by the pure ``render_*`` functions (unit-testable without a
socket); the handler just routes to them.
"""

from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import unquote

from .store import CheckpointStore
from .types import Trajectory

_STYLE = """
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:#0f1419;color:#e6e9ef;line-height:1.5}
.wrap{max-width:960px;margin:0 auto;padding:28px 20px 64px}
a{color:#8ecae6;text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:20px}h2{font-size:15px;color:#8b95a5;text-transform:uppercase;letter-spacing:.5px}
.card{background:#1a2029;border:1px solid #232b36;border-radius:10px;padding:16px 18px}
.card{margin:14px 0}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid #232b36;vertical-align:top}
th{color:#8b95a5;font-size:11px;text-transform:uppercase;letter-spacing:.5px}
code{background:#0c1116;padding:1px 5px;border-radius:4px;font-size:12px;white-space:pre-wrap}
.pill{font-weight:700;padding:3px 10px;border-radius:999px;font-size:12px}
.pass{background:#123020;color:#4ade80}.fail{background:#3a1618;color:#ff6b6b}
.na{background:#232b36;color:#8b95a5}
.tag{font-size:10px;padding:1px 6px;border-radius:999px;background:#3a2f14;color:#e0b050}
.tag{margin-left:6px}
.muted{color:#8b95a5;font-size:12px}
"""


def _outcome_pill(score: Optional[float]) -> str:
    if score is None:
        return '<span class="pill na">n/a</span>'
    cls = "pass" if score >= 0.5 else "fail"
    txt = "PASS" if score >= 0.5 else "FAIL"
    return f'<span class="pill {cls}">{txt} {score:.2f}</span>'


def _page(title: str, body: str) -> str:
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><div class='wrap'>{body}</div></body></html>"
    )


def render_index(store: CheckpointStore) -> str:
    """The session list, grouped so forked branches nest under their parent."""
    sessions = store.list_sessions()
    trajs = {sid: store.load_trajectory(sid) for sid in sessions}
    roots = [t for t in trajs.values() if not t.meta.get("parent_session")]
    orphans = [
        t
        for t in trajs.values()
        if t.meta.get("parent_session") and t.meta["parent_session"] not in trajs
    ]
    rows = []

    def render_node(t: Trajectory, depth: int) -> None:
        pad = depth * 22
        link = f"<a href='/session/{html.escape(t.session_id)}'>{html.escape(t.session_id)}</a>"
        fork_tag = ""
        if t.meta.get("parent_session"):
            label = f"fork@{t.meta.get('fork_step')} {t.meta.get('intervention', '')}"
            fork_tag = f"<span class='tag'>{label}</span>"
        rows.append(
            f"<tr><td style='padding-left:{pad}px'>{link}{fork_tag}</td>"
            f"<td>{len(t)}</td><td>{_outcome_pill(t.outcome_score)}</td></tr>"
        )
        for child_id in store.branches(t.session_id):
            if child_id in trajs:
                render_node(trajs[child_id], depth + 1)

    for r in roots + orphans:
        render_node(r, 0)

    if not rows:
        body = "<h1>Multiverse Console</h1><div class='card muted'>No sessions recorded yet.</div>"
    else:
        body = (
            "<h1>Multiverse Console</h1>"
            "<div class='card'><table><thead><tr><th>Session</th><th>Steps</th>"
            "<th>Outcome</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        )
    return _page("Multiverse Console", body)


def render_session(store: CheckpointStore, session_id: str) -> str:
    """A trajectory's frozen state: per-step inputs/outputs, plus branch links."""
    try:
        t = store.load_trajectory(session_id)
    except KeyError:
        return _page(
            "Not found",
            f"<h1>Session {html.escape(session_id)} not found</h1>"
            "<p><a href='/'>&larr; back</a></p>",
        )

    parent = t.meta.get("parent_session")
    parent_html = (
        f"<p class='muted'>forked from "
        f"<a href='/session/{html.escape(str(parent))}'>{html.escape(str(parent))}</a> "
        f"at step {t.meta.get('fork_step')} ({t.meta.get('intervention', '')})</p>"
        if parent
        else ""
    )
    kids = store.branches(session_id)
    kids_html = ""
    if kids:
        links = ", ".join(f"<a href='/session/{html.escape(k)}'>{html.escape(k)}</a>" for k in kids)
        kids_html = f"<p class='muted'>branches: {links}</p>"

    fork_step = t.meta.get("fork_step")
    step_rows = []
    for s in t.steps:
        tag = "" if s.resamplable else "<span class='tag'>observed-only</span>"
        forked = (
            " style='background:#241416'" if fork_step is not None and s.index == fork_step else ""
        )
        step_rows.append(
            f"<tr{forked}><td>{s.index}</td><td><code>{html.escape(s.kind.value)}</code></td>"
            f"<td>{html.escape(s.name)}{tag}</td>"
            f"<td><code>{html.escape(_short(s.inputs))}</code></td>"
            f"<td><code>{html.escape(_short(s.output))}</code></td></tr>"
        )

    body = (
        f"<p><a href='/'>&larr; all sessions</a></p>"
        f"<h1>{html.escape(session_id)} {_outcome_pill(t.outcome_score)}</h1>"
        f"{parent_html}{kids_html}"
        f"<div class='card'><h2>Frozen state &mdash; {len(t)} steps</h2>"
        f"<table><thead><tr><th>#</th><th>Kind</th><th>Step</th><th>Inputs</th>"
        f"<th>Output</th></tr></thead><tbody>{''.join(step_rows)}</tbody></table></div>"
    )
    return _page(f"{session_id} — Multiverse Console", body)


def _short(value: object, limit: int = 300) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


class _Handler(BaseHTTPRequestHandler):
    store: CheckpointStore

    def do_GET(self) -> None:  # noqa: N802 (stdlib signature)
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("", "/"):
            self._send(render_index(self.store))
        elif path.startswith("/session/"):
            sid = unquote(path[len("/session/") :])
            self._send(render_session(self.store, sid))
        else:
            self._send("<h1>404</h1>", status=404)

    def _send(self, html_text: str, status: int = 200) -> None:
        body = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # silence default stderr logging
        pass


def make_server(db_path: str, host: str = "127.0.0.1", port: int = 8000) -> HTTPServer:
    """Build (but do not start) the console HTTP server bound to ``db_path``."""
    store = CheckpointStore(db_path, check_same_thread=False)
    handler = type("_BoundHandler", (_Handler,), {"store": store})
    return HTTPServer((host, port), handler)


def serve(db_path: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the Multiverse Console until interrupted."""
    httpd = make_server(db_path, host, port)
    print(f"Multiverse Console on http://{host}:{port}  (store: {db_path})  Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
