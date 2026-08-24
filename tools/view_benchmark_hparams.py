#!/usr/bin/env python
"""Offline HTML viewer for benchmark_hparams.py's results: the frontier-walk
chain (logs/benchmark_hparams_<timestamp>.json) plus each chain step's own
Optuna sweep (logs/optuna/<goal>.db) -- matching this project's existing
tools/view_*.py --out foo.html convention: one self-contained local HTML
file, a JSON payload rendered by plain inline JS/canvas, no server, no CDN
(see tools/view_event_compass.py).

Per chain step: a canvas scatter of every COMPLETE trial's steps-to-
saturation against its trial number, with a running-best line overlaid, a
PRUNED/FAIL count (not plotted -- they carry no comparable steps-to-
saturation value, only an in-flight success-rate reading at the point they
were cut), and a sortable table of every trial's full hyperparameters.

Usage:
  python tools/view_benchmark_hparams.py                     # newest logs/benchmark_hparams_*.json
  python tools/view_benchmark_hparams.py --report logs/benchmark_hparams_20260823_211259.json
  python tools/view_benchmark_hparams.py --out report.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _latest_report(logs_dir: Path) -> Path:
    candidates = sorted(logs_dir.glob("benchmark_hparams_*.json"))
    if not candidates:
        raise SystemExit(f"no benchmark_hparams_*.json report found under {logs_dir}")
    return candidates[-1]


def _load_study_trials(goal: str, storage_dir: Path) -> list[dict] | None:
    """None if this goal's study .db doesn't exist (e.g. --storage-dir
    doesn't match the run that produced the report), so the page can say so
    instead of silently rendering an empty chart."""
    db_path = storage_dir / f"{goal}.db"
    if not db_path.is_file():
        return None
    import optuna

    study = optuna.load_study(study_name=goal, storage=f"sqlite:///{db_path.as_posix()}")
    return [
        {
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            "params": t.params,
            "bonus_goals": t.user_attrs.get("bonus_goals"),
        }
        for t in study.trials
    ]


def build_payload(chain: list[dict], storage_dir: Path) -> dict:
    steps = []
    for step in chain:
        steps.append({**step, "trials": _load_study_trials(step["goal"], storage_dir)})
    return {"steps": steps}


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>benchmark_hparams report</title>
<style>
  :root { color-scheme: light; }
  body { font: 14px/1.4 -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #1c1f24; background: #f7f8fa; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  h2 { font-size: 16px; margin: 28px 0 6px; }
  .sub { color: #666; margin-bottom: 20px; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 8px; background: #fff; }
  th, td { border: 1px solid #e0e2e6; padding: 5px 8px; text-align: left; font-size: 12.5px; vertical-align: top; }
  th { background: #eef0f3; cursor: pointer; user-select: none; white-space: nowrap; }
  th:hover { background: #e2e5ea; }
  tr:nth-child(even) td { background: #fafbfc; }
  .params { font-family: ui-monospace, Consolas, monospace; font-size: 11.5px; color: #333; }
  .state-COMPLETE { color: #1a7a3c; font-weight: 600; }
  .state-PRUNED { color: #b8720c; font-weight: 600; }
  .state-FAIL { color: #b02a2a; font-weight: 600; }
  .step { border: 1px solid #e0e2e6; border-radius: 8px; padding: 14px 16px; margin-bottom: 18px; background: #fff; }
  .step h3 { margin: 0 0 2px; font-size: 15px; }
  .counts { color: #555; font-size: 12.5px; margin: 4px 0 10px; }
  .nodata { color: #999; font-style: italic; }
  canvas { border: 1px solid #e0e2e6; border-radius: 4px; background: #fff; }
</style>
</head>
<body>
<h1>benchmark_hparams report</h1>
<div class="sub" id="subtitle"></div>

<h2>Chain</h2>
<table id="chainTable">
  <thead><tr><th>#</th><th>frontier</th><th>&#8594; goal</th><th>steps-to-saturation</th><th>bonus goals</th><th>best params</th></tr></thead>
  <tbody></tbody>
</table>

<div id="steps"></div>

<script>
const DATA = __DATA_JSON__;

function fmtParams(p) {
  return Object.entries(p).map(([k, v]) => k + "=" + (typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(4)) : v)).join(", ");
}

function fillChainTable() {
  const tbody = document.querySelector("#chainTable tbody");
  DATA.steps.forEach((s, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i + 1}</td><td>${s.frontier_from}</td><td>${s.goal}</td>` +
      `<td>${Math.round(s.steps).toLocaleString()}</td><td>${s.bonus_goals}</td>` +
      `<td class="params">${fmtParams(s.params)}</td>`;
    tbody.appendChild(tr);
  });
}

function drawChart(canvas, trials) {
  const completed = trials.filter(t => t.state === "COMPLETE" && t.value !== null)
    .sort((a, b) => a.number - b.number);
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, PAD = 42;
  ctx.clearRect(0, 0, W, H);
  if (completed.length === 0) {
    ctx.fillStyle = "#999"; ctx.font = "12px sans-serif";
    ctx.fillText("no completed trials to plot", PAD, H / 2);
    return;
  }
  const xs = completed.map(t => t.number), ys = completed.map(t => t.value);
  const xMin = 0, xMax = Math.max(...xs, 1);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const yPad = (yMax - yMin) * 0.08 || Math.max(yMax * 0.08, 1);
  const yLo = Math.max(0, yMin - yPad), yHi = yMax + yPad;
  const sx = x => PAD + (x - xMin) / Math.max(xMax - xMin, 1) * (W - PAD - 10);
  const sy = y => H - PAD - (y - yLo) / Math.max(yHi - yLo, 1) * (H - PAD - 10);

  ctx.strokeStyle = "#ccc"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD, 10); ctx.lineTo(PAD, H - PAD); ctx.lineTo(W - 10, H - PAD); ctx.stroke();
  ctx.fillStyle = "#666"; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
  ctx.fillText(Math.round(yHi).toLocaleString(), PAD - 6, 16);
  ctx.fillText(Math.round(yLo).toLocaleString(), PAD - 6, H - PAD);
  ctx.textAlign = "center";
  ctx.fillText("trial #", (PAD + W - 10) / 2, H - 8);

  // running-best line
  let best = Infinity;
  ctx.strokeStyle = "#2f7d5f"; ctx.lineWidth = 1.5; ctx.beginPath();
  completed.forEach((t, i) => {
    best = Math.min(best, t.value);
    const x = sx(t.number), y = sy(best);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // scatter
  ctx.fillStyle = "rgba(47,125,95,0.55)";
  completed.forEach(t => {
    ctx.beginPath(); ctx.arc(sx(t.number), sy(t.value), 3, 0, Math.PI * 2); ctx.fill();
  });
}

function fillTrialsTable(container, trials) {
  const table = document.createElement("table");
  table.innerHTML = `<thead><tr>
      <th data-k="number">#</th><th data-k="state">state</th><th data-k="value">steps</th>
      <th data-k="bonus_goals">bonus</th><th>params</th></tr></thead><tbody></tbody>`;
  const tbody = table.querySelector("tbody");
  const rank = { COMPLETE: 0, PRUNED: 1, FAIL: 2, RUNNING: 3 };
  const render = rows => {
    tbody.innerHTML = "";
    rows.forEach(t => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${t.number}</td><td class="state-${t.state}">${t.state}</td>` +
        `<td>${t.value !== null ? Math.round(t.value).toLocaleString() : "-"}</td>` +
        `<td>${t.bonus_goals ?? "-"}</td><td class="params">${fmtParams(t.params)}</td>`;
      tbody.appendChild(tr);
    });
  };
  let rows = [...trials].sort((a, b) =>
    rank[a.state] - rank[b.state] || (a.value ?? Infinity) - (b.value ?? Infinity));
  render(rows);
  table.querySelectorAll("th[data-k]").forEach(th => {
    th.addEventListener("click", () => {
      const k = th.dataset.k;
      rows = [...rows].sort((a, b) => {
        const av = a[k], bv = b[k];
        if (av === null || av === undefined) return 1;
        if (bv === null || bv === undefined) return -1;
        return av > bv ? 1 : av < bv ? -1 : 0;
      });
      render(rows);
    });
  });
  container.appendChild(table);
}

function fillSteps() {
  const root = document.getElementById("steps");
  DATA.steps.forEach((s, i) => {
    const div = document.createElement("div");
    div.className = "step";
    const trials = s.trials;
    if (trials === null) {
      div.innerHTML = `<h3>${i + 1}. ${s.frontier_from} &#8594; ${s.goal}</h3>` +
        `<div class="nodata">no Optuna study found for this goal under --storage-dir</div>`;
      root.appendChild(div);
      return;
    }
    const counts = trials.reduce((a, t) => (a[t.state] = (a[t.state] || 0) + 1, a), {});
    const countsStr = Object.entries(counts).map(([k, v]) => `${v} ${k.toLowerCase()}`).join(", ");
    div.innerHTML = `<h3>${i + 1}. ${s.frontier_from} &#8594; ${s.goal}</h3>` +
      `<div class="counts">${trials.length} trial(s): ${countsStr}</div>`;
    const canvas = document.createElement("canvas");
    canvas.width = 700; canvas.height = 260;
    div.appendChild(canvas);
    root.appendChild(div);
    drawChart(canvas, trials);
    fillTrialsTable(div, trials);
  });
}

document.getElementById("subtitle").textContent =
  DATA.steps.length + " chain step(s), generated by tools/view_benchmark_hparams.py";
fillChainTable();
fillSteps();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--report", default=None,
        help="Path to a benchmark_hparams_*.json report (default: newest under logs/)",
    )
    parser.add_argument(
        "--storage-dir", default=str(REPO_ROOT / "logs" / "optuna"),
        help="Dir holding <goal>.db Optuna studies (default: logs/optuna)",
    )
    parser.add_argument("--out", default="benchmark_hparams_report.html")
    args = parser.parse_args()

    report_path = Path(args.report) if args.report else _latest_report(REPO_ROOT / "logs")
    with open(report_path, encoding="utf-8") as f:
        chain = json.load(f)
    if not chain:
        raise SystemExit(f"{report_path} is an empty chain -- nothing to render")

    payload = build_payload(chain, Path(args.storage_dir))
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload, separators=(",", ":")))
    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")

    n_trials_total = sum(len(s["trials"] or []) for s in payload["steps"])
    print(f"report:  {report_path}")
    print(f"chain:   {len(chain)} step(s), {n_trials_total} trial(s) total")
    print(f"wrote:   {out_path.resolve()}")


if __name__ == "__main__":
    main()
