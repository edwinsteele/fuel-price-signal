"""Generate a self-contained HTML inspection page from the local SQLite DB."""

import datetime
import json
import logging
import pathlib
import sqlite3
import webbrowser

import click

from fuel_signal import db as _db
from fuel_signal.config import PREFERRED_STATIONS
from fuel_signal.cycle import CycleDetector

logger = logging.getLogger(__name__)

_CSS = """\
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #222; }
h2 { margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }
.stats { display: flex; gap: 2rem; flex-wrap: wrap; margin: 1rem 0; }
.stat { background: #f5f5f5; border-radius: 6px; padding: .8rem 1.4rem; }
.stat-value { font-size: 1.6rem; font-weight: 700; }
.stat-label { font-size: .85rem; color: #666; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: left; padding: .4rem .7rem; border: 1px solid #ddd; }
th { background: #f0f0f0; }
tr:nth-child(even) { background: #fafafa; }
.chart-wrap { height: 460px; margin: 1rem 0; }
.chart-wrap-tall { height: 360px; margin: 1rem 0; }
.peak-legend { font-size: .82rem; color: #555; margin: -.4rem 0 .8rem; }
.peak-legend span { display: inline-block; margin-right: 1.2rem; }
"""

_CHART_JS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4"
_ANNOTATION_CDN = "https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3"

_PREFERRED_COLOURS = [
    "#dc2626",  # red
    "#16a34a",  # green
    "#d97706",  # amber
    "#7c3aed",  # violet
    "#0891b2",  # cyan
    "#db2777",  # pink
    "#ea580c",  # orange
]


def _coverage_by_month(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    return _db.coverage_by_month(conn)


def _recent_prices(conn: sqlite3.Connection, days: int = 14) -> list[tuple]:
    return _db.recent_prices(conn, days=days)


def _preferred_series(conn: sqlite3.Connection, days: int = 365) -> dict[str, list]:
    """Return {station_label: [(date, price), ...]} for preferred stations."""
    cutoff = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    out: dict[str, list] = {}
    for code, label in PREFERRED_STATIONS.items():
        rows = _db.station_price_series(conn, code, start_date=cutoff)
        if rows:
            out[label] = rows
    return out


def _data_boundaries(conn: sqlite3.Connection) -> dict:
    """Return source-boundary dates for the prices table.

    Returns:
        hist_end:    last date of historical-CSV data ('h' source), or None
        snap_start:  first date of snapshot data ('s' source), or None
        gap_start:   day after hist_end (first forward-filled day), or None
        gap_end:     day before snap_start (last forward-filled day), or None

    When hist_end + 1 == snap_start there is no gap (continuous coverage).
    """
    row = conn.execute(
        """SELECT
               MAX(CASE WHEN ps.code='h' THEN p.price_date END),
               MIN(CASE WHEN ps.code='s' THEN p.price_date END)
           FROM prices p JOIN price_sources ps ON p.source_id = ps.id"""
    ).fetchone()
    hist_end_int, snap_start_int = row

    hist_end = _db._date_from_int(hist_end_int) if hist_end_int else None
    snap_start = _db._date_from_int(snap_start_int) if snap_start_int else None

    gap_start = gap_end = None
    if hist_end and snap_start:
        import datetime as _dt
        he = _dt.date.fromisoformat(hist_end)
        ss = _dt.date.fromisoformat(snap_start)
        if ss > he + _dt.timedelta(days=1):
            gap_start = (he + _dt.timedelta(days=1)).isoformat()
            gap_end = (ss - _dt.timedelta(days=1)).isoformat()

    return {
        "hist_end": hist_end,
        "snap_start": snap_start,
        "gap_start": gap_start,
        "gap_end": gap_end,
    }


def _build_annotations(peak_data: dict, labels: list[str],
                       boundaries: dict | None = None) -> dict:
    """Build chartjs-plugin-annotation config for peak overlays + data-gap shading.

    Only annotates dates that exist as labels in the chart (safe for zoomed views).
    scipy-confirmed peaks → red dashed vertical line.
    Boundary plateau peak → orange dashed vertical line with label.
    Last-cycle window    → light-red box.
    Data gap zone        → grey shaded box + seam line at CSV/snapshot boundary.
    """
    label_set = set(labels)
    out: dict = {}

    # Data-gap shading — must appear before peaks so it renders behind them
    if boundaries:
        gs, ge = boundaries.get("gap_start"), boundaries.get("gap_end")
        he = boundaries.get("hist_end")
        ss = boundaries.get("snap_start")
        # Shade the forward-filled gap (if any)
        if gs and ge:
            # Use closest available labels when exact gap dates aren't in the series
            # (gap days have no rows in daily_prices if fill didn't run over them)
            gs_eff = gs if gs in label_set else he
            ge_eff = ge if ge in label_set else ss
            if gs_eff and ge_eff and gs_eff in label_set and ge_eff in label_set:
                out["gap_zone"] = {
                    "type": "box",
                    "xMin": gs_eff,
                    "xMax": ge_eff,
                    "backgroundColor": "rgba(150,150,150,0.13)",
                    "borderColor": "rgba(150,150,150,0.35)",
                    "borderWidth": 1,
                    "label": {
                        "display": True,
                        "content": "gap (forward-fill)",
                        "position": {"x": "center", "y": "center"},
                        "color": "rgba(100,100,100,0.6)",
                        "font": {"size": 10},
                    },
                }
        # Seam line at end of historical CSV data
        if he and he in label_set:
            out["csv_seam"] = {
                "type": "line",
                "scaleID": "x",
                "value": he,
                "borderColor": "rgba(100,100,100,0.5)",
                "borderWidth": 1,
                "borderDash": [2, 4],
                "label": {
                    "display": True,
                    "content": f"CSV end {he}",
                    "position": "end",
                    "color": "rgba(80,80,80,0.7)",
                    "font": {"size": 9},
                },
            }

    for i, date in enumerate(peak_data["peak_dates"]):
        if date not in label_set:
            continue
        out[f"pk{i}"] = {
            "type": "line",
            "scaleID": "x",
            "value": date,
            "borderColor": "rgba(220,38,38,0.55)",
            "borderWidth": 1.5,
            "borderDash": [5, 3],
        }

    plateau_date = peak_data.get("plateau_peak_date")
    if plateau_date and plateau_date in label_set:
        out["plateau"] = {
            "type": "line",
            "scaleID": "x",
            "value": plateau_date,
            "borderColor": "#7c3aed",
            "borderWidth": 3,
            "label": {
                "display": True,
                "content": "△ boundary",
                "position": "start",
                "color": "#7c3aed",
                "font": {"size": 10},
            },
        }

    s, e = peak_data.get("last_cycle_start"), peak_data.get("last_cycle_end")
    if s and e and s in label_set and e in label_set:
        out["last_cycle"] = {
            "type": "box",
            "xMin": s,
            "xMax": e,
            "backgroundColor": "rgba(220,38,38,0.07)",
            "borderWidth": 0,
            "label": {
                "display": True,
                "content": "last cycle",
                "position": {"x": "center", "y": "start"},
                "color": "rgba(180,20,20,0.55)",
                "font": {"size": 10},
            },
        }

    return out


def _chart_script(canvas_id: str, datasets: list[dict], labels: list[str],
                  annotations: dict, y_title: str = "cents/litre",
                  max_ticks: int = 16) -> str:
    """Return a <script> block that creates a Chart.js chart."""
    chart_data = json.dumps({"labels": labels, "datasets": datasets})
    ann_json = json.dumps(annotations)
    return f"""
<script>
new Chart(document.getElementById('{canvas_id}'), {{
  type: 'line',
  data: {chart_data},
  options: {{
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'top' }},
      annotation: {{ annotations: {ann_json} }}
    }},
    scales: {{
      x: {{ ticks: {{ maxTicksLimit: {max_ticks}, maxRotation: 45, minRotation: 45 }} }},
      y: {{ title: {{ display: true, text: '{y_title}' }} }}
    }}
  }}
}});
</script>"""


def _cycle_state_html(state, peak_data: dict) -> str:
    """Render a compact cycle-state summary box."""
    if state is None:
        return "<p style='color:#888'>Cycle state: insufficient data (fewer than 2 peaks).</p>"

    plateau_marker = ""
    if peak_data["plateau_peak_date"]:
        plateau_marker = (
            f"<span style='color:#7c3aed; font-weight:700'>▲ boundary plateau detected "
            f"({peak_data['plateau_peak_date']})</span> &mdash; "
        )
    else:
        plateau_marker = "<span style='color:#888'>no boundary plateau</span> &mdash; "

    return f"""\
<div class="stats">
  <div class="stat">
    <div class="stat-value">{state.pct_through_cycle:.0%}</div>
    <div class="stat-label">through cycle</div>
  </div>
  <div class="stat">
    <div class="stat-value">{state.days_since_last_peak}d</div>
    <div class="stat-label">since last peak</div>
  </div>
  <div class="stat">
    <div class="stat-value">{state.mean_cycle_length:.0f}d</div>
    <div class="stat-label">mean cycle length</div>
  </div>
  <div class="stat">
    <div class="stat-value">{state.peak_count}</div>
    <div class="stat-label">peaks detected</div>
  </div>
  <div class="stat">
    <div class="stat-value">{state.last_cycle_min:.1f}–{state.last_cycle_max:.1f}c</div>
    <div class="stat-label">last cycle min–max</div>
  </div>
</div>
<p style="font-size:.85rem; margin:.4rem 0 0">
  {plateau_marker}
  last 3 gradients: {state.last_3_gradients}
</p>"""


def generate_html(conn: sqlite3.Connection) -> str:
    summary = _db.db_summary(conn)
    coverage = _coverage_by_month(conn)
    recent = _recent_prices(conn)

    # Full gap-filled series — what CycleDetector operates on
    full_series = _db.average_price_series(conn)
    if not full_series:
        # Fall back to raw daily average if fill hasn't been run yet
        full_series = _db.daily_average_e10(conn)

    # Source-boundary metadata (for gap shading)
    boundaries = _data_boundaries(conn)

    # CycleDetector + peaks (over the entire history)
    cd = CycleDetector(full_series)
    peak_data = cd.peaks_for_plot()

    full_labels = [r[0] for r in full_series]
    full_values = [round(r[1], 1) for r in full_series]

    # ---- Chart 1: full history ----
    full_datasets = [{
        "label": "Sydney avg E10",
        "data": full_values,
        "borderColor": "#000000",
        "backgroundColor": "rgba(0,0,0,0.06)",
        "borderDash": [6, 4],
        "pointRadius": 0,
        "tension": 0.3,
        "fill": True,
    }]
    full_annotations = _build_annotations(peak_data, full_labels, boundaries)
    full_chart = _chart_script("fullChart", full_datasets, full_labels,
                               full_annotations, max_ticks=20)

    # ---- Chart 2: last 6 months zoomed (with preferred stations) ----
    cutoff_6m = (datetime.date.today() - datetime.timedelta(days=180)).isoformat()
    series_6m = [(d, v) for d, v in full_series if d >= cutoff_6m]
    labels_6m = [r[0] for r in series_6m]
    values_6m = [round(r[1], 1) for r in series_6m]

    preferred = _preferred_series(conn, days=180)
    datasets_6m: list[dict] = [{
        "label": "Sydney avg E10",
        "data": values_6m,
        "borderColor": "#000000",
        "backgroundColor": "rgba(0,0,0,0.06)",
        "borderDash": [6, 4],
        "pointRadius": 0,
        "tension": 0.3,
        "fill": True,
    }]
    for i, (label, series) in enumerate(preferred.items()):
        series_dict = dict(series)
        colour = _PREFERRED_COLOURS[i % len(_PREFERRED_COLOURS)]
        datasets_6m.append({
            "label": label,
            "data": [series_dict.get(d) for d in labels_6m],
            "borderColor": colour,
            "pointRadius": 0,
            "tension": 0.3,
            "spanGaps": True,
        })

    annotations_6m = _build_annotations(peak_data, labels_6m, boundaries)
    zoom_chart = _chart_script("zoomChart", datasets_6m, labels_6m,
                               annotations_6m, max_ticks=12)

    # ---- Current cycle state (as of latest data point) ----
    today = full_series[-1][0] if full_series else "—"
    cycle_state = cd.detect(today)

    # ---- Peak summary + data-gap note ----
    n_peaks = len(peak_data["peak_dates"])
    plateau_note = (
        f" + boundary plateau on {peak_data['plateau_peak_date']}"
        if peak_data["plateau_peak_date"] else ""
    )
    last_cycle_note = (
        f"Last cycle window: {peak_data['last_cycle_start']} → {peak_data['last_cycle_end']}"
        if peak_data["last_cycle_start"] else "Last cycle window: insufficient data"
    )
    gap_note = ""
    if boundaries["gap_start"] and boundaries["gap_end"]:
        gap_note = (
            f" &mdash; <span style='color:#888'>data gap {boundaries['gap_start']} → "
            f"{boundaries['gap_end']} (forward-filled, shaded grey)</span>"
        )
    elif boundaries["hist_end"] and boundaries["snap_start"]:
        gap_note = (
            f" &mdash; <span style='color:#888'>CSV ends {boundaries['hist_end']}, "
            f"snapshots from {boundaries['snap_start']}</span>"
        )

    # ---- Coverage table ----
    cov_rows = "".join(
        f"<tr><td>{ym}</td><td>{n}</td></tr>" for ym, n in coverage
    )

    # ---- Recent prices table ----
    price_rows = "".join(
        f"<tr><td>{date}</td><td>{name}</td><td>{suburb}</td><td>{price:.1f}c</td></tr>"
        for date, name, suburb, price in recent
    )
    if not price_rows:
        price_rows = "<tr><td colspan='4'>No data in last 14 days</td></tr>"

    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Fuel Price Signal — Inspection</title>
<script src="{_CHART_JS_CDN}"></script>
<script src="{_ANNOTATION_CDN}"></script>
<style>{_CSS}</style>
</head>
<body>
<h1>Fuel Price Signal — Data Inspection</h1>
<p style="color:#888">Generated {generated_at}</p>

<h2>Summary</h2>
<div class="stats">
  <div class="stat">
    <div class="stat-value">{summary["station_count"]:,}</div>
    <div class="stat-label">Stations</div>
  </div>
  <div class="stat">
    <div class="stat-value">{summary["price_count"]:,}</div>
    <div class="stat-label">Price records</div>
  </div>
  <div class="stat">
    <div class="stat-value">{summary["earliest_date"]}</div>
    <div class="stat-label">Earliest E10 date</div>
  </div>
  <div class="stat">
    <div class="stat-value">{summary["latest_date"]}</div>
    <div class="stat-label">Latest E10 date</div>
  </div>
</div>

<h2>Cycle State — as of {today}</h2>
{_cycle_state_html(cycle_state, peak_data)}

<h2>Sydney E10 Average — Full History with Peak Detection</h2>
<p class="peak-legend">
  <span style="border-left:3px solid rgba(220,38,38,0.6); padding-left:5px">scipy peak (red dashed)</span>
  <span style="border-left:4px solid #7c3aed; padding-left:5px">boundary plateau (purple solid)</span>
  <span style="background:rgba(220,38,38,0.12); padding:1px 6px; border-radius:3px">last cycle band</span>
  <span style="background:rgba(150,150,150,0.18); padding:1px 6px; border-radius:3px;
    border:1px solid rgba(150,150,150,0.4)">data gap (forward-fill)</span>
</p>
<p class="peak-legend" style="margin-top:-.6rem">
  {n_peaks} scipy peaks detected{plateau_note} &mdash; {last_cycle_note}{gap_note}
</p>
<div class="chart-wrap">
  <canvas id="fullChart"></canvas>
</div>
{full_chart}

<h2>Last 6 Months — Zoomed (peak detection + preferred stations)</h2>
<div class="chart-wrap">
  <canvas id="zoomChart"></canvas>
</div>
{zoom_chart}

<h2>Data Coverage — E10 stations reporting per month (last 30 months)</h2>
<table>
  <tr><th>Month</th><th>Stations with data</th></tr>
  {cov_rows if cov_rows else "<tr><td colspan='2'>No data yet</td></tr>"}
</table>

<h2>Recent E10 Prices — last 14 days</h2>
<table>
  <tr><th>Date</th><th>Station</th><th>Suburb</th><th>E10 price</th></tr>
  {price_rows}
</table>

</body>
</html>
"""


@click.command("inspect")
@click.option(
    "--db",
    "db_path",
    default=str(_db.DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite database.",
)
@click.option(
    "--out",
    "out_path",
    default="inspect.html",
    show_default=True,
    help="Output HTML file path.",
)
@click.option("--no-open-browser", "open_browser", is_flag=True, default=True, flag_value=False,
              help="Do not open the file in a browser after writing.")
def main(db_path: str, out_path: str, open_browser: bool) -> None:
    """Generate an HTML inspection page from the local SQLite database."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = pathlib.Path(db_path)
    if not path.exists():
        raise click.ClickException(
            f"Database not found: {db_path}. "
            "Run 'uv run python -m fuel_signal.live' then 'uv run python -m fuel_signal.db' first."
        )

    conn = _db.open_db(path)
    html = generate_html(conn)
    conn.close()

    out = pathlib.Path(out_path)
    out.write_text(html, encoding="utf-8")
    click.echo(f"Inspection page written to {out}")

    if open_browser:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
