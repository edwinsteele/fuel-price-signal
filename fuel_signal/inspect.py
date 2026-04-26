"""Generate a self-contained HTML inspection page from the local SQLite DB."""

import datetime
import json
import logging
import pathlib
import sqlite3
import subprocess
import sys
import webbrowser

from fuel_signal import db as _db
from fuel_signal.config import PREFERRED_STATIONS

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
"""

_CHART_JS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4"

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
    rows = conn.execute(
        """SELECT strftime('%Y-%m', price_date) AS ym, COUNT(DISTINCT station_code)
           FROM prices WHERE fuel_code='E10'
           GROUP BY ym ORDER BY ym DESC LIMIT 30"""
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _recent_prices(conn: sqlite3.Connection, days: int = 14) -> list[tuple]:
    cutoff = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    return conn.execute(
        """SELECT p.price_date, s.name, s.suburb, p.price_cents
           FROM prices p JOIN stations s USING(station_code)
           WHERE p.fuel_code='E10' AND p.price_date >= ?
           ORDER BY p.price_date DESC, p.price_cents""",
        (cutoff,),
    ).fetchall()


def _preferred_series(conn: sqlite3.Connection, days: int = 365) -> dict[str, list]:
    """Return {station_name: [(date, price), ...]} for preferred stations."""
    cutoff = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    out: dict[str, list] = {}
    for code, label in PREFERRED_STATIONS.items():
        rows = _db.station_price_series(conn, code, start_date=cutoff)
        if rows:
            out[label] = rows
    return out


def generate_html(conn: sqlite3.Connection) -> str:
    summary = _db.db_summary(conn)
    avg_series = _db.daily_average_e10(
        conn,
        start_date=(datetime.date.today() - datetime.timedelta(days=365)).isoformat(),
    )
    coverage = _coverage_by_month(conn)
    recent = _recent_prices(conn)
    preferred = _preferred_series(conn)

    # --- Chart data ---
    avg_labels = [r[0] for r in avg_series]
    avg_values = [round(r[1], 1) for r in avg_series]

    datasets = [{
        "label": "Sydney avg E10",
        "data": avg_values,
        "borderColor": "#2563eb",
        "backgroundColor": "rgba(37,99,235,0.08)",
        "borderDash": [6, 4],
        "pointRadius": 0,
        "tension": 0.3,
        "fill": True,
    }]
    for i, (label, series) in enumerate(preferred.items()):
        series_dict = dict(series)
        colour = _PREFERRED_COLOURS[i % len(_PREFERRED_COLOURS)]
        datasets.append({
            "label": label,
            "data": [series_dict.get(d) for d in avg_labels],
            "borderColor": colour,
            "pointRadius": 0,
            "tension": 0.3,
            "spanGaps": True,
        })

    chart_data = json.dumps({"labels": avg_labels, "datasets": datasets})

    # --- Coverage table rows ---
    cov_rows = "".join(
        f"<tr><td>{ym}</td><td>{n}</td></tr>" for ym, n in coverage
    )

    # --- Recent prices table rows ---
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

<h2>Sydney E10 Daily Average — last 365 days</h2>
<div class="chart-wrap">
  <canvas id="avgChart"></canvas>
</div>
<script>
new Chart(document.getElementById('avgChart'), {{
  type: 'line',
  data: {chart_data},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'top' }} }},
    scales: {{
      x: {{ ticks: {{ maxTicksLimit: 16, maxRotation: 45, minRotation: 45 }} }},
      y: {{ title: {{ display: true, text: 'cents/litre' }} }}
    }}
  }}
}});
</script>

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


def main(
    db_path: pathlib.Path = _db.DEFAULT_DB_PATH,
    out_path: pathlib.Path = pathlib.Path("inspect.html"),
    open_browser: bool = True,
) -> pathlib.Path:
    if not db_path.exists():
        print(
            f"Database not found at {db_path}.\n"
            "Run 'uv run python -m fuel_signal.live' first to populate stations,\n"
            "then 'uv run python -m fuel_signal.db' to load historical data.",
            file=sys.stderr,
        )
        sys.exit(1)

    conn = _db.open_db(db_path)
    html = generate_html(conn)
    conn.close()

    out_path.write_text(html, encoding="utf-8")
    print(f"Inspection page written to {out_path}")

    if open_browser:
        webbrowser.open(out_path.resolve().as_uri())

    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
