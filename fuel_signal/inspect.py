"""Local Flask analysis workbench for fuel price data.

Start with:
    uv run python -m fuel_signal.inspect

Then open http://localhost:5000 in a browser.
"""

from __future__ import annotations

import datetime
import logging
import pathlib
import sqlite3
import webbrowser

import click
import numpy as np
from flask import Flask, jsonify, render_template, request

from fuel_signal import db as _db
from fuel_signal import series as _series
from fuel_signal.config import PREFERRED_STATIONS
from fuel_signal.cycle import CycleDetector

logger = logging.getLogger(__name__)

_LINE_CAP = 10  # max series on line chart before overflow banner

_SYDNEY_COLOUR = "#9ca3af"  # mid-grey: visible on both light and dark backgrounds

_COLOURS = [
    "#f87171",  # red-400
    "#4ade80",  # green-400
    "#fbbf24",  # amber-400
    "#a78bfa",  # violet-400
    "#22d3ee",  # cyan-400
    "#f472b6",  # pink-400
    "#fb923c",  # orange-400
    "#818cf8",  # indigo-400
    "#2dd4bf",  # teal-400
    "#facc15",  # yellow-400
]

_BRAND_COLOURS: dict[str, str] = {
    "Ampol": "#dc2626",
    "Caltex": "#dc2626",
    "Shell": "#f59e0b",
    "BP": "#16a34a",
    "United": "#7c3aed",
    "Puma": "#0891b2",
    "Coles Express": "#db2777",
    "Viva Energy": "#db2777",
    "Mobil": "#ea580c",
    "7-Eleven": "#6366f1",
    "Metro": "#0d9488",
}


# ---------------------------------------------------------------------------
# Colour helpers (also registered as Jinja2 filters)
# ---------------------------------------------------------------------------

def _gradient_color(slope: float | None, clip: float = 3.0) -> str:
    """Map a slope (cents/day) to a CSS rgb() colour: blue→white→red."""
    if slope is None:
        return "rgb(240,240,240)"
    t = max(0.0, min(1.0, (slope + clip) / (2 * clip)))
    if t < 0.5:
        intensity = int(255 * (2 * t))
        return f"rgb({intensity},{intensity},255)"
    intensity = int(255 * (2 * (1 - t)))
    return f"rgb(255,{intensity},{intensity})"


def _coverage_color(n: int, max_n: int = 30) -> str:
    """Map observation count to CSS rgb() colour: white→green."""
    t = max(0.0, min(1.0, n / max_n))
    r = int(255 * (1 - t * 0.5))
    g = int(255 * (0.5 + t * 0.5))
    b = int(255 * (1 - t * 0.5))
    return f"rgb({r},{g},{b})"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _cutoff_date(window: str) -> str | None:
    today = datetime.date.today()
    offsets = {"6m": 182, "1y": 365, "2y": 730, "4y": 1461}
    if window in offsets:
        return (today - datetime.timedelta(days=offsets[window])).isoformat()
    return None  # "all"


def _slice_points(
    points: list[tuple[str, float]],
    cutoff: str | None,
) -> list[tuple[str, float]]:
    if cutoff is None:
        return points
    return [(d, p) for d, p in points if d >= cutoff]


# Reused from old inspect.py — unchanged logic
def _data_boundaries(conn: sqlite3.Connection) -> dict:
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
        he = datetime.date.fromisoformat(hist_end)
        ss = datetime.date.fromisoformat(snap_start)
        if ss > he + datetime.timedelta(days=1):
            gap_start = (he + datetime.timedelta(days=1)).isoformat()
            gap_end = (ss - datetime.timedelta(days=1)).isoformat()
    return {"hist_end": hist_end, "snap_start": snap_start,
            "gap_start": gap_start, "gap_end": gap_end}


def _build_annotations(peak_data: dict, labels: list[str],
                        boundaries: dict | None = None) -> dict:
    """Build chartjs-plugin-annotation config (ported from old inspect.py)."""
    label_set = set(labels)
    out: dict = {}

    if boundaries:
        gs, ge = boundaries.get("gap_start"), boundaries.get("gap_end")
        he = boundaries.get("hist_end")
        ss = boundaries.get("snap_start")
        if gs and ge:
            gs_eff = gs if gs in label_set else he
            ge_eff = ge if ge in label_set else ss
            if gs_eff and ge_eff and gs_eff in label_set and ge_eff in label_set:
                out["gap_zone"] = {
                    "type": "box", "xMin": gs_eff, "xMax": ge_eff,
                    "backgroundColor": "rgba(150,150,150,0.13)",
                    "borderColor": "rgba(150,150,150,0.35)", "borderWidth": 1,
                    "label": {"display": True, "content": "gap (forward-fill)",
                              "position": {"x": "center", "y": "center"},
                              "color": "rgba(100,100,100,0.6)", "font": {"size": 10}},
                }
        if he and he in label_set:
            out["csv_seam"] = {
                "type": "line", "scaleID": "x", "value": he,
                "borderColor": "rgba(100,100,100,0.5)", "borderWidth": 1,
                "borderDash": [2, 4],
                "label": {"display": True, "content": f"CSV end {he}",
                          "position": "end", "color": "rgba(80,80,80,0.7)",
                          "font": {"size": 9}},
            }

    for i, date in enumerate(peak_data["peak_dates"]):
        if date not in label_set:
            continue
        out[f"pk{i}"] = {
            "type": "line", "scaleID": "x", "value": date,
            "borderColor": "rgba(220,38,38,0.55)", "borderWidth": 1.5,
            "borderDash": [5, 3],
        }

    plateau_date = peak_data.get("plateau_peak_date")
    if plateau_date and plateau_date in label_set:
        out["plateau"] = {
            "type": "line", "scaleID": "x", "value": plateau_date,
            "borderColor": "#7c3aed", "borderWidth": 3,
            "label": {"display": True, "content": "△ boundary", "position": "start",
                      "color": "#7c3aed", "font": {"size": 10}},
        }

    s, e = peak_data.get("last_cycle_start"), peak_data.get("last_cycle_end")
    if s and e and s in label_set and e in label_set:
        out["last_cycle"] = {
            "type": "box", "xMin": s, "xMax": e,
            "backgroundColor": "rgba(220,38,38,0.07)", "borderWidth": 0,
            "label": {"display": True, "content": "last cycle",
                      "position": {"x": "center", "y": "start"},
                      "color": "rgba(180,20,20,0.55)", "font": {"size": 10}},
        }

    return out


# ---------------------------------------------------------------------------
# Chart spec builders
# ---------------------------------------------------------------------------

def _build_line_spec(
    resolved: list[_series.ResolvedSeries],
    peak_data: dict,
    boundaries: dict,
    has_sydney: bool,
) -> dict:
    cap = _LINE_CAP

    def _sort_key(r: _series.ResolvedSeries) -> tuple[int, str]:
        # Aggregate (sydney) → LGA → Brand → Favourites → other individual stations
        if r.kind == "sydney":
            group = 0
        elif r.kind == "lga":
            group = 1
        elif r.kind == "brand":
            group = 2
        elif r.kind == "station" and r.spec.startswith("station:") \
                and int(r.spec.split(":", 1)[1]) in PREFERRED_STATIONS:
            group = 3
        else:
            group = 4
        return (group, r.label)

    ordered = sorted(resolved, key=_sort_key)
    overflow = max(0, len(ordered) - cap)
    displayed = ordered[:cap]

    all_dates = sorted({d for r in displayed for d, _ in r.points})
    if not all_dates:
        return {}

    datasets = []
    colour_idx = -1
    for r in displayed:
        d_map = dict(r.points)
        if r.kind == "sydney":
            colour = _SYDNEY_COLOUR
        else:
            colour_idx += 1
            colour = _COLOURS[colour_idx % len(_COLOURS)]

        ds: dict = {
            "label": r.label,
            "data": [d_map.get(d) for d in all_dates],
            "borderColor": colour,
            "borderWidth": 1.5,
            "pointRadius": 0,
            "tension": 0.3,
            "spanGaps": True,
        }
        # Dash patterns chosen to differ from cycle/event vertical annotations:
        #   scipy peak  = red [5,3] thin vertical
        #   plateau     = purple solid thick vertical
        if r.kind == "sydney":
            ds["borderDash"] = [12, 3, 2, 3]  # dash-dot
            ds["fill"] = True
            ds["backgroundColor"] = "rgba(150,150,150,0.10)"
        elif r.kind == "lga":
            ds["borderDash"] = [10, 4]  # long dash
        elif r.kind == "brand":
            ds["borderDash"] = [1, 3]   # tight dots
        # individual stations (favourite or not) stay solid
        datasets.append(ds)

    annotations = _build_annotations(peak_data, all_dates, boundaries) if has_sydney else {}

    n_peaks = len(peak_data["peak_dates"])
    plateau_note = (
        f" + boundary plateau on {peak_data['plateau_peak_date']}"
        if peak_data["plateau_peak_date"] else ""
    )
    last_cycle_note = (
        f"Last cycle: {peak_data['last_cycle_start']} → {peak_data['last_cycle_end']}"
        if peak_data["last_cycle_start"] else ""
    )
    peak_summary = f"{n_peaks} scipy peaks{plateau_note}"
    if last_cycle_note:
        peak_summary += f" &mdash; {last_cycle_note}"

    return {
        "labels": all_dates,
        "datasets": datasets,
        "annotations": annotations,
        "overflow": overflow,
        "peak_summary": peak_summary,
    }


def _build_scatter_spec(
    conn: sqlite3.Connection,
    resolved: list[_series.ResolvedSeries],
    metric: str,
) -> dict:
    station_series = [r for r in resolved if r.kind == "station"]
    if not station_series:
        return {}

    code_to_brand: dict[int, str] = {}
    for r in station_series:
        code = int(r.spec.split(":")[1])
        row = conn.execute(
            "SELECT brand FROM stations WHERE station_code = ?", (code,)
        ).fetchone()
        code_to_brand[code] = (row[0] if row and row[0] else "Other")

    by_brand: dict[str, list[dict]] = {}
    for r in station_series:
        code = int(r.spec.split(":")[1])
        brand = code_to_brand[code]
        pts = r.points
        if metric == "gradient" and len(pts) >= 2:
            prices = np.array([p for _, p in pts])
            grads = np.gradient(prices)
            data = [{"x": pts[i][0], "y": round(float(grads[i]), 3)} for i in range(len(pts))]
        else:
            data = [{"x": d, "y": round(p, 1)} for d, p in pts]
        by_brand.setdefault(brand, []).extend(data)

    datasets = []
    for brand, data_points in sorted(by_brand.items()):
        colour = _BRAND_COLOURS.get(brand, "#9ca3af")
        datasets.append({
            "label": brand,
            "data": data_points,
            "backgroundColor": colour + "88",
            "borderColor": colour,
            "pointRadius": 3,
        })

    y_title = "gradient (cents/day)" if metric == "gradient" else "cents/litre"
    return {"datasets": datasets, "y_title": y_title}


def _build_gradient_heatmap(
    conn: sqlite3.Connection,
    cutoff: str | None,
    window_days: int = 1,
    councils: list[str] | None = None,
) -> dict:
    raw = _db.gradient_by_lga(conn, window_days=window_days, councils=councils)
    if cutoff:
        raw = [(c, d, s) for c, d, s in raw if d >= cutoff]
    if not raw:
        return {}

    all_dates = sorted({d for _, d, _ in raw})
    council_list = sorted({c for c, _, _ in raw})
    pivot: dict[str, dict[str, float]] = {}
    for council, date, slope in raw:
        pivot.setdefault(council, {})[date] = slope

    rows = [(c, [pivot[c].get(d) for d in all_dates]) for c in council_list]
    return {"dates": all_dates, "rows": rows}


def _build_coverage_heatmap(
    conn: sqlite3.Connection,
    cutoff: str | None,
) -> dict:
    cutoff_ym = cutoff[:7] if cutoff else None
    months_param = 24
    if cutoff_ym:
        # derive months count from cutoff
        today = datetime.date.today()
        cy, cm = int(cutoff_ym[:4]), int(cutoff_ym[5:7])
        ty, tm = today.year, today.month
        months_param = (ty - cy) * 12 + (tm - cm) + 1

    raw = _db.coverage_matrix(conn, months=months_param)
    if cutoff_ym:
        raw = [(code, name, ym, n) for code, name, ym, n in raw if ym >= cutoff_ym]
    if not raw:
        return {}

    all_months = sorted({ym for _, _, ym, _ in raw})
    stations: dict[int, str] = {}
    for code, name, _, _ in raw:
        stations[code] = name
    pivot: dict[int, dict[str, int]] = {}
    for code, _, ym, n in raw:
        pivot.setdefault(code, {})[ym] = n

    rows = [(stations[code], [pivot[code].get(m) for m in all_months])
            for code in sorted(stations)]
    return {"months": all_months, "rows": rows}


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

def _create_app(
    conn: sqlite3.Connection,
    cd: CycleDetector,
    today: str | None,
    cycle_state,
    peak_data: dict,
    summary: dict,
    boundaries: dict,
) -> Flask:
    app = Flask(__name__, template_folder="inspect_templates")

    app.jinja_env.filters["gradient_color"] = _gradient_color
    app.jinja_env.filters["coverage_color"] = _coverage_color

    @app.template_filter("week_label")
    def _week_label_filter(idx: int, weeks: list[str]) -> str:
        return weeks[idx] if idx < len(weeks) else ""

    # Jinja2 doesn't let you call str methods as filters with arguments, so
    # provide a helper for the 'lga:' / 'brand:' prefix check in the template.
    @app.context_processor
    def _inject_helpers():
        def spec_starts_with(iterable, prefix):
            return [s for s in iterable if s.startswith(prefix)]
        return {"spec_starts_with": spec_starts_with}

    @app.route("/")
    def index():
        specs = request.args.getlist("series")
        chart_type = request.args.get("chart", "line")
        window = request.args.get("window", "6m")
        display = request.args.get("display", "mean")
        metric = request.args.get("metric", "price")

        # Default landing: Sydney avg + preferred stations
        if not specs:
            specs = ["sydney"] + [f"station:{code}" for code in PREFERRED_STATIONS]

        cutoff = _cutoff_date(window)
        is_heatmap = chart_type.startswith("heatmap-")

        groups = _series.enumerate_groups(conn)
        preferred_spec_set = {f"station:{code}" for code in PREFERRED_STATIONS} | {"sydney"}
        known_specs = (
            preferred_spec_set
            | {f"lga:{c}" for c in groups["lgas"]}
            | {f"brand:{b}" for b in groups["brands"]}
        )

        # Resolve series, expanding group display if requested. Heatmap views
        # don't use resolved.points — skip the resolver to avoid forcing
        # full-history aggregations for every selected brand/LGA on first load.
        resolved: list[_series.ResolvedSeries] = []
        errors: list[str] = []
        if is_heatmap:
            extra_station_specs = [
                _series.ResolvedSeries(spec=s, label=s, points=[], kind="station")
                for s in specs
                if s.startswith("station:") and s not in known_specs
            ]
        else:
            for spec in specs:
                sl = spec.lower()
                is_group = sl.startswith("lga:") or sl.startswith("council:") or sl.startswith("brand:")
                if is_group and display in ("members", "both") and chart_type in ("line", "scatter"):
                    if display == "both":
                        try:
                            r = _series.resolve(conn, spec)
                            r.points = _slice_points(r.points, cutoff)
                            resolved.append(r)
                        except _series.SeriesError:
                            pass
                    members = _series.resolve_members(conn, spec)
                    for m in members:
                        m.points = _slice_points(m.points, cutoff)
                    resolved.extend(members)
                else:
                    try:
                        r = _series.resolve(conn, spec)
                        r.points = _slice_points(r.points, cutoff)
                        resolved.append(r)
                    except _series.SeriesError as e:
                        errors.append(str(e))

            extra_station_specs = [
                r for r in resolved
                if r.spec not in known_specs and r.kind == "station"
            ]

        # Build chart spec
        chart_spec = None
        heatmap_data = None
        has_sydney = any(r.kind == "sydney" for r in resolved)

        if chart_type == "line":
            chart_spec = _build_line_spec(resolved, peak_data, boundaries, has_sydney) or None
        elif chart_type == "scatter":
            chart_spec = _build_scatter_spec(conn, resolved, metric) or None
        elif chart_type == "heatmap-gradient":
            selected_councils = [s[4:] for s in specs if s.startswith("lga:")]
            if selected_councils:
                heatmap_data = _build_gradient_heatmap(
                    conn, cutoff, councils=selected_councils
                ) or None
            else:
                heatmap_data = None
        elif chart_type == "heatmap-coverage":
            heatmap_data = _build_coverage_heatmap(conn, cutoff) or None

        return render_template(
            "workbench.html",
            now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            summary=summary,
            cycle_state=cycle_state,
            peak_data=peak_data,
            today=today,
            specs=specs,
            resolved=resolved,
            extra_station_specs=extra_station_specs,
            chart_type=chart_type,
            chart_spec=chart_spec,
            heatmap_data=heatmap_data,
            window=window,
            display=display,
            metric=metric,
            groups=groups,
            preferred_stations=PREFERRED_STATIONS,
            errors=errors,
        )

    @app.route("/api/stations/search")
    def stations_search():
        q = request.args.get("q", "").strip()
        if len(q) < 2:
            return jsonify([])
        results = _db.station_search(conn, q)
        return jsonify([
            {"code": code, "name": name, "suburb": suburb, "brand": brand}
            for code, name, suburb, brand in results
        ])

    @app.route("/healthz")
    def healthz():
        return "ok", 200

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command("inspect")
@click.option(
    "--db",
    "db_path",
    default=str(_db.DEFAULT_DB_PATH),
    show_default=True,
    help="Path to SQLite database.",
)
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host.")
@click.option("--port", default=5000, show_default=True, help="Bind port.")
@click.option("--debug", is_flag=True, help="Enable Flask debug mode.")
@click.option(
    "--no-browser",
    is_flag=True,
    default=False,
    help="Do not open the browser automatically.",
)
def main(db_path: str, host: str, port: int, debug: bool, no_browser: bool) -> None:
    """Start the local fuel-price analysis workbench."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = pathlib.Path(db_path)
    if not path.exists():
        raise click.ClickException(
            f"Database not found: {db_path}. "
            "Run 'uv run python -m fuel_signal.db' first."
        )

    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    full_series = _db.average_price_series(conn)
    if not full_series:
        conn.close()
        raise click.ClickException(
            "No gap-filled data found. "
            "Run 'uv run python -m fuel_signal.fill' then 'uv run python -m fuel_signal.db' first."
        )

    logger.info("Loading CycleDetector over %d daily points…", len(full_series))
    cd = CycleDetector(full_series)

    today = cd._series.index[-1].strftime("%Y-%m-%d") if not cd._series.empty else None
    cycle_state = cd.detect(today) if today else None
    peak_data = cd.peaks_for_plot()
    summary = _db.db_summary(conn)
    boundaries = _data_boundaries(conn)

    app = _create_app(conn, cd, today, cycle_state, peak_data, summary, boundaries)

    url = f"http://{host}:{port}/"
    if not no_browser:
        import threading
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    logger.info("Workbench ready at %s", url)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
