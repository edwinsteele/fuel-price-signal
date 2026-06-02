"""Smoke test: derive days_since_my_lga_trough on a small slice and verify
exact equivalence with the broadcast column for each row's station-council."""

from __future__ import annotations

import pandas as pd

from experiments.l1_per_row_lga.run_cv import PER_ROW_COL, derive_per_row_column
from fuel_signal import db as _db
from fuel_signal.lga_leadership import lga_slug


def main() -> None:
    df = pd.read_csv("data/features.csv", nrows=20000)
    print(f"Loaded {len(df):,} sample rows")

    conn = _db.open_db(_db.DEFAULT_DB_PATH)
    council_by_station = dict(
        conn.execute("SELECT station_code, council FROM stations").fetchall()
    )
    df = derive_per_row_column(df, conn)
    conn.close()

    n_total = len(df)
    n_nan = int(df[PER_ROW_COL].isna().sum())
    print(f"per-row NaN: {n_nan:,}/{n_total:,} ({n_nan/n_total*100:.1f}%)")

    # Spot-check: 5 random non-NaN rows — does the per-row value equal the
    # broadcast value for the row's own council?
    sample = df.dropna(subset=[PER_ROW_COL]).sample(n=5, random_state=1)
    print("\nSpot-check (per-row vs broadcast for own council):")
    for _, row in sample.iterrows():
        council = council_by_station.get(row["station_code"])
        broadcast_col = f"days_since_trough_entry_{lga_slug(council)}"
        per_row = row[PER_ROW_COL]
        broadcast = row.get(broadcast_col)
        match = per_row == broadcast
        print(f"  sc={row['station_code']:>5}  date={row['price_date']}  "
              f"council={council:<25}  per_row={per_row}  broadcast={broadcast}  match={match}")

    # Count NaN sources: station with no council vs council not in feature set.
    council_series = df["station_code"].map(council_by_station)
    nan_mask = df[PER_ROW_COL].isna()
    no_council = nan_mask & council_series.isna()
    bad_council = nan_mask & council_series.notna()
    print(f"\nNaN breakdown: {int(no_council.sum())} no-council, "
          f"{int(bad_council.sum())} council-not-in-broadcast-set")

    # For council_series rows where broadcast IS in feature set, the broadcast
    # column itself may carry NaN (PIT trough doesn't exist that early). Check.
    in_feature_set = nan_mask & council_series.notna()
    if in_feature_set.any():
        idx = df[in_feature_set].index[:5]
        print("\nSample rows where council is mapped but per_row is NaN:")
        for i in idx:
            council = council_by_station[df.loc[i, "station_code"]]
            broadcast_col = f"days_since_trough_entry_{lga_slug(council)}"
            print(f"  sc={df.loc[i, 'station_code']}  date={df.loc[i, 'price_date']}  "
                  f"council={council}  broadcast_present={broadcast_col in df.columns}  "
                  f"broadcast_value={df.loc[i, broadcast_col] if broadcast_col in df.columns else 'MISSING_COL'}")


if __name__ == "__main__":
    main()
