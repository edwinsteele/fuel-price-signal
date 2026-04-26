# Future Ideas

Things worth doing but explicitly out of scope for the current phase.

## Data collection

- **Intraday snapshots**: Daily snapshots are right for MVP, but the real question is *when* to purchase on a given day. Some stations reprice in the afternoon (lower) vs morning (higher). Multiple snapshots per day (e.g. 4×) would let us analyse intraday patterns and refine the signal from "buy this week" to "buy this afternoon". The GitHub Actions cron already supports multiple runs per day when ready.

## Signal & analysis

- **Leading indicator analysis**: Identify which LGAs or station groups move first in the Sydney cycle — use them as early-warning signals before the cycle turn is visible in your preferred stations.
- **Terminal gate pricing**: Incorporate wholesale/terminal gate prices as a leading fundamental signal. WA FuelWatch publishes this; NSW equivalent TBD.
- **Benchmark fuel prices**: Reference against a benchmark to understand margin compression/expansion (WA FuelWatch: fuelwatch.wa.gov.au/fuelwatch/pages/public/benchmark_prices.jspx).
- **Acyclic station/LGA classifier**: Automatically flag stations or LGAs that don't follow the Sydney cycle, exclude from cycle detection, and investigate why (large-area LGAs, fixed-cost operators).
- **Cycle duration drift detection**: Alert if the mean cycle length is changing over time.
- **Cross-fuel coupling analysis**: Use historical CSVs to test whether diesel, U91, or premium prices lead E10 cycle turns, or whether they're decoupled.

## Purchasing strategy

- **Additional backtest strategies**: Explore strategies beyond null/signal/half-tank hedge — e.g. "fill to quarter-tank on WAIT, full tank on BUY".
- **Purchase consideration frequency sensitivity**: Model how the value of the signal changes with how often the driver actually checks and could act on it.

## Station discovery

- **Automatic cheap-station discovery**: Learn which stations are consistently cheapest on your routes, rather than manually maintaining a preferred list.

## Notifications

- **Push/email notification**: Send the signal proactively (Pushover or email) rather than requiring a CLI invocation. The ff-aws-backend project already had Pushover wired up.
