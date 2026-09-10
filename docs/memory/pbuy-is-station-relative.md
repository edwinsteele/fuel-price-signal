---
name: pbuy-is-station-relative
description: P(BUY) is measured against each station's OWN trailing percentile, so it cannot rank stations against each other. Ranking by it points at the dearest pump.
metadata:
  type: project
---

`P(BUY)` from the locked model answers "is now a good time to buy **at this
station**", not "is this station the cheapest". Condition 2 of the label
(`labels.py`) gates on `today_price <= percentile(that station's own past
lookback_days, percentile_pct)` — a per-station trailing percentile, default 90
days / 33rd. A station's probability therefore rises when it falls against **its
own** history, with no reference to what any other station charges.

The consequence is counterintuitive and easy to ship by accident: a
cross-sectional ranking by `P(BUY)` routinely puts the **most expensive** station
first, because the pump that just cut a big discount off a high base scores higher
than a consistently cheap one sitting at its normal price. Measured on the five
preferred stations for 2026-09-04: United East Blaxland 206.9c scored 0.844 and EG
Ampol Emu Heights 206.9c scored 0.792, while Shell Blaxland at **195.9c** scored
0.419 — sorting by probability would have sent the driver to a pump 11c/L dearer.

Split the two questions. `P(BUY)` (against tau — see [[tau-is-calibrated-not-raw]]
for which scale tau lives on) answers *whether to buy at all*. Today's price
answers *where*. `fuel_signal/signal.py` orders its table by price for exactly this
reason and carries a test pinning it
(`test_stations_are_ranked_by_price_not_by_probability`); don't "fix" that ordering
to use the model's own number.

Same reasoning blocks reading a cross-station *average* of `P(BUY)` as a network
timing signal — the components have different denominators, the same defect class
as [[screen-and-arbiter-share-no-population]]. For network direction use the
aggregate price series, not a mean of per-station probabilities.
