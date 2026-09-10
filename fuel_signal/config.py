"""Runtime configuration: API credentials, station list, postcode bounds."""

import os

from fuel_signal.postcode_council import (  # noqa: F401 — re-exported for callers
    SYDNEY_METRO_COUNCILS,
    SYDNEY_METRO_POSTCODES,
)

FUELAPI_API_KEY: str = os.environ.get("FUELAPI_API_KEY", "")
FUELAPI_API_SECRET: str = os.environ.get("FUELAPI_API_SECRET", "")

FUELAPI_BASE_URL = "https://api.onegov.nsw.gov.au"
FUELAPI_TOKEN_URL = f"{FUELAPI_BASE_URL}/oauth/client_credential/accesstoken"
FUELAPI_PRICES_URL = f"{FUELAPI_BASE_URL}/FuelPriceCheck/v1/fuel/prices"

# Preferred stations: fill in station_code → label after first live.py run.
# station_code values come from the FuelCheck API (integer, stable across rebrands).
#
# Labels are display-only. The grading identity that experiments stamp
# (`experiments.lib.universe.station_codes_digest`) hashes the sorted CODES and
# never reads the labels, so correcting a label cannot move a noise floor or
# invalidate a banked result — but changing a KEY would.
PREFERRED_STATIONS: dict[int, str] = {
    414: "BP Springwood",
    18517: "Shell Blaxland",
    429: "United East Blaxland",
    585: "EG Ampol Emu Heights",
    261: "7-Eleven Penrith South",
}

# Days of the week each station is actually reachable, as Python weekday numbers
# (`date.weekday()`: Mon=0 … Sun=6). A station ABSENT from this map is on the
# daily commute and reachable every day; only the exceptions are listed.
#
# This is a routing fact, not a price fact, and it belongs here rather than in the
# label: `labels.py` deliberately excludes station availability from the training
# target ("the user's preferred station may not be on their route today") and
# leaves it to the decision layer. `signal.py` is that decision layer.
STATION_ROUTE_DAYS: dict[int, frozenset[int]] = {
    # Passed on the Penrith run, not the daily commute — typically Wed and Sun.
    261: frozenset({2, 6}),
}

# Station codes known to share a normalised address with another code and always
# lose the INSERT OR IGNORE race. Suppresses the duplicate-address WARNING for
# these specific codes; any new collision not listed here will still warn.
KNOWN_DUPLICATE_STATION_CODES: frozenset[int] = frozenset()

# Minimum distinct stations a brand must have across all history to qualify for
# a days_since_trough_entry_<brand> feature column.  Small brands produce noisy
# trough series; brands below this threshold get no column.
MIN_BRAND_SITES: int = 15
