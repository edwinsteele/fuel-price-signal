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
PREFERRED_STATIONS: dict[int, str] = {
    414: "BP Valley Heights",
    18517: "Shell Blaxland",
    429: "United East Blaxland",
    585: "Ampol Emu Heights",
    261: "7-Eleven near Church",
}

# Station codes known to share a normalised address with another code and always
# lose the INSERT OR IGNORE race. Suppresses the duplicate-address WARNING for
# these specific codes; any new collision not listed here will still warn.
KNOWN_DUPLICATE_STATION_CODES: frozenset[int] = frozenset()
