"""Runtime configuration: API credentials, station list, postcode bounds."""

import os

FUELAPI_API_KEY: str = os.environ.get("FUELAPI_API_KEY", "")
FUELAPI_API_SECRET: str = os.environ.get("FUELAPI_API_SECRET", "")

FUELAPI_BASE_URL = "https://api.onegov.nsw.gov.au"
FUELAPI_TOKEN_URL = f"{FUELAPI_BASE_URL}/oauth/client_credential/accesstoken"
FUELAPI_PRICES_URL = f"{FUELAPI_BASE_URL}/FuelPriceCheck/v1/fuel/prices"

# Greater Sydney including Blue Mountains corridor (excludes rural NSW at 2800+)
SYDNEY_METRO_POSTCODES: frozenset[str] = frozenset(str(p) for p in range(2000, 2800))

# Preferred stations: fill in station_code → label after first live.py run.
# station_code values come from the FuelCheck API (integer, stable across rebrands).
PREFERRED_STATIONS: dict[int, str] = {}
