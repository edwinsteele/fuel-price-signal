import pytest

from fuel_signal import series as _series


@pytest.fixture(autouse=True)
def _clear_series_caches():
    """Clear module-level series caches between tests to prevent id(conn) collisions."""
    yield
    _series._SERIES_CACHE.clear()
    _series._GROUPS_CACHE.clear()
