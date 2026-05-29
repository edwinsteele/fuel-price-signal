"""Tests for fuel_signal.postcode_council — LGA mapping and metro filtering."""

from fuel_signal.postcode_council import (
    _PC_MAP,
    SYDNEY_METRO_COUNCILS,
    SYDNEY_METRO_POSTCODES,
    is_sydney_metro,
    primary_council,
)


class TestPrimaryCouncil:
    def test_blue_mountains_home_postcode(self):
        assert primary_council("2777") == "Blue Mountains"

    def test_sydney_cbd(self):
        assert primary_council("2000") == "Sydney"

    def test_penrith(self):
        assert primary_council("2750") == "Penrith"

    def test_unknown_postcode_returns_none(self):
        assert primary_council("9999") is None

    def test_empty_string_returns_none(self):
        assert primary_council("") is None

    def test_newcastle(self):
        assert primary_council("2300") == "Newcastle"

    def test_wollongong(self):
        assert primary_council("2500") == "Wollongong"

    def test_integer_coerced_to_string(self):
        # primary_council calls str() on its argument so int input works
        assert primary_council(str(2777)) == "Blue Mountains"

    def test_border_vic_postcode_present(self):
        # Albury-area postcode that straddles the NSW/VIC border
        assert primary_council("3691") == "Albury"


class TestIsSydneyMetro:
    def test_blue_mountains_is_metro(self):
        assert is_sydney_metro("2777") is True

    def test_sydney_cbd_is_metro(self):
        assert is_sydney_metro("2000") is True

    def test_penrith_is_metro(self):
        assert is_sydney_metro("2750") is True

    def test_inner_west_is_metro(self):
        assert is_sydney_metro("2040") is True

    def test_newcastle_is_not_metro(self):
        assert is_sydney_metro("2300") is False

    def test_wollongong_is_not_metro(self):
        assert is_sydney_metro("2500") is False

    def test_central_coast_is_not_metro(self):
        # Central Coast excluded: cycle decoupled from Sydney metro (issue #133)
        assert is_sydney_metro("2250") is False

    def test_hawkesbury_is_metro(self):
        assert is_sydney_metro("2753") is True

    def test_rural_nsw_is_not_metro(self):
        assert is_sydney_metro("2640") is False  # Albury area

    def test_unknown_postcode_is_not_metro(self):
        assert is_sydney_metro("9999") is False


class TestSydneyMetroPostcodes:
    def test_includes_known_metro_postcodes(self):
        for pc in ("2000", "2777", "2750", "2040", "2148"):
            assert pc in SYDNEY_METRO_POSTCODES, f"{pc} should be in SYDNEY_METRO_POSTCODES"

    def test_excludes_newcastle(self):
        assert "2300" not in SYDNEY_METRO_POSTCODES

    def test_excludes_wollongong(self):
        assert "2500" not in SYDNEY_METRO_POSTCODES

    def test_excludes_rural_nsw(self):
        assert "2640" not in SYDNEY_METRO_POSTCODES  # Albury
        assert "2800" not in SYDNEY_METRO_POSTCODES  # Orange

    def test_consistent_with_is_sydney_metro(self):
        for pc in _PC_MAP:
            expected = is_sydney_metro(pc)
            assert (pc in SYDNEY_METRO_POSTCODES) == expected


class TestSydneyMetroCouncils:
    def test_expected_councils_present(self):
        expected = {
            "Sydney", "Inner West", "Blacktown", "Penrith",
            "Blue Mountains", "Northern Beaches", "Sutherland Shire",
            "Canterbury-Bankstown", "Liverpool", "Campbelltown",
        }
        assert expected <= SYDNEY_METRO_COUNCILS

    def test_regional_councils_absent(self):
        regional = {"Newcastle", "Wollongong", "Lake Macquarie", "Wagga Wagga", "Albury"}
        assert regional.isdisjoint(SYDNEY_METRO_COUNCILS)
