"""Manual squad entry: name resolution and entry-id validation."""

import pytest

from fpl.config import InvalidEntryId, parse_entry_id
from fpl.state.resolve import (
    PlayerNotFound,
    candidates,
    normalise,
    parse_pick,
    resolve_one,
    split_picks,
)

ELEMENTS = [
    {"id": 1, "web_name": "Haaland", "first_name": "Erling", "second_name": "Haaland",
     "team": 1, "element_type": 4, "now_cost": 142},
    {"id": 2, "web_name": "Silva", "first_name": "Bernardo", "second_name": "Silva",
     "team": 1, "element_type": 3, "now_cost": 65},
    {"id": 3, "web_name": "Silva", "first_name": "Marco", "second_name": "Silva",
     "team": 2, "element_type": 3, "now_cost": 55},
    {"id": 4, "web_name": "Sánchez", "first_name": "Robert", "second_name": "Sánchez",
     "team": 2, "element_type": 1, "now_cost": 50},
]
TEAMS = [{"id": 1, "short_name": "MCI"}, {"id": 2, "short_name": "FUL"}]
POOL = candidates(ELEMENTS, TEAMS)


def test_exact_web_name():
    assert resolve_one("Haaland", POOL).element == 1


def test_case_and_accents_are_ignored():
    assert resolve_one("sanchez", POOL).element == 4
    assert resolve_one("SÁNCHEZ", POOL).element == 4


def test_full_name_matches():
    assert resolve_one("Erling Haaland", POOL).element == 1


def test_numeric_id_matches():
    assert resolve_one("3", POOL).element == 3


def test_ambiguity_is_an_error_not_a_guess():
    """Guessing here would silently corrupt the purchase-price ledger."""
    with pytest.raises(PlayerNotFound) as exc:
        resolve_one("Silva", POOL)
    assert "matches 2 players" in str(exc.value)
    assert "MCI" in str(exc.value) and "FUL" in str(exc.value)


def test_club_suffix_disambiguates():
    assert resolve_one("Silva (FUL)", POOL).element == 3
    assert resolve_one("Silva (MCI)", POOL).element == 2


def test_unknown_name_is_an_error():
    with pytest.raises(PlayerNotFound):
        resolve_one("Ronaldo", POOL)


def test_empty_name_is_an_error():
    with pytest.raises(PlayerNotFound):
        resolve_one("   ", POOL)


def test_price_suffix_is_parsed():
    pick = parse_pick("Haaland@14.2")
    assert pick.query == "Haaland"
    assert pick.purchase_price == 142


def test_price_is_optional():
    assert parse_pick("Haaland").purchase_price is None


def test_bad_price_is_an_error():
    with pytest.raises(PlayerNotFound):
        parse_pick("Haaland@lots")


def test_picks_split_on_commas_and_newlines():
    assert split_picks(["Haaland, Salah\nSaka", " Palmer "]) == [
        "Haaland", "Salah", "Saka", "Palmer"
    ]


def test_normalise_strips_punctuation_and_accents():
    assert normalise("Nott'm Forest") == "nottmforest"
    assert normalise("O'Riley") == "oriley"
    assert normalise("Højlund") == "hojlund"


def test_numeric_entry_id_is_accepted():
    assert parse_entry_id("3808385") == 3808385
    assert parse_entry_id(3808385) == 3808385
    assert parse_entry_id(None) is None
    assert parse_entry_id("  ") is None


def test_account_uuid_is_rejected_with_an_explanation():
    with pytest.raises(InvalidEntryId) as exc:
        parse_entry_id("56a89268-c31a-4ea8-87c0-cf7c412ba18d")
    message = str(exc.value)
    assert "account id" in message
    assert "entry/<NUMBER>" in message
    assert "reissues entry ids every season" in message


def test_other_nonsense_is_rejected():
    with pytest.raises(InvalidEntryId):
        parse_entry_id("my-team")
