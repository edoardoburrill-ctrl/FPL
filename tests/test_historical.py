"""The historical data client and bonus-rule verification against real matches.

Parsing is tested offline against embedded fixtures shaped like the real CSVs.
The verification claim itself — that `award_bonus` reproduces every real match
in two full seasons — is checked interactively against the live archive (see
DECISIONS.md); it is not re-run on every test invocation, matching this
project's policy of keeping automated tests off the network.
"""

from fpl.data.historical import iter_match_bps
from fpl.models.bps import award_bonus

SAMPLE_ROWS = [
    # Fixture 10, GW3: three home players, a tie for 2nd.
    {"fixture": "10", "GW": "3", "element": "1", "minutes": "90", "bps": "40", "bonus": "3"},
    {"fixture": "10", "GW": "3", "element": "2", "minutes": "90", "bps": "25", "bonus": "2"},
    {"fixture": "10", "GW": "3", "element": "3", "minutes": "90", "bps": "25", "bonus": "2"},
    # A player who did not feature in this fixture: excluded from the ranking.
    {"fixture": "10", "GW": "3", "element": "4", "minutes": "0", "bps": "0", "bonus": "0"},
    # A different fixture entirely.
    {"fixture": "11", "GW": "3", "element": "5", "minutes": "90", "bps": "50", "bonus": "3"},
]


def test_rows_group_by_fixture_not_gameweek():
    matches = list(iter_match_bps(SAMPLE_ROWS))
    assert {m.fixture for m in matches} == {10, 11}


def test_non_appearances_are_excluded_from_the_match():
    match = next(m for m in iter_match_bps(SAMPLE_ROWS) if m.fixture == 10)
    assert 4 not in match.bps
    assert set(match.bps) == {1, 2, 3}


def test_real_tie_is_reproduced_by_award_bonus():
    match = next(m for m in iter_match_bps(SAMPLE_ROWS) if m.fixture == 10)
    assert award_bonus(match.bps) == match.bonus


def test_rows_with_no_fixture_are_skipped():
    rows = SAMPLE_ROWS + [{"fixture": "", "GW": "3", "element": "9", "minutes": "90",
                           "bps": "10", "bonus": "0"}]
    matches = list(iter_match_bps(rows))
    assert {m.fixture for m in matches} == {10, 11}
