"""The backtest pipeline, tested against a small embedded season — not the live
archive. Automated tests stay off the network, matching the rest of this
project; the real validation claim (see DECISIONS.md) is run interactively and
its result is recorded there, not re-derived on every test run.
"""

from fpl.backtest.adapter import load_season, snapshot_as_of
from fpl.backtest.metrics import GameweekResult, SeasonReport
from fpl.backtest.replay import replay_season
from fpl.data.historical import HistoricalClient
from fpl.models.scoring import DEF, FWD, GK, MID

TEAMS = [
    {"id": "1", "name": "Arsenal", "short_name": "ARS"},
    {"id": "2", "name": "Everton", "short_name": "EVE"},
]

FIXTURES = [
    {"id": "1", "event": "1", "team_h": "1", "team_a": "2",
     "team_h_score": "2", "team_a_score": "0", "finished": "True"},
    {"id": "2", "event": "2", "team_h": "2", "team_a": "1",
     "team_h_score": "1", "team_a_score": "1", "finished": "True"},
    {"id": "3", "event": "3", "team_h": "1", "team_a": "2",
     "team_h_score": "3", "team_a_score": "1", "finished": "True"},
]


def _row(element, name, team, position, gw, minutes, goals=0, assists=0,
        total_points=0, xp=0.0, value=50, bps=0, bonus=0):
    return {
        "element": str(element), "name": name, "team": team, "position": position,
        "GW": str(gw), "minutes": str(minutes), "goals_scored": str(goals),
        "assists": str(assists), "total_points": str(total_points), "xP": str(xp),
        "value": str(value), "bps": str(bps), "bonus": str(bonus),
        "clean_sheets": "0", "goals_conceded": "0", "saves": "0",
        "yellow_cards": "0", "red_cards": "0", "own_goals": "0",
        "penalties_missed": "0", "penalties_saved": "0", "starts": "1" if minutes else "0",
        "expected_goals": "0.1", "expected_assists": "0.1",
    }


ROWS = [
    _row(1, "A Striker", "Arsenal", "FWD", 1, 90, goals=2, total_points=13, xp=6.0),
    _row(2, "A Keeper", "Everton", "GK", 1, 90, total_points=2, xp=3.0),
    _row(1, "A Striker", "Arsenal", "FWD", 2, 90, goals=0, total_points=2, xp=5.5),
    _row(2, "A Keeper", "Everton", "GK", 2, 90, total_points=8, xp=3.2),
    _row(1, "A Striker", "Arsenal", "FWD", 3, 45, goals=1, total_points=8, xp=5.0),
    _row(2, "A Keeper", "Everton", "GK", 3, 90, total_points=2, xp=3.0),
]


class FakeClient:
    def teams(self, season):
        return TEAMS

    def fixtures(self, season):
        return FIXTURES

    def merged_gw(self, season):
        return ROWS


def test_load_season_builds_team_id_map():
    data = load_season(FakeClient(), "test")
    assert data.team_id_by_name == {"Arsenal": 1, "Everton": 2}
    assert data.max_gameweek == 3


def test_snapshot_excludes_the_target_gameweek():
    """This is the entire point of the module: no lookahead."""
    data = load_season(FakeClient(), "test")
    bootstrap, fixtures = snapshot_as_of(data, gameweek=2)

    striker = next(e for e in bootstrap["elements"] if e["id"] == 1)
    assert striker["goals_scored"] == 2.0  # only GW1's goals, not GW2's

    target_fixture = next(f for f in fixtures if f["event"] == 2)
    assert not target_fixture["finished"]
    past_fixture = next(f for f in fixtures if f["event"] == 1)
    assert past_fixture["finished"]


def test_snapshot_at_gameweek_one_has_no_known_players():
    """A player is only known to the snapshot once he has appeared in a played
    gameweek — there is no separate roster listing in the archive the way
    `bootstrap-static` provides one live. This is exactly why `replay_season`
    defaults to starting well after gameweek 1."""
    data = load_season(FakeClient(), "test")
    bootstrap, _ = snapshot_as_of(data, gameweek=1)
    assert bootstrap["elements"] == []


def test_snapshot_takes_the_latest_price():
    rows = ROWS + [_row(1, "A Striker", "Arsenal", "FWD", 2, 90, value=55)]
    data = load_season(FakeClient(), "test")
    data.rows = rows
    bootstrap, _ = snapshot_as_of(data, gameweek=3)
    striker = next(e for e in bootstrap["elements"] if e["id"] == 1)
    assert striker["now_cost"] == 55


def test_actuals_reads_the_real_total_points_and_xp():
    data = load_season(FakeClient(), "test")
    actual = data.actuals(1)
    assert actual[1]["total_points"] == 13
    assert actual[1]["xP"] == 6.0


def test_replay_season_produces_a_scoreable_report():
    report = replay_season("test", start_gameweek=2, end_gameweek=3,
                           client=FakeClient())
    assert len(report.gameweeks) == 2
    assert report.n > 0
    # Every prediction must have a corresponding real outcome, pairwise.
    for gw in report.gameweeks:
        assert len(gw.predicted) == len(gw.actual) == len(gw.xp_baseline)


def test_replay_season_skips_players_who_did_not_play():
    rows = ROWS + [_row(3, "Benchwarmer", "Arsenal", "MID", 2, 0, total_points=0, xp=1.0)]
    class Client(FakeClient):
        def merged_gw(self, season):
            return rows
    report = replay_season("test", start_gameweek=2, end_gameweek=2, client=Client())
    assert 3 not in report.gameweeks[0].elements


def test_season_report_summary_mentions_both_baselines():
    report = SeasonReport("test", [
        GameweekResult(2, predicted=[6.0, 2.0], actual=[8.0, 2.0],
                      xp_baseline=[3.2, 3.0], elements=[1, 2]),
    ])
    text = report.summary()
    assert "ours=" in text and "FPL xP=" in text and "season-mean baseline" in text


def test_top_pick_points_rewards_correct_ranking():
    """If we correctly pick the higher scorer and xP does not, we should win."""
    report = SeasonReport("test", [
        GameweekResult(1, predicted=[9.0, 2.0], actual=[13.0, 2.0],
                      xp_baseline=[3.0, 6.0], elements=[1, 2]),
    ])
    mine, theirs = report.top_pick_points(min_start_pool=1)
    assert mine == 13.0  # we picked element 1, who scored 13
    assert theirs == 2.0  # xP picked element 2, who scored 2
