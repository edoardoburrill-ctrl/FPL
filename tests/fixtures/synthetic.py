"""A synthetic FPL universe, in the exact shape of the real API payloads.

The build environment has no route to ``fantasy.premierleague.com``, and even
where it does, tests that depend on live data are tests that fail on a Tuesday
for reasons unrelated to the code. So the pipeline is exercised against a
generated league whose latent parameters we know, which has the useful side
effect of letting tests assert that the model *recovers* those parameters rather
than merely that it runs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

TEAM_NAMES = [
    ("Arsenal", "ARS"), ("Aston Villa", "AVL"), ("Bournemouth", "BOU"),
    ("Brentford", "BRE"), ("Brighton", "BHA"), ("Burnley", "BUR"),
    ("Chelsea", "CHE"), ("Crystal Palace", "CRY"), ("Everton", "EVE"),
    ("Fulham", "FUL"), ("Leeds", "LEE"), ("Liverpool", "LIV"),
    ("Man City", "MCI"), ("Man Utd", "MUN"), ("Newcastle", "NEW"),
    ("Nott'm Forest", "NFO"), ("Spurs", "TOT"), ("Sunderland", "SUN"),
    ("West Ham", "WHU"), ("Wolves", "WOL"),
]

# Squad shape per club, by element_type (1 GK, 2 DEF, 3 MID, 4 FWD).
SQUAD_SHAPE = {1: 3, 2: 8, 3: 9, 4: 5}
POSITION_BASE_PRICE = {1: 40, 2: 40, 3: 45, 4: 45}


def build_universe(
    seed: int = 7,
    gameweeks_played: int = 2,
    total_gameweeks: int = 38,
) -> tuple[dict, list[dict]]:
    """Return ``(bootstrap_static, fixtures)`` for a plausible season in progress."""
    rng = np.random.default_rng(seed)
    n_teams = len(TEAM_NAMES)

    # Latent team quality, which drives both results and player statistics.
    attack = rng.normal(0.0, 0.28, n_teams)
    defence = rng.normal(0.0, 0.26, n_teams)
    attack -= attack.mean()
    defence -= defence.mean()

    teams = [
        {
            "id": i + 1,
            "name": name,
            "short_name": short,
            "strength": int(np.clip(3 + round(attack[i] * 3), 1, 5)),
            "_attack": float(attack[i]),
            "_defence": float(defence[i]),
        }
        for i, (name, short) in enumerate(TEAM_NAMES)
    ]

    fixtures = _build_fixtures(rng, teams, gameweeks_played, total_gameweeks)
    elements = _build_elements(rng, teams, fixtures, gameweeks_played)
    events = _build_events(gameweeks_played, total_gameweeks)

    bootstrap = {
        "events": events,
        "teams": [{k: v for k, v in t.items() if not k.startswith("_")} for t in teams],
        "elements": elements,
        "element_types": [
            {"id": 1, "singular_name_short": "GKP", "squad_select": 2},
            {"id": 2, "singular_name_short": "DEF", "squad_select": 5},
            {"id": 3, "singular_name_short": "MID", "squad_select": 5},
            {"id": 4, "singular_name_short": "FWD", "squad_select": 3},
        ],
        "total_players": 11_000_000,
        "game_settings": {"squad_squadsize": 15, "transfers_cost": 4},
    }
    return bootstrap, fixtures


def _round_robin(n: int) -> list[list[tuple[int, int]]]:
    """Circle-method schedule: n-1 rounds, each team playing once per round."""
    teams = list(range(n))
    rounds = []
    for _ in range(n - 1):
        pairs = [
            (teams[i], teams[n - 1 - i]) for i in range(n // 2)
        ]
        rounds.append(pairs)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds


def _build_fixtures(rng, teams, gameweeks_played, total_gameweeks) -> list[dict]:
    n = len(teams)
    schedule = _round_robin(n)
    # Second half of the season is the reverse fixture.
    schedule = schedule + [[(a, h) for h, a in rnd] for rnd in schedule]
    schedule = schedule[:total_gameweeks]

    kickoff = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)
    fixtures: list[dict] = []
    fixture_id = 1
    for gw, pairs in enumerate(schedule, start=1):
        for home, away in pairs:
            home_team, away_team = teams[home], teams[away]
            row = {
                "id": fixture_id,
                "event": gw,
                "team_h": home_team["id"],
                "team_a": away_team["id"],
                "team_h_difficulty": int(np.clip(3 - round(away_team["_defence"] * 3), 1, 5)),
                "team_a_difficulty": int(np.clip(3 - round(home_team["_defence"] * 3), 1, 5)),
                "kickoff_time": (kickoff + timedelta(days=7 * (gw - 1))).isoformat(),
                "finished": gw <= gameweeks_played,
                "team_h_score": None,
                "team_a_score": None,
            }
            if row["finished"]:
                lam = np.exp(0.15 + home_team["_attack"] - away_team["_defence"] + 0.25)
                mu = np.exp(0.15 + away_team["_attack"] - home_team["_defence"])
                row["team_h_score"] = int(rng.poisson(lam))
                row["team_a_score"] = int(rng.poisson(mu))
            fixtures.append(row)
            fixture_id += 1
    return fixtures


def _build_events(gameweeks_played: int, total_gameweeks: int) -> list[dict]:
    deadline = datetime(2026, 8, 14, 17, 30, tzinfo=timezone.utc)
    events = []
    for gw in range(1, total_gameweeks + 1):
        finished = gw <= gameweeks_played
        events.append(
            {
                "id": gw,
                "name": f"Gameweek {gw}",
                "deadline_time": (deadline + timedelta(days=7 * (gw - 1))).isoformat(),
                "finished": finished,
                "is_current": gw == gameweeks_played,
                "is_next": gw == gameweeks_played + 1,
                "is_previous": gw == gameweeks_played - 1,
                "chip_plays": [],
            }
        )
    return events


def _build_elements(rng, teams, fixtures, gameweeks_played) -> list[dict]:
    elements: list[dict] = []
    element_id = 1

    for team in teams:
        team_matches = sum(
            1
            for f in fixtures
            if f["finished"] and team["id"] in (f["team_h"], f["team_a"])
        )
        team_goals = sum(
            (f["team_h_score"] if f["team_h"] == team["id"] else f["team_a_score"])
            for f in fixtures
            if f["finished"] and team["id"] in (f["team_h"], f["team_a"])
        )

        for position, count in SQUAD_SHAPE.items():
            # Rank within position determines role: the first are the starters.
            for rank in range(count):
                starter_strength = np.exp(-1.1 * rank)
                start_probability = float(np.clip(starter_strength, 0.02, 0.97))
                starts = int(rng.binomial(team_matches, start_probability))
                cameos = int(
                    rng.binomial(max(0, team_matches - starts), 0.35)
                ) if position != 1 else 0
                minutes = int(
                    starts * rng.integers(62, 91) + cameos * rng.integers(5, 30)
                )

                price = _price(rng, position, rank, team["_attack"])
                share = _output_share(position, rank)
                nineties = max(minutes / 90.0, 1e-9)

                expected_goals = float(
                    max(0.0, rng.gamma(2.0, 0.5) * share * team_goals / max(team_matches, 1) * nineties)
                )
                expected_assists = float(
                    max(0.0, rng.gamma(2.0, 0.5) * share * 0.75 * team_goals / max(team_matches, 1) * nineties)
                )
                defcon_rate = _defcon_rate(position, rank)
                saves = int(rng.poisson(3.1 * nineties)) if position == 1 else 0

                status, chance = _availability(rng)

                elements.append(
                    {
                        "id": element_id,
                        "web_name": f"{team['short_name']}{position}{rank}",
                        "first_name": "Player",
                        "second_name": f"{element_id}",
                        "team": team["id"],
                        "element_type": position,
                        "now_cost": price,
                        "status": status,
                        "chance_of_playing_next_round": chance,
                        "minutes": minutes,
                        "starts": starts,
                        "goals_scored": int(rng.poisson(expected_goals)),
                        "assists": int(rng.poisson(expected_assists)),
                        "clean_sheets": 0,
                        "goals_conceded": 0,
                        "saves": saves,
                        "expected_goals": f"{expected_goals:.2f}",
                        "expected_assists": f"{expected_assists:.2f}",
                        "expected_goal_involvements": f"{expected_goals + expected_assists:.2f}",
                        "clearances_blocks_interceptions": int(
                            rng.poisson(defcon_rate * 0.62 * nineties)
                        ),
                        "tackles": int(rng.poisson(defcon_rate * 0.22 * nineties)),
                        "recoveries": int(rng.poisson(defcon_rate * 0.55 * nineties)),
                        "bps": int(rng.poisson(_bps_rate(position, rank) * nineties)),
                        "yellow_cards": int(rng.poisson(0.16 * nineties)),
                        "red_cards": int(rng.poisson(0.01 * nineties)),
                        "bonus": 0,
                        "total_points": 0,
                        "selected_by_percent": f"{max(0.1, 25 * share):.1f}",
                        "form": "0.0",
                        "ep_next": "0.0",
                        "transfers_in_event": 0,
                        "transfers_out_event": 0,
                    }
                )
                element_id += 1
    return elements


def _price(rng, position: int, rank: int, team_attack: float) -> int:
    base = POSITION_BASE_PRICE[position]
    premium = max(0.0, (4 - rank)) * (18 if position in (3, 4) else 10)
    premium *= 1.0 + team_attack
    return int(np.clip(base + premium + rng.integers(-3, 4), 38, 155))


def _output_share(position: int, rank: int) -> float:
    """Roughly what fraction of a team's goal involvement this role takes."""
    base = {1: 0.001, 2: 0.05, 3: 0.16, 4: 0.24}[position]
    return float(base * np.exp(-0.55 * rank))


def _defcon_rate(position: int, rank: int) -> float:
    base = {1: 1.0, 2: 8.5, 3: 7.0, 4: 3.0}[position]
    return float(base * (1.0 + 0.05 * rank))


def _bps_rate(position: int, rank: int) -> float:
    base = {1: 19.0, 2: 19.0, 3: 20.0, 4: 20.0}[position]
    return float(base * np.exp(-0.12 * rank))


def _availability(rng) -> tuple[str, int | None]:
    draw = rng.random()
    if draw < 0.04:
        return "i", 0
    if draw < 0.09:
        return "d", int(rng.choice([25, 50, 75]))
    return "a", None
