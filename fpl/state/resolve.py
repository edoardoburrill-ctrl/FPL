"""Resolving players by name, and building squad state without an entry id.

The public endpoints need a numeric FPL entry id, which is reissued every season
and is not the same thing as a premierleague.com account id. Waiting on it blocks
everything, and it does not have to: a squad is fifteen players, a bank balance
and a free-transfer count, all of which can be stated directly.

Name matching is deliberately strict about ambiguity. Silently picking the wrong
"Silva" would corrupt the purchase-price ledger, which is the one piece of state
that cannot be reconstructed later — so an ambiguous name is an error listing the
candidates, never a guess.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from fpl.models.scoring import POSITION_NAMES


class PlayerNotFound(LookupError):
    """Raised when a name matches no player, or more than one."""


# Letters that are not accented forms of an ASCII letter and so survive NFKD
# decomposition intact, then get dropped by the ASCII encode. Without these,
# "Højlund" normalises to "hjlund" and never matches anything the user types.
TRANSLITERATIONS = str.maketrans(
    {
        "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
        "ß": "ss", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D", "þ": "th",
        "Þ": "TH", "ł": "l", "Ł": "L", "ı": "i", "İ": "I", "ŋ": "n",
        "å": "a", "Å": "A",
    }
)


def normalise(text: str) -> str:
    """Fold a name to bare lowercase letters and digits, accents included."""
    folded = (text or "").translate(TRANSLITERATIONS)
    stripped = unicodedata.normalize("NFKD", folded)
    stripped = stripped.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]", "", stripped)


@dataclass(frozen=True)
class Candidate:
    element: int
    web_name: str
    full_name: str
    team: str
    position: str
    now_cost: int

    def describe(self) -> str:
        return (
            f"{self.web_name} ({self.team}, {self.position}, "
            f"£{self.now_cost / 10:.1f}m, id {self.element})"
        )


def candidates(elements: Sequence[Mapping], teams: Sequence[Mapping]) -> list[Candidate]:
    team_names = {int(t["id"]): str(t.get("short_name", "?")) for t in teams}
    return [
        Candidate(
            element=int(e["id"]),
            web_name=str(e.get("web_name", "")),
            full_name=f"{e.get('first_name', '')} {e.get('second_name', '')}".strip(),
            team=team_names.get(int(e["team"]), "?"),
            position=POSITION_NAMES.get(int(e["element_type"]), "?"),
            now_cost=int(e["now_cost"]),
        )
        for e in elements
    ]


def resolve_one(query: str, pool: Sequence[Candidate]) -> Candidate:
    """Find the single player ``query`` refers to.

    Accepts a numeric element id, an exact web name, a full name, or a unique
    substring. Team can be disambiguated with a suffix: ``"Silva (FUL)"``.
    """
    text = query.strip()
    if not text:
        raise PlayerNotFound("empty player name")

    if text.isdigit():
        wanted = int(text)
        for candidate in pool:
            if candidate.element == wanted:
                return candidate
        raise PlayerNotFound(f"no player with id {wanted}")

    team_hint = None
    match = re.search(r"\(([A-Za-z]{3})\)\s*$", text)
    if match:
        team_hint = match.group(1).upper()
        text = text[: match.start()].strip()

    pool = [c for c in pool if team_hint is None or c.team.upper() == team_hint]
    needle = normalise(text)
    if not needle:
        raise PlayerNotFound(f"could not read a name from {query!r}")

    for matcher in (
        lambda c: normalise(c.web_name) == needle,
        lambda c: normalise(c.full_name) == needle,
        lambda c: needle in normalise(c.full_name) or needle in normalise(c.web_name),
    ):
        found = [c for c in pool if matcher(c)]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            listed = ", ".join(c.describe() for c in found[:8])
            raise PlayerNotFound(
                f"{query!r} matches {len(found)} players: {listed}. "
                "Add the club in brackets, like \"Silva (FUL)\", or use the id."
            )

    raise PlayerNotFound(f"no player matching {query!r}")


@dataclass(frozen=True)
class ParsedPick:
    query: str
    purchase_price: int | None


def parse_pick(text: str) -> ParsedPick:
    """Parse ``"Haaland"`` or ``"Haaland@14.2"`` — the price you actually paid.

    Recording the real purchase price matters: FPL returns only half of any rise
    when you sell, so the selling price depends on what you paid, and no endpoint
    reports that retrospectively.
    """
    if "@" in text:
        name, _, price = text.rpartition("@")
        try:
            return ParsedPick(name.strip(), int(round(float(price.strip()) * 10)))
        except ValueError as exc:
            raise PlayerNotFound(
                f"could not read a price from {text!r}; expected e.g. 'Haaland@14.2'"
            ) from exc
    return ParsedPick(text.strip(), None)


def split_picks(raw: Iterable[str]) -> list[str]:
    """Accept comma-separated, newline-separated, or repeated arguments."""
    picks: list[str] = []
    for chunk in raw:
        for part in re.split(r"[,\n]", chunk):
            cleaned = part.strip()
            if cleaned:
                picks.append(cleaned)
    return picks
