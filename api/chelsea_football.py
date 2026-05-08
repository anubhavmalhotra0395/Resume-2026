"""
Chelsea-only snapshot built from football-data.org.

Register for a free API token: https://www.football-data.org/client/register
Set env: APP_FOOTBALL_DATA_API_TOKEN

We call the API once per resource type and filter client-side to Chelsea (team id / TLA "CHE").
Transfer rumours are not provided by this provider — use news/RSS separately.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)

FD_BASE = "https://api.football-data.org/v4"

# Extra competitions to try for a single-table row for Chelsea (depends on your API plan).
_EXTRA_STANDINGS = ("CL", "EL", "FAC", "ELC")

# Chelsea team id on football-data.org (stable); used only for demo payload shape.
_CHELSEA_FD_ID = 61


def build_chelsea_demo_snapshot() -> dict[str, Any]:
    """
    Same JSON shape as build_chelsea_snapshot so the desk UI works without a token.
    """
    return {
        "live": False,
        "demo": True,
        "demo_notice": "Demo data. Set APP_FOOTBALL_DATA_API_TOKEN for live football-data.org results.",
        "team": {
            "id": _CHELSEA_FD_ID,
            "name": "Chelsea FC",
            "shortName": "Chelsea",
            "tla": "CHE",
            "crest": "",
        },
        "squad": [
            {"name": "Example Keeper", "position": "Goalkeeper"},
            {"name": "Example Defender", "position": "Defence"},
            {"name": "Example Midfielder", "position": "Midfield"},
            {"name": "Example Forward", "position": "Offence"},
        ],
        "coach": {"name": "—", "nationality": "—"},
        "fixtures_upcoming": [
            {
                "homeTeam": {"shortName": "CHE", "name": "Chelsea FC"},
                "awayTeam": {"shortName": "ARS", "name": "Arsenal FC"},
                "utcDate": "2099-05-01T16:30:00Z",
            },
            {
                "homeTeam": {"shortName": "TOT", "name": "Tottenham Hotspur FC"},
                "awayTeam": {"shortName": "CHE", "name": "Chelsea FC"},
                "utcDate": "2099-05-08T14:00:00Z",
            },
        ],
        "premier_league": {
            "table_row": {
                "position": 6,
                "points": 42,
                "won": 12,
                "draw": 6,
                "lost": 8,
                "goalsFor": 38,
                "goalsAgainst": 34,
                "goalDifference": 4,
                "team": {"id": _CHELSEA_FD_ID, "name": "Chelsea FC", "shortName": "Chelsea"},
            }
        },
        "cups_and_uefa": [
            {
                "competition": "CL",
                "table_row": None,
                "included": False,
                "note": "Live cup tables need APP_FOOTBALL_DATA_API_TOKEN (plan may vary).",
            },
            {
                "competition": "FAC",
                "table_row": None,
                "included": False,
                "note": "—",
            },
        ],
        "source": "demo",
        "disclaimer": "Illustrative snapshot only. Not from football-data.org.",
    }


async def _get_json(client: httpx.AsyncClient, path: str, params: dict | None = None) -> dict[str, Any]:
    r = await client.get(f"{FD_BASE}{path}", params=params or {})
    r.raise_for_status()
    return r.json()


def _find_chelsea_row(standings_payload: dict[str, Any], team_id: int) -> dict[str, Any] | None:
    for standing in standings_payload.get("standings") or []:
        for row in standing.get("table") or []:
            if (row.get("team") or {}).get("id") == team_id:
                return row
    return None


async def build_chelsea_snapshot(token: str) -> dict[str, Any]:
    headers = {"X-Auth-Token": token}
    async with httpx.AsyncClient(timeout=45.0, headers=headers) as client:
        try:
            pl_teams = await _get_json(client, "/competitions/PL/teams")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"football-data.org error: {e.response.text[:500]}") from e

        teams = pl_teams.get("teams") or []
        chelsea = next((t for t in teams if t.get("tla") == "CHE"), None)
        if not chelsea:
            chelsea = next((t for t in teams if "chelsea" in (t.get("name") or "").lower()), None)
        if not chelsea:
            raise HTTPException(status_code=502, detail="Could not resolve Chelsea in Premier League teams list")

        tid = int(chelsea["id"])

        try:
            team_full = await _get_json(client, f"/teams/{tid}")
        except httpx.HTTPStatusError as e:
            team_full = {"detail": str(e), "id": tid, "name": chelsea.get("name")}

        try:
            scheduled = await _get_json(
                client,
                f"/teams/{tid}/matches",
                params={"status": "SCHEDULED", "limit": 25},
            )
        except httpx.HTTPStatusError as e:
            scheduled = {"matches": [], "detail": str(e)}

        try:
            pl_standings = await _get_json(client, "/competitions/PL/standings")
            pl_row = _find_chelsea_row(pl_standings, tid)
        except httpx.HTTPStatusError as e:
            pl_row = None
            pl_standings = {"detail": str(e)}

        cup_rows: list[dict[str, Any]] = []
        for code in _EXTRA_STANDINGS:
            try:
                data = await _get_json(client, f"/competitions/{code}/standings")
                row = _find_chelsea_row(data, tid)
                cup_rows.append(
                    {
                        "competition": code,
                        "table_row": row,
                        "included": row is not None,
                    }
                )
            except httpx.HTTPStatusError as e:
                note = e.response.text[:200] if e.response is not None else str(e)
                if e.response is not None and e.response.status_code == 403:
                    note = "403 — competition may need a higher football-data.org plan"
                cup_rows.append(
                    {
                        "competition": code,
                        "table_row": None,
                        "included": False,
                        "note": note,
                    }
                )

        return {
            "live": True,
            "demo": False,
            "team": {
                "id": tid,
                "name": chelsea.get("name"),
                "shortName": chelsea.get("shortName"),
                "tla": chelsea.get("tla"),
                "crest": chelsea.get("crest"),
            },
            "squad": team_full.get("squad") if isinstance(team_full, dict) else None,
            "coach": team_full.get("coach") if isinstance(team_full, dict) else None,
            "fixtures_upcoming": scheduled.get("matches", []),
            "premier_league": {
                "table_row": pl_row,
            },
            "cups_and_uefa": cup_rows,
            "source": "https://www.football-data.org/",
            "disclaimer": "Filtered to Chelsea only. Transfer news is not available from this API.",
        }
