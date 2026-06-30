#!/usr/bin/env python3
"""
Fetches historical NBA game + team stats via nba_api and stores in SQLite.
Also fetches today's odds from The Odds API.
"""

import os
import time
import datetime
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from nba_api.stats.endpoints import leaguegamefinder, boxscoreadvancedv2
from nba_api.stats.static import teams as nba_teams

from db import get_conn, init_db

load_dotenv(Path(__file__).parent / ".env")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
REQUEST_DELAY = 0.7  # nba_api rate limit


def fetch_season_games(season: str) -> pd.DataFrame:
    """Pull all games for a season, e.g. '2023-24'."""
    print(f"  Fetching games for {season}...")
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable="Regular Season",
    )
    time.sleep(REQUEST_DELAY)
    df = finder.get_data_frames()[0]
    return df


def fetch_advanced_boxscore(game_id: str) -> pd.DataFrame | None:
    """Pull advanced stats (pace, off_rtg, def_rtg) for one game."""
    try:
        box = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
        time.sleep(REQUEST_DELAY)
        return box.get_data_frames()[1]  # team stats frame
    except Exception as e:
        print(f"    Warning: advanced boxscore failed for {game_id}: {e}")
        return None


def load_historical(seasons: list[str]):
    """Pull multiple seasons and upsert into DB."""
    init_db()
    conn = get_conn()

    for season in seasons:
        df = fetch_season_games(season)

        # Each game appears twice (once per team) — deduplicate to one row per game
        games = {}
        for _, row in df.iterrows():
            gid = row["GAME_ID"]
            if gid not in games:
                games[gid] = []
            games[gid].append(row)

        print(f"  {season}: {len(games)} unique games")

        for game_id, rows in games.items():
            if len(rows) != 2:
                continue

            home_row = next((r for r in rows if "vs." in r["MATCHUP"]), None)
            away_row = next((r for r in rows if "@ " in r["MATCHUP"]), None)
            if not home_row or not away_row:
                continue

            home_pts = home_row["PTS"]
            away_pts = away_row["PTS"]
            result = "home" if home_pts > away_pts else "away"

            conn.execute("""
                INSERT OR IGNORE INTO games (game_id, game_date, home_team, away_team, home_score, away_score, result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (game_id, home_row["GAME_DATE"], home_row["TEAM_ABBREVIATION"],
                  away_row["TEAM_ABBREVIATION"], home_pts, away_pts, result))

            for row, is_home in [(home_row, 1), (away_row, 0)]:
                conn.execute("""
                    INSERT OR IGNORE INTO team_stats
                    (game_id, game_date, team, opponent, is_home, pts, ast, reb, fg_pct, fg3_pct, ft_pct, plus_minus)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game_id, row["GAME_DATE"], row["TEAM_ABBREVIATION"],
                    home_row["TEAM_ABBREVIATION"] if is_home == 0 else away_row["TEAM_ABBREVIATION"],
                    is_home, row["PTS"], row["AST"], row["REB"],
                    row["FG_PCT"], row["FG3_PCT"], row["FT_PCT"], row["PLUS_MINUS"]
                ))

        conn.commit()
        print(f"  {season}: saved to DB")

    conn.close()
    print("Historical load complete.")


def fetch_today_odds() -> list[dict]:
    """Fetch today's NBA odds from The Odds API."""
    if not ODDS_API_KEY:
        print("No ODDS_API_KEY set — skipping odds fetch")
        return []

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }
    r = requests.get(ODDS_API_URL, params=params, timeout=10)
    r.raise_for_status()

    remaining = r.headers.get("x-requests-remaining", "?")
    print(f"Odds API: {len(r.json())} games | requests remaining: {remaining}")

    return r.json()


def save_odds(games_odds: list[dict]):
    """Store odds snapshot in DB."""
    now = datetime.datetime.utcnow().isoformat()
    conn = get_conn()

    for game in games_odds:
        game_id = game["id"]
        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                home = game["home_team"]
                away = game["away_team"]
                conn.execute("""
                    INSERT INTO odds_snapshots (game_id, fetched_at, bookmaker, home_odds, away_odds)
                    VALUES (?, ?, ?, ?, ?)
                """, (game_id, now, bookmaker["key"],
                      outcomes.get(home), outcomes.get(away)))

    conn.commit()
    conn.close()
    print(f"Saved odds for {len(games_odds)} games.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "historical":
        seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
        load_historical(seasons)
    else:
        print("Fetching today's odds...")
        odds = fetch_today_odds()
        save_odds(odds)
        for g in odds:
            print(f"  {g['away_team']} @ {g['home_team']}")
