import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "db" / "nba.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS games (
            game_id      TEXT PRIMARY KEY,
            game_date    TEXT NOT NULL,
            home_team    TEXT NOT NULL,
            away_team    TEXT NOT NULL,
            home_score   INTEGER,
            away_score   INTEGER,
            result       TEXT   -- 'home' or 'away'
        );

        CREATE TABLE IF NOT EXISTS team_stats (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id      TEXT NOT NULL,
            game_date    TEXT NOT NULL,
            team         TEXT NOT NULL,
            opponent     TEXT NOT NULL,
            is_home      INTEGER NOT NULL,
            pts          REAL, ast REAL, reb REAL,
            fg_pct       REAL, fg3_pct REAL, ft_pct REAL,
            plus_minus   REAL, pace REAL, off_rtg REAL, def_rtg REAL,
            UNIQUE(game_id, team)
        );

        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id      TEXT NOT NULL,
            fetched_at   TEXT NOT NULL,
            bookmaker    TEXT NOT NULL,
            home_odds    REAL,
            away_odds    REAL
        );

        CREATE TABLE IF NOT EXISTS signals (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id      TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            home_team    TEXT NOT NULL,
            away_team    TEXT NOT NULL,
            model_home_prob REAL,
            implied_home_prob REAL,
            ev_home      REAL,
            implied_away_prob REAL,
            ev_away      REAL,
            best_side    TEXT,
            best_ev      REAL,
            best_odds    REAL,
            bookmaker    TEXT
        );

        CREATE TABLE IF NOT EXISTS bets (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id      TEXT NOT NULL,
            placed_at    TEXT NOT NULL,
            side         TEXT NOT NULL,
            odds         REAL NOT NULL,
            stake        REAL NOT NULL,
            model_prob   REAL NOT NULL,
            ev           REAL NOT NULL,
            result       TEXT,  -- 'win', 'loss', 'void'
            pnl          REAL
        );
        """)
    print(f"DB initialised at {DB_PATH}")


if __name__ == "__main__":
    init_db()
