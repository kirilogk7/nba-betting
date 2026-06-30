#!/usr/bin/env python3
"""
Feature engineering: builds one row per game with rolling team stats for both sides.
All rolling windows use shift(1) so no game data leaks from the future.
"""

import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "db" / "nba.db"
WINDOW = 10  # rolling lookback in games


def build_features() -> tuple[pd.DataFrame, list[str]]:
    conn = sqlite3.connect(DB_PATH)
    games = pd.read_sql("SELECT * FROM games ORDER BY game_date", conn)
    stats = pd.read_sql("SELECT * FROM team_stats ORDER BY game_date", conn)
    conn.close()

    # Add pts_allowed from games table (cleaner than self-join)
    pts_map = {}
    for _, g in games.iterrows():
        pts_map[(g["game_id"], g["home_team"])] = g["away_score"]
        pts_map[(g["game_id"], g["away_team"])] = g["home_score"]
    stats["pts_allowed"] = stats.apply(
        lambda r: pts_map.get((r["game_id"], r["team"])), axis=1
    )

    # Win flag per game row
    stats["won"] = (stats["plus_minus"] > 0).astype(float)

    # Sort for rolling
    stats = stats.sort_values(["team", "game_date"]).reset_index(drop=True)

    # Rolling stats (shift 1 so current game is excluded)
    roll_src = ["pts", "pts_allowed", "fg_pct", "fg3_pct", "ft_pct", "plus_minus", "won"]
    for col in roll_src:
        stats[f"r{WINDOW}_{col}"] = (
            stats.groupby("team")[col]
            .transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=WINDOW // 2).mean())
        )

    # Rest days (capped at 7)
    stats["prev_date"] = stats.groupby("team")["game_date"].shift(1)
    stats["rest_days"] = (
        (pd.to_datetime(stats["game_date"]) - pd.to_datetime(stats["prev_date"]))
        .dt.days.clip(upper=7)
        .fillna(7)
    )

    roll_cols = [f"r{WINDOW}_{c}" for c in roll_src] + ["rest_days"]

    home = stats[stats["is_home"] == 1][["game_id"] + roll_cols].copy()
    away = stats[stats["is_home"] == 0][["game_id"] + roll_cols].copy()

    home = home.rename(columns={c: f"home_{c}" for c in roll_cols})
    away = away.rename(columns={c: f"away_{c}" for c in roll_cols})

    df = games.merge(home, on="game_id").merge(away, on="game_id")
    df["target"] = (df["result"] == "home").astype(int)

    feature_cols = [f"home_{c}" for c in roll_cols] + [f"away_{c}" for c in roll_cols]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    print(f"Feature matrix: {len(df)} games, {len(feature_cols)} features")
    return df, feature_cols
