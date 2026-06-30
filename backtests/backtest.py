#!/usr/bin/env python3
"""
Backtest the EV model on 2024-25 season (out-of-sample).
Assumes fixed odds of 1.91 for both sides (simulated -110).
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import train

ODDS        = 1.91
BREAK_EVEN  = 1 / ODDS        # 52.4%
MIN_EV      = 0.03             # only bet when edge > 3%
FLAT_STAKE  = 0.02             # 2% of bankroll per bet
START_BANK  = 1000.0


def run():
    _, _, test_df, home_probs = train(save=False)

    bankroll = START_BANK
    bets = []

    for i, row in test_df.iterrows():
        hp = home_probs[i]
        ap = 1 - hp

        ev_home = hp * ODDS - 1
        ev_away = ap * ODDS - 1
        best_ev = max(ev_home, ev_away)

        if best_ev < MIN_EV:
            continue

        side = "home" if ev_home >= ev_away else "away"
        prob = hp if side == "home" else ap
        stake = round(bankroll * FLAT_STAKE, 2)
        won = (side == row["result"])
        pnl = stake * (ODDS - 1) if won else -stake
        bankroll += pnl

        bets.append({
            "date":     row["game_date"],
            "home":     row["home_team"],
            "away":     row["away_team"],
            "side":     side,
            "prob":     round(prob, 3),
            "ev":       round(best_ev, 3),
            "odds":     ODDS,
            "stake":    stake,
            "won":      won,
            "pnl":      round(pnl, 2),
            "bankroll": round(bankroll, 2),
        })

    if not bets:
        print("No bets placed — try lowering MIN_EV threshold")
        return

    df = pd.DataFrame(bets)
    total_staked = df["stake"].sum()
    total_pnl    = df["pnl"].sum()
    win_rate     = df["won"].mean()
    roi          = total_pnl / total_staked * 100
    max_dd       = _max_drawdown(df["bankroll"])

    print(f"\n{'='*45}")
    print(f"  BACKTEST — 2024-25 season (OOS, simulated -110)")
    print(f"{'='*45}")
    print(f"  Bets placed : {len(df)}")
    print(f"  Win rate    : {win_rate*100:.1f}%  (break-even: 52.4%)")
    print(f"  Total staked: €{total_staked:,.2f}")
    print(f"  Total P&L   : €{total_pnl:+,.2f}")
    print(f"  ROI         : {roi:+.2f}%")
    print(f"  Max drawdown: €{max_dd:.2f}")
    print(f"  Start bank  : €{START_BANK:,.2f}")
    print(f"  Final bank  : €{bankroll:,.2f}  ({(bankroll/START_BANK-1)*100:+.1f}%)")
    print(f"{'='*45}\n")

    # Monthly breakdown
    df["month"] = df["date"].str[:7]
    monthly = df.groupby("month").agg(
        bets=("pnl", "count"),
        win_rate=("won", "mean"),
        pnl=("pnl", "sum"),
    ).round(2)
    monthly["win_rate"] = (monthly["win_rate"] * 100).round(1).astype(str) + "%"
    print("Monthly breakdown:")
    print(monthly.to_string())

    return df


def _max_drawdown(bankroll_series: pd.Series) -> float:
    peak = bankroll_series.cummax()
    return (peak - bankroll_series).max()


if __name__ == "__main__":
    run()
