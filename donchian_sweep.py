"""
Donchian Channel 參數掃描
─────────────────────────
用法:
    python donchian_sweep.py                # 預設 ADA
    python donchian_sweep.py --symbol ETH
    python donchian_sweep.py --symbol BTC --top 30
"""

import argparse
import itertools
import pandas as pd
import numpy as np
from pathlib import Path

# ── 掃描範圍 ─────────────────────────────────────────────────────────
ENTRY_N_RANGE     = [10, 15, 20, 30, 40, 55]
TRAIL_ATR_RANGE   = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
ATR_SL_MULT_RANGE = [1.5, 2.0, 3.0]

# ── 固定參數 ─────────────────────────────────────────────────────────
FIXED = dict(
    risk_pct    = 0.15,
    leverage    = 1,
    fee_rate    = 0.0005,
    slippage    = 0.001,
    initial_cap = 700,
)


def load_data(symbol: str) -> pd.DataFrame:
    path = Path(__file__).parent / 'data' / f'{symbol}USDT_1h.csv'
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.iloc[:-1]
    return df


def run_backtest(df: pd.DataFrame, entry_n: int, trail_atr: float,
                 atr_sl_mult: float) -> dict:
    risk_pct = FIXED['risk_pct']
    leverage = FIXED['leverage']
    fee_rate = FIXED['fee_rate']
    slippage = FIXED['slippage']
    cap      = FIXED['initial_cap']

    pos = 0; ep = 0.0; sz = 0.0; tsl = 0.0
    hp = 0.0; lp = float('inf')
    entry_fee = 0.0
    wins = 0; losses = 0
    total_win = 0.0; total_loss = 0.0
    peak = cap; max_dd = 0.0

    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    opens  = df['open'].values
    atrs   = df['ATR'].values

    for i in range(entry_n, len(df)):
        O, H, L, C = opens[i], highs[i], lows[i], closes[i]
        atr = atrs[i]

        dc_high = np.max(highs[i - entry_n:i])
        dc_low  = np.min(lows[i - entry_n:i])

        # ── 出場 ─────────────────────────────────────────────────
        if pos != 0:
            closed = False; xp = 0.0
            if pos == 1:
                hp = max(hp, H)
                tsl = max(tsl, hp - trail_atr * atr)
                if L <= tsl:
                    xp = max(tsl, O) * (1 - slippage); closed = True
            elif pos == -1:
                lp = min(lp, L)
                tsl = min(tsl, lp + trail_atr * atr)
                if H >= tsl:
                    xp = min(tsl, O) * (1 + slippage); closed = True

            if closed:
                net = (xp - ep) * sz * pos - entry_fee - xp * sz * fee_rate
                cap += net
                if net > 0:
                    wins += 1; total_win += net
                else:
                    losses += 1; total_loss += net
                pos = 0; sz = 0.0; entry_fee = 0.0

        # ── 入場 ─────────────────────────────────────────────────
        if pos == 0:
            direction = 0
            if C > dc_high:
                direction = 1
            elif C < dc_low:
                direction = -1

            if direction != 0:
                pos = direction
                ep = C * (1 + slippage * direction)
                risk_per_unit = atr_sl_mult * atr
                if risk_per_unit <= 0:
                    pos = 0; continue
                raw_sz = (cap * risk_pct) / risk_per_unit
                max_sz = (cap * leverage) / ep
                sz = min(raw_sz, max_sz)
                entry_fee = ep * sz * fee_rate
                cap -= entry_fee
                if direction == 1:
                    hp = H; lp = float('inf')
                else:
                    lp = L; hp = 0.0
                tsl = ep - trail_atr * atr * direction

        # MDD 追蹤
        equity = cap + ((C - ep) * sz * pos - C * sz * fee_rate if pos != 0 else 0)
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak if peak > 0 else 0
        if dd < max_dd:
            max_dd = dd

    # 強制平倉
    if pos != 0:
        xp = closes[-1] * (1 - slippage if pos == 1 else 1 + slippage)
        net = (xp - ep) * sz * pos - entry_fee - xp * sz * fee_rate
        cap += net
        if net > 0:
            wins += 1; total_win += net
        else:
            losses += 1; total_loss += net

    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    pf = abs(total_win / total_loss) if total_loss != 0 else float('inf')
    avg_win = total_win / wins if wins > 0 else 0
    avg_loss = total_loss / losses if losses > 0 else 0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    return {
        'N': entry_n, 'Trail': trail_atr, 'SL_Mult': atr_sl_mult,
        'Trades': total_trades, 'WinRate%': round(win_rate, 1),
        'PF': round(pf, 2), 'Payoff': round(payoff, 2),
        'Final': round(cap, 2), 'Return%': round((cap / FIXED['initial_cap'] - 1) * 100, 1),
        'MDD%': round(max_dd * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser(description='Donchian Parameter Sweep')
    parser.add_argument('--symbol', default='ADA')
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()
    symbol = args.symbol.upper()

    df = load_data(symbol)
    combos = list(itertools.product(ENTRY_N_RANGE, TRAIL_ATR_RANGE, ATR_SL_MULT_RANGE))
    print(f"Donchian 參數掃描 | {symbol}/USDT 1H")
    print(f"   資料: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df):,} 根 K 棒)")
    print(f"   固定: 風險 {FIXED['risk_pct']*100:.0f}% | 槓桿 {FIXED['leverage']}x | 初始資金 {FIXED['initial_cap']}")
    print(f"   掃描 {len(combos)} 種組合...\n")

    results = []
    for j, (n, t, s) in enumerate(combos, 1):
        r = run_backtest(df, n, t, s)
        results.append(r)
        if j % 18 == 0 or j == len(combos):
            print(f"   進度: {j}/{len(combos)}")

    df_r = pd.DataFrame(results)

    # 按 PF 排序
    df_r.sort_values('PF', ascending=False, inplace=True)
    df_r.reset_index(drop=True, inplace=True)
    df_r.index += 1

    print(f"\n{'='*85}")
    print(f"  前 {args.top} 名（依利潤因子排序）— {symbol}/USDT 1H")
    print(f"{'='*85}")
    print(df_r.head(args.top).to_string())

    # 按最終資金排序
    df_r.sort_values('Final', ascending=False, inplace=True)
    df_r.reset_index(drop=True, inplace=True)
    df_r.index += 1

    print(f"\n{'='*85}")
    print(f"  前 {args.top} 名（依最終資金排序）— {symbol}/USDT 1H")
    print(f"{'='*85}")
    print(df_r.head(args.top).to_string())

    # 綜合排名：PF rank + MDD rank (越小越好)
    df_r['PF_rank'] = df_r['PF'].rank(ascending=False)
    df_r['MDD_rank'] = df_r['MDD%'].abs().rank(ascending=True)
    df_r['Score'] = df_r['PF_rank'] + df_r['MDD_rank']
    df_r.sort_values('Score', ascending=True, inplace=True)
    df_r.reset_index(drop=True, inplace=True)
    df_r.index += 1

    print(f"\n{'='*85}")
    print(f"  前 {args.top} 名（綜合：利潤因子 + MDD 排名）— {symbol}/USDT 1H")
    print(f"{'='*85}")
    print(df_r.drop(columns=['PF_rank', 'MDD_rank', 'Score']).head(args.top).to_string())


if __name__ == '__main__':
    main()
