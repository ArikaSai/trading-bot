"""
fib_sweep.py
════════════
斐波納契回撤策略參數掃描

掃描維度：swing_n / fib_level / trail_atr / fib_tolerance
目的：確認策略穩健性 — 結果應呈「高原」而非「尖峰」

用法:
    python fib_sweep.py                     # 預設 SOL
    python fib_sweep.py --symbol BTC
    python fib_sweep.py --symbol ETH --top 30
    python fib_sweep.py --symbol SOL --cross   # 跨幣種測試
"""

import argparse
import itertools
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ── 掃描範圍 ─────────────────────────────────────────────────────────
SWING_N_RANGE    = [10, 15, 20, 30, 40, 55]
FIB_LEVEL_RANGE  = [0.236, 0.382, 0.500, 0.618, 0.786]
TRAIL_ATR_RANGE  = [2.0, 2.5, 3.0, 4.0, 5.0]
FIB_TOL_RANGE    = [0.003, 0.005, 0.010]

# ── 固定參數 ─────────────────────────────────────────────────────────
FIXED = dict(
    atr_sl_mult = 2.0,
    risk_pct    = 0.15,
    leverage    = 1,
    fee_rate    = 0.0005,
    slippage    = 0.001,
    initial_cap = 500,
)

# 從 config 讀取共用參數
try:
    with open(Path(__file__).parent / 'config.json', 'r', encoding='utf-8') as _f:
        _cfg = json.load(_f)
    FIXED['fee_rate']    = _cfg.get('risk', {}).get('taker_fee_rate', FIXED['fee_rate'])
    FIXED['initial_cap'] = _cfg.get('risk', {}).get('initial_capital', FIXED['initial_cap'])
except FileNotFoundError:
    pass


def load_data(symbol: str) -> pd.DataFrame:
    path = Path(__file__).parent / 'data' / f'{symbol}USDT_1h.csv'
    if not path.exists():
        raise FileNotFoundError(f"找不到: {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.iloc[:-1]
    return df


def run_backtest(df: pd.DataFrame, swing_n: int, fib_level: float,
                 trail_atr: float, fib_tol: float) -> dict:
    atr_sl_mult = FIXED['atr_sl_mult']
    risk_pct    = FIXED['risk_pct']
    leverage    = FIXED['leverage']
    fee_rate    = FIXED['fee_rate']
    slippage    = FIXED['slippage']
    cap         = FIXED['initial_cap']

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
    emas   = df['EMA'].values

    for i in range(swing_n, len(df)):
        O, H, L, C = opens[i], highs[i], lows[i], closes[i]
        atr = atrs[i]
        ema = emas[i]

        swing_high = np.max(highs[i - swing_n:i])
        swing_low  = np.min(lows[i - swing_n:i])
        swing_range = swing_high - swing_low

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
                gross = (xp - ep) * sz * pos
                x_fee = xp * sz * fee_rate
                net = gross - entry_fee - x_fee
                cap += gross - x_fee
                if net > 0:
                    wins += 1; total_win += net
                else:
                    losses += 1; total_loss += net
                pos = 0; sz = 0.0; entry_fee = 0.0

        # ── 進場：Fib 回撤 ───────────────────────────────────────
        if pos == 0 and swing_range > 0 and atr > 0:
            prev_C = closes[i - 1]
            prev_L = lows[i - 1]
            prev_H = highs[i - 1]

            direction = 0

            if C > ema:  # 多頭回撤
                fib_price = swing_high - fib_level * swing_range
                tol = fib_price * fib_tol
                if prev_L <= fib_price + tol and prev_C > fib_price:
                    direction = 1
            elif C < ema:  # 空頭反彈
                fib_price = swing_low + fib_level * swing_range
                tol = fib_price * fib_tol
                if prev_H >= fib_price - tol and prev_C < fib_price:
                    direction = -1

            if direction != 0:
                risk_per_unit = atr_sl_mult * atr
                if risk_per_unit <= 0:
                    continue
                pos = direction
                ep = C * (1 + slippage * direction)
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
        gross = (xp - ep) * sz * pos
        x_fee = xp * sz * fee_rate
        net = gross - entry_fee - x_fee
        cap += gross - x_fee
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
        'Swing': swing_n, 'Fib%': fib_level, 'Trail': trail_atr, 'Tol%': fib_tol,
        'Trades': total_trades, 'WinR%': round(win_rate, 1),
        'PF': round(pf, 2), 'Payoff': round(payoff, 2),
        'Final': round(cap, 2), 'Ret%': round((cap / FIXED['initial_cap'] - 1) * 100, 1),
        'MDD%': round(max_dd * 100, 1),
    }


def run_cross_symbol(top_n: int = 5):
    """用預設參數跨幣種測試，檢驗策略普適性"""
    symbols = ['SOL', 'BTC', 'ETH', 'ADA', 'BNB', 'XRP', 'DOGE', 'AVAX']
    available = []
    for s in symbols:
        path = Path(__file__).parent / 'data' / f'{s}USDT_1h.csv'
        if path.exists():
            available.append(s)

    # 用幾組代表性參數
    test_params = [
        (20, 0.618, 3.0, 0.005),
        (20, 0.500, 3.0, 0.005),
        (20, 0.382, 3.0, 0.005),
        (15, 0.618, 3.0, 0.005),
        (30, 0.618, 3.0, 0.005),
    ]

    print(f"\n{'='*90}")
    print(f"  跨幣種穩健性測試（{len(available)} 幣種 x {len(test_params)} 組參數）")
    print(f"{'='*90}")

    for swing_n, fib_level, trail_atr, fib_tol in test_params:
        print(f"\n  Swing={swing_n} | Fib={fib_level*100:.1f}% | Trail=x{trail_atr}")
        print(f"  {'幣種':<6} {'交易數':>6} {'勝率%':>6} {'PF':>6} {'報酬%':>10} {'MDD%':>7}")
        print(f"  {'-'*48}")

        profitable = 0
        for s in available:
            df = load_data(s)
            r = run_backtest(df, swing_n, fib_level, trail_atr, fib_tol)
            tag = ' +' if r['PF'] > 1.0 else ' -'
            print(f"  {s:<6} {r['Trades']:>6} {r['WinR%']:>5.1f}% {r['PF']:>6.2f} "
                  f"{r['Ret%']:>+9.1f}% {r['MDD%']:>+6.1f}%{tag}")
            if r['PF'] > 1.0:
                profitable += 1

        print(f"  盈利幣種: {profitable}/{len(available)}")


def main():
    parser = argparse.ArgumentParser(description='Fibonacci Retracement Parameter Sweep')
    parser.add_argument('--symbol', default='SOL', help='幣種')
    parser.add_argument('--top', type=int, default=20, help='顯示前 N 名')
    parser.add_argument('--cross', action='store_true', help='跨幣種穩健性測試')
    args = parser.parse_args()

    if args.cross:
        run_cross_symbol(args.top)
        return

    symbol = args.symbol.upper()
    df = load_data(symbol)

    combos = list(itertools.product(SWING_N_RANGE, FIB_LEVEL_RANGE, TRAIL_ATR_RANGE, FIB_TOL_RANGE))
    print(f"Fib 回撤參數掃描 | {symbol}/USDT 1H")
    print(f"   資料: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df):,} 根 K 棒)")
    print(f"   固定: SL x{FIXED['atr_sl_mult']} | 風險 {FIXED['risk_pct']*100:.0f}% | "
          f"槓桿 {FIXED['leverage']}x | 初始 {FIXED['initial_cap']}")
    print(f"   掃描 {len(combos)} 種組合...\n")

    results = []
    for j, (sw, fl, ta, ft) in enumerate(combos, 1):
        r = run_backtest(df, sw, fl, ta, ft)
        results.append(r)
        if j % 50 == 0 or j == len(combos):
            print(f"   進度: {j}/{len(combos)}")

    df_r = pd.DataFrame(results)

    # ── 1. 依利潤因子排序 ────────────────────────────────────────
    df_pf = df_r.sort_values('PF', ascending=False).reset_index(drop=True)
    df_pf.index += 1
    print(f"\n{'='*95}")
    print(f"  前 {args.top} 名（依利潤因子排序）— {symbol}/USDT 1H")
    print(f"{'='*95}")
    print(df_pf.head(args.top).to_string())

    # ── 2. 依最終資金排序 ────────────────────────────────────────
    df_cap = df_r.sort_values('Final', ascending=False).reset_index(drop=True)
    df_cap.index += 1
    print(f"\n{'='*95}")
    print(f"  前 {args.top} 名（依最終資金排序）— {symbol}/USDT 1H")
    print(f"{'='*95}")
    print(df_cap.head(args.top).to_string())

    # ── 3. 綜合排名：PF + MDD ───────────────────────────────────
    df_r['PF_rank']  = df_r['PF'].rank(ascending=False)
    df_r['MDD_rank'] = df_r['MDD%'].abs().rank(ascending=True)
    df_r['Score']    = df_r['PF_rank'] + df_r['MDD_rank']
    df_comp = df_r.sort_values('Score', ascending=True).reset_index(drop=True)
    df_comp.index += 1
    print(f"\n{'='*95}")
    print(f"  前 {args.top} 名（綜合：利潤因子 + MDD 排名）— {symbol}/USDT 1H")
    print(f"{'='*95}")
    print(df_comp.drop(columns=['PF_rank', 'MDD_rank', 'Score']).head(args.top).to_string())

    # ── 4. 參數高原分析 ──────────────────────────────────────────
    print(f"\n{'='*95}")
    print(f"  參數高原分析（各維度平均利潤因子）")
    print(f"{'='*95}")

    for col, label, vals in [
        ('Swing', 'Swing N', SWING_N_RANGE),
        ('Fib%',  'Fib 水平', FIB_LEVEL_RANGE),
        ('Trail', 'Trail ATR', TRAIL_ATR_RANGE),
        ('Tol%',  '容差', FIB_TOL_RANGE),
    ]:
        print(f"\n  {label}:")
        grp = df_r.groupby(col).agg(
            平均PF=('PF', 'mean'),
            平均MDD=('MDD%', 'mean'),
            平均報酬=('Ret%', 'mean'),
            盈利組合=('PF', lambda x: (x > 1.0).sum()),
            總組合=('PF', 'count'),
        ).round(2)
        grp['盈利率%'] = (grp['盈利組合'] / grp['總組合'] * 100).round(1)
        print(grp.to_string())

    # ── 5. 虧損組合統計 ──────────────────────────────────────────
    total = len(df_r)
    profitable = (df_r['PF'] > 1.0).sum()
    print(f"\n  總計: {total} 組合 | 盈利: {profitable} ({profitable/total*100:.1f}%) | "
          f"虧損: {total - profitable} ({(total-profitable)/total*100:.1f}%)")


if __name__ == '__main__':
    main()
