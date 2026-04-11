"""
squeeze_sweep.py
════════════════
BB Squeeze 策略參數掃描

掃描維度：kc_period / kc_mult / trail_atr / bb_std（MomP 固定 12）
目的：確認策略穩健性 — 結果應呈「高原」而非「尖峰」

用法:
    python squeeze_sweep.py                  # 預設 LINK 1h
    python squeeze_sweep.py --symbol DOGE
    python squeeze_sweep.py --symbol AVAX --top 30
    python squeeze_sweep.py --cross          # 跨幣種穩健性測試
"""

import argparse
import itertools
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ── 掃描範圍 ─────────────────────────────────────────────────────────
KC_PERIOD_RANGE = [10, 15, 20, 25]
KC_MULT_RANGE   = [1.0, 1.25, 1.5, 1.75, 2.0]
TRAIL_ATR_RANGE = [2.0, 2.5, 3.0, 3.5, 4.0]
BB_STD_RANGE    = [1.5, 2.0, 2.5]

# ── 固定參數（MomP 不影響結果，固定為 12）────────────────────────────
FIXED = dict(
    bb_period   = 20,
    mom_period  = 12,
    atr_sl_mult = 2.0,
    risk_pct    = 0.15,
    leverage    = 1,
    fee_rate    = 0.0005,
    slippage    = 0.001,
    initial_cap = 500,
)

try:
    with open(Path(__file__).parent / 'config.json', 'r', encoding='utf-8') as _f:
        _cfg = json.load(_f)
    FIXED['fee_rate']    = _cfg.get('risk', {}).get('taker_fee_rate', FIXED['fee_rate'])
    FIXED['initial_cap'] = _cfg.get('risk', {}).get('initial_capital', FIXED['initial_cap'])
except FileNotFoundError:
    pass


def load_data(symbol: str, timeframe: str = '1h') -> pd.DataFrame:
    path = Path(__file__).parent / 'data' / f'{symbol}USDT_{timeframe}.csv'
    if not path.exists():
        raise FileNotFoundError(f"找不到: {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df.iloc[:-1]


def _precompute_base(df: pd.DataFrame) -> dict:
    """預計算不隨參數變動的基礎序列（原始 TR）"""
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values

    # True Range（kc_period 不同時 ATR/kc_mid 需重算，只預存 TR）
    prev_c = np.empty_like(closes)
    prev_c[0] = closes[0]
    prev_c[1:] = closes[:-1]
    tr = np.maximum.reduce([highs - lows,
                             np.abs(highs - prev_c),
                             np.abs(lows  - prev_c)])

    return {'tr': tr,
            'closes': closes, 'highs': highs, 'lows': lows,
            'opens': df['open'].values}


def run_backtest(base: dict, kc_period: int, kc_mult: float,
                 trail_atr: float, bb_std: float) -> dict:
    bb_p        = FIXED['bb_period']
    mom_period  = FIXED['mom_period']
    atr_sl_mult = FIXED['atr_sl_mult']
    risk_pct    = FIXED['risk_pct']
    leverage    = FIXED['leverage']
    fee_rate    = FIXED['fee_rate']
    slippage    = FIXED['slippage']
    cap         = float(FIXED['initial_cap'])

    closes  = base['closes']
    highs   = base['highs']
    lows    = base['lows']
    opens   = base['opens']
    tr      = base['tr']
    n       = len(closes)

    # ── 預計算指標（含 kc_period 相關序列）────────────────────────
    close_s = pd.Series(closes)

    # ATR 與 KC 中線（依 kc_period 計算）
    atr_arr = pd.Series(tr).rolling(kc_period).mean().values
    kc_mid  = close_s.rolling(kc_period).mean().values

    # BB
    bb_mid_s  = close_s.rolling(bb_p).mean()
    bb_std_s  = close_s.rolling(bb_p).std(ddof=0)
    bb_upper  = (bb_mid_s + bb_std * bb_std_s).values
    bb_lower  = (bb_mid_s - bb_std * bb_std_s).values

    # KC bounds
    kc_upper = kc_mid + kc_mult * atr_arr
    kc_lower = kc_mid - kc_mult * atr_arr

    # Squeeze
    squeeze = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(np.int8)

    # Momentum
    roll_h = pd.Series(highs).rolling(mom_period).max().values
    roll_l = pd.Series(lows).rolling(mom_period).min().values
    mom    = closes - ((roll_h + roll_l) / 2 + kc_mid) / 2

    # ── 回測循環 ──────────────────────────────────────────────────
    pos = 0; ep = 0.0; sz = 0.0; tsl = 0.0
    hp = 0.0; lp = float('inf'); entry_fee = 0.0
    pending_dir = 0
    wins = 0; losses = 0
    total_win = 0.0; total_loss = 0.0
    peak = cap; max_dd = 0.0

    for i in range(1, n):
        O = opens[i];  H = highs[i]
        L = lows[i];   C = closes[i]
        atr      = atr_arr[i]      # 本根收盤後 ATR（用於 TSL 更新）
        atr_prev = atr_arr[i - 1]  # 上根收盤 ATR（開盤時已知，用於進場計算）
        if np.isnan(atr) or atr <= 0 or np.isnan(atr_prev) or atr_prev <= 0:
            continue

        # 進場執行
        if pos == 0 and pending_dir != 0:
            rpu = atr_sl_mult * atr_prev
            if rpu > 0:
                size = min((cap * risk_pct) / rpu,
                           (cap * leverage) / O)
                if size > 0:
                    entry_p = O * (1 + slippage * pending_dir)
                    f = size * entry_p * fee_rate
                    cap -= f
                    pos = pending_dir; ep = entry_p; sz = size; entry_fee = f
                    if pos == 1:
                        hp = H; lp = float('inf')
                        tsl = ep - trail_atr * atr_prev
                    else:
                        lp = L; hp = 0.0
                        tsl = ep + trail_atr * atr_prev
            pending_dir = 0

        # 出場
        if pos != 0:
            closed = False; xp = 0.0
            if pos == 1:
                hp  = max(hp, H)
                tsl = max(tsl, hp - trail_atr * atr)
                if L <= tsl:
                    xp = max(tsl, O) * (1 - slippage); closed = True
            else:
                lp  = min(lp, L)
                tsl = min(tsl, lp + trail_atr * atr)
                if H >= tsl:
                    xp = min(tsl, O) * (1 + slippage); closed = True

            if closed:
                gross = (xp - ep) * sz * pos
                x_fee = xp * sz * fee_rate
                net   = gross - entry_fee - x_fee
                cap  += gross - x_fee
                if net > 0:
                    wins += 1; total_win += net
                else:
                    losses += 1; total_loss += net
                pos = 0; sz = 0.0; entry_fee = 0.0

        # 訊號偵測
        if pos == 0 and not np.isnan(squeeze[i]) and not np.isnan(squeeze[i-1]):
            if squeeze[i-1] == 1 and squeeze[i] == 0:
                pending_dir = 1 if mom[i] > 0 else -1

        # MDD 追蹤
        equity = cap + ((C - ep) * sz * pos - C * sz * fee_rate if pos != 0 else 0)
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak if peak > 0 else 0
        if dd < max_dd:
            max_dd = dd

    # 強制平倉
    if pos != 0:
        xp    = closes[-1] * (1 - slippage if pos == 1 else 1 + slippage)
        gross = (xp - ep) * sz * pos
        x_fee = xp * sz * fee_rate
        net   = gross - entry_fee - x_fee
        cap  += gross - x_fee
        if net > 0:
            wins += 1; total_win += net
        else:
            losses += 1; total_loss += net

    total = wins + losses
    wr  = wins / total * 100 if total > 0 else 0
    pf  = abs(total_win / total_loss) if total_loss != 0 else float('inf')
    avg_w = total_win  / wins   if wins   > 0 else 0
    avg_l = total_loss / losses if losses > 0 else 0
    rr  = abs(avg_w / avg_l) if avg_l != 0 else 0

    return {
        'KC_period': kc_period,
        'KC_mult':   kc_mult,
        'Trail':     trail_atr,
        'BB_std':    bb_std,
        'Trades':   total,
        'WinR%':    round(wr, 1),
        'PF':       round(pf, 2),
        'RR':       round(rr, 2),
        'Final':    round(cap, 2),
        'Ret%':     round((cap / FIXED['initial_cap'] - 1) * 100, 1),
        'MDD%':     round(max_dd * 100, 1),
    }


def run_sweep(df: pd.DataFrame, symbol: str, timeframe: str, top_n: int):
    base   = _precompute_base(df)
    combos = list(itertools.product(
        KC_PERIOD_RANGE, KC_MULT_RANGE, TRAIL_ATR_RANGE, BB_STD_RANGE
    ))

    print(f"BB Squeeze 參數掃描 | {symbol}/USDT {timeframe}")
    print(f"  資料: {df.index[0].date()} ~ {df.index[-1].date()}  ({len(df):,} 根 K 棒)")
    print(f"  固定: BB週期{FIXED['bb_period']}  MomP{FIXED['mom_period']}  "
          f"SL×{FIXED['atr_sl_mult']}  風險{FIXED['risk_pct']*100:.0f}%  槓桿{FIXED['leverage']}×")
    print(f"  掃描 {len(combos)} 種組合...\n")

    results = []
    for j, (kc_p, kc_m, ta, bs) in enumerate(combos, 1):
        r = run_backtest(base, kc_p, kc_m, ta, bs)
        results.append(r)
        if j % 100 == 0 or j == len(combos):
            print(f"  進度: {j}/{len(combos)}")

    df_r = pd.DataFrame(results)

    # ── 1. 依利潤因子排序 ────────────────────────────────────────
    df_pf = df_r.sort_values('PF', ascending=False).reset_index(drop=True)
    df_pf.index += 1
    print(f"\n{'='*90}")
    print(f"  前 {top_n} 名（依利潤因子）— {symbol}/USDT {timeframe}")
    print(f"{'='*90}")
    print(df_pf.head(top_n).to_string())

    # ── 2. 依最終資金排序 ────────────────────────────────────────
    df_cap = df_r.sort_values('Final', ascending=False).reset_index(drop=True)
    df_cap.index += 1
    print(f"\n{'='*90}")
    print(f"  前 {top_n} 名（依最終資金）— {symbol}/USDT {timeframe}")
    print(f"{'='*90}")
    print(df_cap.head(top_n).to_string())

    # ── 3. 綜合排名：PF + MDD ─────────────────────────────────────
    df_r['PF_rank']  = df_r['PF'].rank(ascending=False)
    df_r['MDD_rank'] = df_r['MDD%'].abs().rank(ascending=True)
    df_r['Score']    = df_r['PF_rank'] + df_r['MDD_rank']
    df_comp = df_r.sort_values('Score').reset_index(drop=True)
    df_comp.index += 1
    print(f"\n{'='*90}")
    print(f"  前 {top_n} 名（綜合：PF + MDD）— {symbol}/USDT {timeframe}")
    print(f"{'='*90}")
    print(df_comp.drop(columns=['PF_rank', 'MDD_rank', 'Score']).head(top_n).to_string())

    # ── 4. 參數高原分析 ──────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  參數高原分析（各維度平均利潤因子）")
    print(f"{'='*90}")

    for col, label in [('KC_period', 'KC 週期'), ('KC_mult', 'KC 倍數'),
                       ('Trail', 'Trail ATR'), ('BB_std', 'BB 標準差')]:
        print(f"\n  {label}:")
        grp = df_r.groupby(col).agg(
            平均PF  =('PF',   'mean'),
            平均MDD =('MDD%', 'mean'),
            平均報酬=('Ret%', 'mean'),
            盈利組合=('PF',   lambda x: (x > 1.0).sum()),
            總組合  =('PF',   'count'),
        ).round(2)
        grp['盈利率%'] = (grp['盈利組合'] / grp['總組合'] * 100).round(1)
        print(grp.to_string())

    # ── 5. 總計 ──────────────────────────────────────────────────
    total      = len(df_r)
    profitable = (df_r['PF'] > 1.0).sum()
    print(f"\n  總計: {total} 組合 | 盈利: {profitable} ({profitable/total*100:.1f}%) | "
          f"虧損: {total-profitable} ({(total-profitable)/total*100:.1f}%)")


def run_cross_symbol(timeframe: str = '1h'):
    """用代表性參數跨幣種測試，檢驗策略普適性"""
    candidates = ['LINK', 'DOGE', 'AVAX', 'SOL', 'BTC', 'ETH', 'BNB',
                  'XRP', 'ADA', 'DOT', 'MATIC', 'LTC', 'ATOM']
    available  = [s for s in candidates
                  if (Path(__file__).parent / 'data' / f'{s}USDT_{timeframe}.csv').exists()]

    # 代表性參數組合（kc_period / kc_mult / trail / bb_std）
    test_params = [
        (10, 1.5, 3.0, 2.0),
        (10, 1.8, 2.5, 2.0),
        (15, 1.5, 3.0, 2.0),
        (20, 1.5, 3.0, 2.0),
        (20, 1.75, 2.5, 2.5),
    ]

    print(f"\n{'='*90}")
    print(f"  跨幣種穩健性測試  {len(available)} 幣種 × {len(test_params)} 組參數  ({timeframe})")
    print(f"{'='*90}")

    for kc_p, kc_m, ta, bs in test_params:
        print(f"\n  KC({kc_p},{kc_m})  Trail×{ta}  BB_std{bs}")
        print(f"  {'幣種':<6} {'交易':>5} {'勝率%':>6} {'PF':>6} {'報酬%':>10} {'MDD%':>7}")
        print(f"  {'-'*48}")
        profitable = 0
        for s in available:
            try:
                df   = load_data(s, timeframe)
                base = _precompute_base(df)
                r    = run_backtest(base, kc_p, kc_m, ta, bs)
                tag  = ' +' if r['PF'] > 1.0 else ' -'
                print(f"  {s:<6} {r['Trades']:>5} {r['WinR%']:>5.1f}% "
                      f"{r['PF']:>6.2f} {r['Ret%']:>+9.1f}% {r['MDD%']:>+6.1f}%{tag}")
                if r['PF'] > 1.0:
                    profitable += 1
            except FileNotFoundError:
                print(f"  {s:<6} (無資料)")
        print(f"  盈利幣種: {profitable}/{len(available)}")


def main():
    parser = argparse.ArgumentParser(description='BB Squeeze Parameter Sweep')
    parser.add_argument('--symbol',    default='LINK', help='幣種')
    parser.add_argument('--timeframe', default='1h',   help='時間框架')
    parser.add_argument('--top',  type=int, default=20, help='顯示前 N 名')
    parser.add_argument('--cross', action='store_true', help='跨幣種穩健性測試')
    args = parser.parse_args()

    if args.cross:
        run_cross_symbol(args.timeframe)
        return

    symbol    = args.symbol.upper()
    timeframe = args.timeframe
    df = load_data(symbol, timeframe)
    run_sweep(df, symbol, timeframe, args.top)


if __name__ == '__main__':
    main()
