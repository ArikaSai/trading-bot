"""
explore_60pct.py
════════════════
探索突破 60% 勝率的三種方向（基於 F1_RSI30_MACD 基準 58.5%）：

  1. 市場時段過濾（亞洲盤 / 歐洲盤 / 美盤）
  2. ADX 低（盤整市場）過濾
  3. 更長 hlpct 窗口（4h / 6h）

用法：
    python explore_60pct.py
    python explore_60pct.py --start 2024-01-01
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from event_contract_backtest import (
    load_and_build, backtest_combined,
    DEFAULT_PAYOUT, SETTLE_BARS,
    MAX_BET_BTC, MAX_BET_ETH, MIN_BET,
)

TF         = '5m'
BAR_MIN    = 5
TF_SCALE   = 15 // BAR_MIN          # 3
SETTLE_B   = SETTLE_BARS * TF_SCALE  # 6
PAYOUT     = DEFAULT_PAYOUT
BREAK_EVEN = 1 / (1 + PAYOUT) * 100
MAX_CONC   = 5


# ══════════════════════════════════════════════════════════════
#  共用：F1 + RSI30 + MACD 的基底 mask
# ══════════════════════════════════════════════════════════════
def f1_rsi30_macd_mask(df: pd.DataFrame) -> pd.Series:
    """回傳 F1_RSI30_MACD 的 Boolean mask（True = 此 bar 符合基準條件）"""
    pct12    = df['hlpct_12']
    base_l   = pct12 < 0.10
    base_s   = pct12 > 0.90
    f1       = df['ATR_ratio'] < 1.20
    rsi      = df['RSI_14']
    rsi30    = ((base_l & (rsi < 30)) | (base_s & (rsi > 70)))
    macd_ok  = ((base_l & (df['MACD_hist'] < 0)) |
                (base_s & (df['MACD_hist'] > 0)))
    return f1 & rsi30 & macd_ok


def base_signal(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    """給定 Boolean mask，產生 +1/-1/0 訊號 Series"""
    pct12 = df['hlpct_12']
    s = pd.Series(0, index=df.index)
    s[(pct12 < 0.10) & mask] =  1
    s[(pct12 > 0.90) & mask] = -1
    return s


# ══════════════════════════════════════════════════════════════
#  方向一：市場時段過濾
# ══════════════════════════════════════════════════════════════
def time_signals(df: pd.DataFrame) -> dict:
    """
    CST（UTC+8）時段劃分：
      亞洲盤   08:00–16:00
      歐洲盤   15:00–22:00（與亞尾重疊）
      美盤     21:00–04:00（跨日）
      亞歐重疊 15:00–16:00
      歐美重疊 21:00–22:00
    """
    mask_base = f1_rsi30_macd_mask(df)
    hour = df.index.hour   # index 已是 UTC，需加 8
    cst_hour = (hour + 8) % 24

    sessions = {
        'Asia  (08-16)': (cst_hour >= 8)  & (cst_hour < 16),
        'EU    (15-22)': (cst_hour >= 15) & (cst_hour < 22),
        'US    (21-04)': (cst_hour >= 21) | (cst_hour < 4),
        'Asia+EU(15-16)': (cst_hour >= 15) & (cst_hour < 16),
        'EU+US (21-22)': (cst_hour >= 21) & (cst_hour < 22),
    }
    sigs = {}
    for name, hour_mask in sessions.items():
        m = mask_base & hour_mask
        sigs[name] = base_signal(df, m)
    return sigs


# ══════════════════════════════════════════════════════════════
#  方向二：ADX 低（盤整）過濾
# ══════════════════════════════════════════════════════════════
def adx_signals(df: pd.DataFrame) -> dict:
    mask_base = f1_rsi30_macd_mask(df)
    sigs = {}
    for th in [15, 20, 25, 30]:
        m = mask_base & (df['ADX'] < th)
        sigs[f'ADX<{th}'] = base_signal(df, m)
    # 反向：只在趨勢期（ADX 高）進場
    sigs['ADX≥25'] = base_signal(df, mask_base & (df['ADX'] >= 25))
    return sigs


# ══════════════════════════════════════════════════════════════
#  方向三：更長 hlpct 窗口（4h / 6h）
# ══════════════════════════════════════════════════════════════
def hlpct_long_signals(df: pd.DataFrame) -> dict:
    """
    在 df 上動態計算更長窗口的 hlpct，不修改原 df。
    4h = 48×5m，6h = 72×5m
    """
    mask_f1  = df['ATR_ratio'] < 1.20
    rsi      = df['RSI_14']
    mh       = df['MACD_hist']
    sigs = {}

    for label, n_bars in [('4h', 48), ('6h', 72), ('8h', 96)]:
        rh  = df['high'].rolling(n_bars).max()
        rl  = df['low'].rolling(n_bars).min()
        pct = (df['close'] - rl) / (rh - rl).replace(0, np.nan)

        base_l = pct < 0.10
        base_s = pct > 0.90
        rsi30  = (base_l & (rsi < 30)) | (base_s & (rsi > 70))
        macd_ok = (base_l & (mh < 0)) | (base_s & (mh > 0))
        m = mask_f1 & rsi30 & macd_ok

        s = pd.Series(0, index=df.index)
        s[base_l & m] =  1
        s[base_s & m] = -1
        sigs[label] = s

    return sigs


# ══════════════════════════════════════════════════════════════
#  執行並顯示
# ══════════════════════════════════════════════════════════════
def run_group(title: str, sigs_btc: dict, sigs_eth: dict,
              df_btc: pd.DataFrame, df_eth: pd.DataFrame,
              min_n: int = 500):
    print(f"\n{'─'*72}")
    print(f"  {title}")
    print(f"{'─'*72}")
    hdr = (f"  {'名稱':<18} {'交易數':>7} {'勝率%':>7} {'期望值':>8} "
           f"{'模擬終值':>14} {'MDD%':>7}")
    print(hdr)
    print(f"  {'─'*64}")

    rows = []
    for name in sigs_btc:
        if name not in sigs_eth:
            continue
        r = backtest_combined(
            df_btc, sigs_btc[name],
            df_eth, sigs_eth[name],
            PAYOUT, SETTLE_B, max_concurrent=MAX_CONC)
        if r is None or r['n'] < min_n:
            continue
        r['name'] = name
        rows.append(r)

    if not rows:
        print("  （無足夠樣本的訊號）")
        return

    rows.sort(key=lambda x: x['win_rate'], reverse=True)
    for r in rows:
        flag = '★' if r['win_rate'] >= BREAK_EVEN else ' '
        hi   = '🔥' if r['win_rate'] >= 60.0 else ('⭐' if r['win_rate'] >= 59.0 else ' ')
        print(f"{flag} {hi} {r['name']:<16} {r['n']:>7,} {r['win_rate']:>6.1f}% "
              f"{r['ev']:>+7.2f}% {r['final']:>13,.0f} {r['mdd%']:>+6.1f}%")

    print(f"  {'─'*64}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default=None)
    parser.add_argument('--end',   default=None)
    parser.add_argument('--min_n', type=int, default=500,
                        help='最少交易筆數（預設 500）')
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  探索 60%+ 勝率  |  基準：F1_RSI30_MACD 58.5% / 24,972 筆")
    print(f"  損益平衡：{BREAK_EVEN:.1f}%  |  最少樣本：{args.min_n} 筆")
    print(f"{'='*72}")

    print("\n  載入資料...", end='', flush=True)
    df_btc = load_and_build('BTC', tf=TF, start=args.start, end=args.end)
    df_eth = load_and_build('ETH', tf=TF, start=args.start, end=args.end)
    print(f" BTC {len(df_btc):,} / ETH {len(df_eth):,} 根")

    # 基準（對照組）
    base_btc = {'F1_RSI30_MACD': base_signal(df_btc, f1_rsi30_macd_mask(df_btc))}
    base_eth = {'F1_RSI30_MACD': base_signal(df_eth, f1_rsi30_macd_mask(df_eth))}
    run_group('【基準】F1_RSI30_MACD（對照組）',
              base_btc, base_eth, df_btc, df_eth, min_n=1)

    # ADX≥25 基準組合
    def adx_combo(df):
        base  = f1_rsi30_macd_mask(df)
        adx   = df['ADX']
        pct12 = df['hlpct_12']
        rsi   = df['RSI_14']
        volz  = df['vol_z']
        # RSI 開始反轉：超賣後 RSI 已在回升（做多），超買後 RSI 已在回落（做空）
        rsi_turn = (((pct12 < 0.10) & (rsi > rsi.shift(1))) |
                    ((pct12 > 0.90) & (rsi < rsi.shift(1))))

        return {
            'ADX≥25（基準）':          base_signal(df, base & (adx >= 25)),
            'ADX 25-40':               base_signal(df, base & (adx >= 25) & (adx < 40)),
            'ADX≥25 + RSI轉向':        base_signal(df, base & (adx >= 25) & rsi_turn),
            'ADX≥25 + vol_z<1':        base_signal(df, base & (adx >= 25) & (volz < 1)),
            'ADX 25-40 + RSI轉向':     base_signal(df, base & (adx >= 25) & (adx < 40) & rsi_turn),
            'ADX 25-40 + vol_z<1':     base_signal(df, base & (adx >= 25) & (adx < 40) & (volz < 1)),
            'ADX≥25 + RSI轉向 + vol<1':base_signal(df, base & (adx >= 25) & rsi_turn & (volz < 1)),
        }

    run_group('【組合】ADX≥25 + F1_RSI30_MACD 延伸',
              adx_combo(df_btc), adx_combo(df_eth),
              df_btc, df_eth, min_n=1)

    # 方向一：時段
    run_group('【方向一】市場時段過濾',
              time_signals(df_btc), time_signals(df_eth),
              df_btc, df_eth, min_n=args.min_n)

    # 方向二：ADX
    run_group('【方向二】ADX 盤整過濾',
              adx_signals(df_btc), adx_signals(df_eth),
              df_btc, df_eth, min_n=args.min_n)

    # 方向三：更長窗口
    run_group('【方向三】更長 hlpct 窗口',
              hlpct_long_signals(df_btc), hlpct_long_signals(df_eth),
              df_btc, df_eth, min_n=args.min_n)

    print()


if __name__ == '__main__':
    main()
