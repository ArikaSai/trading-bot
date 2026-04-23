"""
event_contract_backtest.py
══════════════════════════
幣安事件合約回測 — 對應實盤 event_signal_bot.py 邏輯

訊號：hlpct_12 極端位置（≤10% 或 ≥90%）+ ATR_ratio < 1.20（同實盤）
死水：ADX < 15 直接跳過，不進場
下注：階梯式固定注，每 STEP_UNIT 本金升一階（每階 +BET_UNIT），上限依幣種

用法：
    python event_contract_backtest.py            # 預設 BTC+ETH 5m
    python event_contract_backtest.py --sym BTC
    python event_contract_backtest.py --start 2023-01-01 --end 2024-01-01
    python event_contract_backtest.py --sym BOTH --max_concurrent 5
"""

import argparse
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# ── 合約參數（對應實盤）────────────────────────────────────────
PAYOUT_30      = 0.85    # 30 分鐘合約賠率
DEFAULT_PAYOUT = PAYOUT_30   # 舊版相容名稱
SETTLE_BARS    = 2       # 30 分鐘 = 2 × 15m（5m 時自動 ×3 = 6）
INITIAL_CAP    = 1000.0
MAX_BET_ETH    = 125.0
MAX_BET_BTC    = 250.0
MIN_BET        = 5.0

# ── 市場分類閾值（對應 event_signal_bot.py）──────────────────────
ADX_TREND_THRESH = 25
ADX_DEAD_THRESH  = 15
ATR_DEAD_THRESH  = 0.80
VOLZ_DEAD_THRESH = -0.50
ATR_RATIO_MAX    = 1.20
THRESHOLD_LO     = 0.10
THRESHOLD_HI     = 0.90
EMA_FAST         = 20
EMA_SLOW         = 60

# ── 階梯式下注參數 ────────────────────────────────────────────
STEP_UNIT = 250    # 每 250U 本金升一階
BET_UNIT  = 5      # 每階增加 5U（5, 10, 15, 20...）
TIER_ZH = {
    'trend_aligned': '趨勢順勢',
    'trend_counter': '趨勢逆向',
    'ranging':       '震盪',
    'dead':          '死水',
}


# ══════════════════════════════════════════════════════════════
#  資料載入與指標計算
# ══════════════════════════════════════════════════════════════
def load_and_build(sym: str, tf: str = '5m', config_path="config.json",
                   start: str = None, end: str = None) -> pd.DataFrame:
    path = Path(__file__).parent / 'data' / f'{sym}USDT_{tf}.csv'
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.iloc[:-1].copy()

    bar_min = int(tf[:-1])
    scale   = 15 // bar_min   # 15m→1, 5m→3

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        bt = cfg.get('backtest', {})
        cfg_start = bt.get('start_date', '2020-01-01')
        cfg_end   = bt.get('end_date',   '2099-01-01')
    except Exception:
        cfg_start, cfg_end = '2020-01-01', '2099-01-01'

    t_start = pd.Timestamp(start if start else cfg_start)
    t_end   = pd.Timestamp(end   if end   else cfg_end)
    df = df[(df.index >= t_start) & (df.index <= t_end)]

    c = df['close']
    h = df['high']
    l = df['low']

    # ── 基礎指標 ───────────────────────────────────────────────
    df['RSI_14'] = df['RSI']
    df['MACD']        = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']

    bb_std = c.rolling(20).std(ddof=0)
    df['BB_upper'] = df['BB_Mid'] + 2 * bb_std
    df['BB_lower'] = df['BB_Mid'] - 2 * bb_std
    df['BB_pct']   = (c - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # ATR_ratio：當前 ATR / 近 60×5m 均 ATR（對應實盤 ATR_MA=60）
    df['ATR_ratio'] = df['ATR'] / df['ATR'].rolling(20 * scale).mean()

    # vol_z（對應實盤 VOL_Z_N=60）
    df['vol_z'] = (df['volume'] - df['volume'].rolling(20 * scale).mean()) / \
                  df['volume'].rolling(20 * scale).std(ddof=0)

    # hlpct_12（對應實盤 LOOKBACK_N=36 在 5m = 12×15m = 3小時）
    for n in [4, 8, 12]:
        rh = h.rolling(n * scale).max()
        rl = l.rolling(n * scale).min()
        df[f'hlpct_{n}'] = (c - rl) / (rh - rl).replace(0, np.nan)

    # ── 趨勢方向（對應實盤 EMA_FAST=20, EMA_SLOW=60）────────────
    # EMA span 固定不 scale，與實盤 calc_trend_dir() 一致
    df['ema20'] = c.ewm(span=EMA_FAST, adjust=False).mean()
    df['ema60'] = c.ewm(span=EMA_SLOW, adjust=False).mean()
    df['trend_dir'] = np.where(df['ema20'] > df['ema60'], 1, -1)

    # ── 市場狀態分類（對應實盤 classify_market()）────────────────
    is_dead     = df['ADX'] < ADX_DEAD_THRESH
    is_trending = (df['ADX'] >= ADX_TREND_THRESH) & (~is_dead)
    df['market_state'] = 'ranging'
    df.loc[is_dead,     'market_state'] = 'dead'
    df.loc[is_trending, 'market_state'] = 'trending'

    # 結算收盤價（預先移位，供模擬使用）
    settle_bars_5m = SETTLE_BARS * scale   # 30 min on this timeframe
    df['settle_30'] = c.shift(-settle_bars_5m)

    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════════════
#  訊號建構（對應實盤邏輯）
# ══════════════════════════════════════════════════════════════
def build_signal(df: pd.DataFrame) -> pd.Series:
    """
    hlpct_12 極端位置 + ATR_ratio 過濾，與實盤 check_signals() 完全一致。
    回傳 Series：1=做多, -1=做空, 0=不開單
    """
    pct12   = df['hlpct_12']
    atr_ok  = df['ATR_ratio'] < ATR_RATIO_MAX
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[atr_ok & (pct12 <= THRESHOLD_LO)] =  1
    sig[atr_ok & (pct12 >= THRESHOLD_HI)] = -1
    return sig


def classify_tier_series(df: pd.DataFrame, sig: pd.Series) -> pd.Series:
    """
    為每個有訊號的時間點分配 tier，對應實盤 classify_tier()。
    回傳 Series（index 同 sig，非訊號點為 NaN）。
    """
    ms = df['market_state']
    td = df['trend_dir']
    active = sig[sig != 0]
    tiers = {}
    for ts, d in active.items():
        m = ms.loc[ts]
        if m == 'dead':
            tiers[ts] = 'dead'
        elif m == 'ranging':
            tiers[ts] = 'ranging'
        else:
            t = td.loc[ts]
            tiers[ts] = ('trend_aligned'
                         if (d == 1 and t == 1) or (d == -1 and t == -1)
                         else 'trend_counter')
    return pd.Series(tiers)


# ══════════════════════════════════════════════════════════════
#  單幣快速統計（無 Kelly，直接看勝率）
# ══════════════════════════════════════════════════════════════
def calc_staged_bet(capital: float, symbol: str) -> float:
    """
    階梯式固定注：floor(本金 / STEP_UNIT) × BET_UNIT，最低 MIN_BET，上限依幣種。
    250U→5, 500U→10, 750U→15, 1000U→20 ...
    """
    step = int(capital // STEP_UNIT)
    bet  = max(MIN_BET, step * BET_UNIT)
    cap_map = {'BTC': MAX_BET_BTC, 'ETH': MAX_BET_ETH}
    return min(bet, cap_map.get(symbol, MAX_BET_ETH))


def quick_stats(df: pd.DataFrame, sig: pd.Series) -> dict | None:
    mask = (sig != 0) & df['settle_30'].notna()
    n = mask.sum()
    if n < 10:
        return None
    entry   = df.loc[mask, 'close']
    settle  = df.loc[mask, 'settle_30']
    d       = sig[mask]
    wins    = (((settle > entry) & (d ==  1)) |
               ((settle < entry) & (d == -1))) & (settle != entry)
    wr      = wins.sum() / n * 100
    be      = 1 / (1 + PAYOUT_30) * 100
    ev      = (wr / 100 * PAYOUT_30 - (1 - wr / 100)) * 100

    tiers   = classify_tier_series(df, sig)
    tier_stats = {}
    for tier in ['trend_aligned', 'trend_counter', 'ranging', 'dead']:
        idx = tiers[tiers == tier].index
        idx = [i for i in idx if i in wins.index]
        if not idx:
            continue
        tw = wins.loc[idx].sum()
        tn = len(idx)
        tier_stats[tier] = {'n': tn, 'wr': tw / tn * 100}

    return {'n': n, 'wr': wr, 'ev': ev, 'be': be, 'tier': tier_stats}


# ══════════════════════════════════════════════════════════════
#  BTC + ETH 分層 Kelly 合併回測（完全對應實盤）
# ══════════════════════════════════════════════════════════════
def build_events_tiered(df: pd.DataFrame, sym: str, max_bet: float) -> list:
    """
    建構含 tier 與指標值的事件列表，對應實盤訊號與分層邏輯。
    """
    close  = df['close']
    settle = df['settle_30']
    sig    = build_signal(df)
    ms_col = df['market_state']
    td_col = df['trend_dir']
    adx_col  = df['ADX']
    atr_col  = df['ATR_ratio']
    volz_col = df['vol_z']
    pct_col  = df['hlpct_12']

    events = []
    for ts, direction in sig[sig != 0].items():
        sp = settle.loc[ts]
        if pd.isna(sp):
            continue
        ep  = close.loc[ts]
        win = (((sp > ep) and direction ==  1) or
               ((sp < ep) and direction == -1)) and sp != ep

        ms = ms_col.loc[ts]
        td = td_col.loc[ts]
        if ms == 'dead':
            tier = 'dead'
        elif ms == 'ranging':
            tier = 'ranging'
        else:
            tier = ('trend_aligned'
                    if (direction == 1 and td == 1) or (direction == -1 and td == -1)
                    else 'trend_counter')

        if tier == 'dead':
            continue   # 死水盤（ADX < 15）跳過，不進場

        events.append({
            'time':         ts,
            'sym':          sym,
            'direction':    'LONG' if direction == 1 else 'SHORT',
            'market_state': ms,
            'trend_dir':    'up' if td == 1 else 'down',
            'tier':         tier,
            'max_bet':      max_bet,
            'win':          win,
            'entry_price':  ep,
            'settle_price': sp,
            'hlpct_12':     round(pct_col.loc[ts], 4),
            'adx':          round(adx_col.loc[ts], 2),
            'atr_ratio':    round(atr_col.loc[ts], 4),
            'vol_z':        round(volz_col.loc[ts], 4),
        })
    return events


def backtest_tiered_live(events_btc: list, events_eth: list,
                         bar_dur, max_concurrent: int = 5) -> dict | None:
    """
    分層 Half-Kelly 回測，完全對應實盤 event_signal_bot.py 邏輯：
    - dead（ADX < 15）→ 進場前已跳過，不出現在 events 裡
    - 其他 tier → 階梯式固定注（calc_staged_bet）
    回傳結果包含 'trades' 欄位（list of dict），可直接存 CSV。
    """
    events = sorted(events_btc + events_eth, key=lambda x: x['time'])
    if not events:
        return None

    cap          = INITIAL_CAP
    equity       = [cap]
    open_settles = []
    per_tier     = {t: {'n': 0, 'wins': 0, 'total_bet': 0.0}
                    for t in TIER_ZH}
    n_played     = n_wins = 0
    trades       = []

    for ev in events:
        ts   = ev['time']
        tier = ev['tier']
        win  = ev['win']

        if max_concurrent > 0:
            open_settles = [t for t in open_settles if t > ts]
            if len(open_settles) >= max_concurrent:
                continue
        if cap < MIN_BET:
            break

        bet  = calc_staged_bet(cap, ev['sym'])
        pnl  = bet * PAYOUT_30 if win else -bet
        cap += pnl
        equity.append(cap)
        n_played += 1
        if win:
            n_wins += 1

        trades.append({
            'time':         ts,
            'sym':          ev['sym'],
            'direction':    ev['direction'],
            'market_state': ev['market_state'],
            'trend_dir':    ev['trend_dir'],
            'tier':         tier,
            'hlpct_12':     ev['hlpct_12'],
            'adx':          ev['adx'],
            'atr_ratio':    ev['atr_ratio'],
            'vol_z':        ev['vol_z'],
            'entry_price':  ev['entry_price'],
            'settle_price': ev['settle_price'],
            'win':          win,
            'bet':          round(bet, 4),
            'pnl':          round(pnl, 4),
            'capital_after': round(cap, 4),
        })

        per_tier[tier]['n']         += 1
        per_tier[tier]['wins']      += 1 if win else 0
        per_tier[tier]['total_bet'] += bet
        if max_concurrent > 0:
            open_settles.append(ts + bar_dur)

    if n_played == 0:
        return None

    equity_s = pd.Series(equity)
    peak     = equity_s.cummax()
    return {
        'n':        n_played,
        'wr':       n_wins / n_played * 100,
        'final':    cap,
        'ret_pct':  (cap / INITIAL_CAP - 1) * 100,
        'mdd_pct':  ((equity_s - peak) / peak.replace(0, np.nan)).min() * 100,
        'equity':   equity_s,
        'per_tier': per_tier,
        'trades':   trades,
    }


# ══════════════════════════════════════════════════════════════
#  單幣分層 Kelly 回測（供 --sym BTC/ETH 使用）
# ══════════════════════════════════════════════════════════════
def backtest_single_tiered(df: pd.DataFrame, sym: str) -> dict | None:
    """單幣階梯式下注回測。"""
    close  = df['close']
    settle = df['settle_30']
    sig    = build_signal(df)
    ms_col = df['market_state']
    td_col = df['trend_dir']

    cap          = INITIAL_CAP
    equity       = [cap]
    per_tier     = {t: {'n': 0, 'wins': 0, 'total_bet': 0.0} for t in TIER_ZH}
    n_played     = n_wins = 0

    for ts, direction in sig[sig != 0].items():
        sp = settle.loc[ts]
        if pd.isna(sp):
            continue
        ep  = close.loc[ts]
        win = (((sp > ep) and direction == 1) or
               ((sp < ep) and direction == -1)) and sp != ep

        ms = ms_col.loc[ts]
        if ms == 'dead':
            continue   # 死水盤跳過
        elif ms == 'ranging':
            tier = 'ranging'
        else:
            td = td_col.loc[ts]
            tier = ('trend_aligned'
                    if (direction == 1 and td == 1) or (direction == -1 and td == -1)
                    else 'trend_counter')

        if cap < MIN_BET:
            break

        bet  = calc_staged_bet(cap, sym)
        pnl  = bet * PAYOUT_30 if win else -bet
        cap += pnl
        equity.append(cap)
        n_played += 1
        if win:
            n_wins += 1

        per_tier[tier]['n']         += 1
        per_tier[tier]['wins']      += 1 if win else 0
        per_tier[tier]['total_bet'] += bet

    if n_played == 0:
        return None

    equity_s = pd.Series(equity)
    peak     = equity_s.cummax()
    return {
        'n':        n_played,
        'wr':       n_wins / n_played * 100,
        'final':    cap,
        'ret_pct':  (cap / INITIAL_CAP - 1) * 100,
        'mdd_pct':  ((equity_s - peak) / peak.replace(0, np.nan)).min() * 100,
        'equity':   equity_s,
        'per_tier': per_tier,
    }


# ══════════════════════════════════════════════════════════════
#  主程式
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sym',            default='BOTH', help='ETH、BTC 或 BOTH')
    parser.add_argument('--tf',             default='5m',   help='K 棒時間框架（預設 5m）')
    parser.add_argument('--start',          default=None)
    parser.add_argument('--end',            default=None)
    parser.add_argument('--max_concurrent', type=int, default=5,
                        help='同時持倉上限（預設 5）')
    args = parser.parse_args()

    sym = args.sym.upper()
    tf  = args.tf
    mc  = args.max_concurrent
    be  = 1 / (1 + PAYOUT_30) * 100

    print(f"\n{'='*64}")
    print(f"  事件合約回測（對應實盤邏輯）｜{tf}｜30min 85%｜同時持倉 {mc}")
    print(f"  損益平衡勝率：{be:.1f}%  初始本金：{INITIAL_CAP:.0f} USDT")
    print(f"  期間：{args.start or 'config'} → {args.end or 'config'}")
    print(f"{'='*64}\n")

    syms_to_load = ['BTC', 'ETH'] if sym == 'BOTH' else [sym]
    dfs = {}
    for s in syms_to_load:
        print(f"  載入 {s} 資料...", end='', flush=True)
        df = load_and_build(s, tf=tf, start=args.start, end=args.end)
        dfs[s] = df
        print(f" {len(df):,} 根  [{df.index[0].date()} ~ {df.index[-1].date()}]")

    # ── 市場狀態分布 ─────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  市場狀態分布（K 棒比例）")
    print(f"  {'─'*60}")
    for s, df in dfs.items():
        total = len(df)
        n_d = (df['market_state'] == 'dead').sum()
        n_r = (df['market_state'] == 'ranging').sum()
        n_t = (df['market_state'] == 'trending').sum()
        print(f"  {s}/USDT：死水 {n_d/total*100:.1f}%  震盪 {n_r/total*100:.1f}%  趨勢 {n_t/total*100:.1f}%"
              f"  ({n_d:,} / {n_r:,} / {n_t:,} 根)")

    # ── 訊號快速統計 ─────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  訊號統計（hlpct12 極端 + ATR 過濾，30min）")
    print(f"  {'─'*60}")
    hdr = f"  {'幣種':<8} {'訊號數':>7} {'勝率%':>7} {'期望值':>8} {'損益平衡':>9}"
    print(hdr)
    print(f"  {'─'*50}")
    for s, df in dfs.items():
        sig = build_signal(df)
        st  = quick_stats(df, sig)
        if st:
            flag = '★' if st['wr'] >= be + 2 else ('✓' if st['wr'] >= be else ' ')
            print(f"  {flag} {s+'/USDT':<8} {st['n']:>7,} {st['wr']:>6.1f}%"
                  f" {st['ev']:>+7.2f}%   {st['be']:>6.1f}%")
            for tier in ['trend_aligned', 'trend_counter', 'ranging', 'dead']:
                ts_t = st['tier'].get(tier)
                if ts_t:
                    print(f"      {TIER_ZH[tier]:<8}：{ts_t['n']:>6,} 筆  勝率 {ts_t['wr']:.1f}%")
    print(f"  {'─'*50}")

    # ── BTC + ETH 合併分層 Kelly 模擬 ────────────────────────
    if sym == 'BOTH' and 'BTC' in dfs and 'ETH' in dfs:
        bar_dur_30 = (dfs['BTC'].index[1] - dfs['BTC'].index[0]) * (SETTLE_BARS * (15 // int(tf[:-1])))

        print(f"\n  {'─'*60}")
        print(f"  BTC + ETH 合併模擬（分層 Half-Kelly，初始 {INITIAL_CAP:.0f} USDT）")
        print(f"  {'─'*60}")

        evts_btc = build_events_tiered(dfs['BTC'], 'BTC', MAX_BET_BTC)
        evts_eth = build_events_tiered(dfs['ETH'], 'ETH', MAX_BET_ETH)
        r = backtest_tiered_live(evts_btc, evts_eth, bar_dur_30, mc)

        if r:
            print(f"\n  總覽：{r['n']:,} 筆  勝率 {r['wr']:.1f}%  "
                  f"終值 {r['final']:,.0f} USDT  報酬 {r['ret_pct']:+.0f}%  MDD {r['mdd_pct']:+.1f}%")
            print(f"\n  {'Tier':<10} {'筆數':>7} {'勝率%':>7} {'平均下注':>10}")
            print(f"  {'─'*40}")
            for tier in ['trend_aligned', 'trend_counter', 'ranging', 'dead']:
                pt = r['per_tier'][tier]
                n  = pt['n']
                if n == 0:
                    continue
                wr_t     = pt['wins'] / n * 100
                avg_bet  = pt['total_bet'] / n
                be_flag  = '★' if wr_t >= be + 2 else ('✓' if wr_t >= be else ' ')
                print(f"  {be_flag} {TIER_ZH[tier]:<10} {n:>7,} {wr_t:>6.1f}%  {avg_bet:>8.1f} USDT")
            print(f"  {'─'*40}")

            # 繪圖
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle('事件合約回測（對應實盤）BTC+ETH 分層 Kelly', fontsize=12, fontweight='bold')

            ax = axes[0]
            ax.plot(r['equity'].values, color='#f39c12', linewidth=1.3,
                    label=f"分層Kelly  WR={r['wr']:.1f}%  MDD{r['mdd_pct']:.0f}%")
            ax.axhline(INITIAL_CAP, color='gray', linestyle=':', linewidth=0.7)
            ax.set_yscale('log')
            ax.set_title('合併資金曲線（對數）')
            ax.set_xlabel('交易筆數')
            ax.set_ylabel('資金 (USDT)')
            ax.legend(fontsize=8)

            ax2 = axes[1]
            tiers_plot = ['trend_aligned', 'trend_counter', 'ranging', 'dead']
            colors = ['#2ecc71', '#3498db', '#f39c12', '#95a5a6']
            ns   = [r['per_tier'][t]['n'] for t in tiers_plot]
            wrs  = [r['per_tier'][t]['wins'] / r['per_tier'][t]['n'] * 100
                    if r['per_tier'][t]['n'] > 0 else 0 for t in tiers_plot]
            labels = [TIER_ZH[t] for t in tiers_plot]
            bars = ax2.barh(labels, wrs, color=colors, alpha=0.8)
            ax2.axvline(be, color='red', linestyle='--', linewidth=1, label=f'損益平衡 {be:.1f}%')
            ax2.set_xlabel('勝率 %')
            ax2.set_title('各 Tier 勝率')
            ax2.legend(fontsize=8)
            for i, (_, n, wr) in enumerate(zip(bars, ns, wrs)):
                ax2.text(wr + 0.1, i, f'{wr:.1f}% ({n:,}筆)', va='center', fontsize=8)

            plt.tight_layout()
            out = 'event_contract_tiered_live.png'
            plt.savefig(out, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n  圖表已儲存：{out}")

            # ── CSV 輸出 ──────────────────────────────────────
            if r.get('trades'):
                csv_path = 'trades_tiered_live.csv'
                pd.DataFrame(r['trades']).to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"  交易明細已儲存：{csv_path}（{len(r['trades']):,} 筆）")

    else:
        # 單幣模式
        s  = syms_to_load[0]
        df = dfs[s]
        r  = backtest_single_tiered(df, s)
        if r:
            print(f"\n  {s}/USDT 分層 Kelly 結果：")
            print(f"  總覽：{r['n']:,} 筆  勝率 {r['wr']:.1f}%  "
                  f"終值 {r['final']:,.0f} USDT  報酬 {r['ret_pct']:+.0f}%  MDD {r['mdd_pct']:+.1f}%")
            print(f"\n  {'Tier':<10} {'筆數':>7} {'勝率%':>7} {'平均下注':>10}")
            print(f"  {'─'*40}")
            for tier in ['trend_aligned', 'trend_counter', 'ranging', 'dead']:
                pt = r['per_tier'][tier]
                n  = pt['n']
                if n == 0:
                    continue
                wr_t    = pt['wins'] / n * 100
                avg_bet = pt['total_bet'] / n
                be_flag = '★' if wr_t >= be + 2 else ('✓' if wr_t >= be else ' ')
                print(f"  {be_flag} {TIER_ZH[tier]:<10} {n:>7,} {wr_t:>6.1f}%  {avg_bet:>8.1f} USDT")
            print(f"  {'─'*40}")

            # 繪圖
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(r['equity'].values, color='#f39c12', linewidth=1.3)
            ax.axhline(INITIAL_CAP, color='gray', linestyle=':', linewidth=0.7)
            ax.set_yscale('log')
            ax.set_title(f'{s} 分層 Kelly 資金曲線  WR={r["wr"]:.1f}%  MDD{r["mdd_pct"]:.0f}%')
            ax.set_xlabel('交易筆數')
            ax.set_ylabel('資金 (USDT)')
            plt.tight_layout()
            out = f'event_contract_{s.lower()}_tiered.png'
            plt.savefig(out, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n  圖表已儲存：{out}")

    print()


if __name__ == '__main__':
    main()
