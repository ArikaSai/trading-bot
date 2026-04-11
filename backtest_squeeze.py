"""
backtest_squeeze.py
═══════════════════
BB Squeeze 波動率壓縮爆發策略回測

策略邏輯：
  1. 偵測 BB Squeeze：布林帶（BB）縮進 Keltner Channel（KC）時為壓縮狀態
  2. Squeeze 解除（前根壓縮、本根擴張）時產生進場訊號
  3. 方向判斷：動量指標（收盤 - 近期高低中軸 - KC中線）
     正動量 → 做多；負動量 → 做空
  4. 出場：ATR Trailing Stop

用法:
    python backtest_squeeze.py                        # 預設 XRP 1h
    python backtest_squeeze.py --symbol BTC
    python backtest_squeeze.py --symbol SOL --timeframe 15m
    python backtest_squeeze.py --use_trend 1          # 開啟 EMA200 趨勢過濾
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

DEFAULTS = dict(
    bb_period   = 20,     # 布林帶週期
    bb_std      = 2.0,    # 布林帶標準差倍數
    kc_period   = 10,     # Keltner Channel 週期
    kc_mult     = 1.25,    # Keltner Channel ATR 倍數
    mom_period  = 12,     # 動量回溯週期
    trail_atr   = 3.5,    # ATR trailing 倍數
    atr_sl_mult = 2.0,    # 初始 SL = atr_sl_mult × ATR（倉位計算用）
    risk_pct    = 0.15,   # 每筆風險佔資金比例
    leverage    = 1,      # 槓桿
    fee_rate    = 0.0005, # taker 手續費
    slippage    = 0.001,  # 滑價
    initial_cap = 500,    # 初始資金
    use_trend   = 0,      # 1=開啟 EMA200 趨勢過濾
)


def _load_defaults_from_config():
    try:
        with open(Path(__file__).parent / 'config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        risk = config.get('risk', {})
        return {
            **DEFAULTS,
            'fee_rate':    risk.get('taker_fee_rate', DEFAULTS['fee_rate']),
            'initial_cap': risk.get('initial_capital', DEFAULTS['initial_cap']),
        }
    except FileNotFoundError:
        return dict(DEFAULTS)


def load_data(symbol: str, timeframe: str = '1h') -> pd.DataFrame:
    path = Path(__file__).parent / 'data' / f'{symbol}USDT_{timeframe}.csv'
    if not path.exists():
        raise FileNotFoundError(
            f"找不到資料檔: {path}\n請先執行 download_data.py"
        )
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df.iloc[:-1]


def compute_squeeze_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """計算 BB、Keltner Channel、動量、EMA200"""
    df = df.copy()
    bb_p  = cfg['bb_period']
    bb_s  = cfg['bb_std']
    kc_p  = cfg['kc_period']
    kc_m  = cfg['kc_mult']
    mom_p = cfg['mom_period']

    closes = df['close']
    highs  = df['high']
    lows   = df['low']

    # ── Bollinger Bands ─────────────────────────────────────────────
    bb_mid    = closes.rolling(bb_p).mean()
    bb_std_s  = closes.rolling(bb_p).std(ddof=0)
    bb_upper  = bb_mid + bb_s * bb_std_s
    bb_lower  = bb_mid - bb_s * bb_std_s

    # ── True Range & ATR（用於 KC 與止損） ──────────────────────────
    tr = pd.concat([
        highs - lows,
        (highs - closes.shift(1)).abs(),
        (lows  - closes.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(kc_p).mean()

    # ── Keltner Channel ─────────────────────────────────────────────
    kc_mid   = closes.rolling(kc_p).mean()
    kc_upper = kc_mid + kc_m * atr
    kc_lower = kc_mid - kc_m * atr

    # ── Squeeze 偵測：BB 縮進 KC 時為壓縮狀態 ───────────────────────
    squeeze = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(int)

    # ── 動量指標（Lazy Bear TTM Squeeze 動量簡化版） ──────────────────
    # mom = close - (近期高低點中值 + KC中線) / 2
    roll_high = highs.rolling(mom_p).max()
    roll_low  = lows.rolling(mom_p).min()
    mom = closes - ((roll_high + roll_low) / 2 + kc_mid) / 2

    # ── EMA 200（趨勢過濾用） ────────────────────────────────────────
    ema200 = closes.ewm(span=200, adjust=False).mean()

    df['ATR_sq']  = atr
    df['KC_mid']  = kc_mid
    df['squeeze'] = squeeze
    df['mom']     = mom
    df['EMA200']  = ema200

    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════
#  回測核心
# ══════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, cfg: dict) -> tuple[list, list]:
    trail_atr   = cfg['trail_atr']
    atr_sl_mult = cfg['atr_sl_mult']
    risk_pct    = cfg['risk_pct']
    leverage    = cfg['leverage']
    fee_rate    = cfg['fee_rate']
    slippage    = cfg['slippage']
    cap         = float(cfg['initial_cap'])
    use_trend   = bool(cfg.get('use_trend', 0))

    # 倉位狀態
    pos = 0; ep = 0.0; sz = 0.0
    tsl = 0.0; hp = 0.0; lp = float('inf')
    entry_time = None; entry_fee = 0.0

    # 下根開盤待命進場
    pending_dir = 0

    trades = []
    equity = []

    opens   = df['open'].values
    highs   = df['high'].values
    lows    = df['low'].values
    closes  = df['close'].values
    atrs    = df['ATR_sq'].values
    squeeze = df['squeeze'].values
    moms    = df['mom'].values
    emas    = df['EMA200'].values
    times   = df.index

    for i in range(1, len(df)):
        O  = opens[i];  H = highs[i]
        L  = lows[i];   C = closes[i]
        atr      = atrs[i]      # 本根收盤後的 ATR（用於 TSL 更新）
        atr_prev = atrs[i - 1]  # 上根收盤的 ATR（開盤時已知，用於進場計算）

        # ── 進場執行（上一根訊號，本根開盤市價單） ───────────────────
        if pos == 0 and pending_dir != 0:
            risk_per_unit = atr_sl_mult * atr_prev
            if risk_per_unit > 0:
                size = min(
                    (cap * risk_pct) / risk_per_unit,
                    (cap * leverage) / O,
                )
                if size > 0:
                    entry_p = O * (1 + slippage * pending_dir)
                    f = size * entry_p * fee_rate
                    cap -= f
                    pos = pending_dir; ep = entry_p; sz = size
                    entry_time = times[i]; entry_fee = f
                    if pos == 1:
                        hp = H; lp = float('inf')
                        tsl = ep - trail_atr * atr_prev
                    else:
                        lp = L; hp = 0.0
                        tsl = ep + trail_atr * atr_prev
            pending_dir = 0

        # ── ATR Trailing Stop 出場 ───────────────────────────────────
        if pos != 0:
            closed = False; xp = 0.0

            if pos == 1:
                hp  = max(hp, H)
                tsl = max(tsl, hp - trail_atr * atr)
                if L <= tsl:
                    xp = max(tsl, O) * (1 - slippage)
                    closed = True
            else:
                lp  = min(lp, L)
                tsl = min(tsl, lp + trail_atr * atr)
                if H >= tsl:
                    xp = min(tsl, O) * (1 + slippage)
                    closed = True

            if closed:
                gross = (xp - ep) * sz * pos
                x_fee = xp * sz * fee_rate
                net   = gross - entry_fee - x_fee
                cap  += gross - x_fee
                trades.append({
                    'Entry_Time':  entry_time,
                    'Exit_Time':   times[i],
                    'Type':        'Long' if pos == 1 else 'Short',
                    'Entry_Price': ep,
                    'Exit_Price':  xp,
                    'PnL':         net,
                    'Capital':     cap,
                })
                pos = 0; sz = 0.0; entry_fee = 0.0

        # ── 訊號偵測：Squeeze Fire ────────────────────────────────────
        if pos == 0:
            prev_sq = squeeze[i - 1]
            curr_sq = squeeze[i]
            # 前根壓縮、本根擴張 → 爆發訊號
            if prev_sq == 1 and curr_sq == 0:
                direction = 1 if moms[i] > 0 else -1

                if use_trend:
                    if direction == 1  and C < emas[i]:
                        direction = 0
                    elif direction == -1 and C > emas[i]:
                        direction = 0

                pending_dir = direction

        # ── 淨值快照 ─────────────────────────────────────────────────
        unr = (C - ep) * sz * pos - C * sz * fee_rate if pos != 0 else 0
        equity.append({'time': times[i], 'capital': cap + unr})

    # ── 強制平倉 ──────────────────────────────────────────────────────
    if pos != 0:
        xp    = closes[-1] * (1 - slippage if pos == 1 else 1 + slippage)
        gross = (xp - ep) * sz * pos
        x_fee = xp * sz * fee_rate
        net   = gross - entry_fee - x_fee
        cap  += gross - x_fee
        trades.append({
            'Entry_Time':  entry_time,
            'Exit_Time':   times[-1],
            'Type':        'Long' if pos == 1 else 'Short',
            'Entry_Price': ep,
            'Exit_Price':  xp,
            'PnL':         net,
            'Capital':     cap,
        })

    return trades, equity


# ══════════════════════════════════════════════════════════════════════
#  統計與輸出
# ══════════════════════════════════════════════════════════════════════

def print_stats(trades: list, equity: list, cfg: dict,
                symbol: str, timeframe: str):
    initial_cap = cfg['initial_cap']
    if not trades:
        print("  無交易紀錄")
        return

    df_t = pd.DataFrame(trades)
    df_e = pd.DataFrame(equity)

    wins   = df_t[df_t['PnL'] > 0]
    losses = df_t[df_t['PnL'] <= 0]
    pf     = (abs(wins['PnL'].sum() / losses['PnL'].sum())
              if len(losses) and losses['PnL'].sum() != 0 else float('inf'))
    avg_w  = wins['PnL'].mean()   if len(wins)   else 0.0
    avg_l  = losses['PnL'].mean() if len(losses) else 0.0
    final  = df_t['Capital'].iloc[-1]
    peak   = df_e['capital'].cummax()
    mdd    = ((df_e['capital'] - peak) / peak * 100).min()

    trend_str = '開啟(EMA200)' if cfg.get('use_trend') else '關閉'

    print(f"\n{'='*62}")
    print(f"  BB Squeeze — {symbol}/USDT {timeframe}")
    print(f"  BB({cfg['bb_period']},{cfg['bb_std']:.1f})  KC({cfg['kc_period']},{cfg['kc_mult']:.1f})  Mom({cfg['mom_period']})")
    print(f"  Trail×{cfg['trail_atr']}  SL×{cfg['atr_sl_mult']}  風險{cfg['risk_pct']*100:.0f}%  "
          f"槓桿{cfg['leverage']}×  趨勢過濾:{trend_str}")
    print(f"{'='*62}")
    print(f"  交易次數:  {len(df_t)}  (多:{len(df_t[df_t['Type']=='Long'])} / 空:{len(df_t[df_t['Type']=='Short'])})")
    print(f"  勝率:      {len(wins)/len(df_t)*100:.1f}%")
    print(f"  利潤因子:  {pf:.2f}")
    print(f"  盈虧比:    {abs(avg_w/avg_l):.2f}" if avg_l else "  盈虧比:    ∞")
    print(f"  總損益:    {df_t['PnL'].sum():+,.2f}")
    print(f"  最終資金:  {final:,.2f}")
    print(f"  報酬率:    {(final/initial_cap-1)*100:+.1f}%")
    print(f"  最大回撤:  {mdd:.1f}%")
    print(f"{'='*62}")

    # ── 年度明細 ──────────────────────────────────────────────────────
    df_t['Year'] = pd.to_datetime(df_t['Exit_Time']).dt.year
    years = sorted(df_t['Year'].unique())
    W = 68
    print(f"\n  {'─'*W}")
    print(f"  年度明細")
    print(f"  {'─'*W}")
    print(f"  {'年度':>4}  {'交易':>5}  {'勝率%':>6}  {'損益':>14}  {'PF':>5}  {'盈虧比':>6}")
    print(f"  {'─'*W}")
    for y in years:
        yt = df_t[df_t['Year'] == y]
        yw = yt[yt['PnL'] > 0]; yl = yt[yt['PnL'] <= 0]
        ypf = (abs(yw['PnL'].sum() / yl['PnL'].sum())
               if len(yl) and yl['PnL'].sum() != 0 else float('inf'))
        yrr = abs(yw['PnL'].mean() / yl['PnL'].mean()) if len(yl) and len(yw) else 0
        print(f"  {y:>4}  {len(yt):>5}  {len(yw)/len(yt)*100:>6.1f}  "
              f"{yt['PnL'].sum():>+14,.2f}  {ypf:>5.2f}  {yrr:>6.2f}")
    print(f"  {'─'*W}")


def plot_equity(equity: list, trades: list,
                symbol: str, timeframe: str, cfg: dict):
    if not equity:
        return

    df_e = pd.DataFrame(equity)
    df_t = pd.DataFrame(trades) if trades else pd.DataFrame()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [3, 1]}
    )

    ax1.plot(df_e['time'], df_e['capital'].clip(lower=1.0),
             linewidth=1.2, color='#2f8ccb')
    ax1.axhline(cfg['initial_cap'], color='gray', linestyle='--',
                alpha=0.4, linewidth=0.8)
    trend_str = ' + EMA200趨勢' if cfg.get('use_trend') else ''
    ax1.set_title(
        f"BB Squeeze{trend_str} — {symbol}/USDT {timeframe}  |  "
        f"BB({cfg['bb_period']},{cfg['bb_std']:.1f})  "
        f"KC({cfg['kc_period']},{cfg['kc_mult']:.1f})  "
        f"Mom({cfg['mom_period']})",
        fontsize=13
    )
    ax1.set_ylabel('資金 (USDT)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    if not df_t.empty:
        df_t['Exit_Time'] = pd.to_datetime(df_t['Exit_Time'])
        colors = ['#2ecc71' if p > 0 else '#e74c3c' for p in df_t['PnL']]
        ax2.bar(df_t['Exit_Time'], df_t['PnL'],
                color=colors, alpha=0.75, width=0.8)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_ylabel('單筆損益')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'squeeze_{symbol}_{timeframe}.png'
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"\n  圖表已儲存: {fname}")


# ══════════════════════════════════════════════════════════════════════
#  主程式
# ══════════════════════════════════════════════════════════════════════

def main():
    defaults = _load_defaults_from_config()

    try:
        with open(Path(__file__).parent / 'config.json', 'r', encoding='utf-8') as f:
            _cfg = json.load(f)
        bt_start = _cfg.get('backtest', {}).get('start_date', '')
        bt_end   = _cfg.get('backtest', {}).get('end_date', '')
    except FileNotFoundError:
        bt_start = bt_end = ''

    parser = argparse.ArgumentParser(description='BB Squeeze Backtest')
    parser.add_argument('--symbol',    default='DOGE', help='幣種')
    parser.add_argument('--timeframe', default='1h',  help='時間框架')
    parser.add_argument('--start',     default=bt_start)
    parser.add_argument('--end',       default=bt_end)
    for k, v in defaults.items():
        t = float if isinstance(v, float) else int if isinstance(v, int) else str
        parser.add_argument(f'--{k}', type=t, default=v)
    args = parser.parse_args()

    symbol    = args.symbol.upper()
    timeframe = args.timeframe
    cfg = {k: getattr(args, k) for k in defaults}

    df = load_data(symbol, timeframe)
    if args.start:
        df = df[df.index >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df.index <= pd.Timestamp(args.end)]

    print(f"\nBB Squeeze 回測 | {symbol}/USDT {timeframe}")
    print(f"  BB({cfg['bb_period']},{cfg['bb_std']:.1f})  KC({cfg['kc_period']},{cfg['kc_mult']:.1f})  Mom({cfg['mom_period']})")
    print(f"  趨勢過濾: {'開啟(EMA200)' if cfg['use_trend'] else '關閉'}")
    print(f"  資料: {df.index[0].date()} ~ {df.index[-1].date()}  ({len(df):,} 根 K 棒)")

    df_ind = compute_squeeze_indicators(df, cfg)
    squeeze_pct  = df_ind['squeeze'].mean() * 100
    fire_count   = int(((df_ind['squeeze'].shift(1) == 1) & (df_ind['squeeze'] == 0)).sum())
    print(f"  壓縮比例: {squeeze_pct:.1f}%  |  Squeeze Fire 次數: {fire_count}")

    trades, equity = run_backtest(df_ind, cfg)
    print_stats(trades, equity, cfg, symbol, timeframe)
    plot_equity(equity, trades, symbol, timeframe, cfg)


if __name__ == '__main__':
    main()
