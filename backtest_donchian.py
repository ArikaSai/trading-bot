"""
Donchian Channel Breakout 回測（ATR Trailing 出場）
───────────────────────────────────────────────────
用法:
    python backtest_donchian.py              # 預設 ETH
    python backtest_donchian.py --symbol BTC
    python backtest_donchian.py --symbol BTC --entry_n 55
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# ── 預設參數 ─────────────────────────────────────────────────────────
DEFAULTS = dict(
    entry_n        = 10,        # 入場通道長度
    trail_atr      = 3.0,      # ATR trailing 倍數
    atr_sl_mult    = 2.0,      # 初始 SL = atr_sl_mult × ATR（倉位計算用）
    risk_pct       = 0.15,     # 每筆風險佔資金比例
    leverage       = 1,        # 槓桿（1 = 現貨等效）
    fee_rate       = 0.0005,   # taker 手續費
    slippage       = 0.001,    # 滑價
    initial_cap    = 700,      # 初始資金
    max_trade_cap  = 200000.0, # 單筆最大名目價值（TWAP 拆單門檻）
)


def load_data(symbol: str) -> pd.DataFrame:
    path = Path(__file__).parent / 'data' / f'{symbol}USDT_1h.csv'
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.iloc[:-1]  # 去掉最後一筆未收盤
    return df


def run_backtest(df: pd.DataFrame, cfg: dict) -> tuple[list, list]:
    entry_n       = cfg['entry_n']
    trail_atr     = cfg['trail_atr']
    atr_sl_mult   = cfg['atr_sl_mult']
    risk_pct      = cfg['risk_pct']
    leverage      = cfg['leverage']
    fee_rate      = cfg['fee_rate']
    slippage      = cfg['slippage']
    cap           = cfg['initial_cap']
    max_trade_cap = cfg['max_trade_cap']

    pos = 0; ep = 0.0; sz = 0.0
    tsl = 0.0; hp = 0.0; lp = float('inf')
    entry_time = None; entry_fee = 0.0

    # 自適應 TWAP 狀態
    twap_active    = False
    twap_remaining = 0
    twap_size_each = 0.0
    twap_direction = 0
    twap_pending   = False   # 第一份子單待執行

    trades = []
    equity = []

    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    opens  = df['open'].values
    atrs   = df['ATR'].values
    times  = df.index

    for i in range(entry_n, len(df)):
        ts   = times[i]
        O, H, L, C = opens[i], highs[i], lows[i], closes[i]
        atr  = atrs[i]

        dc_high = np.max(highs[i - entry_n:i])
        dc_low  = np.min(lows[i - entry_n:i])

        # ── TWAP 第一份子單（訊號下一根 K 棒 open 執行）──────────
        if pos == 0 and twap_active and twap_pending:
            sub_ep = O * (1 + slippage * twap_direction)
            sub_fee = twap_size_each * sub_ep * fee_rate
            cap -= sub_fee
            entry_fee = sub_fee
            pos = twap_direction
            sz  = twap_size_each
            ep  = sub_ep
            entry_time = ts
            if pos == 1:
                hp = H; lp = float('inf')
                tsl = ep - trail_atr * atr
            else:
                lp = L; hp = 0.0
                tsl = ep + trail_atr * atr
            twap_remaining -= 1
            twap_pending = False

        # ── TWAP 後續子單（持倉中、每根新 K 棒追加一份）─────────
        elif pos != 0 and twap_active and twap_remaining > 0 and pos == twap_direction:
            sub_ep  = O * (1 + slippage * pos)
            sub_fee = twap_size_each * sub_ep * fee_rate
            cap -= sub_fee
            entry_fee += sub_fee
            total_sz = sz + twap_size_each
            ep = (ep * sz + sub_ep * twap_size_each) / total_sz
            sz = total_sz
            twap_remaining -= 1

        if twap_active and twap_remaining == 0 and not twap_pending:
            twap_active = False

        # ── 出場：ATR trailing stop ──────────────────────────────
        if pos != 0:
            closed = False
            xp = 0.0

            if pos == 1:
                hp = max(hp, H)
                tsl = max(tsl, hp - trail_atr * atr)
                if L <= tsl:
                    xp = max(tsl, O) * (1 - slippage)
                    closed = True
            elif pos == -1:
                lp = min(lp, L)
                tsl = min(tsl, lp + trail_atr * atr)
                if H >= tsl:
                    xp = min(tsl, O) * (1 + slippage)
                    closed = True

            if closed:
                gross = (xp - ep) * sz * pos
                x_fee = xp * sz * fee_rate
                net   = gross - entry_fee - x_fee   # 完整 PnL（含進出場費）
                cap  += gross - x_fee                # 進場費已在入場時扣過
                trades.append({
                    'Entry_Time': entry_time, 'Exit_Time': ts,
                    'Type': 'Long' if pos == 1 else 'Short',
                    'Entry_Price': ep, 'Exit_Price': xp,
                    'PnL': net, 'Capital': cap, 'Exit_Reason': 'Trailing_ATR',
                })
                pos = 0; sz = 0.0; entry_fee = 0.0
                twap_active = False; twap_remaining = 0; twap_pending = False

        # ── 入場訊號：突破 Donchian 通道 → 啟動 TWAP ────────────
        if pos == 0 and not twap_active:
            direction = 0
            if C > dc_high:
                direction = 1
            elif C < dc_low:
                direction = -1

            if direction != 0:
                risk_per_unit = atr_sl_mult * atr
                if risk_per_unit <= 0:
                    continue
                # 計算拆單數 N
                N = max(1, int(cap * leverage / max_trade_cap))
                if N == 1:
                    size_each = min(
                        (cap * risk_pct) / risk_per_unit,
                        (cap * leverage) / C
                    )
                else:
                    size_each = max_trade_cap / C

                twap_size_each = size_each
                twap_direction = direction
                twap_active    = True
                twap_remaining = N
                twap_pending   = True

        equity.append({'time': ts, 'capital': cap + (
            (C - ep) * sz * pos - C * sz * fee_rate if pos != 0 else 0)})

    # 強制平倉
    if pos != 0:
        xp = closes[-1] * (1 - slippage if pos == 1 else 1 + slippage)
        gross = (xp - ep) * sz * pos
        x_fee = xp * sz * fee_rate
        net   = gross - entry_fee - x_fee
        cap  += gross - x_fee
        trades.append({
            'Entry_Time': entry_time, 'Exit_Time': times[-1],
            'Type': 'Long' if pos == 1 else 'Short',
            'Entry_Price': ep, 'Exit_Price': xp,
            'PnL': net, 'Capital': cap, 'Exit_Reason': 'End_of_Data',
        })

    return trades, equity


def calc_stats(df_t: pd.DataFrame, df_e: pd.DataFrame, initial_cap: float) -> dict:
    """計算單一區間的績效指標，回傳 dict"""
    if len(df_t) == 0:
        return None
    win_t  = df_t[df_t['PnL'] > 0]
    lose_t = df_t[df_t['PnL'] <= 0]
    avg_w  = win_t['PnL'].mean() if len(win_t) > 0 else 0
    avg_l  = lose_t['PnL'].mean() if len(lose_t) > 0 else 0
    pf     = abs(win_t['PnL'].sum() / lose_t['PnL'].sum()) if lose_t['PnL'].sum() != 0 else float('inf')

    if df_e is not None and len(df_e) > 0:
        peak = df_e['capital'].cummax()
        dd   = ((df_e['capital'] - peak) / peak * 100).min()
    else:
        dd = 0.0

    return {
        'Trades':   len(df_t),
        'Long':     len(df_t[df_t['Type'] == 'Long']),
        'Short':    len(df_t[df_t['Type'] == 'Short']),
        'WinRate%': round(len(win_t) / len(df_t) * 100, 1),
        'PnL':      round(df_t['PnL'].sum(), 2),
        'Final':    round(df_t['Capital'].iloc[-1], 2),
        'AvgWin':   round(avg_w, 2),
        'AvgLoss':  round(avg_l, 2),
        'Payoff':   round(abs(avg_w / avg_l), 2) if avg_l != 0 else 0,
        'PF':       round(pf, 2),
        'MDD%':     round(dd, 1),
    }


def print_stats(trades: list, equity: list, cfg: dict):
    initial_cap = cfg['initial_cap']
    if not trades:
        print("\n  無交易")
        return

    df_t = pd.DataFrame(trades)
    df_e = pd.DataFrame(equity)

    s = calc_stats(df_t, df_e, initial_cap)

    print(f"\n{'='*55}")
    print(f"  ATR Trailing 出場 (×{cfg['trail_atr']})")
    print(f"{'='*55}")
    print(f"  交易次數:    {s['Trades']} (多:{s['Long']} / 空:{s['Short']})")
    print(f"  勝率:        {s['WinRate%']:.1f}%")
    print(f"  總淨利:      {s['PnL']:,.2f}")
    print(f"  最終資金:    {s['Final']:,.2f}")
    print(f"  報酬率:      {(s['Final'] / initial_cap - 1) * 100:.1f}%")
    print(f"  平均獲利:    {s['AvgWin']:,.2f}")
    print(f"  平均虧損:    {s['AvgLoss']:,.2f}")
    print(f"  盈虧比:      {s['Payoff']:.2f}")
    print(f"  利潤因子:    {s['PF']:.2f}")
    print(f"  最大回撤:    {s['MDD%']:.1f}%")
    print(f"{'='*55}")

    # ── 年度拆分 ─────────────────────────────────────────────────
    df_t['Exit_Time'] = pd.to_datetime(df_t['Exit_Time'])
    df_t['Year'] = df_t['Exit_Time'].dt.year
    df_e['time'] = pd.to_datetime(df_e['time'])
    df_e['Year'] = df_e['time'].dt.year

    years = sorted(df_t['Year'].unique())
    rows = []
    for y in years:
        yt = df_t[df_t['Year'] == y]
        ye = df_e[df_e['Year'] == y]
        ys = calc_stats(yt, ye, initial_cap)
        if ys:
            ys['Year'] = y
            rows.append(ys)

    if rows:
        df_y = pd.DataFrame(rows)
        df_y = df_y[['Year', 'Trades', 'WinRate%', 'PnL', 'Payoff', 'PF', 'MDD%']]
        print(f"\n{'='*75}")
        print(f"  年度績效")
        print(f"{'='*75}")
        print(df_y.to_string(index=False))
        print(f"{'='*75}")


def plot_equity(equity: list, symbol: str, cfg: dict):
    if not equity:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    df_e = pd.DataFrame(equity)
    ax.plot(df_e['time'], df_e['capital'], linewidth=1)
    ax.axhline(cfg['initial_cap'], color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f"Donchian 突破回測 — {symbol}/USDT 1H（N={cfg['entry_n']}, Trail x{cfg['trail_atr']}）")
    ax.set_ylabel('資金 (USDT)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _load_defaults_from_config():
    """從 config.json 讀取 ada_donchian 區塊，合併到 DEFAULTS"""
    try:
        with open(Path(__file__).parent / 'config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        ada = config.get('ada_donchian', {})
        risk = config.get('risk', {})
        return {
            'entry_n':       ada.get('entry_n', DEFAULTS['entry_n']),
            'trail_atr':     ada.get('trail_atr', DEFAULTS['trail_atr']),
            'atr_sl_mult':   ada.get('atr_sl_mult', DEFAULTS['atr_sl_mult']),
            'risk_pct':      ada.get('risk_pct', DEFAULTS['risk_pct']),
            'leverage':      DEFAULTS['leverage'],
            'fee_rate':      risk.get('taker_fee_rate', DEFAULTS['fee_rate']),
            'slippage':      DEFAULTS['slippage'],
            'initial_cap':   risk.get('initial_capital', DEFAULTS['initial_cap']),
            'max_trade_cap': ada.get('max_trade_cap', DEFAULTS['max_trade_cap']),
        }
    except FileNotFoundError:
        return dict(DEFAULTS)


def main():
    defaults = _load_defaults_from_config()
    parser = argparse.ArgumentParser(description='Donchian Channel Breakout Backtest')
    parser.add_argument('--symbol', default='ADA', help='BTC or ETH')
    for k, v in defaults.items():
        parser.add_argument(f'--{k}', type=type(v), default=v)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    cfg = {k: getattr(args, k) for k in defaults}

    print(f"Donchian 通道回測 | {symbol}/USDT 1H")
    print(f"   入場 N={cfg['entry_n']} | ATR trailing ×{cfg['trail_atr']} | TWAP 門檻 {cfg['max_trade_cap']:,.0f}")
    print(f"   風險 {cfg['risk_pct']*100:.0f}% | 槓桿 {cfg['leverage']}x | 初始資金 {cfg['initial_cap']:,}")

    df = load_data(symbol)
    print(f"   資料: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df):,} 根 K 棒)")

    trades, equity = run_backtest(df, cfg)
    print_stats(trades, equity, cfg)
    plot_equity(equity, symbol, cfg)


if __name__ == '__main__':
    main()
