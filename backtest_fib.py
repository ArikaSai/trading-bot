"""
backtest_fib.py
═══════════════
斐波納契回撤策略回測

策略邏輯：
  1. 用 swing_n 根 K 棒找出近期擺動高低點
  2. 在高低點之間畫出 Fib 回撤水平（23.6%, 38.2%, 50%, 61.8%）
  3. 趨勢判斷：收盤價 > EMA → 多頭，反之 → 空頭
  4. 多頭回調進場：價格從上方跌到 fib_level 附近並收回上方 → 做多
     空頭反彈進場：價格從下方漲到 fib_level 附近並收回下方 → 做空
  5. 出場：ATR Trailing Stop

進場模式（--mode）：
  market   ─ 市價單（原始邏輯，含 TWAP）
  limit_a  ─ 方案A：不需確認，直接在 Fib 水平掛限價單（每根 K 棒更新）
  limit_b  ─ 方案B：等收盤確認後掛限價單，逾時（limit_timeout 根 K 棒）自動取消
  compare  ─ 同時跑三種模式並排比較（market 模式關閉 TWAP 以公平比較）

用法:
    python backtest_fib.py                            # 預設 XRP 1h, market 模式
    python backtest_fib.py --mode compare             # 三模式比較
    python backtest_fib.py --mode limit_b --limit_timeout 8
    python backtest_fib.py --symbol BTC --swing_n 30 --fib_level 0.5
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
    swing_n         = 20,        # 擺動高低點回溯長度
    fib_level       = 0.618,     # 回撤進場位（0.382 / 0.5 / 0.618）
    fib_tolerance   = 0.005,     # 價格碰觸 Fib 線的容差（±0.5%）
    trail_atr       = 3.0,       # ATR trailing 倍數
    atr_sl_mult     = 2.0,       # 初始 SL = atr_sl_mult x ATR（倉位計算用）
    risk_pct        = 0.15,      # 每筆風險佔資金比例
    leverage        = 1,         # 槓桿
    fee_rate        = 0.0005,    # taker 手續費（市價單）
    maker_fee_rate  = 0.0002,    # maker 手續費（限價單）
    slippage        = 0.001,     # 滑價
    initial_cap     = 500,       # 初始資金
    max_trade_cap   = 200000.0,  # 單筆最大名目價值（TWAP 拆單門檻，market 模式）
    limit_timeout   = 5,         # 方案B：限價單最多等幾根 K 棒（逾時取消）
)


def _load_defaults_from_config():
    """從 config.json 讀取共用參數"""
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


def load_data(symbol: str) -> pd.DataFrame:
    path = Path(__file__).parent / 'data' / f'{symbol}USDT_1h.csv'
    if not path.exists():
        raise FileNotFoundError(f"找不到資料檔: {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.iloc[:-1]
    return df


# ══════════════════════════════════════════════════════════════════════
#  回測核心（統一支援三種進場模式）
# ══════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, cfg: dict, mode: str = 'market') -> tuple:
    """
    mode:
      'market'  ─ 市價進場（含 TWAP）
      'limit_a' ─ 無確認限價單，掛在 Fib 水平，每根 K 棒更新
      'limit_b' ─ 確認後限價單，逾時取消
    """
    swing_n        = cfg['swing_n']
    fib_level      = cfg['fib_level']
    fib_tol        = cfg['fib_tolerance']
    trail_atr      = cfg['trail_atr']
    atr_sl_mult    = cfg['atr_sl_mult']
    risk_pct       = cfg['risk_pct']
    leverage       = cfg['leverage']
    fee_rate       = cfg['fee_rate']
    maker_fee      = cfg.get('maker_fee_rate', cfg['fee_rate'])
    slippage       = cfg['slippage']
    cap            = cfg['initial_cap']
    max_trade_cap  = cfg['max_trade_cap']
    limit_timeout  = cfg.get('limit_timeout', 5)

    # ── 倉位狀態 ──
    pos = 0; ep = 0.0; sz = 0.0
    tsl = 0.0; hp = 0.0; lp = float('inf')
    entry_time = None; entry_fee = 0.0

    # ── TWAP 狀態（market 模式） ──
    twap_active    = False
    twap_remaining = 0
    twap_count     = 0
    max_twap_n     = 1
    twap_size_each = 0.0
    twap_direction = 0
    twap_pending   = False

    # ── 限價單狀態（limit_a / limit_b） ──
    limit_pending    = False
    limit_price      = 0.0
    limit_dir        = 0
    limit_bars_left  = 0   # 僅 limit_b 使用

    trades = []
    equity = []

    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    opens  = df['open'].values
    atrs   = df['ATR'].values
    emas   = df['EMA'].values
    times  = df.index

    for i in range(swing_n, len(df)):
        ts   = times[i]
        O, H, L, C = opens[i], highs[i], lows[i], closes[i]
        atr  = atrs[i]
        ema  = emas[i]

        # ── 擺動高低點 ──────────────────────────────────────────────
        _win_h      = highs[i - swing_n:i]
        _win_l      = lows[i - swing_n:i]
        swing_high  = np.max(_win_h)
        swing_low   = np.min(_win_l)
        swing_range = swing_high - swing_low
        _hi_idx     = int(np.argmax(_win_h))
        _lo_idx     = int(np.argmin(_win_l))

        uptrend   = C > ema
        downtrend = C < ema

        # ══════════════════════════════════════════════════════════
        #  MARKET 模式：TWAP 執行
        # ══════════════════════════════════════════════════════════
        if mode == 'market':
            if pos == 0 and twap_active and twap_pending:
                sub_ep  = O * (1 + slippage * twap_direction)
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

        # ══════════════════════════════════════════════════════════
        #  LIMIT 模式：檢查是否成交
        # ══════════════════════════════════════════════════════════
        elif pos == 0 and limit_pending:
            filled = False

            if limit_dir == 1 and L <= limit_price:
                # 多頭限價單成交（以 limit_price 成交，無額外滑價）
                fill_price = limit_price
                risk_per_unit = atr_sl_mult * atr
                if risk_per_unit > 0:
                    size = min(
                        (cap * risk_pct) / risk_per_unit,
                        (cap * leverage) / fill_price
                    )
                    f = size * fill_price * maker_fee
                    cap -= f
                    pos = 1; ep = fill_price; sz = size
                    entry_time = ts; entry_fee = f
                    hp = H; lp = float('inf')
                    tsl = ep - trail_atr * atr
                    filled = True

            elif limit_dir == -1 and H >= limit_price:
                # 空頭限價單成交
                fill_price = limit_price
                risk_per_unit = atr_sl_mult * atr
                if risk_per_unit > 0:
                    size = min(
                        (cap * risk_pct) / risk_per_unit,
                        (cap * leverage) / fill_price
                    )
                    f = size * fill_price * maker_fee
                    cap -= f
                    pos = -1; ep = fill_price; sz = size
                    entry_time = ts; entry_fee = f
                    lp = L; hp = 0.0
                    tsl = ep + trail_atr * atr
                    filled = True

            if filled:
                limit_pending = False
                limit_bars_left = 0
            elif mode == 'limit_b':
                limit_bars_left -= 1
                if limit_bars_left <= 0:
                    limit_pending = False  # 逾時取消

        # ══════════════════════════════════════════════════════════
        #  出場：ATR Trailing Stop（所有模式共用）
        # ══════════════════════════════════════════════════════════
        if pos != 0:
            closed = False; xp = 0.0

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
                x_fee = xp * sz * fee_rate   # 出場用市價單費率
                net   = gross - entry_fee - x_fee
                cap  += gross - x_fee
                trades.append({
                    'Entry_Time': entry_time, 'Exit_Time': ts,
                    'Type': 'Long' if pos == 1 else 'Short',
                    'Entry_Price': ep, 'Exit_Price': xp,
                    'PnL': net, 'Capital': cap, 'Exit_Reason': 'Trailing_ATR',
                })
                pos = 0; sz = 0.0; entry_fee = 0.0
                if mode == 'market':
                    twap_active = False; twap_remaining = 0; twap_pending = False

        # ══════════════════════════════════════════════════════════
        #  進場訊號
        # ══════════════════════════════════════════════════════════
        if pos == 0 and swing_range > 0 and atr > 0:

            # ── MARKET 模式：收盤確認 → 下一根開盤市價進場 ──────────
            if mode == 'market' and not twap_active:
                prev_C = closes[i - 1]
                prev_L = lows[i - 1]
                prev_H = highs[i - 1]
                direction = 0

                if uptrend and _hi_idx > _lo_idx:
                    fib_price = swing_high - fib_level * swing_range
                    tol = fib_price * fib_tol
                    if prev_L <= fib_price + tol and prev_C > fib_price:
                        direction = 1
                elif downtrend and _lo_idx > _hi_idx:
                    fib_price = swing_low + fib_level * swing_range
                    tol = fib_price * fib_tol
                    if prev_H >= fib_price - tol and prev_C < fib_price:
                        direction = -1

                if direction != 0:
                    risk_per_unit = atr_sl_mult * atr
                    if risk_per_unit > 0:
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
                        if N > 1:
                            twap_count += 1
                            max_twap_n = max(max_twap_n, N)

            # ── LIMIT_A 模式：無確認，直接掛 Fib 水平限價單（每根更新）──
            elif mode == 'limit_a':
                if uptrend and _hi_idx > _lo_idx:
                    new_fib = swing_high - fib_level * swing_range
                    limit_price = new_fib
                    limit_dir   = 1
                    limit_pending = True
                elif downtrend and _lo_idx > _hi_idx:
                    new_fib = swing_low + fib_level * swing_range
                    limit_price = new_fib
                    limit_dir   = -1
                    limit_pending = True
                else:
                    # 趨勢不明 → 取消掛單
                    limit_pending = False

            # ── LIMIT_B 模式：確認後掛限價單，逾時取消 ──────────────
            elif mode == 'limit_b' and not limit_pending:
                prev_C = closes[i - 1]
                prev_L = lows[i - 1]
                prev_H = highs[i - 1]
                direction = 0

                if uptrend and _hi_idx > _lo_idx:
                    fib_price = swing_high - fib_level * swing_range
                    tol = fib_price * fib_tol
                    if prev_L <= fib_price + tol and prev_C > fib_price:
                        direction = 1
                elif downtrend and _lo_idx > _hi_idx:
                    fib_price = swing_low + fib_level * swing_range
                    tol = fib_price * fib_tol
                    if prev_H >= fib_price - tol and prev_C < fib_price:
                        direction = -1

                if direction != 0:
                    fib_price = (swing_high - fib_level * swing_range
                                 if direction == 1
                                 else swing_low + fib_level * swing_range)
                    limit_price     = fib_price
                    limit_dir       = direction
                    limit_bars_left = limit_timeout
                    limit_pending   = True

        equity.append({'time': ts, 'capital': cap + (
            (C - ep) * sz * pos - C * sz * fee_rate if pos != 0 else 0)})

    # ── 強制平倉 ──────────────────────────────────────────────────────
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
            'PnL': net, 'Capital': cap, 'Exit_Reason': '資料結束',
        })

    return trades, equity, {'twap_count': twap_count, 'max_twap_n': max_twap_n}


# ══════════════════════════════════════════════════════════════════════
#  績效統計
# ══════════════════════════════════════════════════════════════════════

def calc_stats(df_t: pd.DataFrame, df_e: pd.DataFrame, initial_cap: float) -> dict:
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
        '交易次數':  len(df_t),
        '多':        len(df_t[df_t['Type'] == 'Long']),
        '空':        len(df_t[df_t['Type'] == 'Short']),
        '勝率%':     round(len(win_t) / len(df_t) * 100, 1),
        '總損益':    round(df_t['PnL'].sum(), 2),
        '最終資金':  round(df_t['Capital'].iloc[-1], 2),
        '平均獲利':  round(avg_w, 2),
        '平均虧損':  round(avg_l, 2),
        '盈虧比':    round(abs(avg_w / avg_l), 2) if avg_l != 0 else 0,
        '利潤因子':  round(pf, 2),
        '最大回撤%': round(dd, 1),
    }


def print_stats(trades: list, equity: list, cfg: dict,
                label: str = '', twap_info: dict = None):
    initial_cap = cfg['initial_cap']
    if not trades:
        print(f"\n  [{label}] 無交易")
        return

    df_t = pd.DataFrame(trades)
    df_e = pd.DataFrame(equity)
    s = calc_stats(df_t, df_e, initial_cap)
    title = f"Fib 回撤策略 ─ {label}（{cfg['fib_level']*100:.1f}% | Trail x{cfg['trail_atr']}）"

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  交易次數:    {s['交易次數']} (多:{s['多']} / 空:{s['空']})")
    print(f"  勝率:        {s['勝率%']:.1f}%")
    print(f"  總損益:      {s['總損益']:,.2f}")
    print(f"  最終資金:    {s['最終資金']:,.2f}")
    print(f"  報酬率:      {(s['最終資金'] / initial_cap - 1) * 100:.1f}%")
    print(f"  平均獲利:    {s['平均獲利']:,.2f}")
    print(f"  平均虧損:    {s['平均虧損']:,.2f}")
    print(f"  盈虧比:      {s['盈虧比']:.2f}")
    print(f"  利潤因子:    {s['利潤因子']:.2f}")
    print(f"  最大回撤:    {s['最大回撤%']:.1f}%")
    if twap_info and twap_info['twap_count'] > 0:
        print(f"  TWAP 拆單:   {twap_info['twap_count']} 次（最大 N={twap_info['max_twap_n']}）")
    print(f"{'='*60}")

    # ── 年度拆分 ──────────────────────────────────────────────────
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
            ys['年度'] = y
            rows.append(ys)

    if rows:
        df_y = pd.DataFrame(rows)
        df_y = df_y[['年度', '交易次數', '勝率%', '總損益', '盈虧比', '利潤因子', '最大回撤%']]
        print(f"\n{'='*75}")
        print(f"  年度績效 [{label}]")
        print(f"{'='*75}")
        print(df_y.to_string(index=False))
        print(f"{'='*75}")


def print_compare(results: dict, cfg: dict):
    """並排顯示三種模式的核心指標"""
    labels = list(results.keys())
    stats  = {}
    for lb in labels:
        trades, equity, _ = results[lb]
        if trades:
            df_t = pd.DataFrame(trades)
            df_e = pd.DataFrame(equity)
            stats[lb] = calc_stats(df_t, df_e, cfg['initial_cap'])
        else:
            stats[lb] = None

    keys = ['交易次數', '勝率%', '總損益', '最終資金', '平均獲利', '平均虧損',
            '盈虧比', '利潤因子', '最大回撤%']
    col_w = 18

    print(f"\n{'='*(12 + col_w * len(labels))}")
    print(f"  三模式比較  Fib {cfg['fib_level']*100:.1f}% | Swing {cfg['swing_n']} | Trail x{cfg['trail_atr']}")
    print(f"  limit_b timeout = {cfg['limit_timeout']} 根 K 棒")
    print(f"{'='*(12 + col_w * len(labels))}")

    header = f"{'指標':<12}" + "".join(f"{lb:>{col_w}}" for lb in labels)
    print(header)
    print('-' * (12 + col_w * len(labels)))

    for k in keys:
        row = f"{k:<12}"
        for lb in labels:
            s = stats[lb]
            if s is None:
                row += f"{'─':>{col_w}}"
            else:
                v = s[k]
                if isinstance(v, float):
                    row += f"{v:>{col_w}.2f}"
                else:
                    row += f"{v:>{col_w}}"
        print(row)

    print(f"{'='*(12 + col_w * len(labels))}")

    # 報酬率行
    row = f"{'報酬率%':<12}"
    for lb in labels:
        s = stats[lb]
        if s is None:
            row += f"{'─':>{col_w}}"
        else:
            ret = (s['最終資金'] / cfg['initial_cap'] - 1) * 100
            row += f"{ret:>{col_w}.1f}"
    print(row)
    print(f"{'='*(12 + col_w * len(labels))}")


def plot_compare(results: dict, symbol: str, cfg: dict):
    """繪製三模式權益曲線"""
    colors = {'market': '#4C72B0', 'limit_a': '#DD8452', 'limit_b': '#55A868'}
    fig, ax = plt.subplots(figsize=(16, 6))

    for lb, (trades, equity, _) in results.items():
        if equity:
            df_e = pd.DataFrame(equity)
            ax.plot(df_e['time'], df_e['capital'],
                    label=lb, color=colors.get(lb), linewidth=1.2)

    ax.axhline(cfg['initial_cap'], color='gray', linestyle='--', alpha=0.5, label='初始資金')
    ax.set_title(
        f"Fib 回撤策略 三模式比較 — {symbol}/USDT 1H\n"
        f"Fib {cfg['fib_level']*100:.1f}% | Swing {cfg['swing_n']} | "
        f"Trail x{cfg['trail_atr']} | limit_b timeout={cfg['limit_timeout']}根"
    )
    ax.set_ylabel('資金 (USDT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_equity(equity: list, symbol: str, cfg: dict, label: str = ''):
    if not equity:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    df_e = pd.DataFrame(equity)
    ax.plot(df_e['time'], df_e['capital'], linewidth=1)
    ax.axhline(cfg['initial_cap'], color='gray', linestyle='--', alpha=0.5)
    ax.set_title(
        f"Fib 回撤策略 [{label}] — {symbol}/USDT 1H\n"
        f"（{cfg['fib_level']*100:.1f}% | Swing {cfg['swing_n']} | Trail x{cfg['trail_atr']}）"
    )
    ax.set_ylabel('資金 (USDT)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


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

    parser = argparse.ArgumentParser(description='Fibonacci Retracement Backtest')
    parser.add_argument('--symbol', default='XRP', help='幣種（如 XRP, SOL, BTC）')
    parser.add_argument('--start',  default=bt_start, help='回測起始日')
    parser.add_argument('--end',    default=bt_end,   help='回測結束日')
    parser.add_argument('--mode',   default='market',
                        choices=['market', 'limit_a', 'limit_b', 'compare'],
                        help='進場模式（market/limit_a/limit_b/compare）')
    for k, v in defaults.items():
        parser.add_argument(f'--{k}', type=type(v), default=v)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    cfg = {k: getattr(args, k) for k in defaults}

    df = load_data(symbol)
    if args.start:
        df = df[df.index >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df.index <= pd.Timestamp(args.end)]

    print(f"\nFib 回撤回測 | {symbol}/USDT 1H | 模式: {args.mode}")
    print(f"   Swing N={cfg['swing_n']} | Fib {cfg['fib_level']*100:.1f}% | 容差 {cfg['fib_tolerance']*100:.1f}%")
    print(f"   ATR trailing x{cfg['trail_atr']} | SL x{cfg['atr_sl_mult']}")
    print(f"   風險 {cfg['risk_pct']*100:.0f}% | 槓桿 {cfg['leverage']}x | 初始資金 {cfg['initial_cap']:,}")
    print(f"   Taker 費率 {cfg['fee_rate']*100:.3f}% | Maker 費率 {cfg['maker_fee_rate']*100:.3f}%")
    print(f"   資料: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df):,} 根 K 棒)")

    if args.mode == 'compare':
        # ── 三模式並排比較（market 也關閉 TWAP 影響不大，但 N=1 時相同）──
        print("\n  執行三模式回測中...")
        results = {}
        for m in ('market', 'limit_a', 'limit_b'):
            results[m] = run_backtest(df.copy(), cfg, mode=m)
            print(f"    [{m}] 完成，共 {len(results[m][0])} 筆交易")

        print_compare(results, cfg)

        # 各模式年度明細
        for m, (trades, equity, twap_info) in results.items():
            mode_labels = {
                'market':  '市價單（Market）',
                'limit_a': '方案A：無確認限價',
                'limit_b': f"方案B：確認後限價（timeout={cfg['limit_timeout']}根）",
            }
            print_stats(trades, equity, cfg,
                        label=mode_labels[m], twap_info=twap_info)

        plot_compare(results, symbol, cfg)

    else:
        mode_labels = {
            'market':  '市價單（Market）',
            'limit_a': '方案A：無確認限價',
            'limit_b': f"方案B：確認後限價（timeout={cfg['limit_timeout']}根）",
        }
        trades, equity, twap_info = run_backtest(df, cfg, mode=args.mode)
        print_stats(trades, equity, cfg,
                    label=mode_labels[args.mode], twap_info=twap_info)
        plot_equity(equity, symbol, cfg, label=mode_labels[args.mode])


if __name__ == '__main__':
    main()
