import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from strategy import CoreStrategy

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False


def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 找不到設定檔：{config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_date_range(df: pd.DataFrame,
                      start_date: str = None,
                      end_date:   str = None) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]

    if df.empty:
        raise ValueError(f"❌ 時間過濾後資料為空，請確認區間：{start_date} ~ {end_date}")

    print(f"🗓️  時間過濾器啟用：{df.index.min().date()} ～ {df.index.max().date()} "
          f"（共 {len(df)} 筆）")
    return df


def run_backtest(df_slice: pd.DataFrame, config: dict,
                 adx_override:     float = None,
                 slippage_pct:     float = 0.0010,
                 max_position_cap: float = 40_000_000.0):

    required_cols = ['open', 'high', 'low', 'close', 'ATR', 'RSI', 'EMA', 'BB_Mid']
    has_adx = 'ADX' in df_slice.columns
    if not has_adx:
        print("⚠️ 未發現 ADX 欄位，使用 ADX=100.0 代替")
    if any(c not in df_slice.columns for c in required_cols) or len(df_slice) < 50:
        return pd.DataFrame(), pd.DataFrame()

    initial_capital    = config['risk']['initial_capital']
    risk_per_trade     = config['risk']['risk_per_trade']
    max_pos_ratio      = config['risk']['max_pos_ratio']
    leverage           = config['risk'].get('leverage', 1)
    max_trade_usdt_cap = config['risk'].get('max_trade_usdt_cap', 200000.0)
    max_consec_losses  = config['risk'].get('max_consec_losses', 3)
    fee_rate           = config['risk'].get('taker_fee_rate', 0.0005)
    mmr                = config['risk'].get('maintenance_margin_rate', 0.005)
    trailing_atr       = config['strategy']['trailing_atr']
    initial_sl_atr     = config['strategy']['initial_sl_atr']
    adx_threshold      = adx_override if adx_override is not None else config['strategy']['adx_threshold']

    capital        = initial_capital
    position       = 0
    entry_price    = 0.0
    position_size  = 0.0
    liq_price      = 0.0
    initial_risk   = 0.0
    entry_time     = None
    entry_fee      = 0.0
    stop_loss      = 0.0
    trailing_stop  = 0.0
    highest_price  = 0.0
    lowest_price   = float('inf')

    enter_long_signal_next_bar  = False
    enter_short_signal_next_bar = False
    consecutive_losses          = 0
    skip_next_trade             = False
    in_skip_zone                = False
    saved_sl_dist               = 0.0
    be_activated                = False

    # 自適應 TWAP 狀態
    from collections import deque as _deque
    twap_active    = False
    twap_remaining = 0
    twap_size_each = 0.0
    twap_direction = 0

    # 盤整縮緊：N=8, X=1.25, tight=1.0×
    _consol_highs = _deque(maxlen=8)
    _consol_lows  = _deque(maxlen=8)

    trade_log    = []
    equity_curve = []

    for row in df_slice.itertuples():
        cur_open  = row.open;   cur_high  = row.high
        cur_low   = row.low;    cur_close = row.close
        cur_atr   = row.ATR;    cur_rsi   = row.RSI
        timestamp = row.Index;  bb_mid    = row.BB_Mid
        ema       = row.EMA
        cur_adx   = row.ADX if has_adx else 100.0

        _consol_highs.append(cur_high)
        _consol_lows.append(cur_low)

        if capital < 5.0:
            equity_curve.append({'timestamp': timestamp, 'equity': 0.0}); continue

        # ── 首筆進場 ─────────────────────────────────────────────
        if position == 0 and (enter_long_signal_next_bar or enter_short_signal_next_bar):
            sl_dist       = saved_sl_dist if saved_sl_dist > 0 else max(initial_sl_atr * cur_atr, cur_open * 0.001)
            saved_sl_dist = 0.0

            if twap_active and twap_remaining > 0:
                position_size = twap_size_each
                initial_risk  = twap_remaining * twap_size_each * sl_dist
            else:
                position_size, initial_risk = CoreStrategy.calculate_position_size(
                    capital, risk_per_trade, sl_dist, cur_open, max_pos_ratio, leverage, max_trade_usdt_cap
                )

            if enter_long_signal_next_bar:
                entry_price   = cur_open * (1 + slippage_pct)
                position      = 1
                stop_loss     = trailing_stop = entry_price - sl_dist
                highest_price = cur_open
                liq_price     = CoreStrategy.calc_liquidation_price(entry_price, 1, leverage, mmr)
            else:
                entry_price   = cur_open * (1 - slippage_pct)
                position      = -1
                stop_loss     = trailing_stop = entry_price + sl_dist
                lowest_price  = cur_open
                liq_price     = CoreStrategy.calc_liquidation_price(entry_price, -1, leverage, mmr)

            entry_fee  = position_size * entry_price * fee_rate
            capital   -= entry_fee
            entry_time   = timestamp
            be_activated = False
            enter_long_signal_next_bar = enter_short_signal_next_bar = False
            if twap_active:
                twap_remaining -= 1
                if twap_remaining == 0:
                    twap_active = False

        # ── TWAP 後續子單 ────────────────────────────────────────
        elif position != 0 and twap_active and twap_remaining > 0 and position == twap_direction:
            sub_ep    = cur_open * (1 + slippage_pct) if position == 1 else cur_open * (1 - slippage_pct)
            sub_fee   = twap_size_each * sub_ep * fee_rate
            capital  -= sub_fee;  entry_fee += sub_fee
            total_sz  = position_size + twap_size_each
            entry_price = (entry_price * position_size + sub_ep * twap_size_each) / total_sz
            position_size = total_sz
            liq_price = CoreStrategy.calc_liquidation_price(entry_price, position, leverage, mmr)
            twap_remaining -= 1
            if twap_remaining == 0:
                twap_active = False

        trade_closed = False; closed_pnl = 0.0; exit_price = 0.0; exit_reason = ""

        # 盤整縮緊（N=8, X=1.25× ATR, tight=1.0× ATR）
        if position != 0 and len(_consol_highs) == 8 and cur_atr > 0:
            _is_consol = (max(_consol_highs) - min(_consol_lows)) < 1.25 * cur_atr
            _eff_trail = 1.0 if _is_consol else trailing_atr
        else:
            _eff_trail = trailing_atr

        # 插針過濾 + 追蹤止損更新（4.0× ATR 插針過濾：插針K棒改用收盤價更新追蹤參考點）
        is_spike = cur_atr > 0 and (cur_high - cur_low) > 4.0 * cur_atr
        ref_high = cur_close if (is_spike and position == 1)  else cur_high
        ref_low  = cur_close if (is_spike and position == -1) else cur_low
        trailing_stop, highest_price, lowest_price = CoreStrategy.update_trailing_stop(
            position, trailing_stop, highest_price, lowest_price,
            ref_high, ref_low, cur_atr, _eff_trail
        )

        # 2.0R 保本
        if not be_activated and position != 0:
            sl_dist_be = abs(entry_price - stop_loss)
            if position == 1 and cur_high >= entry_price + 2.0 * sl_dist_be:
                be_price      = entry_price / ((1 - fee_rate) * (1 - slippage_pct))
                trailing_stop = max(trailing_stop, be_price)
                be_activated  = True
            elif position == -1 and cur_low <= entry_price - 2.0 * sl_dist_be:
                be_price      = entry_price / ((1 + fee_rate) * (1 + slippage_pct))
                trailing_stop = min(trailing_stop, be_price)
                be_activated  = True

        # 判斷出場
        trade_closed, exit_price, closed_pnl, exit_reason = CoreStrategy.check_exit(
            position, cur_low, cur_high, cur_open,
            liq_price, trailing_stop, stop_loss,
            entry_price, position_size, fee_rate, slippage_pct
        )
        if trade_closed and exit_reason == '💀 Liquidation':
            capital = max(0.0, capital + closed_pnl)
        elif trade_closed:
            capital += closed_pnl

        if trade_closed:
            net_pnl = closed_pnl - entry_fee
            trade_log.append({'Entry_Time': entry_time, 'Exit_Time': timestamp,
                               'Type': 'Long' if position == 1 else 'Short',
                               'Entry_Price': entry_price, 'Exit_Price': exit_price,
                               'PnL': net_pnl, 'Capital': capital,
                               'Exit_Reason': exit_reason, 'Initial_Risk': initial_risk})
            position     = 0
            be_activated = False
            entry_fee    = 0.0
            twap_active  = False; twap_remaining = 0
            _consol_highs.clear(); _consol_lows.clear()
            if net_pnl < 0:
                consecutive_losses += 1
                if consecutive_losses >= max_consec_losses:
                    skip_next_trade = True; in_skip_zone = False; consecutive_losses = 0
            else:
                consecutive_losses = 0

        if position == 0 and not enter_long_signal_next_bar and not enter_short_signal_next_bar:
            l_cond, s_cond, _ = CoreStrategy.check_signals(row, adx_threshold)

            if skip_next_trade:
                if l_cond or s_cond: skip_next_trade = False; in_skip_zone = True
            elif in_skip_zone:
                if not l_cond and not s_cond: in_skip_zone = False
            else:
                if l_cond or s_cond:
                    _sl_dist = max(initial_sl_atr * cur_atr, cur_open * 0.001)
                    _N = min(max(1, int(capital / max_trade_usdt_cap)),
                             int(max_position_cap / max_trade_usdt_cap))
                    twap_size_each = (CoreStrategy.calculate_position_size(
                                         capital, risk_per_trade, _sl_dist, cur_open,
                                         max_pos_ratio, leverage, max_trade_usdt_cap)[0]
                                     if _N == 1 else max_trade_usdt_cap / cur_open)
                    saved_sl_dist  = _sl_dist
                    twap_remaining = _N
                    twap_active    = True
                    twap_direction = 1 if l_cond else -1
                    if l_cond:
                        enter_long_signal_next_bar  = True
                    else:
                        enter_short_signal_next_bar = True

        unr = (cur_close - entry_price) * position_size * position if position != 0 else 0.0
        equity_curve.append({'timestamp': timestamp, 'equity': capital + unr})

    return pd.DataFrame(trade_log), pd.DataFrame(equity_curve)


def calc_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame, initial_capital: float) -> dict:
    if trades_df.empty or equity_df.empty:
        return {"Total_Trades": 0, "Return_%": 0.0, "Win_Rate_%": 0.0,
                "True_MDD_%": 0.0, "Max_Consecutive_Losses": 0,
                "Worst_Trade_USDT": 0.0, "Worst_Trade_%": 0.0,
                "Avg_R": 0.0, "Final_Cap": initial_capital}

    final_capital = equity_df['equity'].iloc[-1]
    total_return  = (final_capital - initial_capital) / initial_capital * 100
    win_rate      = (len(trades_df[trades_df['PnL'] > 0]) / len(trades_df)) * 100

    equity_series = equity_df['equity']
    true_mdd      = ((equity_series - equity_series.cummax()) / equity_series.cummax()).min() * 100

    max_consec = cur_consec = 0
    for pnl in trades_df['PnL']:
        cur_consec = cur_consec + 1 if pnl <= 0 else 0
        max_consec = max(max_consec, cur_consec)

    worst_row       = trades_df.loc[trades_df['PnL'].idxmin()]
    worst_trade     = round(worst_row['PnL'], 2)
    capital_before  = worst_row['Capital'] - worst_row['PnL']
    worst_trade_pct = round(worst_trade / capital_before * 100, 2) if capital_before > 0 else 0.0

    valid = trades_df[trades_df['Initial_Risk'] > 0].copy()
    avg_r = (valid['PnL'] / valid['Initial_Risk']).mean() if len(valid) > 0 else 0.0

    equity_df = equity_df.copy()
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    daily_eq  = equity_df.set_index('timestamp')['equity'].resample('D').last().ffill()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe    = round(float(daily_ret.mean() / daily_ret.std() * np.sqrt(365)), 3) \
                if daily_ret.std() > 0 else 0.0

    return {
        "Total_Trades":           len(trades_df),
        "Return_%":               round(total_return, 2),
        "Win_Rate_%":             round(win_rate, 2),
        "True_MDD_%":             round(true_mdd, 2),
        "Sharpe":                 sharpe,
        "Max_Consecutive_Losses": max_consec,
        "Worst_Trade_USDT":       worst_trade,
        "Worst_Trade_%":          worst_trade_pct,
        "Avg_R":                  round(avg_r, 2),
        "Final_Cap":              round(final_capital, 2)
    }


def plot_equity_curve(equity_df: pd.DataFrame, title: str, use_log_scale: bool = False):
    if equity_df.empty:
        print("❌ 淨值數據為空"); return
    plot_df = equity_df.set_index('timestamp')
    plt.figure(figsize=(10, 5))
    plt.plot(plot_df.index, plot_df['equity'], color="#502980", linewidth=1.2, label='True Equity')
    if use_log_scale:
        plt.yscale('log'); plt.ylabel('Capital (USDT) - Log', fontsize=12)
        plt.grid(True, which='both', linestyle='-', alpha=0.3)
    else:
        plt.ylabel('Capital (USDT) - Linear', fontsize=12)
        plt.grid(True, linestyle='-', alpha=0.6)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.legend(); plt.gcf().autofmt_xdate(); plt.tight_layout(); plt.show()


def run_walk_forward(df: pd.DataFrame, config: dict):
    periods = [
        {"name": "區間一", "test": ("2021-01-01", "2022-12-31"), "val": ("2023-01-01", "2023-12-31")},
        {"name": "區間二", "test": ("2022-01-01", "2023-12-31"), "val": ("2024-01-01", "2024-12-31")},
        {"name": "區間三", "test": ("2023-01-01", "2024-12-31"), "val": ("2025-01-01", "2025-12-31")},
    ]
    init_cap = config['risk']['initial_capital']
    t_atr    = config['strategy']['trailing_atr']
    sl_atr   = config['strategy']['initial_sl_atr']
    adx      = config['strategy']['adx_threshold']

    df_index = pd.to_datetime(df.index) if not isinstance(df.index, pd.DatetimeIndex) else df.index
    print(f"\n📊 數據範圍: {df_index.min().date()} 至 {df_index.max().date()}")
    print("=" * 70)
    print(f"   [策略] Trailing ATR: {t_atr} | SL ATR: {sl_atr} | ADX: {adx}")
    print(f"   [風控] 本金: {init_cap} | 風險: {config['risk']['risk_per_trade']*100}% | "
          f"持倉上限: {config['risk']['max_pos_ratio']*100}% | 槓桿: {config['risk'].get('leverage',1)}x")
    print("=" * 70)

    for p in periods:
        print(f"\n📍 【{p['name']}】")
        try:
            df_test = df.loc[p['test'][0]:p['test'][1]]
            if len(df_test) == 0: print("  ⚠️ 測試區間無數據"); continue
            t, e = run_backtest(df_test, config)
            m    = calc_metrics(t, e, init_cap)
            print(f"  [測試 {p['test'][0]}~{p['test'][1]}] 報酬: {m['Return_%']}% | MDD: {m['True_MDD_%']}%")

            df_val = df.loc[p['val'][0]:p['val'][1]]
            if len(df_val) == 0: print("  ⚠️ 驗證區間無數據"); continue
            tv, ev = run_backtest(df_val, config)
            mv     = calc_metrics(tv, ev, init_cap)
            print(f"  [驗證 {p['val'][0]}~{p['val'][1]}]")
            print(f"  💰 報酬率: {mv['Return_%']}% | 🩸 MDD: {mv['True_MDD_%']}%")
            print(f"  🎯 勝率: {mv['Win_Rate_%']}% | ⚖️ 平均R: {mv['Avg_R']} | 📉 最大連虧: {mv['Max_Consecutive_Losses']} 次 | 🔄 總交易: {mv['Total_Trades']}")
            print(f"  💥 單筆最慘: {mv['Worst_Trade_USDT']} USDT ({mv['Worst_Trade_%']}%)")
            print("-" * 70)
            plot_equity_curve(ev, title=f"True Equity Curve - {p['name']}")
        except Exception as e:
            import traceback; print(f"  ❌ 執行出錯: {e}"); traceback.print_exc()


def run_stress_test(df: pd.DataFrame, config: dict,
                    start_date: str, end_date: str,
                    label: str = "壓力測試"):
    init_cap = config['risk']['initial_capital']
    df_clean = filter_date_range(df, start_date, end_date)

    t, e = run_backtest(df_clean, config)
    m    = calc_metrics(t, e, init_cap)

    print(f"\n{'='*70}")
    print(f"🔬 【{label}】{start_date} ～ {end_date}")
    print(f"{'='*70}")
    print(f"  💰 報酬率:    {m['Return_%']}%")
    print(f"  🩸 MDD:       {m['True_MDD_%']}%")
    print(f"  🎯 勝率:      {m['Win_Rate_%']}%")
    print(f"  ⚖️  平均R:     {m['Avg_R']}")
    print(f"  📉 最大連虧:  {m['Max_Consecutive_Losses']} 次")
    print(f"  💥 單筆最慘:  {m['Worst_Trade_USDT']} USDT ({m['Worst_Trade_%']}%)")
    print(f"  🔄 總交易:    {m['Total_Trades']} 筆")
    print(f"  💵 最終餘額:  {m['Final_Cap']} USDT")
    print(f"{'='*70}")

    plot_equity_curve(e, title=f"{label}：{start_date} ～ {end_date}", use_log_scale=True)
    return m


if __name__ == "__main__":
    try:
        app_config = load_config("config.json")
    except FileNotFoundError as e:
        print(e); exit(1)

    # ══════════════════════════════════════════════════════════════
    #  SOL 趨勢策略回測
    # ══════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("█  SOL 趨勢策略回測")
    print("█" * 70)

    sol_csv = "data/SOLUSDT_15m.csv"
    try:
        df_main = pd.read_csv(sol_csv, index_col='timestamp', parse_dates=True)
        print(f"✅ 成功讀取 SOL 資料：{len(df_main)} 筆")
        df_main = df_main.iloc[:-1]
        df_main = CoreStrategy.prepare_data(df_main)

        run_walk_forward(df_main, config=app_config)
        run_stress_test(df_main, app_config,
                        start_date="2026-01-01", end_date="2026-03-31",
                        label="SOL 壓力測試")
    except FileNotFoundError:
        print(f"❌ 找不到 {sol_csv}，跳過 SOL 回測")

    # ══════════════════════════════════════════════════════════════
    #  XRP MR 策略回測
    # ══════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("█  XRP MR 策略回測")
    print("█" * 70)

    xrp_cfg = app_config.get('xrp_mr', {})
    from train_mr_kelly import run_rolling_kelly
    run_rolling_kelly(
        csv_path     = "ml_experiments/XRP_mr_training_data.csv",
        coin         = "XRP",
        start_date   = "2023-01-01",
        end_date     = "2026-01-01",
        train_months = 8,
        test_months  = 1,
    )