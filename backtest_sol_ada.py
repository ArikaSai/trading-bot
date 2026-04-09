"""
backtest_sol_ada.py
═══════════════════
SOL 趨勢策略 + ADA Donchian 突破策略 共享資金聯合回測

時間軸：SOL 15m + ADA 1h 合併排序
資金池：共享，各自按比例計算倉位
風控  ：同時持倉時合計保證金 ≤ 總資金
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import json
import warnings
from collections import deque as _deque
from strategy import CoreStrategy

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════
#  固定參數
# ══════════════════════════════════════════════════════════════
SLIPPAGE          = 0.0010
MAX_TOTAL_RISK    = 0.3
MAX_POSITION_CAP  = 40_000_000.0   # SOL 幣安最大持倉名目價值

# ADA Donchian 參數（從 config.json 讀取）
# — 在 run_combined_backtest() 中賦值 —


def load_config(path="config.json") -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════
#  聯合回測
# ══════════════════════════════════════════════════════════════

def run_combined_backtest(df_sol, df_ada, config,
                          start_date="2023-01-01", end_date="2026-03-31"):
    # ── SOL 參數 ─────────────────────────────────────────────
    initial_capital    = config['risk']['initial_capital']
    sol_risk           = config['risk']['risk_per_trade']
    sol_max_pos        = config['risk']['max_pos_ratio']
    sol_leverage       = config['risk'].get('leverage', 1)
    sol_max_cap        = config['risk'].get('max_trade_usdt_cap', 200000.0)
    sol_max_loss       = config['risk'].get('max_consec_losses', 3)
    sol_fee            = config['risk'].get('taker_fee_rate', 0.0005)
    sol_mmr            = config['risk'].get('maintenance_margin_rate', 0.005)
    sol_trail_atr      = config['strategy']['trailing_atr']
    sol_sl_atr         = config['strategy']['initial_sl_atr']
    sol_adx_th         = config['strategy']['adx_threshold']

    # ── ADA 參數 ─────────────────────────────────────────────
    ada_cfg            = config['ada_donchian']
    ADA_ENTRY_N        = ada_cfg['entry_n']
    ADA_TRAIL_ATR      = ada_cfg['trail_atr']
    ADA_ATR_SL_MULT    = ada_cfg['atr_sl_mult']
    ADA_MAX_TRADE_CAP  = ada_cfg.get('max_trade_cap', 200000.0)
    ADA_MAX_CONSEC     = ada_cfg.get('max_consec_losses', 3)
    ADA_FEE            = config['risk'].get('taker_fee_rate', 0.0005)
    ada_risk_pct       = ada_cfg.get('risk_pct', 0.15)

    # ── 時間過濾 ─────────────────────────────────────────────
    t_start = pd.Timestamp(start_date)
    t_end   = pd.Timestamp(end_date)
    df_sol  = df_sol[(df_sol.index >= t_start) & (df_sol.index <= t_end)].copy()
    df_ada  = df_ada[(df_ada.index >= t_start) & (df_ada.index <= t_end)].copy()

    # ── 建立查詢字典 ─────────────────────────────────────────
    sol_dict = {row.Index: row for row in df_sol.itertuples()}
    ada_dict = {row.Index: row for row in df_ada.itertuples()}

    # ADA Donchian 通道預計算
    ada_highs  = df_ada['high'].values
    ada_lows   = df_ada['low'].values
    ada_idx    = df_ada.index
    ada_idx_to_pos = {ts: i for i, ts in enumerate(ada_idx)}

    # ── 建立統一時間軸 ───────────────────────────────────────
    all_ts = sorted(set(list(df_sol.index) + list(df_ada.index)))

    # ── 共享資金 ─────────────────────────────────────────────
    capital = initial_capital

    # ── SOL 狀態 ─────────────────────────────────────────────
    sol_pos = 0; sol_size = 0.0; sol_entry = 0.0
    sol_sl = 0.0; sol_trail = 0.0; sol_liq = 0.0
    sol_high = 0.0; sol_low = float('inf')
    sol_init_risk = 0.0; sol_entry_ts = None; sol_entry_fee = 0.0
    sol_long_sig = False; sol_short_sig = False
    sol_consec = 0; sol_skip = False; sol_in_skip = False
    sol_be_activated = False
    sol_saved_sl_dist = 0.0
    sol_trades = []
    sol_margin_used = 0.0

    # SOL TWAP
    sol_twap_active = False; sol_twap_remaining = 0
    sol_twap_size_each = 0.0; sol_twap_direction = 0

    # SOL 盤整縮緊
    _sol_consol_highs = _deque(maxlen=8)
    _sol_consol_lows  = _deque(maxlen=8)

    # ── ADA 狀態 ─────────────────────────────────────────────
    ada_pos = 0; ada_size = 0.0; ada_entry = 0.0
    ada_tsl = 0.0; ada_hp = 0.0; ada_lp = float('inf')
    ada_entry_ts = None; ada_entry_fee = 0.0
    ada_consec = 0; ada_skip = False; ada_in_skip = False
    ada_trades = []
    ada_margin_used = 0.0

    # ADA TWAP
    ada_twap_active = False; ada_twap_remaining = 0
    ada_twap_size_each = 0.0; ada_twap_direction = 0
    ada_twap_pending = False

    # ── 淨值曲線 ─────────────────────────────────────────────
    equity_curve = []
    prev_sol_ts = None
    simultaneous_holds = 0

    # ══════════════════════════════════════════════════════════
    #  主循環
    # ══════════════════════════════════════════════════════════

    for ts in all_ts:
        if capital < 5.0:
            equity_curve.append({'timestamp': ts, 'equity': 0.0})
            continue

        is_sol_bar = ts in sol_dict
        is_ada_bar = ts in ada_dict

        # ══════════════════════════════════════════════════════
        #  SOL 15m 事件
        # ══════════════════════════════════════════════════════
        if is_sol_bar:
            row = sol_dict[ts]
            cur_open  = float(row.open);  cur_high = float(row.high)
            cur_low   = float(row.low);   cur_close = float(row.close)
            cur_atr   = float(row.ATR)

            # 進場執行
            if sol_pos == 0 and (sol_long_sig or sol_short_sig):
                sl_dist = sol_saved_sl_dist if sol_saved_sl_dist > 0 else max(sol_sl_atr * cur_atr, cur_open * 0.001)
                sol_saved_sl_dist = 0.0
                if sol_twap_active and sol_twap_remaining > 0:
                    sol_size      = sol_twap_size_each
                    sol_init_risk = sol_twap_remaining * sol_twap_size_each * sl_dist
                else:
                    sol_size, sol_init_risk = CoreStrategy.calculate_position_size(
                        capital, sol_risk, sl_dist, cur_open,
                        sol_max_pos, sol_leverage, sol_max_cap
                    )
                if sol_long_sig:
                    sol_entry = cur_open * (1 + SLIPPAGE)
                    sol_pos   = 1
                    sol_sl    = sol_trail = sol_entry - sl_dist
                    sol_high  = cur_open
                    sol_liq   = CoreStrategy.calc_liquidation_price(sol_entry, 1, sol_leverage, sol_mmr)
                else:
                    sol_entry = cur_open * (1 - SLIPPAGE)
                    sol_pos   = -1
                    sol_sl    = sol_trail = sol_entry + sl_dist
                    sol_low   = cur_open
                    sol_liq   = CoreStrategy.calc_liquidation_price(sol_entry, -1, sol_leverage, sol_mmr)
                sol_entry_fee   = sol_size * sol_entry * sol_fee
                capital -= sol_entry_fee
                sol_margin_used = sol_size * sol_entry / sol_leverage
                sol_entry_ts     = ts
                sol_be_activated = False
                sol_long_sig     = sol_short_sig = False
                if sol_twap_active:
                    sol_twap_remaining -= 1
                    if sol_twap_remaining == 0:
                        sol_twap_active = False

            # SOL TWAP 後續子單
            elif sol_pos != 0 and sol_twap_active and sol_twap_remaining > 0 and sol_pos == sol_twap_direction:
                sub_ep      = cur_open * (1 + SLIPPAGE) if sol_pos == 1 else cur_open * (1 - SLIPPAGE)
                sub_fee     = sol_twap_size_each * sub_ep * sol_fee
                capital    -= sub_fee
                sol_entry_fee += sub_fee
                total_sz   = sol_size + sol_twap_size_each
                sol_entry  = (sol_entry * sol_size + sub_ep * sol_twap_size_each) / total_sz
                sol_size   = total_sz
                sol_liq    = CoreStrategy.calc_liquidation_price(sol_entry, sol_pos, sol_leverage, sol_mmr)
                sol_margin_used = sol_size * sol_entry / sol_leverage
                sol_twap_remaining -= 1
                if sol_twap_remaining == 0:
                    sol_twap_active = False

            # 盤整縮緊 + 追蹤止損
            _sol_consol_highs.append(cur_high)
            _sol_consol_lows.append(cur_low)
            if sol_pos != 0:
                if len(_sol_consol_highs) == 8 and cur_atr > 0:
                    _is_consol = (max(_sol_consol_highs) - min(_sol_consol_lows)) < 1.25 * cur_atr
                    _eff_trail = 1.0 if _is_consol else sol_trail_atr
                else:
                    _eff_trail = sol_trail_atr
                is_spike = cur_atr > 0 and (cur_high - cur_low) > 4.0 * cur_atr
                ref_high = cur_close if (is_spike and sol_pos == 1)  else cur_high
                ref_low  = cur_close if (is_spike and sol_pos == -1) else cur_low
                sol_trail, sol_high, sol_low = CoreStrategy.update_trailing_stop(
                    sol_pos, sol_trail, sol_high, sol_low,
                    ref_high, ref_low, cur_atr, _eff_trail
                )

            # 2.0R 保本
            if not sol_be_activated and sol_pos != 0:
                sol_sl_dist = abs(sol_entry - sol_sl)
                if sol_pos == 1 and cur_high >= sol_entry + 2.0 * sol_sl_dist:
                    be_price  = sol_entry / ((1 - sol_fee) * (1 - SLIPPAGE))
                    sol_trail = max(sol_trail, be_price)
                    sol_be_activated = True
                elif sol_pos == -1 and cur_low <= sol_entry - 2.0 * sol_sl_dist:
                    be_price  = sol_entry / ((1 + sol_fee) * (1 + SLIPPAGE))
                    sol_trail = min(sol_trail, be_price)
                    sol_be_activated = True

            # 出場判斷
            closed, exit_p, pnl, reason = CoreStrategy.check_exit(
                sol_pos, cur_low, cur_high, cur_open,
                sol_liq, sol_trail, sol_sl,
                sol_entry, sol_size, sol_fee, SLIPPAGE
            )
            if closed:
                if reason == '💀 Liquidation':
                    capital = max(0.0, capital + pnl)
                else:
                    capital += pnl
                sol_trades.append({
                    'Entry_Time': sol_entry_ts, 'Exit_Time': ts,
                    'Strategy': 'SOL',
                    'PnL': pnl - sol_entry_fee, 'Capital': capital,
                    'Exit_Reason': reason, 'Initial_Risk': sol_init_risk,
                })
                net_pnl = pnl - sol_entry_fee
                sol_entry_fee      = 0.0
                sol_pos            = 0
                sol_be_activated   = False
                sol_margin_used    = 0.0
                sol_twap_active    = False
                sol_twap_remaining = 0
                _sol_consol_highs.clear()
                _sol_consol_lows.clear()
                if net_pnl < 0:
                    sol_consec += 1
                    if sol_consec >= sol_max_loss:
                        sol_skip = True; sol_in_skip = False; sol_consec = 0
                else:
                    sol_consec = 0

            # 信號偵測
            if sol_pos == 0 and not sol_twap_active and ts != prev_sol_ts:
                l_cond, s_cond, _ = CoreStrategy.check_signals(row, sol_adx_th)
                if sol_skip:
                    if l_cond or s_cond:
                        sol_skip = False; sol_in_skip = True
                elif sol_in_skip:
                    if not l_cond and not s_cond:
                        sol_in_skip = False
                else:
                    if l_cond or s_cond:
                        _sl_dist = max(sol_sl_atr * cur_atr, cur_open * 0.001)
                        _N = min(max(1, int(capital / sol_max_cap)),
                                 int(MAX_POSITION_CAP / sol_max_cap))
                        if _N == 1:
                            _sz_each = CoreStrategy.calculate_position_size(
                                capital, sol_risk, _sl_dist, cur_open,
                                sol_max_pos, sol_leverage, sol_max_cap
                            )[0]
                        else:
                            _sz_each = sol_max_cap / cur_open
                        sol_saved_sl_dist  = _sl_dist
                        sol_twap_size_each = _sz_each
                        sol_twap_remaining = _N
                        sol_twap_active    = True
                        if l_cond:
                            sol_long_sig       = True
                            sol_twap_direction = 1
                        else:
                            sol_short_sig      = True
                            sol_twap_direction = -1

            prev_sol_ts = ts

        # ══════════════════════════════════════════════════════
        #  ADA 1h 事件
        # ══════════════════════════════════════════════════════
        if is_ada_bar:
            ai = ada_idx_to_pos[ts]
            a_row = ada_dict[ts]
            a_O = float(a_row.open);  a_H = float(a_row.high)
            a_L = float(a_row.low);   a_C = float(a_row.close)
            a_atr = float(a_row.ATR)

            # Donchian 通道（前 N 根）
            if ai >= ADA_ENTRY_N:
                dc_high = np.max(ada_highs[ai - ADA_ENTRY_N:ai])
                dc_low  = np.min(ada_lows[ai - ADA_ENTRY_N:ai])
            else:
                dc_high = dc_low = None

            # ── ADA TWAP 第一份子單 ──────────────────────────
            if ada_pos == 0 and ada_twap_active and ada_twap_pending:
                sub_ep = a_O * (1 + SLIPPAGE * ada_twap_direction)
                sub_fee = ada_twap_size_each * sub_ep * ADA_FEE
                capital -= sub_fee
                ada_entry_fee = sub_fee
                ada_pos  = ada_twap_direction
                ada_size = ada_twap_size_each
                ada_entry = sub_ep
                ada_entry_ts = ts
                ada_margin_used = ada_size * ada_entry
                if ada_pos == 1:
                    ada_hp = a_H; ada_lp = float('inf')
                    ada_tsl = ada_entry - ADA_TRAIL_ATR * a_atr
                else:
                    ada_lp = a_L; ada_hp = 0.0
                    ada_tsl = ada_entry + ADA_TRAIL_ATR * a_atr
                ada_twap_remaining -= 1
                ada_twap_pending = False
                if ada_pos != 0 and sol_pos != 0:
                    simultaneous_holds += 1

            # ── ADA TWAP 後續子單 ────────────────────────────
            elif ada_pos != 0 and ada_twap_active and ada_twap_remaining > 0 and ada_pos == ada_twap_direction:
                sub_ep  = a_O * (1 + SLIPPAGE * ada_pos)
                sub_fee = ada_twap_size_each * sub_ep * ADA_FEE
                capital -= sub_fee
                ada_entry_fee += sub_fee
                total_sz = ada_size + ada_twap_size_each
                ada_entry = (ada_entry * ada_size + sub_ep * ada_twap_size_each) / total_sz
                ada_size = total_sz
                ada_margin_used = ada_size * ada_entry
                ada_twap_remaining -= 1

            if ada_twap_active and ada_twap_remaining == 0 and not ada_twap_pending:
                ada_twap_active = False

            # ── ADA 出場：ATR trailing ───────────────────────
            if ada_pos != 0:
                if ada_pos == 1:
                    ada_hp = max(ada_hp, a_H)
                    ada_tsl = max(ada_tsl, ada_hp - ADA_TRAIL_ATR * a_atr)
                    if a_L <= ada_tsl:
                        xp = max(ada_tsl, a_O) * (1 - SLIPPAGE)
                        gross = (xp - ada_entry) * ada_size
                        x_fee = xp * ada_size * ADA_FEE
                        net   = gross - ada_entry_fee - x_fee   # 完整 PnL（含進出場費）
                        capital += gross - x_fee                 # 進場費已在入場時扣過
                        ada_trades.append({
                            'Entry_Time': ada_entry_ts, 'Exit_Time': ts,
                            'Strategy': 'ADA', 'Type': 'Long',
                            'PnL': net, 'Capital': capital,
                            'Exit_Reason': 'Trailing_ATR',
                        })
                        ada_pos = 0; ada_size = 0.0; ada_entry_fee = 0.0
                        ada_margin_used = 0.0
                        ada_twap_active = False; ada_twap_remaining = 0; ada_twap_pending = False
                        if net < 0:
                            ada_consec += 1
                            if ada_consec >= ADA_MAX_CONSEC:
                                ada_skip = True; ada_in_skip = False; ada_consec = 0
                        else:
                            ada_consec = 0

                elif ada_pos == -1:
                    ada_lp = min(ada_lp, a_L)
                    ada_tsl = min(ada_tsl, ada_lp + ADA_TRAIL_ATR * a_atr)
                    if a_H >= ada_tsl:
                        xp = min(ada_tsl, a_O) * (1 + SLIPPAGE)
                        gross = (ada_entry - xp) * ada_size
                        x_fee = xp * ada_size * ADA_FEE
                        net   = gross - ada_entry_fee - x_fee   # 完整 PnL（含進出場費）
                        capital += gross - x_fee                 # 進場費已在入場時扣過
                        ada_trades.append({
                            'Entry_Time': ada_entry_ts, 'Exit_Time': ts,
                            'Strategy': 'ADA', 'Type': 'Short',
                            'PnL': net, 'Capital': capital,
                            'Exit_Reason': 'Trailing_ATR',
                        })
                        ada_pos = 0; ada_size = 0.0; ada_entry_fee = 0.0
                        ada_margin_used = 0.0
                        ada_twap_active = False; ada_twap_remaining = 0; ada_twap_pending = False
                        if net < 0:
                            ada_consec += 1
                            if ada_consec >= ADA_MAX_CONSEC:
                                ada_skip = True; ada_in_skip = False; ada_consec = 0
                        else:
                            ada_consec = 0

            # ── ADA 入場訊號：Donchian 突破 ──────────────────
            if ada_pos == 0 and not ada_twap_active and dc_high is not None and a_atr > 0:
                has_signal = (a_C > dc_high) or (a_C < dc_low)

                if ada_skip:
                    if has_signal:
                        ada_skip = False; ada_in_skip = True
                elif ada_in_skip:
                    if not has_signal:
                        ada_in_skip = False
                elif has_signal:
                    direction = 1 if a_C > dc_high else -1
                    risk_per_unit = ADA_ATR_SL_MULT * a_atr
                    if risk_per_unit > 0:
                        available = capital - sol_margin_used
                        if available > 10:
                            N = max(1, int(available / ADA_MAX_TRADE_CAP))
                            if N == 1:
                                raw_sz = (available * ada_risk_pct) / risk_per_unit
                                max_sz = available / a_C
                                size_each = min(raw_sz, max_sz)
                            else:
                                size_each = ADA_MAX_TRADE_CAP / a_C

                            ada_twap_size_each = size_each
                            ada_twap_direction = direction
                            ada_twap_active    = True
                            ada_twap_remaining = N
                            ada_twap_pending   = True

        # ── 淨值快照 ─────────────────────────────────────────
        if is_sol_bar or is_ada_bar:
            unr = 0.0
            if sol_pos != 0 and ts in sol_dict:
                sol_c = float(sol_dict[ts].close)
                unr += (sol_c - sol_entry) * sol_size * sol_pos
            if ada_pos != 0 and ts in ada_dict:
                ada_c = float(ada_dict[ts].close)
                unr += (ada_c - ada_entry) * ada_size * ada_pos
            equity_curve.append({'timestamp': ts, 'equity': capital + unr})

    trades_df = pd.DataFrame(sol_trades + ada_trades)
    equity_df = pd.DataFrame(equity_curve).drop_duplicates('timestamp').set_index('timestamp')

    return trades_df, equity_df, {'simultaneous_holds': simultaneous_holds}


# ══════════════════════════════════════════════════════════════
#  指標計算
# ══════════════════════════════════════════════════════════════

def calc_metrics(trades_df, equity_df, initial_capital):
    if trades_df.empty or equity_df.empty:
        return {}
    eq  = equity_df['equity']
    mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    ret = (eq.iloc[-1] - initial_capital) / initial_capital * 100

    daily_eq      = eq.resample('D').last().ffill()
    daily_returns = daily_eq.pct_change().dropna()
    sharpe        = float(daily_returns.mean() / daily_returns.std() * np.sqrt(365)) \
                    if daily_returns.std() > 0 else 0.0

    sol_t = trades_df[trades_df['Strategy'] == 'SOL']
    ada_t = trades_df[trades_df['Strategy'] == 'ADA']

    def _wr(t): return (t['PnL'] > 0).mean() * 100 if len(t) > 0 else 0

    return {
        'final_capital':  round(eq.iloc[-1], 2),
        'total_return':   round(ret, 2),
        'sharpe':         round(sharpe, 3),
        'true_mdd':       round(mdd, 2),
        'total_trades':   len(trades_df),
        'sol_trades':     len(sol_t),
        'ada_trades':     len(ada_t),
        'sol_win_rate':   round(_wr(sol_t), 2),
        'ada_win_rate':   round(_wr(ada_t), 2),
        'sol_pnl':        round(sol_t['PnL'].sum(), 2) if len(sol_t) > 0 else 0,
        'ada_pnl':        round(ada_t['PnL'].sum(), 2) if len(ada_t) > 0 else 0,
    }


# ══════════════════════════════════════════════════════════════
#  繪圖
# ══════════════════════════════════════════════════════════════

def plot_combined(equity_combined, equity_sol_only, equity_ada_only,
                  m_combined, m_sol, m_ada, initial_capital):

    def _to_ts_series(eq):
        if eq is None or eq.empty:
            return None, None
        if 'timestamp' in eq.columns:
            return pd.to_datetime(eq['timestamp']), eq['equity'].values
        return pd.to_datetime(eq.index), eq['equity'].values

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    for eq, label, color in [
        (equity_combined, f"聯合策略（Sharpe {m_combined.get('sharpe','N/A')}）", "#15b959"),
        (equity_sol_only, f"SOL 獨立（報酬 {m_sol.get('total_return','N/A')}%）", "#2d82bb"),
        (equity_ada_only, f"ADA 獨立（報酬 {m_ada.get('total_return','N/A')}%）", "#df8b41"),
    ]:
        ts, val = _to_ts_series(eq)
        if ts is not None:
            ax1.plot(ts, val, label=label, linewidth=1.2, color=color)

    ax1.set_title('\n聯合 vs 獨立策略淨值曲線（對數刻度）\n', fontsize=12, fontweight='bold')
    ax1.set_ylabel('資金 (USDT) - 對數刻度')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

    ax2 = fig.add_subplot(gs[1, 0])
    metrics_labels = ['總報酬率 %', 'Sharpe', '|最大回撤 %|']
    combined_vals  = [m_combined.get('total_return', 0), m_combined.get('sharpe', 0), abs(m_combined.get('true_mdd', 0))]
    sol_vals       = [m_sol.get('total_return', 0), m_sol.get('sharpe', 0), abs(m_sol.get('true_mdd', 0))]
    ada_vals       = [m_ada.get('total_return', 0), m_ada.get('sharpe', 0), abs(m_ada.get('true_mdd', 0))]
    x = np.arange(len(metrics_labels)); w = 0.25
    ax2.bar(x - w, sol_vals,      w, label='SOL 獨立',  color='#3498db', alpha=0.8)
    ax2.bar(x,     ada_vals,      w, label='ADA 獨立',  color='#e67e22', alpha=0.8)
    ax2.bar(x + w, combined_vals, w, label='聯合策略', color='#2ecc71', alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(metrics_labels)
    ax2.set_title('關鍵指標對比', fontsize=11, fontweight='bold')
    ax2.legend(); ax2.grid(True, axis='y', linestyle='--', alpha=0.4)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    lines = [
        "【聯合策略績效報告】",
        f"初始資金:   {initial_capital} U",
        f"最終資金:   {m_combined.get('final_capital', 'N/A')} U",
        f"總報酬率:   {m_combined.get('total_return', 'N/A')}%",
        f"Sharpe:     {m_combined.get('sharpe', 'N/A')}",
        f"最大回撤:   {m_combined.get('true_mdd', 'N/A')}%",
        f"總交易數:   {m_combined.get('total_trades', 'N/A')} 筆",
        f"  SOL:      {m_combined.get('sol_trades', 0)} 筆 "
        f"(勝率 {m_combined.get('sol_win_rate', 0)}%)",
        f"  ADA:      {m_combined.get('ada_trades', 0)} 筆 "
        f"(勝率 {m_combined.get('ada_win_rate', 0)}%)",
        "",
        f"SOL 貢獻:   {m_combined.get('sol_pnl', 0):,.2f} U "
        f"({m_combined.get('sol_pnl', 0) / (m_combined.get('sol_pnl', 0) + m_combined.get('ada_pnl', 0)) * 100:.1f}%)"
        if (m_combined.get('sol_pnl', 0) + m_combined.get('ada_pnl', 0)) != 0
        else f"SOL 貢獻:   {m_combined.get('sol_pnl', 0):,.2f} U",
        f"ADA 貢獻:   {m_combined.get('ada_pnl', 0):,.2f} U "
        f"({m_combined.get('ada_pnl', 0) / (m_combined.get('sol_pnl', 0) + m_combined.get('ada_pnl', 0)) * 100:.1f}%)"
        if (m_combined.get('sol_pnl', 0) + m_combined.get('ada_pnl', 0)) != 0
        else f"ADA 貢獻:   {m_combined.get('ada_pnl', 0):,.2f} U",
    ]
    for i, line in enumerate(lines):
        ax3.text(0.05, 0.95 - i * 0.07, line, transform=ax3.transAxes,
                 fontsize=11, verticalalignment='top',
                 fontweight='bold' if i == 0 else 'normal')

    plt.suptitle('SOL 趨勢 + ADA Donchian 聯合回測', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════
#  ADA 獨立回測（基準對比用）
# ══════════════════════════════════════════════════════════════

def _run_ada_solo(df_ada, config, start_date, end_date):
    """用 backtest_donchian 的邏輯跑 ADA 獨立"""
    from backtest_donchian import run_backtest as donchian_bt
    t_start = pd.Timestamp(start_date); t_end = pd.Timestamp(end_date)
    df = df_ada[(df_ada.index >= t_start) & (df_ada.index <= t_end)].copy()

    ada_cfg = config['ada_donchian']
    cfg = {
        'entry_n': ada_cfg['entry_n'], 'trail_atr': ada_cfg['trail_atr'],
        'atr_sl_mult': ada_cfg['atr_sl_mult'], 'risk_pct': ada_cfg.get('risk_pct', 0.15),
        'leverage': 1, 'fee_rate': config['risk'].get('taker_fee_rate', 0.0005),
        'slippage': SLIPPAGE,
        'initial_cap': config['risk']['initial_capital'],
        'max_trade_cap': ada_cfg.get('max_trade_cap', 200000.0),
    }
    trades, equity = donchian_bt(df, cfg)
    eq_df = pd.DataFrame(equity)
    eq_df.index = pd.to_datetime(eq_df['time'])
    eq_df.rename(columns={'capital': 'equity'}, inplace=True)

    daily_eq  = eq_df['equity'].resample('D').last().ffill()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(365)) if daily_ret.std() > 0 else 0.0

    init_cap = cfg['initial_cap']
    final = eq_df['equity'].iloc[-1]
    mdd   = ((eq_df['equity'] - eq_df['equity'].cummax()) / eq_df['equity'].cummax()).min() * 100
    t_df  = pd.DataFrame(trades) if trades else pd.DataFrame()
    wr    = (t_df['PnL'] > 0).mean() * 100 if len(t_df) > 0 else 0

    return eq_df, {
        'total_return': round((final - init_cap) / init_cap * 100, 2),
        'true_mdd': round(mdd, 2), 'sharpe': round(sharpe, 3),
        'sol_win_rate': 0, 'ada_win_rate': round(wr, 2),
        'total_trades': len(t_df), 'sol_trades': 0, 'ada_trades': len(t_df),
        'sol_pnl': 0, 'ada_pnl': round(t_df['PnL'].sum(), 2) if len(t_df) > 0 else 0,
        'final_capital': round(final, 2),
    }


# ══════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    config = load_config("config.json")

    START   = config.get('backtest', {}).get('start_date', '2023-01-01')
    END     = config.get('backtest', {}).get('end_date', '2026-04-09')
    INITIAL = config['risk']['initial_capital']

    # 載入資料
    print("[讀取] 載入 SOL 15m 資料...")
    df_sol = pd.read_csv("data/SOLUSDT_15m.csv", index_col='timestamp', parse_dates=True)
    df_sol = df_sol.iloc[:-1]
    df_sol = CoreStrategy.prepare_data(df_sol)
    print(f"  [OK] SOL {len(df_sol)} 根 K 棒")

    print("[讀取] 載入 ADA 1h 資料...")
    df_ada = pd.read_csv("data/ADAUSDT_1h.csv", index_col='timestamp', parse_dates=True)
    df_ada = df_ada.iloc[:-1]
    print(f"  [OK] ADA {len(df_ada)} 根 K 棒")

    # ── 聯合回測 ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[執行] SOL + ADA 聯合回測...")
    trades_df, equity_combined, diag = run_combined_backtest(
        df_sol, df_ada, config, START, END
    )
    m_combined = calc_metrics(trades_df, equity_combined, INITIAL)

    # ── SOL 獨立基準 ─────────────────────────────────────────
    print("\n[SOL] 獨立基準回測...")
    from backtest_sol import run_backtest as sol_bt, calc_metrics as sol_calc, filter_date_range
    sol_only_df = filter_date_range(df_sol.copy(), START, END)
    t_sol, e_sol, _ = sol_bt(sol_only_df, config)
    m_sol_raw = sol_calc(t_sol, e_sol, INITIAL)
    e_sol_ts = e_sol.copy()
    e_sol_ts.index = pd.to_datetime(e_sol_ts['timestamp'])
    sol_daily     = e_sol_ts['equity'].resample('D').last().ffill()
    sol_daily_ret = sol_daily.pct_change().dropna()
    sol_sharpe    = float(sol_daily_ret.mean() / sol_daily_ret.std() * np.sqrt(365)) \
                    if sol_daily_ret.std() > 0 else 0.0
    m_sol = {
        'total_return': m_sol_raw['Return_%'], 'true_mdd': m_sol_raw['True_MDD_%'],
        'sharpe': round(sol_sharpe, 3), 'sol_win_rate': m_sol_raw['Win_Rate_%'],
        'ada_win_rate': 0, 'total_trades': m_sol_raw['Total_Trades'],
        'sol_trades': m_sol_raw['Total_Trades'], 'ada_trades': 0,
        'sol_pnl': round(m_sol_raw['Final_Cap'] - INITIAL, 2), 'ada_pnl': 0,
        'final_capital': m_sol_raw['Final_Cap'],
    }

    # ── ADA 獨立基準 ─────────────────────────────────────────
    print("\n[ADA] 獨立基準回測...")
    equity_ada_only, m_ada = _run_ada_solo(df_ada, config, START, END)

    # ── 報告 ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[總覽] SOL + ADA 聯合 vs 獨立")
    print('='*60)
    print(f"{'指標':<20} {'SOL 獨立':>12} {'ADA 獨立':>12} {'聯合策略':>12}")
    print('-'*60)
    for k, label in [
        ('total_return',  '總報酬率 %'),
        ('true_mdd',      '最大回撤 %'),
        ('sharpe',        'Sharpe'),
        ('total_trades',  '總交易數'),
        ('sol_trades',    'SOL 交易數'),
        ('ada_trades',    'ADA 交易數'),
        ('sol_win_rate',  'SOL 勝率 %'),
        ('ada_win_rate',  'ADA 勝率 %'),
        ('sol_pnl',       'SOL 損益'),
        ('ada_pnl',       'ADA 損益'),
        ('final_capital', '最終資金'),
    ]:
        vs = m_sol.get(k, '-'); va = m_ada.get(k, '-'); vc = m_combined.get(k, '-')
        print(f"  {label:<18} {vs:>12} {va:>12} {vc:>12}")
    print('='*60)
    print(f"  同時持倉次數: {diag['simultaneous_holds']}")

    # ── 個別幣種貢獻量 ───────────────────────────────────────
    sol_pnl = m_combined.get('sol_pnl', 0)
    ada_pnl = m_combined.get('ada_pnl', 0)
    total_pnl = sol_pnl + ada_pnl
    sol_pct = sol_pnl / total_pnl * 100 if total_pnl != 0 else 0
    ada_pct = ada_pnl / total_pnl * 100 if total_pnl != 0 else 0

    print(f"\n{'='*60}")
    print("[貢獻] 個別幣種貢獻")
    print('='*60)
    print(f"  SOL 貢獻損益:  {sol_pnl:>14,.2f} U  ({sol_pct:+.1f}%)")
    print(f"  ADA 貢獻損益:  {ada_pnl:>14,.2f} U  ({ada_pct:+.1f}%)")
    print(f"  合計淨利:       {total_pnl:>14,.2f} U")

    # ── 理論收益 vs 實際收益驗證 ─────────────────────────────
    expected_capital = INITIAL + sol_pnl + ada_pnl
    actual_capital   = m_combined.get('final_capital', 0)
    discrepancy      = expected_capital - actual_capital

    print(f"\n{'='*60}")
    print("[驗證] 理論收益 vs 實際收益")
    print('='*60)
    print(f"  初始資金:       {INITIAL:>14,.2f} U")
    print(f"  SOL 淨利:       {sol_pnl:>+14,.2f} U")
    print(f"  ADA 淨利:       {ada_pnl:>+14,.2f} U")
    print(f"  理論最終資金:   {expected_capital:>14,.2f} U")
    print(f"  實際最終資金:   {actual_capital:>14,.2f} U")
    print(f"  差異:           {discrepancy:>+14,.2f} U")

    if abs(discrepancy) > 1.0:
        print(f"\n  [!] 差異超過 1 USDT，可能原因：")
        print(f"      1. 費用未完全計入（進場費用在 capital 扣除但未記入 PnL）")
        print(f"      2. 還有倉位未平倉（未實現盈虧）")
    else:
        print(f"\n  [OK] 帳務一致")
    print('='*60)

    # ── 繪圖 ─────────────────────────────────────────────────
    plot_combined(equity_combined, e_sol_ts[['equity']], equity_ada_only,
                  m_combined, m_sol, m_ada, INITIAL)
