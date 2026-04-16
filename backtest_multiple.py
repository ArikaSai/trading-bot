"""
backtest_multiple.py
════════════════════
SOL 趨勢 + ADA 唐奇安 + XRP 斐波 + DOGE Squeeze  四策略共用資金聯合回測

資金方案：free_b（可用保證金 × 40%，直接用）
時間軸  ：SOL 15m + ADA / XRP / DOGE 1h 合併排序

用法:
    python backtest_multiple.py
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from collections import deque as _deque
from pathlib import Path
from strategy import CoreStrategy

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

SLIPPAGE         = 0.001
MAX_POSITION_CAP = 100_000_000.0   # SOL 幣安最大持倉名目價值
_FREE_PCT        = 0.40            # 每次取可用保證金的固定 40%


def load_config(path="config.json") -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_csv(symbol: str, timeframe: str) -> pd.DataFrame:
    path = Path(__file__).parent / 'data' / f'{symbol}USDT_{timeframe}.csv'
    if not path.exists():
        raise FileNotFoundError(f"找不到: {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.iloc[:-1]
    return df


def run_triple(config, label: str = "",
               consol_n: int = 6, consol_x: float = 1.5,
               tight_trail: float = 0.5, spike_mult: float = 3.0):
    # ── 載入資料 ─────────────────────────────────────────────
    sol_tf  = config['trading'].get('timeframe', '15m')
    df_sol  = load_csv('SOL',  sol_tf)
    df_ada  = load_csv('ADA',  '1h')
    df_xrp  = load_csv('XRP',  '1h')
    df_doge = load_csv('DOGE', '1h')

    # ── 時間過濾 ─────────────────────────────────────────────
    bt      = config.get('backtest', {})
    t_start = pd.Timestamp(bt.get('start_date', '2020-01-01'))
    t_end   = pd.Timestamp(bt.get('end_date',   '2026-04-09'))
    df_sol  = df_sol[ (df_sol.index  >= t_start) & (df_sol.index  <= t_end)]
    df_ada  = df_ada[ (df_ada.index  >= t_start) & (df_ada.index  <= t_end)]
    df_xrp  = df_xrp[ (df_xrp.index  >= t_start) & (df_xrp.index  <= t_end)]
    df_doge = df_doge[(df_doge.index >= t_start) & (df_doge.index <= t_end)]

    # ── 共用參數 ─────────────────────────────────────────────
    initial_cap = config['risk']['initial_capital']
    fee_rate    = config['risk'].get('taker_fee_rate', 0.0005)

    # SOL 趨勢參數
    sol_adx_th    = config['strategy']['adx_threshold']
    sol_trail_atr = config['strategy']['trailing_atr']
    sol_sl_atr    = config['strategy']['initial_sl_atr']
    sol_risk      = config['risk']['risk_per_trade']
    sol_max_pos   = config['risk']['max_pos_ratio']
    sol_leverage  = config['risk'].get('leverage', 2)
    sol_max_cap   = config['risk'].get('max_trade_usdt_cap', 200000.0)
    sol_max_loss  = config['risk'].get('max_consec_losses', 3)
    sol_fee       = fee_rate
    sol_mmr       = config['risk'].get('maintenance_margin_rate', 0.005)

    # ADA 唐奇安參數
    ada_cfg      = config['ada_donchian']
    ada_entry_n  = ada_cfg['entry_n']
    ada_trail    = ada_cfg['trail_atr']
    ada_leverage = ada_cfg.get('leverage', 1)
    ada_max_cap  = ada_cfg.get('max_trade_cap', 200000.0)
    ada_max_loss = ada_cfg.get('max_consec_losses', 3)
    ada_fee      = fee_rate

    # XRP 斐波參數
    xrp_cfg       = config.get('xrp_fib', {})
    xrp_swing_n   = xrp_cfg.get('swing_n', 20)
    xrp_fib_level = xrp_cfg.get('fib_level', 0.618)
    xrp_trail     = xrp_cfg.get('trail_atr', 3.0)
    xrp_fib_tol   = xrp_cfg.get('fib_tol', 0.005)
    xrp_leverage  = xrp_cfg.get('leverage', 1)
    xrp_max_cap   = xrp_cfg.get('max_trade_cap', 200000.0)
    xrp_max_loss        = xrp_cfg.get('max_consec_losses', 3)
    xrp_limit_max_bars  = xrp_cfg.get('xrp_limit_max_hours', 0)  # 0 = 無逾時
    xrp_fee             = fee_rate

    # DOGE Squeeze 參數
    doge_cfg        = config.get('doge_squeeze', {})
    doge_bb_period  = doge_cfg.get('bb_period',        20)
    doge_bb_std_v   = doge_cfg.get('bb_std',           2.0)
    doge_kc_period  = doge_cfg.get('kc_period',        10)
    doge_kc_mult_v  = doge_cfg.get('kc_mult',          1.25)
    doge_mom_period = doge_cfg.get('mom_period',        12)
    doge_trail      = doge_cfg.get('trail_atr',         3.5)
    doge_atr_sl     = doge_cfg.get('atr_sl_mult',       2.0)
    doge_leverage   = doge_cfg.get('leverage',          1)
    doge_max_cap    = doge_cfg.get('max_trade_cap',     200000.0)
    doge_max_loss   = doge_cfg.get('max_consec_losses', 3)
    doge_fee        = fee_rate

    # ── DOGE Squeeze 指標計算 ────────────────────────────────
    _dc = df_doge['close']
    _dh = df_doge['high']
    _dl = df_doge['low']
    _bb_mid   = _dc.rolling(doge_bb_period).mean()
    _bb_std   = _dc.rolling(doge_bb_period).std(ddof=0)
    _bb_upper = _bb_mid + doge_bb_std_v * _bb_std
    _bb_lower = _bb_mid - doge_bb_std_v * _bb_std
    _tr = pd.concat([
        _dh - _dl,
        (_dh - _dc.shift(1)).abs(),
        (_dl - _dc.shift(1)).abs(),
    ], axis=1).max(axis=1)
    _d_atr    = _tr.rolling(doge_kc_period).mean()
    _kc_mid   = _dc.rolling(doge_kc_period).mean()
    _kc_upper = _kc_mid + doge_kc_mult_v * _d_atr
    _kc_lower = _kc_mid - doge_kc_mult_v * _d_atr
    _sq       = ((_bb_upper < _kc_upper) & (_bb_lower > _kc_lower)).astype(int)
    _rl_hi    = _dh.rolling(doge_mom_period).max()
    _rl_lo    = _dl.rolling(doge_mom_period).min()
    _mom      = _dc - ((_rl_hi + _rl_lo) / 2 + _kc_mid) / 2
    df_doge           = df_doge.copy()
    df_doge['_atr']   = _d_atr
    df_doge['_sq']    = _sq
    df_doge['_mom']   = _mom
    df_doge.dropna(inplace=True)

    # ── 建立查詢字典 ─────────────────────────────────────────
    sol_dict  = {row.Index: row for row in df_sol.itertuples()}
    ada_dict  = {row.Index: row for row in df_ada.itertuples()}
    xrp_dict  = {row.Index: row for row in df_xrp.itertuples()}
    doge_dict = {row.Index: row for row in df_doge.itertuples()}

    all_ts = sorted(set(
        list(df_sol.index) + list(df_ada.index) +
        list(df_xrp.index) + list(df_doge.index)
    ))

    # ADA Donchian 預計算
    ada_highs = df_ada['high'].values
    ada_lows  = df_ada['low'].values
    ada_idx   = df_ada.index
    ada_i_map = {ts: i for i, ts in enumerate(ada_idx)}

    # XRP Fib 預計算
    xrp_highs  = df_xrp['high'].values
    xrp_lows   = df_xrp['low'].values
    xrp_idx    = df_xrp.index
    xrp_i_map  = {ts: i for i, ts in enumerate(xrp_idx)}

    # DOGE Squeeze 陣列
    doge_sq_arr  = df_doge['_sq'].values
    doge_mom_arr = df_doge['_mom'].values
    doge_atr_arr = df_doge['_atr'].values
    doge_idx     = df_doge.index
    doge_i_map   = {ts: i for i, ts in enumerate(doge_idx)}

    # ── 共享資金 ─────────────────────────────────────────────
    capital = initial_cap

    # ── SOL 狀態 ─────────────────────────────────────────────
    sol_pos = 0; sol_size = 0.0; sol_entry = 0.0
    sol_sl = 0.0; sol_tsl = 0.0; sol_liq = 0.0
    sol_high = 0.0; sol_low = float('inf')
    sol_entry_fee = 0.0
    sol_long_sig = False; sol_short_sig = False
    sol_consec = 0; sol_skip = False; sol_in_skip = False
    sol_be_activated = False
    sol_saved_sl_dist = 0.0
    sol_margin_used = 0.0
    sol_trades = []
    sol_twap_active = False; sol_twap_remaining = 0
    sol_twap_size_each = 0.0; sol_twap_direction = 0
    _sol_consol_highs = _deque(maxlen=8)
    _sol_consol_lows  = _deque(maxlen=8)
    prev_sol_ts = None

    # ── Feature deque（盤整縮緊，三策略共用參數）─────────────
    _cn = consol_n if consol_n > 0 else 1  # deque 最小 1，disabled 時不讀取
    _ada_ch  = _deque(maxlen=_cn); _ada_cl  = _deque(maxlen=_cn)
    _xrp_ch  = _deque(maxlen=_cn); _xrp_cl  = _deque(maxlen=_cn)
    _doge_ch = _deque(maxlen=_cn); _doge_cl = _deque(maxlen=_cn)

    # ── ADA 狀態 ─────────────────────────────────────────────
    ada_pos = 0; ada_size = 0.0; ada_entry = 0.0
    ada_tsl = 0.0; ada_hp = 0.0; ada_lp = float('inf')
    ada_entry_fee = 0.0; ada_margin_used = 0.0
    ada_consec = 0; ada_skip = False; ada_in_skip = False
    ada_trades = []
    ada_twap_active = False; ada_twap_remaining = 0
    ada_twap_size_each = 0.0; ada_twap_direction = 0
    ada_twap_pending = False

    # ── XRP 狀態（限價進場，直接 pending） ───────────────────
    xrp_pos = 0; xrp_size = 0.0; xrp_entry = 0.0
    xrp_tsl = 0.0; xrp_hp = 0.0; xrp_lp = float('inf')
    xrp_entry_fee = 0.0; xrp_margin_used = 0.0
    xrp_consec = 0; xrp_skip = False; xrp_in_skip = False
    xrp_pending_dir   = 0
    xrp_pending_price = 0.0   # 掛單 Fib 水平
    xrp_pending_bars  = 0     # 掛單後已過的 bar 數
    xrp_trades = []

    # ── DOGE 狀態 ─────────────────────────────────────────────
    doge_pos = 0; doge_size = 0.0; doge_entry = 0.0
    doge_tsl = 0.0; doge_hp = 0.0; doge_lp = float('inf')
    doge_entry_fee = 0.0; doge_margin_used = 0.0
    doge_consec = 0; doge_skip = False; doge_in_skip = False
    doge_trades = []
    doge_twap_active = False; doge_twap_remaining = 0
    doge_twap_size_each = 0.0; doge_twap_direction = 0
    doge_twap_pending = False

    # ── 淨值 / 統計 ─────────────────────────────────────────
    equity_curve = []
    peak = initial_cap; max_dd = 0.0
    simul_2 = 0; simul_3 = 0; simul_4 = 0

    # ══════════════════════════════════════════════════════════
    #  主循環
    # ══════════════════════════════════════════════════════════
    for ts in all_ts:
        if capital < 5.0:
            equity_curve.append({'timestamp': ts, 'equity': 0.0})
            continue

        is_sol  = ts in sol_dict
        is_ada  = ts in ada_dict
        is_xrp  = ts in xrp_dict
        is_doge = ts in doge_dict

        # ══════════════════════════════════════════════════════
        #  SOL 15m 事件
        # ══════════════════════════════════════════════════════
        if is_sol:
            row = sol_dict[ts]
            cur_open  = float(row.open);  cur_high = float(row.high)
            cur_low   = float(row.low);   cur_close = float(row.close)
            cur_atr   = float(row.ATR)

            # ── 進場執行（上一根訊號，本根開盤執行）──────────
            if sol_pos == 0 and (sol_long_sig or sol_short_sig):
                other_margin = ada_margin_used + xrp_margin_used + doge_margin_used
                free_margin  = max(0.0, capital - other_margin)
                alloc_margin = free_margin * _FREE_PCT

                sl_dist = sol_saved_sl_dist if sol_saved_sl_dist > 0 else max(sol_sl_atr * cur_atr, cur_open * 0.001)
                sol_saved_sl_dist = 0.0

                if sol_twap_active and sol_twap_remaining > 0:
                    sol_size = sol_twap_size_each
                else:
                    sol_size = alloc_margin * sol_leverage / cur_open

                sol_size = min(sol_size, max(0.0, free_margin * sol_leverage / cur_open))

                if sol_size > 0:
                    if sol_long_sig:
                        sol_entry = cur_open * (1 + SLIPPAGE)
                        sol_pos   = 1
                        sol_sl    = sol_tsl = sol_entry - sl_dist
                        sol_high  = cur_open
                        sol_liq   = CoreStrategy.calc_liquidation_price(sol_entry, 1, sol_leverage, sol_mmr)
                    else:
                        sol_entry = cur_open * (1 - SLIPPAGE)
                        sol_pos   = -1
                        sol_sl    = sol_tsl = sol_entry + sl_dist
                        sol_low   = cur_open
                        sol_liq   = CoreStrategy.calc_liquidation_price(sol_entry, -1, sol_leverage, sol_mmr)
                    sol_entry_fee    = sol_size * sol_entry * sol_fee
                    capital         -= sol_entry_fee
                    sol_margin_used  = sol_size * sol_entry / sol_leverage
                    sol_be_activated = False
                    if sol_twap_active:
                        sol_twap_remaining -= 1
                        if sol_twap_remaining == 0:
                            sol_twap_active = False
                else:
                    # 保證金不足無法進場，重置 TWAP 等待下次訊號
                    sol_twap_active    = False
                    sol_twap_remaining = 0

                sol_long_sig = sol_short_sig = False

            # ── SOL TWAP 後續子單 ────────────────────────────
            elif sol_pos != 0 and sol_twap_active and sol_twap_remaining > 0 and sol_pos == sol_twap_direction:
                sub_ep    = cur_open * (1 + SLIPPAGE) if sol_pos == 1 else cur_open * (1 - SLIPPAGE)
                sub_fee   = sol_twap_size_each * sub_ep * sol_fee
                capital  -= sub_fee
                sol_entry_fee += sub_fee
                total_sz  = sol_size + sol_twap_size_each
                sol_entry = (sol_entry * sol_size + sub_ep * sol_twap_size_each) / total_sz
                sol_size  = total_sz
                sol_liq   = CoreStrategy.calc_liquidation_price(sol_entry, sol_pos, sol_leverage, sol_mmr)
                sol_margin_used    = sol_size * sol_entry / sol_leverage
                sol_twap_remaining -= 1
                if sol_twap_remaining == 0:
                    sol_twap_active = False

            # ── 盤整縮緊 + 追蹤止損 ─────────────────────────
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
                sol_tsl, sol_high, sol_low = CoreStrategy.update_trailing_stop(
                    sol_pos, sol_tsl, sol_high, sol_low,
                    ref_high, ref_low, cur_atr, _eff_trail
                )

            # ── 2.0R 保本 ───────────────────────────────────
            if not sol_be_activated and sol_pos != 0:
                sol_sl_dist = abs(sol_entry - sol_sl)
                if sol_pos == 1 and cur_high >= sol_entry + 2.0 * sol_sl_dist:
                    be_price  = sol_entry / ((1 - sol_fee) * (1 - SLIPPAGE))
                    sol_tsl   = max(sol_tsl, be_price)
                    sol_be_activated = True
                elif sol_pos == -1 and cur_low <= sol_entry - 2.0 * sol_sl_dist:
                    be_price  = sol_entry / ((1 + sol_fee) * (1 + SLIPPAGE))
                    sol_tsl   = min(sol_tsl, be_price)
                    sol_be_activated = True

            # ── 出場判斷 ─────────────────────────────────────
            closed, _, pnl, reason = CoreStrategy.check_exit(
                sol_pos, cur_low, cur_high, cur_open,
                sol_liq, sol_tsl, sol_sl,
                sol_entry, sol_size, sol_fee, SLIPPAGE
            )
            if closed:
                if reason == '\U0001f480 Liquidation':
                    capital = max(0.0, capital + pnl)
                else:
                    capital += pnl
                net_pnl = pnl - sol_entry_fee
                sol_trades.append({'Time': ts, 'Strategy': 'SOL', 'PnL': net_pnl, 'Cap': capital})
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

            # ── 信號偵測（本根偵測，下根執行）────────────────
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
        if is_ada:
            ai    = ada_i_map[ts]
            a_row = ada_dict[ts]
            a_O   = float(a_row.open);  a_H = float(a_row.high)
            a_L   = float(a_row.low);   a_C = float(a_row.close)
            a_atr = float(a_row.ATR)

            if ai >= ada_entry_n:
                dc_high = np.max(ada_highs[ai - ada_entry_n:ai])
                dc_low  = np.min(ada_lows[ai - ada_entry_n:ai])
            else:
                dc_high = dc_low = None

            # ── ADA TWAP 第一份子單 ──────────────────────────
            if ada_pos == 0 and ada_twap_active and ada_twap_pending:
                sub_ep  = a_O * (1 + SLIPPAGE * ada_twap_direction)
                sub_fee = ada_twap_size_each * sub_ep * ada_fee
                capital -= sub_fee
                ada_entry_fee   = sub_fee
                ada_pos         = ada_twap_direction
                ada_size        = ada_twap_size_each
                ada_entry       = sub_ep
                ada_margin_used = ada_size * ada_entry / ada_leverage
                if ada_pos == 1:
                    ada_hp = a_H; ada_lp = float('inf')
                    ada_tsl = ada_entry - ada_trail * a_atr
                else:
                    ada_lp = a_L; ada_hp = 0.0
                    ada_tsl = ada_entry + ada_trail * a_atr
                ada_twap_remaining -= 1
                ada_twap_pending    = False

            # ── ADA TWAP 後續子單 ────────────────────────────
            elif ada_pos != 0 and ada_twap_active and ada_twap_remaining > 0 and ada_pos == ada_twap_direction:
                sub_ep  = a_O * (1 + SLIPPAGE * ada_pos)
                sub_fee = ada_twap_size_each * sub_ep * ada_fee
                capital -= sub_fee
                ada_entry_fee += sub_fee
                total_sz  = ada_size + ada_twap_size_each
                ada_entry = (ada_entry * ada_size + sub_ep * ada_twap_size_each) / total_sz
                ada_size  = total_sz
                ada_margin_used    = ada_size * ada_entry / ada_leverage
                ada_twap_remaining -= 1

            if ada_twap_active and ada_twap_remaining == 0 and not ada_twap_pending:
                ada_twap_active = False

            # ── ADA 出場 ─────────────────────────────────────
            if ada_pos != 0:
                closed_ada = False; xp = 0.0
                # 插針防護
                _a_spike = spike_mult > 0 and a_atr > 0 and (a_H - a_L) > spike_mult * a_atr
                a_ref_H  = a_C if (_a_spike and ada_pos == 1)  else a_H
                a_ref_L  = a_C if (_a_spike and ada_pos == -1) else a_L
                # 盤整縮緊
                _ada_ch.append(a_H); _ada_cl.append(a_L)
                if consol_n > 0 and len(_ada_ch) == consol_n and a_atr > 0:
                    _a_eff = tight_trail if (max(_ada_ch) - min(_ada_cl)) < consol_x * a_atr else ada_trail
                else:
                    _a_eff = ada_trail
                if ada_pos == 1:
                    ada_hp  = max(ada_hp, a_ref_H)
                    ada_tsl = max(ada_tsl, ada_hp - _a_eff * a_atr)
                    if a_L <= ada_tsl:
                        xp = max(ada_tsl, a_O) * (1 - SLIPPAGE); closed_ada = True
                elif ada_pos == -1:
                    ada_lp  = min(ada_lp, a_ref_L)
                    ada_tsl = min(ada_tsl, ada_lp + _a_eff * a_atr)
                    if a_H >= ada_tsl:
                        xp = min(ada_tsl, a_O) * (1 + SLIPPAGE); closed_ada = True

                if closed_ada:
                    gross = (xp - ada_entry) * ada_size * ada_pos
                    x_fee = xp * ada_size * ada_fee
                    net   = gross - ada_entry_fee - x_fee
                    capital += gross - x_fee
                    ada_trades.append({'Time': ts, 'Strategy': 'ADA', 'PnL': net, 'Cap': capital})
                    ada_pos = 0; ada_size = 0.0; ada_entry_fee = 0.0
                    ada_margin_used  = 0.0
                    ada_twap_active  = False; ada_twap_remaining = 0; ada_twap_pending = False
                    _ada_ch.clear(); _ada_cl.clear()
                    if net < 0:
                        ada_consec += 1
                        if ada_consec >= ada_max_loss:
                            ada_skip = True; ada_in_skip = False; ada_consec = 0
                    else:
                        ada_consec = 0

            # ── ADA 入場訊號 ─────────────────────────────────
            if ada_pos == 0 and not ada_twap_active and dc_high is not None and a_atr > 0:
                has_signal = (a_C > dc_high) or (a_C < dc_low)

                if ada_skip:
                    if has_signal:
                        ada_skip = False; ada_in_skip = True
                elif ada_in_skip:
                    if not has_signal:
                        ada_in_skip = False
                elif has_signal:
                    direction    = 1 if a_C > dc_high else -1
                    other_margin = sol_margin_used + xrp_margin_used + doge_margin_used
                    free_margin  = max(0.0, capital - other_margin)

                    if free_margin > 10:
                        N            = max(1, int(free_margin / ada_max_cap))
                        alloc_margin = free_margin * _FREE_PCT
                        size_each    = alloc_margin * ada_leverage / a_C if N == 1 else ada_max_cap / a_C

                        ada_twap_size_each = size_each
                        ada_twap_direction = direction
                        ada_twap_active    = True
                        ada_twap_remaining = N
                        ada_twap_pending   = True

        # ══════════════════════════════════════════════════════
        #  XRP 1h 事件（限價進場：等待 Fib 觸價 + 逾時取消）
        # ══════════════════════════════════════════════════════
        if is_xrp:
            xi    = xrp_i_map[ts]
            x_row = xrp_dict[ts]
            x_O   = float(x_row.open);  x_H = float(x_row.high)
            x_L   = float(x_row.low);   x_C = float(x_row.close)
            x_atr = float(x_row.ATR)
            x_ema = float(x_row.EMA)

            # ── 限價單成交確認 & 逾時取消 ────────────────────
            if xrp_pos == 0 and xrp_pending_dir != 0:
                xrp_pending_bars += 1
                tol = xrp_pending_price * xrp_fib_tol
                filled = False
                # 做多：本根 Low 觸及 Fib → 限價單成交（不需收盤確認，與交易所行為一致）
                if xrp_pending_dir == 1 and x_L <= xrp_pending_price + tol:
                    fill_p = min(x_O, xrp_pending_price) * (1 + SLIPPAGE)
                    filled = True
                # 做空：本根 High 觸及 Fib → 限價單成交
                elif xrp_pending_dir == -1 and x_H >= xrp_pending_price - tol:
                    fill_p = max(x_O, xrp_pending_price) * (1 - SLIPPAGE)
                    filled = True

                if filled:
                    other_margin = sol_margin_used + ada_margin_used + doge_margin_used
                    free_margin  = max(0.0, capital - other_margin)
                    if free_margin > 10:
                        N            = max(1, int(free_margin / xrp_max_cap))
                        alloc_margin = free_margin * _FREE_PCT
                        size = (alloc_margin * xrp_leverage / fill_p
                                if N == 1 else xrp_max_cap * xrp_leverage / fill_p)
                        if size > 0:
                            xrp_entry     = fill_p
                            xrp_entry_fee = size * xrp_entry * xrp_fee
                            capital      -= xrp_entry_fee
                            xrp_pos       = xrp_pending_dir
                            xrp_size      = size
                            xrp_margin_used = xrp_size * xrp_entry / xrp_leverage
                            if xrp_pos == 1:
                                xrp_hp = x_H; xrp_lp = float('inf')
                                xrp_tsl = xrp_entry - xrp_trail * x_atr
                            else:
                                xrp_lp = x_L; xrp_hp = 0.0
                                xrp_tsl = xrp_entry + xrp_trail * x_atr
                    xrp_pending_dir = 0; xrp_pending_price = 0.0; xrp_pending_bars = 0

                elif xrp_limit_max_bars > 0 and xrp_pending_bars >= xrp_limit_max_bars:
                    # 逾時取消掛單
                    xrp_pending_dir = 0; xrp_pending_price = 0.0; xrp_pending_bars = 0

            # ── XRP 出場 ─────────────────────────────────────
            if xrp_pos != 0:
                closed_xrp = False; xp = 0.0
                # 插針防護
                _x_spike = spike_mult > 0 and x_atr > 0 and (x_H - x_L) > spike_mult * x_atr
                x_ref_H  = x_C if (_x_spike and xrp_pos == 1)  else x_H
                x_ref_L  = x_C if (_x_spike and xrp_pos == -1) else x_L
                # 盤整縮緊
                _xrp_ch.append(x_H); _xrp_cl.append(x_L)
                if consol_n > 0 and len(_xrp_ch) == consol_n and x_atr > 0:
                    _x_eff = tight_trail if (max(_xrp_ch) - min(_xrp_cl)) < consol_x * x_atr else xrp_trail
                else:
                    _x_eff = xrp_trail
                if xrp_pos == 1:
                    xrp_hp  = max(xrp_hp, x_ref_H)
                    xrp_tsl = max(xrp_tsl, xrp_hp - _x_eff * x_atr)
                    if x_L <= xrp_tsl:
                        xp = max(xrp_tsl, x_O) * (1 - SLIPPAGE); closed_xrp = True
                elif xrp_pos == -1:
                    xrp_lp  = min(xrp_lp, x_ref_L)
                    xrp_tsl = min(xrp_tsl, xrp_lp + _x_eff * x_atr)
                    if x_H >= xrp_tsl:
                        xp = min(xrp_tsl, x_O) * (1 + SLIPPAGE); closed_xrp = True

                if closed_xrp:
                    gross = (xp - xrp_entry) * xrp_size * xrp_pos
                    x_fee = xp * xrp_size * xrp_fee
                    net   = gross - xrp_entry_fee - x_fee
                    capital += gross - x_fee
                    xrp_trades.append({'Time': ts, 'Strategy': 'XRP', 'PnL': net, 'Cap': capital})
                    xrp_pos = 0; xrp_size = 0.0; xrp_entry_fee = 0.0
                    xrp_margin_used = 0.0
                    _xrp_ch.clear(); _xrp_cl.clear()
                    if net < 0:
                        xrp_consec += 1
                        if xrp_consec >= xrp_max_loss:
                            xrp_skip = True; xrp_in_skip = False; xrp_consec = 0
                    else:
                        xrp_consec = 0

            # ── XRP 入場訊號：計算 Fib → 掛出限價單 ────────
            # 條件：趨勢 + Swing 方向一致 + 當前價格在 Fib 外側（尚未觸及）
            if xrp_pos == 0 and xrp_pending_dir == 0 and xi >= xrp_swing_n and x_atr > 0:
                _win_xh    = xrp_highs[xi - xrp_swing_n:xi]
                _win_xl    = xrp_lows[xi - xrp_swing_n:xi]
                swing_high = np.max(_win_xh)
                swing_low  = np.min(_win_xl)
                swing_rng  = swing_high - swing_low
                _xhi_idx   = int(np.argmax(_win_xh))
                _xlo_idx   = int(np.argmin(_win_xl))

                direction = 0; fib_price = 0.0
                if swing_rng > 0:
                    if x_C > x_ema and _xhi_idx > _xlo_idx:
                        fp = swing_high - xrp_fib_level * swing_rng
                        if x_C > fp:          # 當前價在 Fib 上方，等待回撤
                            direction = 1; fib_price = fp
                    elif x_C < x_ema and _xlo_idx > _xhi_idx:
                        fp = swing_low + xrp_fib_level * swing_rng
                        if x_C < fp:          # 當前價在 Fib 下方，等待反彈
                            direction = -1; fib_price = fp

                has_sig = direction != 0
                if xrp_skip:
                    if has_sig:
                        xrp_skip = False; xrp_in_skip = True
                elif xrp_in_skip:
                    if not has_sig:
                        xrp_in_skip = False
                elif has_sig:
                    xrp_pending_dir   = direction
                    xrp_pending_price = fib_price
                    xrp_pending_bars  = 0

        # ══════════════════════════════════════════════════════
        #  DOGE 1h 事件（Squeeze 爆發，市價 TWAP）
        # ══════════════════════════════════════════════════════
        if is_doge:
            di    = doge_i_map[ts]
            d_row = doge_dict[ts]
            d_O   = float(d_row.open);  d_H = float(d_row.high)
            d_L   = float(d_row.low);   d_C = float(d_row.close)
            d_atr      = doge_atr_arr[di]
            d_atr_prev = doge_atr_arr[di - 1] if di > 0 else d_atr

            # ── DOGE TWAP 第一份子單 ─────────────────────────
            if doge_pos == 0 and doge_twap_active and doge_twap_pending:
                sub_ep  = d_O * (1 + SLIPPAGE * doge_twap_direction)
                sub_fee = doge_twap_size_each * sub_ep * doge_fee
                capital -= sub_fee
                doge_entry_fee   = sub_fee
                doge_pos         = doge_twap_direction
                doge_size        = doge_twap_size_each
                doge_entry       = sub_ep
                doge_margin_used = doge_size * doge_entry / doge_leverage
                if doge_pos == 1:
                    doge_hp = d_H; doge_lp = float('inf')
                    doge_tsl = doge_entry - doge_trail * d_atr_prev
                else:
                    doge_lp = d_L; doge_hp = 0.0
                    doge_tsl = doge_entry + doge_trail * d_atr_prev
                doge_twap_remaining -= 1
                doge_twap_pending    = False

            # ── DOGE TWAP 後續子單 ───────────────────────────
            elif doge_pos != 0 and doge_twap_active and doge_twap_remaining > 0 and doge_pos == doge_twap_direction:
                sub_ep  = d_O * (1 + SLIPPAGE * doge_pos)
                sub_fee = doge_twap_size_each * sub_ep * doge_fee
                capital -= sub_fee
                doge_entry_fee += sub_fee
                total_sz   = doge_size + doge_twap_size_each
                doge_entry = (doge_entry * doge_size + sub_ep * doge_twap_size_each) / total_sz
                doge_size  = total_sz
                doge_margin_used    = doge_size * doge_entry / doge_leverage
                doge_twap_remaining -= 1

            if doge_twap_active and doge_twap_remaining == 0 and not doge_twap_pending:
                doge_twap_active = False

            # ── DOGE 出場 ────────────────────────────────────
            if doge_pos != 0:
                closed_doge = False; xp = 0.0
                # 插針防護
                _d_spike = spike_mult > 0 and d_atr > 0 and (d_H - d_L) > spike_mult * d_atr
                d_ref_H  = d_C if (_d_spike and doge_pos == 1)  else d_H
                d_ref_L  = d_C if (_d_spike and doge_pos == -1) else d_L
                # 盤整縮緊
                _doge_ch.append(d_H); _doge_cl.append(d_L)
                if consol_n > 0 and len(_doge_ch) == consol_n and d_atr > 0:
                    _d_eff = tight_trail if (max(_doge_ch) - min(_doge_cl)) < consol_x * d_atr else doge_trail
                else:
                    _d_eff = doge_trail
                if doge_pos == 1:
                    doge_hp  = max(doge_hp, d_ref_H)
                    doge_tsl = max(doge_tsl, doge_hp - _d_eff * d_atr)
                    if d_L <= doge_tsl:
                        xp = max(doge_tsl, d_O) * (1 - SLIPPAGE); closed_doge = True
                elif doge_pos == -1:
                    doge_lp  = min(doge_lp, d_ref_L)
                    doge_tsl = min(doge_tsl, doge_lp + _d_eff * d_atr)
                    if d_H >= doge_tsl:
                        xp = min(doge_tsl, d_O) * (1 + SLIPPAGE); closed_doge = True

                if closed_doge:
                    gross = (xp - doge_entry) * doge_size * doge_pos
                    x_fee = xp * doge_size * doge_fee
                    net   = gross - doge_entry_fee - x_fee
                    capital += gross - x_fee
                    doge_trades.append({'Time': ts, 'Strategy': 'DOGE', 'PnL': net, 'Cap': capital})
                    doge_pos = 0; doge_size = 0.0; doge_entry_fee = 0.0
                    _doge_ch.clear(); _doge_cl.clear()
                    doge_margin_used  = 0.0
                    doge_twap_active  = False; doge_twap_remaining = 0; doge_twap_pending = False
                    if net < 0:
                        doge_consec += 1
                        if doge_consec >= doge_max_loss:
                            doge_skip = True; doge_in_skip = False; doge_consec = 0
                    else:
                        doge_consec = 0

            # ── DOGE 入場訊號：Squeeze Fire ──────────────────
            if doge_pos == 0 and not doge_twap_active and di > 0:
                is_fire = (doge_sq_arr[di - 1] == 1 and doge_sq_arr[di] == 0
                           and not np.isnan(d_atr) and d_atr > 0)
                direction = (1 if doge_mom_arr[di] > 0 else -1) if is_fire else 0

                if doge_skip:
                    if is_fire:
                        doge_skip = False; doge_in_skip = True
                elif doge_in_skip:
                    if not is_fire:
                        doge_in_skip = False
                elif is_fire:
                    other_margin = sol_margin_used + ada_margin_used + xrp_margin_used
                    free_margin  = max(0.0, capital - other_margin)
                    if free_margin > 10:
                        N            = max(1, int(free_margin / doge_max_cap))
                        alloc_margin = free_margin * _FREE_PCT
                        size_each    = alloc_margin * doge_leverage / d_C if N == 1 else doge_max_cap / d_C

                        doge_twap_size_each = size_each
                        doge_twap_direction = direction
                        doge_twap_active    = True
                        doge_twap_remaining = N
                        doge_twap_pending   = True

        # ── 同時持倉統計 ─────────────────────────────────────
        active = (sol_pos != 0) + (ada_pos != 0) + (xrp_pos != 0) + (doge_pos != 0)
        if   active == 2: simul_2 += 1
        elif active == 3: simul_3 += 1
        elif active == 4: simul_4 += 1

        # ── 淨值快照 ─────────────────────────────────────────
        if is_sol or is_ada or is_xrp or is_doge:
            unr = 0.0
            if sol_pos  != 0 and is_sol:
                unr += (float(sol_dict[ts].close)  - sol_entry)  * sol_size  * sol_pos
            if ada_pos  != 0 and is_ada:
                unr += (float(ada_dict[ts].close)  - ada_entry)  * ada_size  * ada_pos
            if xrp_pos  != 0 and is_xrp:
                unr += (float(xrp_dict[ts].close)  - xrp_entry)  * xrp_size  * xrp_pos
            if doge_pos != 0 and is_doge:
                unr += (float(doge_dict[ts].close) - doge_entry) * doge_size * doge_pos
            equity = capital + unr
            equity_curve.append({'timestamp': ts, 'equity': equity})
            if equity > peak:
                peak = equity
            dd = (equity - peak) / peak if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd

    # ── 強制平倉 ─────────────────────────────────────────────
    if sol_pos != 0:
        sc  = df_sol['close'].iloc[-1]
        xp  = sc * (1 - SLIPPAGE if sol_pos == 1 else 1 + SLIPPAGE)
        pnl = (xp - sol_entry) * sol_size * sol_pos - xp * sol_size * sol_fee
        net = pnl - sol_entry_fee
        capital += pnl
        sol_trades.append({'Time': df_sol.index[-1], 'Strategy': 'SOL', 'PnL': net, 'Cap': capital})

    if ada_pos != 0:
        ac    = df_ada['close'].iloc[-1]
        xp    = ac * (1 - SLIPPAGE if ada_pos == 1 else 1 + SLIPPAGE)
        gross = (xp - ada_entry) * ada_size * ada_pos
        x_fee = xp * ada_size * ada_fee
        net   = gross - ada_entry_fee - x_fee
        capital += gross - x_fee
        ada_trades.append({'Time': df_ada.index[-1], 'Strategy': 'ADA', 'PnL': net, 'Cap': capital})

    if xrp_pos != 0:
        xc    = df_xrp['close'].iloc[-1]
        xp    = xc * (1 - SLIPPAGE if xrp_pos == 1 else 1 + SLIPPAGE)
        gross = (xp - xrp_entry) * xrp_size * xrp_pos
        x_fee = xp * xrp_size * xrp_fee
        net   = gross - xrp_entry_fee - x_fee
        capital += gross - x_fee
        xrp_trades.append({'Time': df_xrp.index[-1], 'Strategy': 'XRP', 'PnL': net, 'Cap': capital})

    if doge_pos != 0:
        dc    = df_doge['close'].iloc[-1]
        xp    = dc * (1 - SLIPPAGE if doge_pos == 1 else 1 + SLIPPAGE)
        gross = (xp - doge_entry) * doge_size * doge_pos
        x_fee = xp * doge_size * doge_fee
        net   = gross - doge_entry_fee - x_fee
        capital += gross - x_fee
        doge_trades.append({'Time': df_doge.index[-1], 'Strategy': 'DOGE', 'PnL': net, 'Cap': capital})

    # ── 統計 ─────────────────────────────────────────────────
    eq_df = pd.DataFrame(equity_curve)

    def _stats(trades):
        if not trades:
            return 0, 0.0, 0.0, 0.0, 0.0, 0.0
        wins   = [t['PnL'] for t in trades if t['PnL'] > 0]
        losses = [t['PnL'] for t in trades if t['PnL'] <= 0]
        total  = len(trades)
        wr     = len(wins) / total * 100 if total > 0 else 0
        pf     = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        pnl    = sum(t['PnL'] for t in trades)
        avg_w  = sum(wins)   / len(wins)   if wins   else 0.0
        avg_l  = sum(losses) / len(losses) if losses else 0.0
        return total, wr, pf, pnl, avg_w, avg_l

    sol_n,  sol_wr,  sol_pf,  sol_pnl,  sol_avgw,  sol_avgl  = _stats(sol_trades)
    ada_n,  ada_wr,  ada_pf,  ada_pnl,  ada_avgw,  ada_avgl  = _stats(ada_trades)
    xrp_n,  xrp_wr,  xrp_pf,  xrp_pnl,  xrp_avgw,  xrp_avgl  = _stats(xrp_trades)
    doge_n, doge_wr, doge_pf, doge_pnl, doge_avgw, doge_avgl = _stats(doge_trades)
    all_n,  all_wr,  all_pf,  _,         _,         _         = _stats(
        sol_trades + ada_trades + xrp_trades + doge_trades)

    # Sharpe
    sharpe = 0.0
    if len(eq_df) > 1:
        eq_s      = eq_df.set_index('timestamp')['equity']
        daily     = eq_s.resample('D').last().ffill()
        daily_ret = daily.pct_change().dropna()
        if daily_ret.std() > 0:
            sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(365))

    return {
        'label':  label,
        'final':  capital,
        'ret%':   (capital / initial_cap - 1) * 100,
        'mdd%':   max_dd * 100,
        'sharpe': sharpe,
        'sol_n':  sol_n,  'sol_wr':  sol_wr,  'sol_pf':  sol_pf,  'sol_pnl':  sol_pnl,
        'sol_avgw':  sol_avgw,  'sol_avgl':  sol_avgl,
        'ada_n':  ada_n,  'ada_wr':  ada_wr,  'ada_pf':  ada_pf,  'ada_pnl':  ada_pnl,
        'ada_avgw':  ada_avgw,  'ada_avgl':  ada_avgl,
        'xrp_n':  xrp_n,  'xrp_wr':  xrp_wr,  'xrp_pf':  xrp_pf,  'xrp_pnl':  xrp_pnl,
        'xrp_avgw':  xrp_avgw,  'xrp_avgl':  xrp_avgl,
        'doge_n': doge_n, 'doge_wr': doge_wr, 'doge_pf': doge_pf, 'doge_pnl': doge_pnl,
        'doge_avgw': doge_avgw, 'doge_avgl': doge_avgl,
        'all_n':  all_n,  'all_wr':  all_wr,  'all_pf':  all_pf,
        'simul_2': simul_2, 'simul_3': simul_3, 'simul_4': simul_4,
        'equity_df':   eq_df,
        'sol_trades':  sol_trades,
        'ada_trades':  ada_trades,
        'xrp_trades':  xrp_trades,
        'doge_trades': doge_trades,
    }


def _yearly_table(trades, label):
    if not trades:
        return
    df   = pd.DataFrame(trades)
    df['Year'] = pd.to_datetime(df['Time']).dt.year
    rows = []
    for y, g in df.groupby('Year'):
        wins   = g[g['PnL'] > 0]['PnL']
        losses = g[g['PnL'] <= 0]['PnL']
        total  = len(g)
        wr     = len(wins) / total * 100 if total > 0 else 0
        pf     = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float('inf')
        avgw   = wins.mean()   if len(wins)   > 0 else 0
        avgl   = losses.mean() if len(losses) > 0 else 0
        rr     = abs(avgw / avgl) if avgl != 0 else 0
        rows.append({'年度': y, '交易': total, '勝率%': round(wr, 1),
                     '損益': round(g['PnL'].sum(), 2),
                     'PF': round(pf, 2), '盈虧比': round(rr, 2)})
    df_y = pd.DataFrame(rows)
    W = 62
    print(f"\n  {'─'*W}")
    print(f"  {label} 年度明細")
    print(f"  {'─'*W}")
    print(f"  {'年度':>4}  {'交易':>5}  {'勝率%':>6}  {'損益':>14}  {'PF':>5}  {'盈虧比':>6}")
    print(f"  {'─'*W}")
    for _, row in df_y.iterrows():
        print(f"  {int(row['年度']):>4}  {int(row['交易']):>5}  {row['勝率%']:>6.1f}  "
              f"{row['損益']:>+14,.2f}  {row['PF']:>5.2f}  {row['盈虧比']:>6.2f}")
    print(f"  {'─'*W}")


def print_result(r):
    print(f"\n  {'='*72}")
    print(f"  {r['label']}")
    print(f"  {'='*72}")
    print(f"  最終資金:   ${r['final']:,.2f}")
    print(f"  總報酬:     {r['ret%']:+,.1f}%")
    print(f"  最大回撤:   {r['mdd%']:+.1f}%")
    print(f"  Sharpe:     {r['sharpe']:.3f}")
    print(f"  總 PF:      {r['all_pf']:.2f}")
    print()
    print(f"  {'策略':<6} {'交易數':>6} {'勝率%':>8} {'PF':>7} {'盈虧比':>7} {'損益':>16}")
    print(f"  {'-'*58}")
    for key, name in [('sol', 'SOL'), ('ada', 'ADA'), ('xrp', 'XRP'), ('doge', 'DOGE')]:
        n    = r[f'{key}_n']
        wr   = r[f'{key}_wr']
        pf   = r[f'{key}_pf']
        pnl  = r[f'{key}_pnl']
        avgw = r[f'{key}_avgw']
        avgl = r[f'{key}_avgl']
        rr   = abs(avgw / avgl) if avgl != 0 else 0
        print(f"  {name:<6} {n:>6} {wr:>7.1f}% {pf:>7.2f} {rr:>7.2f} {pnl:>+15,.2f}")
    print(f"  {'合計':<6} {r['all_n']:>6} {r['all_wr']:>7.1f}% {r['all_pf']:>7.2f}")
    print()
    print(f"  同時持倉 2 策略: {r['simul_2']:,} 根 K 棒")
    print(f"  同時持倉 3 策略: {r['simul_3']:,} 根 K 棒")
    print(f"  同時持倉 4 策略: {r['simul_4']:,} 根 K 棒")

    _yearly_table(r['sol_trades'],  'SOL 趨勢')
    _yearly_table(r['ada_trades'],  'ADA 唐奇安')
    _yearly_table(r['xrp_trades'],  'XRP 斐波')
    _yearly_table(r['doge_trades'], 'DOGE Squeeze')


def _build_single_equity(trades: list, initial_cap: float,
                         time_index: pd.DatetimeIndex) -> pd.Series:
    if not trades:
        return pd.Series(initial_cap, index=time_index)
    times   = pd.to_datetime([t['Time'] for t in trades])
    returns = pd.Series(
        [t['PnL'] / t['Cap'] if t.get('Cap', 0) > 0 else 0.0 for t in trades],
        index=times
    )
    returns    = returns.groupby(level=0).sum()
    cum_equity = initial_cap * (1 + returns).cumprod()
    return cum_equity.reindex(time_index).ffill().fillna(initial_cap)


def main():
    parser = argparse.ArgumentParser(description='四策略聯合回測')
    parser.add_argument('--xrp_timeout', type=int, default=None,
                        help='XRP 限價單最大逾時小時數（0=不逾時，覆蓋 config 設定）')
    args = parser.parse_args()

    config      = load_config()
    bt          = config.get('backtest', {})
    initial_cap = config['risk']['initial_capital']

    # CLI 覆蓋 config 設定
    if args.xrp_timeout is not None:
        config.setdefault('xrp_fib', {})['xrp_limit_max_hours'] = args.xrp_timeout

    xrp_timeout = config.get('xrp_fib', {}).get('xrp_limit_max_hours', 0)
    timeout_str = f"{xrp_timeout}h" if xrp_timeout > 0 else "不逾時"

    print(f"四策略聯合回測 — SOL 趨勢 + ADA 唐奇安 + XRP 斐波 + DOGE Squeeze")
    print(f"  資金方案: 可用保證金 × {_FREE_PCT*100:.0f}%（free_b）")
    print(f"  期間: {bt.get('start_date','?')} ~ {bt.get('end_date','?')}")
    print(f"  初始資金: ${initial_cap}")
    print(f"  XRP 限價逾時: {timeout_str}")

    label = f"可用×{_FREE_PCT*100:.0f}% XRP逾時={timeout_str}"
    r     = run_triple(config, label=label)
    print_result(r)

    # ── 淨值曲線 ─────────────────────────────────────────────
    eq         = r['equity_df']
    time_index = pd.DatetimeIndex(eq['timestamp'])
    _floor     = 1.0

    sol_eq  = _build_single_equity(r['sol_trades'],  initial_cap, time_index)
    ada_eq  = _build_single_equity(r['ada_trades'],  initial_cap, time_index)
    xrp_eq  = _build_single_equity(r['xrp_trades'],  initial_cap, time_index)
    doge_eq = _build_single_equity(r['doge_trades'], initial_cap, time_index)

    _, ax = plt.subplots(figsize=(16, 7))
    ax.plot(sol_eq.index,  sol_eq.clip(lower=_floor),  color="#D58035", linewidth=1.0,
            alpha=0.7, label=f"SOL  (獨立) +{r['sol_pnl']:,.0f}")
    ax.plot(ada_eq.index,  ada_eq.clip(lower=_floor),  color="#44C744", linewidth=1.0,
            alpha=0.7, label=f"ADA  (獨立) +{r['ada_pnl']:,.0f}")
    ax.plot(xrp_eq.index,  xrp_eq.clip(lower=_floor),  color="#E069E0", linewidth=1.0,
            alpha=0.7, label=f"XRP  (獨立) +{r['xrp_pnl']:,.0f}")
    ax.plot(doge_eq.index, doge_eq.clip(lower=_floor), color="#5DADE2", linewidth=1.0,
            alpha=0.7, label=f"DOGE (獨立) +{r['doge_pnl']:,.0f}")
    ax.plot(pd.DatetimeIndex(eq['timestamp']), eq['equity'].clip(lower=_floor),
            color="#2ecc71", linewidth=1.5,
            label=f"四策略聯合  +{r['ret%']:.0f}%")

    ax.axhline(initial_cap, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_title(
        f"四策略聯合 {label} | MDD {r['mdd%']:+.1f}% | Sharpe {r['sharpe']:.3f}",
        fontsize=14
    )
    ax.set_ylabel('資金 (USDT)')
    ax.set_xlabel('日期')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('multiple_backtest.png', dpi=150)
    plt.show()
    print(f"\n  圖表已儲存: multiple_backtest.png")


if __name__ == '__main__':
    main()
