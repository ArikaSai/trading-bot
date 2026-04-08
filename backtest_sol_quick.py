"""
backtest_sol_quick.py
══════════════════════
快速查看 SOL 當前策略在指定時間區間的淨值曲線。
調整下方 START / END 即可。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from backtest import load_config
from strategy import CoreStrategy

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# ── 調整這裡 ──────────────────────────────────────────────────
START = "2020-08-19"
END   = "2026-04-04"
WEEKLY_WITHDRAWAL_PCT   = 0.01  # 每週提現比例上限（相對於當前資金）
WEEKLY_WITHDRAWAL_AMOUNT  =  30_000.0  # 每週提現金額上限（U）
WEEKLY_WITHDRAWAL_START   = 100_000.0  # 帳戶達此金額後才啟動每週提現
MAX_POSITION_CAP = 40_000_000.0  # 幣安最大持倉限制（8× 槓桿下）
# ─────────────────────────────────────────────────────────────

SLIPPAGE_PCT = 0.0010
SPIKE_N      = 4.0
BREAKEVEN_R  = 2.0
CONSOL_N     = 8
CONSOL_X     = 1.25
CONSOL_TIGHT = 1.0



def run_backtest_adaptive_twap(df_slice, config, slippage_pct=SLIPPAGE_PCT,
                               enable_withdrawal=False):
    """
    自適應 TWAP：
    - 資金 < max_trade_usdt_cap → 正常單筆進場（與一般版本完全相同）
    - 資金 = N × max_trade_usdt_cap → 拆成 N 根 K 棒，每份吃滿 max_trade_usdt_cap
    止損距離由 ATR 計算，停損價格從第一份子單起生效。
    """
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
    adx_threshold      = config['strategy']['adx_threshold']

    capital       = initial_capital
    position      = 0
    position_size = 0.0
    entry_price   = 0.0
    liq_price     = 0.0
    initial_risk  = 0.0
    entry_time    = None
    stop_loss     = 0.0
    trailing_stop = 0.0
    highest_price = 0.0
    lowest_price  = float('inf')
    be_activated  = False
    entry_fee     = 0.0

    # 自適應 TWAP 狀態
    atw_active        = False   # 是否有 TWAP 任務進行中
    atw_remaining     = 0       # 剩餘未執行子單數
    atw_size_each     = 0.0     # 每份子單大小（單位）
    atw_direction     = 0       # 方向
    atw_sl_dist       = 0.0     # 止損距離（鎖定於訊號時）
    atw_first_pending = False   # 第一份子單待執行

    consecutive_losses          = 0
    skip_next_trade             = False
    in_skip_zone                = False

    consol_highs = deque(maxlen=CONSOL_N)
    consol_lows  = deque(maxlen=CONSOL_N)

    trade_log          = []
    equity_curve       = []
    total_withdrawn    = 0.0
    last_withdrawal_ts = None

    for row in df_slice.itertuples():
        cur_open  = row.open;   cur_high  = row.high
        cur_low   = row.low;    cur_close = row.close
        cur_atr   = row.ATR
        timestamp = row.Index

        consol_highs.append(cur_high)
        consol_lows.append(cur_low)

        if capital < 5.0:
            equity_curve.append({'timestamp': timestamp, 'equity': 0.0})
            continue

        # ── 第一份子單（訊號後下一根 K 棒）──────────────────────
        if position == 0 and atw_active and atw_first_pending:
            ep = cur_open * (1 + slippage_pct) if atw_direction == 1 \
                 else cur_open * (1 - slippage_pct)
            entry_fee     = atw_size_each * ep * fee_rate
            capital      -= entry_fee
            position      = atw_direction
            position_size = atw_size_each
            entry_price   = ep
            entry_time    = timestamp
            stop_loss     = ep - atw_sl_dist if position == 1 else ep + atw_sl_dist
            trailing_stop = stop_loss
            highest_price = cur_open
            lowest_price  = cur_open
            liq_price     = CoreStrategy.calc_liquidation_price(ep, position, leverage, mmr)
            be_activated  = False
            atw_remaining    -= 1
            atw_first_pending = False

        # ── 後續子單（每根 K 棒自動加一份）──────────────────────
        elif position != 0 and atw_active and atw_remaining > 0 and not atw_first_pending:
            if position == atw_direction:
                ep = cur_open * (1 + slippage_pct) if position == 1 \
                     else cur_open * (1 - slippage_pct)
                sub_fee       = atw_size_each * ep * fee_rate
                capital      -= sub_fee
                entry_fee    += sub_fee
                total_size    = position_size + atw_size_each
                entry_price   = (entry_price * position_size + ep * atw_size_each) / total_size
                position_size = total_size
                liq_price     = CoreStrategy.calc_liquidation_price(entry_price, position, leverage, mmr)
                atw_remaining -= 1

        if atw_active and atw_remaining == 0 and not atw_first_pending:
            atw_active = False

        # ── 盤整縮緊 ──────────────────────────────────────────
        if position != 0 and len(consol_highs) == CONSOL_N and cur_atr > 0:
            is_consol  = (max(consol_highs) - min(consol_lows)) < CONSOL_X * cur_atr
            eff_trail  = CONSOL_TIGHT if is_consol else trailing_atr
        else:
            eff_trail  = trailing_atr

        # ── 插針過濾 + 追蹤止損更新 ──────────────────────────
        is_spike = cur_atr > 0 and (cur_high - cur_low) > SPIKE_N * cur_atr
        ref_high = cur_close if (is_spike and position == 1)  else cur_high
        ref_low  = cur_close if (is_spike and position == -1) else cur_low
        trailing_stop, highest_price, lowest_price = CoreStrategy.update_trailing_stop(
            position, trailing_stop, highest_price, lowest_price,
            ref_high, ref_low, cur_atr, eff_trail
        )

        # ── 2.0R 保本 ─────────────────────────────────────────
        if not be_activated and position != 0:
            sl_dist_be = abs(entry_price - stop_loss)
            if position == 1 and cur_high >= entry_price + BREAKEVEN_R * sl_dist_be:
                be_price      = entry_price / ((1 - fee_rate) * (1 - slippage_pct))
                trailing_stop = max(trailing_stop, be_price)
                be_activated  = True
            elif position == -1 and cur_low <= entry_price - BREAKEVEN_R * sl_dist_be:
                be_price      = entry_price / ((1 + fee_rate) * (1 + slippage_pct))
                trailing_stop = min(trailing_stop, be_price)
                be_activated  = True

        # ── 出場判斷 ──────────────────────────────────────────
        trade_closed, _, closed_pnl, exit_reason = CoreStrategy.check_exit(
            position, cur_low, cur_high, cur_open,
            liq_price, trailing_stop, stop_loss,
            entry_price, position_size, fee_rate, slippage_pct
        )
        if trade_closed:
            capital = max(0.0, capital + closed_pnl) if exit_reason == '💀 Liquidation' \
                      else capital + closed_pnl
            net_pnl = closed_pnl - entry_fee
            trade_log.append({
                'Entry_Time':  entry_time, 'Exit_Time': timestamp,
                'Type':        'Long' if position == 1 else 'Short',
                'PnL':         net_pnl, 'Capital':   capital,
                'Exit_Reason': exit_reason, 'Initial_Risk': initial_risk,
            })
            position      = 0
            position_size = 0.0
            be_activated  = False
            entry_fee     = 0.0
            atw_active    = False
            atw_remaining = 0
            atw_first_pending = False
            consol_highs.clear()
            consol_lows.clear()
            if net_pnl < 0:
                consecutive_losses += 1
                if consecutive_losses >= max_consec_losses:
                    skip_next_trade = True; in_skip_zone = False; consecutive_losses = 0
            else:
                consecutive_losses = 0

        # ── 訊號偵測 ──────────────────────────────────────────
        if position == 0 and not atw_active:
            l_cond, s_cond, _ = CoreStrategy.check_signals(row, adx_threshold)
            if skip_next_trade:
                if l_cond or s_cond:
                    skip_next_trade = False; in_skip_zone = True
            elif in_skip_zone:
                if not l_cond and not s_cond:
                    in_skip_zone = False
            else:
                if l_cond or s_cond:
                    sl_dist = max(initial_sl_atr * cur_atr, cur_open * 0.001)
                    direction = 1 if l_cond else -1

                    # ── N=1：用一份 TWAP 進場（時機與一般版相同）──────
                    # N>1：啟動 N 份自適應 TWAP
                    N = min(
                        max(1, int(capital / max_trade_usdt_cap)),
                        int(MAX_POSITION_CAP / max_trade_usdt_cap)
                    )
                    if N == 1:
                        size_each = CoreStrategy.calculate_position_size(
                            capital, risk_per_trade, sl_dist, cur_open,
                            max_pos_ratio, leverage, max_trade_usdt_cap
                        )[0]
                    else:
                        size_each = max_trade_usdt_cap / cur_open
                    initial_risk  = N * size_each * sl_dist
                    atw_size_each = size_each
                    atw_sl_dist   = sl_dist
                    atw_direction = direction
                    atw_active    = True
                    atw_remaining = N
                    atw_first_pending = True

        # ── 每週固定提現
        if enable_withdrawal and capital > WEEKLY_WITHDRAWAL_START:
            if last_withdrawal_ts is None or (timestamp - last_withdrawal_ts).days >= 7:
                withdraw_amt = min(capital * WEEKLY_WITHDRAWAL_PCT, WEEKLY_WITHDRAWAL_AMOUNT, capital - WEEKLY_WITHDRAWAL_START)
                if withdraw_amt > 0:
                    total_withdrawn   += withdraw_amt
                    capital           -= withdraw_amt
                    last_withdrawal_ts = timestamp

        unr = (cur_close - entry_price) * position_size * position if position != 0 else 0.0
        equity_curve.append({'timestamp': timestamp, 'equity': capital + unr})

    return pd.DataFrame(trade_log), pd.DataFrame(equity_curve), total_withdrawn


def calc_metrics(trades_df, equity_df, initial_capital):
    if trades_df.empty or equity_df.empty:
        return {"Total_Trades": 0, "Return_%": 0.0, "Win_Rate_%": 0.0,
                "True_MDD_%": 0.0, "Sharpe": 0.0, "Avg_R": 0.0, "Final_Cap": initial_capital}
    eq           = equity_df['equity']
    final_cap    = eq.iloc[-1]
    total_return = (final_cap - initial_capital) / initial_capital * 100
    win_rate     = (trades_df['PnL'] > 0).mean() * 100
    true_mdd     = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    valid        = trades_df[trades_df['Initial_Risk'] > 0]
    avg_r        = (valid['PnL'] / valid['Initial_Risk']).mean() if len(valid) > 0 else 0.0

    eq_df = equity_df.copy()
    eq_df['timestamp'] = pd.to_datetime(eq_df['timestamp'])
    daily_eq  = eq_df.set_index('timestamp')['equity'].resample('D').last().ffill()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe    = round(float(daily_ret.mean() / daily_ret.std() * np.sqrt(365)), 3) \
                if daily_ret.std() > 0 else 0.0

    return {
        "Total_Trades": len(trades_df),
        "Return_%":     round(total_return, 2),
        "Win_Rate_%":   round(win_rate, 2),
        "True_MDD_%":   round(true_mdd, 2),
        "Sharpe":       sharpe,
        "Avg_R":        round(avg_r, 2),
        "Final_Cap":    round(final_cap, 2),
    }


# ══════════════════════════════════════════════════════════════
config  = load_config("config.json")
INITIAL = config['risk']['initial_capital']

df = pd.read_csv("data/SOLUSDT_15m.csv", index_col='timestamp', parse_dates=True)
df = df.iloc[:-1]
df = CoreStrategy.prepare_data(df)
df = df[(df.index >= pd.Timestamp(START)) & (df.index <= pd.Timestamp(END))]
print(f"[OK] {len(df)} bars（{START} ~ {END}）")

# ── 自適應 TWAP 版本 ──────────────────────────────────────────
trades_a, equity_a, _ = run_backtest_adaptive_twap(df, config)
m_a = calc_metrics(trades_a, equity_a, INITIAL)

# ── 自適應 TWAP + 每週提現版本 ────────────────────────────────
trades_wk, equity_wk, total_withdrawn_wk = run_backtest_adaptive_twap(
    df, config, enable_withdrawal=True)
m_wk = calc_metrics(trades_wk, equity_wk, INITIAL)
total_wealth_wk = m_wk['Final_Cap'] + total_withdrawn_wk

print(f"\n── 每週提現 {WEEKLY_WITHDRAWAL_PCT*100:.1f} % 上限 {WEEKLY_WITHDRAWAL_AMOUNT:,.0f} U（門檻 {WEEKLY_WITHDRAWAL_START:,.0f} U）")
print(f"  報酬率：{m_wk['Return_%']}%")
print(f"  MDD：  {m_wk['True_MDD_%']}%")
print(f"  Sharpe：{m_wk['Sharpe']}")
print(f"  勝率：  {m_wk['Win_Rate_%']}%")
print(f"  平均R：  {m_wk['Avg_R']}")
print(f"  交易數：{m_wk['Total_Trades']} 筆")
print(f"  帳戶餘額：{m_wk['Final_Cap']:,.2f} U")
print(f"  累計提現：{total_withdrawn_wk:,.2f} U")
print(f"  總財富：  {total_wealth_wk:,.2f} U（初始 {INITIAL} U）")

print(f"\n── 自適應 TWAP（資金達 N×{config['risk'].get('max_trade_usdt_cap',200000):.0f} U 才啟動，每份吃滿上限）")
print(f"  報酬率：{m_a['Return_%']}%")
print(f"  MDD：  {m_a['True_MDD_%']}%")
print(f"  Sharpe：{m_a['Sharpe']}")
print(f"  勝率：  {m_a['Win_Rate_%']}%")
print(f"  平均R：  {m_a['Avg_R']}")
print(f"  交易數：{m_a['Total_Trades']} 筆")
print(f"  最終資金：{m_a['Final_Cap']:,.2f} U（初始 {INITIAL} U）")

# ── 翻倍里程碑（以自適應 TWAP 為準）────────────────────────
eq = equity_a.set_index('timestamp')['equity']
milestones = []
target = INITIAL * 2
multiple = 2
while target <= eq.max() * 1.01:
    reached = eq[eq >= target].index
    if len(reached) > 0:
        milestones.append((reached[0], target, multiple))
    target   *= 2
    multiple *= 2

if milestones:
    print("\n  翻倍里程碑(自適應TWAP)：")
    for ts, cap, mult in milestones:
        print(f"    ×{mult:>6}  {cap:>14.2f} U  →  {ts.strftime('%Y-%m-%d %H:%M')}")
else:
    print("\n  此區間內未達翻倍")

# ── 淨值曲線（雙線對比）──────────────────────────────────────
eq_wk = equity_wk.set_index('timestamp')['equity']
eq_a  = equity_a.set_index('timestamp')['equity']

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(eq_wk.index, eq_wk.values, color='#e74c3c', linewidth=1.0, label=f'每週提現 {WEEKLY_WITHDRAWAL_PCT*100:.1f} % | 上限 {WEEKLY_WITHDRAWAL_AMOUNT:,.0f} U（門檻 {WEEKLY_WITHDRAWAL_START:,.0f} U）')
ax.plot(eq_a.index,  eq_a.values,  color='#1abc9c', linewidth=1.2, label='自適應 TWAP')
ax.axhline(INITIAL, color='gray', linewidth=0.8, label=f'初始 {INITIAL} U')

colors = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db', '#9b59b6']
for i, (ts, cap, mult) in enumerate(milestones):
    c = colors[i % len(colors)]
    ax.axhline(cap, color=c, linewidth=0.8, linestyle=':')
    ax.axvline(ts,  color=c, linewidth=0.8, linestyle=':')
    ax.annotate(f'×{mult} ({ts.strftime("%m/%d")})',
                xy=(ts, cap), xytext=(8, 4), textcoords='offset points',
                fontsize=8, color=c)

ax.set_title(f'SOL 策略淨值曲線  {START} ～ {END}', fontsize=13, fontweight='bold')
ax.set_xlabel('日期'); ax.set_ylabel('資金 (USDT)')
ax.legend(); ax.grid(True, alpha=0.4)
fig.autofmt_xdate(); plt.tight_layout(); plt.show()
