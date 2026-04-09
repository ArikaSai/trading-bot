"""
rolling_mdd.py
══════════════
每月滾動回測（回測半年），找出 MDD 最大的時段。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest_sol import load_config, run_backtest, calc_metrics
from strategy import CoreStrategy

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

WINDOW_MONTHS = 3   # 每次回測幾個月
INITIAL = None       # 從 config 讀取

config  = load_config("config.json")
INITIAL = config['risk']['initial_capital']

# 載入資料
df = pd.read_csv("data/SOLUSDT_15m.csv", index_col='timestamp', parse_dates=True)
df = df.iloc[:-1]
df = CoreStrategy.prepare_data(df)

# 建立滾動窗口：每月 1 號開始，往後推 6 個月
first_date = df.index[0]
last_date  = df.index[-1]

# 產生所有可能的起始月份
starts = pd.date_range(
    start=first_date.replace(day=1),
    end=last_date - pd.DateOffset(months=WINDOW_MONTHS),
    freq='MS'   # Month Start
)

print(f"[OK] {len(df)} 根 K 棒 | {first_date.strftime('%Y-%m-%d')} ~ {last_date.strftime('%Y-%m-%d')}")
print(f"[執行] {len(starts)} 個滾動窗口（每窗 {WINDOW_MONTHS} 個月）\n")

results = []
for s in starts:
    e = s + pd.DateOffset(months=WINDOW_MONTHS)
    df_slice = df[(df.index >= s) & (df.index < e)]
    if len(df_slice) < 100:
        continue

    trades, equity, _ = run_backtest(df_slice, config)
    m = calc_metrics(trades, equity, INITIAL)

    results.append({
        'Start':      s.strftime('%Y-%m-%d'),
        'End':        e.strftime('%Y-%m-%d'),
        'Bars':       len(df_slice),
        'Trades':     m['Total_Trades'],
        'Return_%':   m['Return_%'],
        'MDD_%':      m['True_MDD_%'],
        'Sharpe':     m['Sharpe'],
        'Win_Rate_%': m['Win_Rate_%'],
        'Final_Cap':  m['Final_Cap'],
    })

rdf = pd.DataFrame(results)

# 排序：MDD 最差排最前
rdf_sorted = rdf.sort_values('MDD_%').reset_index(drop=True)

print(f"{'#':>3}  {'起始':<12} {'結束':<12} {'交易數':>6} {'報酬%':>9} {'MDD%':>8} {'Sharpe':>7} {'勝率%':>6}")
print("-" * 72)
for i, r in rdf_sorted.iterrows():
    print(f"{i+1:>3}  {r['Start']:<12} {r['End']:<12} {r['Trades']:>6} "
          f"{r['Return_%']:>+8.1f}% {r['MDD_%']:>+7.1f}% {r['Sharpe']:>7.3f} {r['Win_Rate_%']:>5.1f}%")

print(f"\n{'='*72}")
print(f"MDD 最差時段: {rdf_sorted.iloc[0]['Start']} ~ {rdf_sorted.iloc[0]['End']} | MDD: {rdf_sorted.iloc[0]['MDD_%']}%")
print(f"MDD 最佳時段: {rdf_sorted.iloc[-1]['Start']} ~ {rdf_sorted.iloc[-1]['End']} | MDD: {rdf_sorted.iloc[-1]['MDD_%']}%")

# 統計
print(f"\n{'='*72}")
print(f"[統計] {len(rdf)} 個窗口")
print(f"  MDD  平均: {rdf['MDD_%'].mean():.2f}% | 中位: {rdf['MDD_%'].median():.2f}% | 最差: {rdf['MDD_%'].min():.2f}%")
print(f"  收益 平均: {rdf['Return_%'].mean():.1f}% | 中位: {rdf['Return_%'].median():.1f}%")
print(f"  虧損窗口數: {(rdf['Return_%'] < 0).sum()} / {len(rdf)}")

# 繪圖
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

x_dates = pd.to_datetime(rdf['Start'])
colors  = ['#e74c3c' if m < -30 else '#f39c12' if m < -15 else '#2ecc71' for m in rdf['MDD_%']]

ax1.bar(x_dates, rdf['MDD_%'].values, width=25, color=colors, alpha=0.8)
ax1.axhline(rdf['MDD_%'].mean(), color='gray', linestyle='--', linewidth=0.8, label=f"平均 {rdf['MDD_%'].mean():.1f}%")
ax1.set_ylabel('最大回撤 %')
ax1.set_title(f'SOL {WINDOW_MONTHS}M 滾動 MDD（紅 < -30%, 橘 < -15%, 綠 > -15%）', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.bar(x_dates, rdf['Return_%'].values, width=25,
        color=['#2ecc71' if r > 0 else '#e74c3c' for r in rdf['Return_%']], alpha=0.8)
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_ylabel('報酬率 %')
ax2.set_title(f'SOL {WINDOW_MONTHS}M 滾動報酬', fontweight='bold')
ax2.grid(True, alpha=0.3)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()
