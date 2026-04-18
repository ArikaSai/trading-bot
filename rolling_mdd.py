"""
rolling_mdd.py
══════════════
四策略聯合滾動回測 — 每月滾動，找出歷史最差區間

用法:
    python rolling_mdd.py                  # 預設 3 個月窗口
    python rolling_mdd.py --window 6       # 6 個月窗口
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from backtest_multiple import load_csv, run_triple

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False


def load_config(path="config.json") -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='四策略聯合滾動 MDD')
    parser.add_argument('--window', type=int, default=3, help='滾動窗口（月）')
    args = parser.parse_args()

    WINDOW = args.window

    config = load_config()
    initial_cap = config['risk']['initial_capital']

    # 載入資料取得時間範圍
    sol_tf = config['trading'].get('timeframe', '15m')
    df_sol = load_csv('SOL', sol_tf)
    first_date = df_sol.index[0]
    last_date  = df_sol.index[-1]

    # 產生滾動窗口起始日
    starts = pd.date_range(
        start=first_date.replace(day=1),
        end=last_date - pd.DateOffset(months=WINDOW),
        freq='MS'
    )

    print(f"四策略聯合滾動 MDD")
    print(f"  方案: 可用×40% 直接用 | 窗口: {WINDOW} 個月")
    print(f"  期間: {first_date.strftime('%Y-%m-%d')} ~ {last_date.strftime('%Y-%m-%d')}")
    print(f"  SOL 槓桿: {config['risk'].get('leverage', 2)}x | 初始: ${initial_cap}")
    print(f"  共 {len(starts)} 個滾動窗口\n")

    results = []
    for j, s in enumerate(starts, 1):
        e = s + pd.DateOffset(months=WINDOW)

        cfg = json.loads(json.dumps(config))
        cfg['backtest'] = {
            'start_date': s.strftime('%Y-%m-%d'),
            'end_date':   e.strftime('%Y-%m-%d'),
        }

        try:
            r = run_triple(cfg, label=f"{s.strftime('%Y-%m')} ~ {e.strftime('%Y-%m')}",
                           consol_n=6, consol_x=1.5, tight_trail=0.5, spike_mult=3.0)
        except Exception as ex:
            print(f"  [{j}/{len(starts)}] {s.strftime('%Y-%m')} 跳過: {ex}")
            continue

        results.append({
            'Start':    s.strftime('%Y-%m-%d'),
            'End':      e.strftime('%Y-%m-%d'),
            'Trades':   r['all_n'],
            'Return_%': round(r['ret%'], 1),
            'MDD_%':    round(r['mdd%'], 1),
            'Sharpe':   round(r['sharpe'], 3),
            'PF':       round(r['all_pf'], 2),
            'Final':    round(r['final'], 2),
            'SOL_n':    r['sol_n'],
            'ADA_n':    r['ada_n'],
            'XRP_n':    r['xrp_n'],
            'DOGE_n':   r['doge_n'],
        })

        if j % 5 == 0 or j == len(starts):
            print(f"  進度: {j}/{len(starts)}")

    if not results:
        print("沒有有效窗口")
        return

    rdf = pd.DataFrame(results)

    # ── 排序：MDD 最差 → 最前 ────────────────────────────────
    rdf_sorted = rdf.sort_values('MDD_%').reset_index(drop=True)

    print(f"\n{'='*90}")
    print(f"  滾動 MDD 排名（{WINDOW} 個月窗口，可用×40%）")
    print(f"{'='*90}")
    print(f"{'#':>3}  {'起始':<12} {'結束':<12} {'交易':>5} {'報酬%':>9} {'MDD%':>8} {'PF':>6} {'Sharpe':>7}")
    print("-" * 75)
    for i, r in rdf_sorted.iterrows():
        tag = ' [!]' if r['MDD_%'] < -30 else ''
        print(f"{i+1:>3}  {r['Start']:<12} {r['End']:<12} {r['Trades']:>5} "
              f"{r['Return_%']:>+8.1f}% {r['MDD_%']:>+7.1f}% {r['PF']:>5.2f} {r['Sharpe']:>7.3f}{tag}")

    # ── 統計摘要 ─────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  統計摘要 — {len(rdf)} 個窗口")
    print(f"{'='*90}")
    print(f"  MDD   平均: {rdf['MDD_%'].mean():.1f}% | 中位: {rdf['MDD_%'].median():.1f}% | 最差: {rdf['MDD_%'].min():.1f}%")
    print(f"  報酬  平均: {rdf['Return_%'].mean():.1f}% | 中位: {rdf['Return_%'].median():.1f}%")
    print(f"  PF    平均: {rdf['PF'].mean():.2f} | 中位: {rdf['PF'].median():.2f}")
    n_loss = (rdf['Return_%'] < 0).sum()
    print(f"  虧損窗口: {n_loss} / {len(rdf)} ({n_loss/len(rdf)*100:.1f}%)")

    worst = rdf_sorted.iloc[0]
    best  = rdf_sorted.iloc[-1]
    print(f"\n  最差時段: {worst['Start']} ~ {worst['End']} | MDD: {worst['MDD_%']}% | 報酬: {worst['Return_%']}%")
    print(f"  最佳時段: {best['Start']} ~ {best['End']} | MDD: {best['MDD_%']}% | 報酬: {best['Return_%']}%")

    # ── 繪圖 ─────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    x_dates   = pd.to_datetime(rdf['Start'])
    mdd_colors = ['#e74c3c' if m < -30 else '#f39c12' if m < -15 else '#2ecc71'
                  for m in rdf['MDD_%']]

    ax1.bar(x_dates, rdf['MDD_%'].values, width=25, color=mdd_colors, alpha=0.8)
    ax1.axhline(rdf['MDD_%'].mean(), color='gray', linestyle='--', linewidth=0.8,
                label=f"平均 {rdf['MDD_%'].mean():.1f}%")
    ax1.axhline(-40, color='red', linestyle=':', linewidth=1, label='警戒線 -40%')
    ax1.set_ylabel('最大回撤 %')
    ax1.set_title(f'四策略聯合 {WINDOW}M 滾動 MDD（可用×40%）', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ret_colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in rdf['Return_%']]
    ax2.bar(x_dates, rdf['Return_%'].values, width=25, color=ret_colors, alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('報酬率 %')
    ax2.set_title(f'四策略聯合 {WINDOW}M 滾動報酬', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig('multiple_rolling_mdd.png', dpi=150)
    plt.show()
    print(f"\n  圖表已儲存: multiple_rolling_mdd.png")


if __name__ == '__main__':
    main()
