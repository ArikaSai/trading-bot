"""
stress_test.py
══════════════
漏單壓力測試：隨機漏掉 N% 的單，策略還能賺錢嗎？

方法：Monte Carlo — 對回測產出的交易清單隨機跳過 MISS_RATE 比例，
      重新計算資金曲線，重複 N_SIMULATIONS 次，報告報酬分佈。

注意：使用固定 PnL（不重新計算倉位大小），結果略為保守估計，
      因為真實情境下跳過虧損單會讓後續倉位更大。

用法:
    python stress_test.py              # 預設漏單率 20%，3000 次模擬
    python stress_test.py --miss 0.30  # 漏 30%
    python stress_test.py --n 5000     # 5000 次模擬
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False


def _simulate(trades: list, miss_rate: float, initial_cap: float):
    """隨機跳過 miss_rate 比例的交易，回傳 (最終資金, MDD)。
    資金歸零視為爆倉，後續交易停止。MDD 最大為 100%。"""
    capital = initial_cap
    peak    = initial_cap
    mdd     = 0.0
    for t in trades:
        if capital <= 0:
            break  # 爆倉，後續停止
        if np.random.random() < miss_rate:
            continue
        capital = max(0.0, capital + t['PnL'])
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak if peak > 0 else 1.0
        if dd > mdd:
            mdd = dd
    return capital, mdd


def _report(name: str, caps: np.ndarray, mdds: np.ndarray,
            initial_cap: float, n_sim: int, miss_rate: float):
    profitable = float(np.sum(caps > initial_cap)) / n_sim * 100
    print(f"\n  {'─'*56}")
    print(f"  {name}（漏單率 {miss_rate*100:.0f}%，{n_sim} 次模擬）")
    print(f"  {'─'*56}")
    print(f"  獲利模擬比例  {profitable:>7.1f}%")
    print(f"  最終資金 中位 ${np.median(caps):>15,.2f}")
    print(f"  最終資金  5%  ${np.percentile(caps,  5):>15,.2f}  ← 最差 5%")
    print(f"  最終資金 95%  ${np.percentile(caps, 95):>15,.2f}  ← 最佳 5%")
    print(f"  最終資金 最差 ${np.min(caps):>15,.2f}")
    print(f"  最終資金 最佳 ${np.max(caps):>15,.2f}")
    print(f"  MDD 中位      {np.median(mdds)*100:>7.1f}%")
    print(f"  MDD 95%       {np.percentile(mdds, 95)*100:>7.1f}%  ← 最差 5%")


def main():
    parser = argparse.ArgumentParser(description='漏單壓力測試')
    parser.add_argument('--miss', type=float, default=0.20, help='漏單率（0~1，預設 0.20）')
    parser.add_argument('--n',    type=int,   default=3000, help='模擬次數')
    args = parser.parse_args()

    miss_rate = args.miss
    n_sim     = args.n

    # ── 執行基準回測 ────────────────────────────────────────────
    from backtest_multiple import run_triple, load_config
    config = load_config()

    initial_cap = config['risk'].get('initial_capital', 500.0)

    print("執行基準回測，請稍候...")
    r = run_triple(config, consol_n=6, consol_x=1.5, tight_trail=0.5, spike_mult=3.0)
    print(f"基準完成：最終資金 ${r['final']:,.2f}  報酬 +{r['ret%']:.1f}%  MDD -{r['mdd%']:.1f}%")

    # 各策略交易清單（依時間排序）
    all_trades = sorted(
        r['sol_trades'] + r['ada_trades'] + r['xrp_trades'] + r['doge_trades'],
        key=lambda x: x['Time']
    )
    # 只對「四策略合計」做 Monte Carlo。
    # 個別策略的 PnL 是基於混合後的大額資金計算，
    # 對著 $500 單跑會讓資金跑到負值（MDD > 100%），結果無意義。
    strats = {
        '四策略合計': all_trades,
    }

    # ── Monte Carlo ─────────────────────────────────────────────
    print(f"\n執行 Monte Carlo（漏單率 {miss_rate*100:.0f}%，{n_sim} 次）...")
    results = {}
    for name, trades in strats.items():
        caps = np.empty(n_sim)
        mdds = np.empty(n_sim)
        for i in range(n_sim):
            caps[i], mdds[i] = _simulate(trades, miss_rate, initial_cap)
        results[name] = (caps, mdds)
        print(f"  {name} 完成")

    # ── 文字報告 ────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  壓力測試結果（初始資金 ${initial_cap:,.0f}）")
    print(f"{'═'*60}")
    for name, (caps, mdds) in results.items():
        _report(name, caps, mdds, initial_cap, n_sim, miss_rate)

    # ── 圖表（只畫四策略合計）───────────────────────────────────
    caps = results['四策略合計'][0]
    mdds = results['四策略合計'][1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 左：最終資金分佈
    profitable = float(np.sum(caps > initial_cap)) / n_sim * 100
    ax1.hist(caps, bins=60, color='steelblue', alpha=0.75, edgecolor='white')
    ax1.axvline(initial_cap,            color='red',    linestyle='--', linewidth=1.2,
                label=f'初始 ${initial_cap:,.0f}')
    ax1.axvline(np.median(caps),        color='lime',   linestyle='-',  linewidth=1.5,
                label=f'中位 ${np.median(caps):,.0f}')
    ax1.axvline(np.percentile(caps,  5),color='orange', linestyle=':',  linewidth=1.2,
                label=f'最差5% ${np.percentile(caps,5):,.0f}')
    ax1.axvline(np.percentile(caps, 95),color='cyan',   linestyle=':',  linewidth=1.2,
                label=f'最佳5% ${np.percentile(caps,95):,.0f}')
    ax1.set_title(f'四策略合計 最終資金分佈  獲利率 {profitable:.1f}%', fontsize=11)
    ax1.set_xlabel('最終資金 ($)')
    ax1.legend(fontsize=8)

    # 右：MDD 分佈
    ax2.hist(mdds * 100, bins=60, color='tomato', alpha=0.75, edgecolor='white')
    ax2.axvline(np.median(mdds)*100,         color='darkred', linestyle='-',  linewidth=1.5,
                label=f'中位 {np.median(mdds)*100:.1f}%')
    ax2.axvline(np.percentile(mdds, 95)*100, color='orange',  linestyle=':',  linewidth=1.2,
                label=f'最壞5% {np.percentile(mdds,95)*100:.1f}%')
    ax2.set_title('四策略合計 MDD 分佈', fontsize=11)
    ax2.set_xlabel('最大回撤 (%)')
    ax2.legend(fontsize=8)

    fig.suptitle(
        f'漏單壓力測試 — 隨機漏掉 {miss_rate*100:.0f}% 的單  ({n_sim} 次 Monte Carlo)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    out = 'stress_test_result.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n圖表已存: {out}")

if __name__ == '__main__':
    main()
