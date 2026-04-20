"""
feature_sweep.py
════════════════
掃描「盤整縮緊」與「插針防護」兩項功能對 ADA / XRP / DOGE 策略的影響。
SOL 已有這兩項功能，此腳本只測試其餘三個策略。

輸出：按照總報酬排序的結果表，以及三條最佳曲線對比圖。

用法:
    python feature_sweep.py
"""

import json
import itertools
import pandas as pd
from backtest_multiple import run_triple

CONFIG_PATH = "config.json"

def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    cfg = load_config()

    # ── 掃描參數 ────────────────────────────────────────────
    # spike_mult: 0 = 關閉
    spike_vals   = [0.0, 3.0, 4.0, 5.0]
    # consol_n: 0 = 關閉
    consol_n_vals = [0, 6, 8, 10, 12]
    consol_x_vals = [1.0, 1.25, 1.5]
    tight_vals    = [0.5, 1.0, 1.5]

    # 先生成所有組合（consol 關閉時不掃 x / tight）
    combos = []
    for sp in spike_vals:
        # case A: consol 關閉
        combos.append(dict(spike_mult=sp, consol_n=0, consol_x=1.25, tight_trail=1.0))
        # case B: consol 開啟
        for cn, cx, tt in itertools.product(consol_n_vals[1:], consol_x_vals, tight_vals):
            combos.append(dict(spike_mult=sp, consol_n=cn, consol_x=cx, tight_trail=tt))

    # 去重（consol_n=0 時 x/tight 無意義，已統一固定）
    seen = set(); uniq = []
    for c in combos:
        key = (c['spike_mult'], c['consol_n'], c['consol_x'], c['tight_trail'])
        if key not in seen:
            seen.add(key); uniq.append(c)

    total = len(uniq)
    print(f"共 {total} 組組合，開始掃描...\n")

    results = []
    for i, p in enumerate(uniq, 1):
        sp = p['spike_mult']; cn = p['consol_n']
        cx = p['consol_x'];   tt = p['tight_trail']

        spike_str  = f"spike={sp:.1f}" if sp > 0 else "spike=off"
        consol_str = f"consol=N{cn},X{cx},T{tt}" if cn > 0 else "consol=off"
        label      = f"{spike_str} | {consol_str}"

        r = run_triple(cfg, label=label,
                       consol_n=cn, consol_x=cx,
                       tight_trail=tt, spike_mult=sp)

        results.append({
            'label':      label,
            'spike_mult': sp,
            'consol_n':   cn,
            'consol_x':   cx,
            'tight_trail': tt,
            'ret%':       r['ret%'],
            'mdd%':       r['mdd%'],
            'sharpe':     r['sharpe'],
            'all_n':      r['all_n'],
            'all_wr':     r['all_wr'],
            'ada_pnl':    r['ada_pnl'],
            'xrp_pnl':    r['xrp_pnl'],
            'doge_pnl':   r['doge_pnl'],
            'equity_df':  r['equity_df'],
        })

        if i % 20 == 0 or i == total:
            print(f"  [{i}/{total}] {label}  ret={r['ret%']:.1f}%  mdd={r['mdd%']:.1f}%")

    # ── 整理結果表 ───────────────────────────────────────────
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'equity_df'}
                        for r in results])
    df.sort_values('ret%', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("\n" + "═"*90)
    print("TOP 20 組合（依總報酬排序）")
    print("═"*90)
    cols = ['label', 'ret%', 'mdd%', 'sharpe', 'all_n', 'all_wr', 'ada_pnl', 'xrp_pnl', 'doge_pnl']
    print(df[cols].head(20).to_string(index=True,
          float_format=lambda x: f"{x:.2f}"))

    # baseline = spike=off, consol=off
    baseline = df[(df.spike_mult == 0.0) & (df.consol_n == 0)].iloc[0]
    print(f"\n── Baseline（無功能）──")
    print(f"  ret={baseline['ret%']:.2f}%  mdd={baseline['mdd%']:.2f}%  "
          f"sharpe={baseline['sharpe']:.2f}  trades={baseline['all_n']}")

    best = df.iloc[0]
    print(f"\n── 最佳組合 ──")
    print(f"  {best['label']}")
    print(f"  ret={best['ret%']:.2f}%  mdd={best['mdd%']:.2f}%  "
          f"sharpe={best['sharpe']:.2f}  trades={best['all_n']}")
    print(f"  ADA={best['ada_pnl']:.0f}U  XRP={best['xrp_pnl']:.0f}U  DOGE={best['doge_pnl']:.0f}U")

    # ── 繪圖：baseline vs best_spike_only vs best_consol_only vs best_both ─
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False

    def _best_for(mask):
        sub = df[mask]
        return sub.iloc[0] if len(sub) > 0 else None

    cases = {
        'Baseline':       df[(df.spike_mult == 0.0) & (df.consol_n == 0)].iloc[0],
        'Spike only':     _best_for((df.spike_mult > 0) & (df.consol_n == 0)),
        'Consol only':    _best_for((df.spike_mult == 0.0) & (df.consol_n > 0)),
        'Spike + Consol': _best_for((df.spike_mult > 0) & (df.consol_n > 0)),
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    for name, row in cases.items():
        if row is None:
            continue
        eq = results[df[df.label == row['label']].index[0]]['equity_df']
        eq_s = eq.set_index('timestamp')['equity']
        ax.plot(eq_s.index, eq_s.values, label=f"{name}  ret={row['ret%']:.1f}%  mdd={row['mdd%']:.1f}%")

    ax.set_title("ADA/XRP/DOGE 盤整縮緊 & 插針防護功能比較")
    ax.set_ylabel("淨值 (USDT)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = "feature_sweep_result.png"
    fig.savefig(out, dpi=120)
    print(f"\n圖表已儲存：{out}")

    # ── 最佳 spike / consol 參數小結 ────────────────────────
    print("\n── 分項最佳參數 ──")
    for mode, mask in [
        ("Spike only",     (df.spike_mult > 0) & (df.consol_n == 0)),
        ("Consol only",    (df.spike_mult == 0.0) & (df.consol_n > 0)),
        ("Spike + Consol", (df.spike_mult > 0) & (df.consol_n > 0)),
    ]:
        row = _best_for(mask)
        if row is not None:
            print(f"  {mode:20s}: {row['label']}  ret={row['ret%']:.2f}%  mdd={row['mdd%']:.2f}%")

if __name__ == "__main__":
    main()