"""
ml_trade_analysis.py
════════════════════
分析 trades_tiered_live.csv，找出可優化的訊號過濾條件。

用法：
    python ml_trade_analysis.py
    python ml_trade_analysis.py --csv trades_tiered_live.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

PAYOUT   = 0.85
BE       = 1 / (1 + PAYOUT) * 100   # 54.1%
ROLL_N   = 500                        # 滾動勝率視窗
TIER_ZH  = {
    'trend_aligned': '趨勢順勢',
    'trend_counter': '趨勢逆向',
    'ranging':       '震盪',
    'dead':          '死水',
}
TIER_ORDER = ['trend_aligned', 'trend_counter', 'ranging', 'dead']


# ══════════════════════════════════════════════════════════════
#  工具
# ══════════════════════════════════════════════════════════════
def wr_ev(df: pd.DataFrame) -> tuple[float, float]:
    if len(df) == 0:
        return 0.0, 0.0
    wr = df['win'].mean() * 100
    ev = wr / 100 * PAYOUT - (1 - wr / 100)
    return wr, ev * 100


def flag(wr: float) -> str:
    if wr >= BE + 3:  return '★★'
    if wr >= BE + 1:  return '★ '
    if wr >= BE:      return '✓ '
    return '✗ '


def print_table(rows: list[tuple], headers: list[str], title: str):
    print(f"\n  {'─'*64}")
    print(f"  {title}")
    print(f"  {'─'*64}")
    col_w = [max(len(h), 10) for h in headers]
    hdr = '  ' + '  '.join(f'{h:>{col_w[i]}}' for i, h in enumerate(headers))
    print(hdr)
    print(f"  {'─'*64}")
    for row in rows:
        line = '  ' + '  '.join(f'{str(v):>{col_w[i]}}' for i, v in enumerate(row))
        print(line)
    print(f"  {'─'*64}")


# ══════════════════════════════════════════════════════════════
#  各維度勝率分析
# ══════════════════════════════════════════════════════════════
def analyze_breakdowns(df: pd.DataFrame):

    # ── 1. 總覽 ────────────────────────────────────────────────
    wr, ev = wr_ev(df)
    print(f"\n  總覽：{len(df):,} 筆  勝率 {wr:.1f}%  期望值 {ev:+.2f}%  損益平衡 {BE:.1f}%")

    # ── 2. 各 Tier ─────────────────────────────────────────────
    rows = []
    for tier in TIER_ORDER:
        sub = df[df['tier'] == tier]
        if len(sub) == 0:
            continue
        wr_t, ev_t = wr_ev(sub)
        avg_bet     = sub['bet'].mean()
        total_pnl   = sub['pnl'].sum()
        rows.append((flag(wr_t), TIER_ZH[tier], f'{len(sub):,}',
                     f'{wr_t:.1f}%', f'{ev_t:+.2f}%',
                     f'{avg_bet:.1f}U', f'{total_pnl:+,.0f}U'))
    print_table(rows, ['', 'Tier', '筆數', '勝率', '期望值', '均注', '總損益'],
                'Tier 分析')

    # ── 3. 幣種 × 方向 ─────────────────────────────────────────
    rows = []
    for sym in ['BTC', 'ETH']:
        for direction in ['LONG', 'SHORT']:
            sub = df[(df['sym'] == sym) & (df['direction'] == direction)]
            if len(sub) == 0:
                continue
            wr_t, ev_t = wr_ev(sub)
            rows.append((flag(wr_t), sym, direction, f'{len(sub):,}',
                         f'{wr_t:.1f}%', f'{ev_t:+.2f}%'))
    print_table(rows, ['', '幣種', '方向', '筆數', '勝率', '期望值'],
                '幣種 × 方向')

    # ── 4. ADX 分段 ────────────────────────────────────────────
    adx_bins   = [0, 15, 20, 25, 30, 35, 45, 999]
    adx_labels = ['<15', '15-20', '20-25', '25-30', '30-35', '35-45', '>45']
    df['adx_bin'] = pd.cut(df['adx'], bins=adx_bins, labels=adx_labels, right=False)
    rows = []
    for lbl in adx_labels:
        sub = df[df['adx_bin'] == lbl]
        if len(sub) < 50:
            continue
        wr_t, ev_t = wr_ev(sub)
        rows.append((flag(wr_t), lbl, f'{len(sub):,}', f'{wr_t:.1f}%', f'{ev_t:+.2f}%'))
    print_table(rows, ['', 'ADX 範圍', '筆數', '勝率', '期望值'], 'ADX 分段勝率')

    # ── 5. ATR_ratio 分段 ─────────────────────────────────────
    atr_bins   = [0, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 999]
    atr_labels = ['<0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0', '1.0-1.1', '1.1-1.2', '>1.2']
    df['atr_bin'] = pd.cut(df['atr_ratio'], bins=atr_bins, labels=atr_labels, right=False)
    rows = []
    for lbl in atr_labels:
        sub = df[df['atr_bin'] == lbl]
        if len(sub) < 50:
            continue
        wr_t, ev_t = wr_ev(sub)
        rows.append((flag(wr_t), lbl, f'{len(sub):,}', f'{wr_t:.1f}%', f'{ev_t:+.2f}%'))
    print_table(rows, ['', 'ATR_ratio', '筆數', '勝率', '期望值'], 'ATR_ratio 分段勝率')

    # ── 6. vol_z 分段 ─────────────────────────────────────────
    vz_bins   = [-999, -1, -0.5, 0, 0.5, 1, 2, 999]
    vz_labels = ['<-1', '-1~-0.5', '-0.5~0', '0~0.5', '0.5~1', '1~2', '>2']
    df['vz_bin'] = pd.cut(df['vol_z'], bins=vz_bins, labels=vz_labels, right=False)
    rows = []
    for lbl in vz_labels:
        sub = df[df['vz_bin'] == lbl]
        if len(sub) < 50:
            continue
        wr_t, ev_t = wr_ev(sub)
        rows.append((flag(wr_t), lbl, f'{len(sub):,}', f'{wr_t:.1f}%', f'{ev_t:+.2f}%'))
    print_table(rows, ['', 'vol_z', '筆數', '勝率', '期望值'], 'vol_z 分段勝率')

    # ── 7. hlpct_12 極端程度（非死水）──────────────────────────
    sub_nd = df[df['tier'] != 'dead'].copy()
    sub_nd['hlpct_extreme'] = sub_nd.apply(
        lambda r: r['hlpct_12'] if r['direction'] == 'LONG' else 1 - r['hlpct_12'], axis=1)
    pct_bins   = [0, 0.02, 0.04, 0.06, 0.08, 0.10]
    pct_labels = ['0-2%', '2-4%', '4-6%', '6-8%', '8-10%']
    sub_nd['pct_bin'] = pd.cut(sub_nd['hlpct_extreme'], bins=pct_bins,
                                labels=pct_labels, right=False)
    rows = []
    for lbl in pct_labels:
        sub = sub_nd[sub_nd['pct_bin'] == lbl]
        if len(sub) < 50:
            continue
        wr_t, ev_t = wr_ev(sub)
        rows.append((flag(wr_t), lbl, f'{len(sub):,}', f'{wr_t:.1f}%', f'{ev_t:+.2f}%'))
    print_table(rows, ['', '區間位置（極端程度）', '筆數', '勝率', '期望值'],
                'hlpct_12 極端程度（0% = 最極端）')

    # ── 8. 時段分析 ────────────────────────────────────────────
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    rows = []
    for h in range(0, 24, 2):
        sub = df[(df['hour'] >= h) & (df['hour'] < h + 2)]
        if len(sub) < 30:
            continue
        wr_t, ev_t = wr_ev(sub)
        rows.append((flag(wr_t), f'{h:02d}:00-{h+2:02d}:00',
                     f'{len(sub):,}', f'{wr_t:.1f}%', f'{ev_t:+.2f}%'))
    print_table(rows, ['', '時段（UTC+8）', '筆數', '勝率', '期望值'], '時段分析')


# ══════════════════════════════════════════════════════════════
#  Tier × 指標 交叉分析（找最佳過濾條件）
# ══════════════════════════════════════════════════════════════
def analyze_tier_cross(df: pd.DataFrame):
    print(f"\n  {'='*64}")
    print(f"  Tier × 條件 交叉分析")
    print(f"  {'='*64}")

    for tier in TIER_ORDER:
        sub = df[df['tier'] == tier]
        if len(sub) < 100:
            continue
        wr_base, _ = wr_ev(sub)
        print(f"\n  ── {TIER_ZH[tier]}（共 {len(sub):,} 筆，基準勝率 {wr_base:.1f}%）──")

        # ADX × tier
        adx_bins   = [0, 15, 20, 25, 30, 35, 45, 999]
        adx_labels = ['<15', '15-20', '20-25', '25-30', '30-35', '35-45', '>45']
        sub_adx = sub.copy()
        sub_adx['adx_bin'] = pd.cut(sub['adx'], bins=adx_bins,
                                     labels=adx_labels, right=False)
        rows = []
        for lbl in adx_labels:
            s2 = sub_adx[sub_adx['adx_bin'] == lbl]
            if len(s2) < 30:
                continue
            wr_t, ev_t = wr_ev(s2)
            diff = wr_t - wr_base
            rows.append((flag(wr_t), f'ADX {lbl}', f'{len(s2):,}',
                         f'{wr_t:.1f}%', f'{ev_t:+.2f}%', f'{diff:+.1f}%'))
        if rows:
            print(f"  {'':2} {'ADX 範圍':<12} {'筆數':>7} {'勝率':>7} {'EV':>8} {'vs基準':>8}")
            for row in rows:
                print(f"  {'  '.join(f'{str(v):>{[2,12,7,7,8,8][i]}}' for i, v in enumerate(row))}")

        # ATR_ratio × tier
        atr_bins   = [0, 0.8, 0.9, 1.0, 1.1, 1.2]
        atr_labels = ['<0.8', '0.8-0.9', '0.9-1.0', '1.0-1.1', '1.1-1.2']
        sub_atr = sub.copy()
        sub_atr['atr_bin'] = pd.cut(sub['atr_ratio'], bins=atr_bins,
                                     labels=atr_labels, right=False)
        rows = []
        for lbl in atr_labels:
            s2 = sub_atr[sub_atr['atr_bin'] == lbl]
            if len(s2) < 30:
                continue
            wr_t, ev_t = wr_ev(s2)
            diff = wr_t - wr_base
            rows.append((flag(wr_t), f'ATR {lbl}', f'{len(s2):,}',
                         f'{wr_t:.1f}%', f'{ev_t:+.2f}%', f'{diff:+.1f}%'))
        if rows:
            print()
            for row in rows:
                print(f"  {'  '.join(f'{str(v):>{[2,12,7,7,8,8][i]}}' for i, v in enumerate(row))}")


# ══════════════════════════════════════════════════════════════
#  濾掉低勝率條件的影響試算
# ══════════════════════════════════════════════════════════════
def simulate_filter_impact(df: pd.DataFrame):
    """根據 pnl 試算各種過濾條件對總損益的影響。"""
    print(f"\n  {'='*64}")
    print(f"  過濾條件影響試算（基準總損益 {df['pnl'].sum():+,.0f} U）")
    print(f"  {'='*64}")

    base_pnl = df['pnl'].sum()
    rows = []

    candidates = [
        ('排除 ATR_ratio > 1.10',  df['atr_ratio'] > 1.10),
        ('排除 ATR_ratio > 1.05',  df['atr_ratio'] > 1.05),
        ('排除 vol_z > 2',          df['vol_z']     > 2.0),
        ('排除 vol_z < -1',         df['vol_z']     < -1.0),
        ('排除 ADX < 12',           df['adx']       < 12.0),
        ('排除 ADX 15-20（模糊區）', (df['adx'] >= 15) & (df['adx'] < 20)),
        ('排除 hlpct 5-10% 長尾',
            df.apply(lambda r: (r['direction'] == 'LONG'  and 0.05 <= r['hlpct_12'] <= 0.10) or
                                (r['direction'] == 'SHORT' and 0.90 <= r['hlpct_12'] <= 0.95),
                     axis=1)),
        ('僅保留 hlpct ≤ 3% / ≥ 97%',
            df.apply(lambda r: not ((r['direction'] == 'LONG'  and r['hlpct_12'] <= 0.03) or
                                    (r['direction'] == 'SHORT' and r['hlpct_12'] >= 0.97)),
                     axis=1)),
    ]

    for label, mask in candidates:
        filtered    = df[~mask]
        removed     = df[mask]
        new_pnl     = filtered['pnl'].sum()
        removed_pnl = removed['pnl'].sum()
        removed_wr, _ = wr_ev(removed)
        rows.append((label,
                     f'{mask.sum():,}筆',
                     f'{removed_wr:.1f}%',
                     f'{removed_pnl:+,.0f}U',
                     f'{new_pnl:+,.0f}U',
                     f'{new_pnl - base_pnl:+,.0f}U'))

    print_table(rows,
                ['過濾條件', '排除筆數', '排除WR', '排除損益', '過濾後損益', '變化'],
                '各過濾條件試算')


# ══════════════════════════════════════════════════════════════
#  繪圖
# ══════════════════════════════════════════════════════════════
def plot_analysis(df: pd.DataFrame, out: str):
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle('事件合約交易分析', fontsize=13, fontweight='bold')

    # 1. 滾動勝率（全部 + 各 tier）
    ax = fig.add_subplot(gs[0, :2])
    df_sorted = df.sort_values('time').reset_index(drop=True)
    roll_wr = df_sorted['win'].rolling(ROLL_N).mean() * 100
    ax.plot(roll_wr, color='black', lw=1.2, label=f'整體 ({ROLL_N}筆滾動)')
    colors = ['#2ecc71', '#3498db', '#f39c12', '#95a5a6']
    for tier, color in zip(TIER_ORDER, colors):
        sub = df_sorted[df_sorted['tier'] == tier].reset_index(drop=True)
        if len(sub) < ROLL_N:
            continue
        rw = sub['win'].rolling(ROLL_N).mean() * 100
        ax.plot(rw.values, color=color, lw=0.9, alpha=0.8,
                label=f'{TIER_ZH[tier]}')
    ax.axhline(BE, color='red', linestyle='--', lw=1, label=f'損益平衡 {BE:.1f}%')
    ax.set_title(f'滾動勝率（視窗 {ROLL_N} 筆）')
    ax.set_xlabel('累計交易筆數')
    ax.set_ylabel('勝率 %')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(48, 68)

    # 2. Tier 勝率橫條
    ax2 = fig.add_subplot(gs[0, 2])
    tier_wrs = []
    tier_labels = []
    for tier in TIER_ORDER:
        sub = df[df['tier'] == tier]
        if len(sub) == 0:
            continue
        wr_t, _ = wr_ev(sub)
        tier_wrs.append(wr_t)
        tier_labels.append(f'{TIER_ZH[tier]}\n({len(sub):,}筆)')
    bar_colors = ['#2ecc71' if w >= BE else '#e74c3c' for w in tier_wrs]
    ax2.barh(tier_labels, tier_wrs, color=bar_colors, alpha=0.85)
    ax2.axvline(BE, color='red', linestyle='--', lw=1)
    ax2.set_xlabel('勝率 %')
    ax2.set_title('Tier 勝率')
    for i, w in enumerate(tier_wrs):
        ax2.text(w + 0.1, i, f'{w:.1f}%', va='center', fontsize=8)

    # 3. ADX vs 勝率（散點/分段）
    ax3 = fig.add_subplot(gs[1, 0])
    adx_bins   = [0, 15, 20, 25, 30, 35, 45, 999]
    adx_labels = ['<15', '15-20', '20-25', '25-30', '30-35', '35-45', '>45']
    df['adx_bin'] = pd.cut(df['adx'], bins=adx_bins, labels=adx_labels, right=False)
    adx_wrs = []
    adx_ns  = []
    adx_used = []
    for lbl in adx_labels:
        sub = df[df['adx_bin'] == lbl]
        if len(sub) < 50:
            continue
        wr_t, _ = wr_ev(sub)
        adx_wrs.append(wr_t)
        adx_ns.append(len(sub))
        adx_used.append(lbl)
    colors_a = ['#2ecc71' if w >= BE else '#e74c3c' for w in adx_wrs]
    ax3.bar(adx_used, adx_wrs, color=colors_a, alpha=0.8)
    ax3.axhline(BE, color='red', linestyle='--', lw=1)
    ax3.set_title('ADX 分段勝率')
    ax3.set_ylabel('勝率 %')
    ax3.set_ylim(50, 65)
    for i, (w, n) in enumerate(zip(adx_wrs, adx_ns)):
        ax3.text(i, w + 0.1, f'{w:.1f}%\n({n//1000}k)', ha='center', fontsize=7)

    # 4. ATR_ratio vs 勝率
    ax4 = fig.add_subplot(gs[1, 1])
    atr_bins   = [0, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    atr_labels = ['<0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0', '1.0-1.1', '1.1-1.2']
    df['atr_bin'] = pd.cut(df['atr_ratio'], bins=atr_bins, labels=atr_labels, right=False)
    atr_wrs = []
    atr_ns  = []
    atr_used = []
    for lbl in atr_labels:
        sub = df[df['atr_bin'] == lbl]
        if len(sub) < 50:
            continue
        wr_t, _ = wr_ev(sub)
        atr_wrs.append(wr_t)
        atr_ns.append(len(sub))
        atr_used.append(lbl)
    colors_b = ['#2ecc71' if w >= BE else '#e74c3c' for w in atr_wrs]
    ax4.bar(atr_used, atr_wrs, color=colors_b, alpha=0.8)
    ax4.axhline(BE, color='red', linestyle='--', lw=1)
    ax4.set_title('ATR_ratio 分段勝率')
    ax4.set_ylabel('勝率 %')
    ax4.set_ylim(50, 65)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)
    for i, (w, n) in enumerate(zip(atr_wrs, atr_ns)):
        ax4.text(i, w + 0.1, f'{w:.1f}%', ha='center', fontsize=7)

    # 5. vol_z vs 勝率
    ax5 = fig.add_subplot(gs[1, 2])
    vz_bins   = [-999, -1, -0.5, 0, 0.5, 1, 2, 999]
    vz_labels = ['<-1', '-1~-0.5', '-0.5~0', '0~0.5', '0.5~1', '1~2', '>2']
    df['vz_bin'] = pd.cut(df['vol_z'], bins=vz_bins, labels=vz_labels, right=False)
    vz_wrs = []
    vz_ns  = []
    vz_used = []
    for lbl in vz_labels:
        sub = df[df['vz_bin'] == lbl]
        if len(sub) < 50:
            continue
        wr_t, _ = wr_ev(sub)
        vz_wrs.append(wr_t)
        vz_ns.append(len(sub))
        vz_used.append(lbl)
    colors_c = ['#2ecc71' if w >= BE else '#e74c3c' for w in vz_wrs]
    ax5.bar(vz_used, vz_wrs, color=colors_c, alpha=0.8)
    ax5.axhline(BE, color='red', linestyle='--', lw=1)
    ax5.set_title('vol_z 分段勝率')
    ax5.set_ylabel('勝率 %')
    ax5.set_ylim(50, 65)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)
    for i, (w, n) in enumerate(zip(vz_wrs, vz_ns)):
        ax5.text(i, w + 0.1, f'{w:.1f}%', ha='center', fontsize=7)

    # 6. 時段勝率（熱圖風格）
    ax6 = fig.add_subplot(gs[2, 0])
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    hour_wrs = [df[df['hour'] == h]['win'].mean() * 100 if (df['hour'] == h).sum() > 30 else np.nan
                for h in range(24)]
    colors_h = ['#2ecc71' if (w is not None and not np.isnan(w) and w >= BE)
                else '#e74c3c' for w in hour_wrs]
    ax6.bar(range(24), hour_wrs, color=colors_h, alpha=0.8)
    ax6.axhline(BE, color='red', linestyle='--', lw=1)
    ax6.set_title('各小時勝率（UTC+8）')
    ax6.set_xlabel('小時')
    ax6.set_ylabel('勝率 %')
    ax6.set_xticks(range(0, 24, 2))
    ax6.set_ylim(50, 65)

    # 7. hlpct_12 極端程度（LONG 方向）
    ax7 = fig.add_subplot(gs[2, 1])
    sub_l = df[df['direction'] == 'LONG'].copy()
    sub_s = df[df['direction'] == 'SHORT'].copy()
    pct_bins   = np.arange(0, 0.105, 0.01)
    pct_labels = [f'{int(x*100)}-{int((x+0.01)*100)}%' for x in pct_bins[:-1]]
    sub_l['pb'] = pd.cut(sub_l['hlpct_12'], bins=pct_bins, labels=pct_labels, right=False)
    sub_s['pb'] = pd.cut(1 - sub_s['hlpct_12'], bins=pct_bins, labels=pct_labels, right=False)
    combined = pd.concat([sub_l, sub_s])
    pb_wrs  = []
    pb_ns   = []
    pb_used = []
    for lbl in pct_labels:
        sub = combined[combined['pb'] == lbl]
        if len(sub) < 50:
            continue
        wr_t, _ = wr_ev(sub)
        pb_wrs.append(wr_t)
        pb_ns.append(len(sub))
        pb_used.append(lbl)
    colors_p = ['#2ecc71' if w >= BE else '#e74c3c' for w in pb_wrs]
    ax7.bar(pb_used, pb_wrs, color=colors_p, alpha=0.8)
    ax7.axhline(BE, color='red', linestyle='--', lw=1)
    ax7.set_title('hlpct 極端程度（0% = 最極端）')
    ax7.set_ylabel('勝率 %')
    ax7.set_ylim(50, 65)
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)

    # 8. 每月累積損益
    ax8 = fig.add_subplot(gs[2, 2])
    df['month'] = pd.to_datetime(df['time']).dt.to_period('M')
    monthly = df.groupby('month')['pnl'].sum()
    colors_m = ['#2ecc71' if v >= 0 else '#e74c3c' for v in monthly.values]
    ax8.bar(range(len(monthly)), monthly.values, color=colors_m, alpha=0.8)
    ax8.axhline(0, color='black', lw=0.8)
    ax8.set_title('每月損益')
    ax8.set_xlabel('月份（索引）')
    ax8.set_ylabel('損益 USDT')
    # 每 6 個月標一個 x 刻度
    tick_pos  = list(range(0, len(monthly), 6))
    tick_lbls = [str(monthly.index[i]) for i in tick_pos if i < len(monthly)]
    ax8.set_xticks(tick_pos[:len(tick_lbls)])
    ax8.set_xticklabels(tick_lbls, rotation=30, ha='right', fontsize=7)

    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  圖表已儲存：{out}")


# ══════════════════════════════════════════════════════════════
#  里程碑模擬（從 100U 開始，看多久能升到下注上限）
# ══════════════════════════════════════════════════════════════
STEP_UNIT = 250.0
BET_UNIT  = 5.0
MIN_BET   = 5.0
MAX_BET   = {'BTC': 250.0, 'ETH': 125.0}

MILESTONES = [250, 500, 1000, 2500, 6250, 12500]
MILESTONE_LABELS = {
    250:   '第一階段（250U → 下注 5U）',
    500:   '第二階段（500U → 下注 10U）',
    1000:  '第三階段（1000U → 下注 20U）',
    2500:  '第四階段（2500U → 下注 50U）',
    6250:  'ETH 下注上限（6250U → 125U/次）',
    12500: 'BTC 下注上限（12500U → 250U/次）',
}


def _staged_bet(capital: float, sym: str) -> float:
    step = int(capital // STEP_UNIT)
    bet  = max(MIN_BET, step * BET_UNIT)
    return min(bet, MAX_BET.get(sym, MAX_BET['ETH']))


def simulate_milestones(df: pd.DataFrame, start_capital: float = 100.0,
                        start_date: str | None = None):
    print(f"\n  {'='*64}")
    print(f"  里程碑模擬：從 {start_capital:.0f}U 出發（使用真實勝負序列）")
    print(f"  {'='*64}")

    trades = df.sort_values('time').reset_index(drop=True)

    if start_date is not None:
        cutoff = pd.Timestamp(start_date)
        trades = trades[trades['time'] >= cutoff].reset_index(drop=True)
        if trades.empty:
            print(f"  指定開始日期 {start_date} 之後無資料")
            return

    cap      = start_capital
    start_dt = trades['time'].iloc[0]
    remaining = list(MILESTONES)
    hit = []

    for i, row in trades.iterrows():
        bet = _staged_bet(cap, row['sym'])
        if row['win']:
            cap += bet * PAYOUT
        else:
            cap -= bet

        while remaining and cap >= remaining[0]:
            ms      = remaining.pop(0)
            elapsed = (row['time'] - start_dt).days
            hit.append((ms, row['time'].date(), i + 1, elapsed, bet))

        if not remaining:
            break

    if not hit:
        print(f"  資金不足以達到任何里程碑（最終資金 {cap:.1f}U）")
        return

    rows = []
    for ms, dt, trade_no, days, bet_at in hit:
        label = MILESTONE_LABELS.get(ms, f'{ms}U')
        rows.append((f'{ms:,}U', str(dt), f'{trade_no:,}', f'{days:,}天',
                     f'{bet_at:.0f}U', label))

    print_table(rows,
                ['里程碑', '達到日期', '第幾筆', '歷經天數', '當時下注', '說明'],
                f'里程碑達到時間（起始 {start_capital:.0f}U，'
                f'資料期間 {trades["time"].iloc[0].date()} ~ {trades["time"].iloc[-1].date()}）')

    # 若有里程碑未達到
    if remaining:
        print(f"\n  未達到的里程碑：{', '.join(str(m)+'U' for m in remaining)}")
        print(f"  模擬結束後資金：{cap:,.1f}U")


# ══════════════════════════════════════════════════════════════
#  壓力測試：從指定本金出發，找出歷史最壞情境
# ══════════════════════════════════════════════════════════════
def simulate_stress_test(df: pd.DataFrame, start_capital: float = 600.0):
    print(f"\n  {'='*64}")
    print(f"  壓力測試：從 {start_capital:.0f}U 出發")
    print(f"  {'='*64}")

    trades = df.sort_values('time').reset_index(drop=True)

    # ── 1. 全序列模擬，計算資金曲線 ───────────────────────────
    cap    = start_capital
    peak   = cap
    caps   = [cap]
    bets   = []
    losses = []

    for _, row in trades.iterrows():
        bet = _staged_bet(cap, row['sym'])
        bets.append(bet)
        if row['win']:
            cap += bet * PAYOUT
        else:
            cap -= bet
            losses.append(cap)
        peak = max(peak, cap)
        caps.append(cap)

    caps_s = pd.Series(caps)
    running_max = caps_s.cummax()
    dd_series   = (caps_s - running_max) / running_max * 100

    max_dd   = dd_series.min()
    min_cap  = caps_s.min()
    final    = caps_s.iloc[-1]

    print(f"\n  全歷史模擬（從 {start_capital:.0f}U 出發，{len(trades):,} 筆）")
    print(f"    最終資金：{final:,.1f}U")
    print(f"    最大回撤：{max_dd:.1f}%（最低 {min_cap:.1f}U）")

    # ── 2. 找出連續最多敗仗 ───────────────────────────────────
    max_streak = 0
    cur_streak = 0
    streak_start_idx = 0
    best_streak_start = 0
    for i, row in trades.iterrows():
        if not row['win']:
            if cur_streak == 0:
                streak_start_idx = i
            cur_streak += 1
            if cur_streak > max_streak:
                max_streak = cur_streak
                best_streak_start = streak_start_idx
        else:
            cur_streak = 0

    streak_trades = trades.iloc[best_streak_start:best_streak_start + max_streak]
    print(f"\n  歷史最長連敗：{max_streak} 連敗")
    print(f"    發生時間：{streak_trades['time'].iloc[0].date()} ~ "
          f"{streak_trades['time'].iloc[-1].date()}")

    # ── 3. 從目前本金模擬最壞連敗段 ──────────────────────────
    print(f"\n  情境模擬：從 {start_capital:.0f}U 遭遇 {max_streak} 連敗")
    cap_sim = start_capital
    rows_sc = []
    for k in range(max_streak):
        bet = _staged_bet(cap_sim, 'ETH')
        cap_before = cap_sim
        cap_sim -= bet
        step_lv = int(cap_before // STEP_UNIT)
        rows_sc.append((f'第{k+1}敗', f'{cap_before:.1f}U', f'{bet:.0f}U',
                        f'{cap_sim:.1f}U', f'下注檔位 {step_lv}'))
        if cap_sim < 0:
            break

    print_table(rows_sc,
                ['', '下注前', '下注', '下注後', '備註'],
                f'{max_streak} 連敗資金變化（以 ETH 計，自動降注）')

    # ── 4. 安全邊際分析 ──────────────────────────────────────
    print(f"\n  安全邊際分析（目前 {start_capital:.0f}U）")
    thresholds = [250 * i for i in range(1, int(start_capital // 250) + 2)]
    for th in thresholds:
        step = int(th // STEP_UNIT)
        bet  = max(MIN_BET, step * BET_UNIT)
        losses_to_drop = (start_capital - (th - 1)) / _staged_bet(start_capital, 'ETH')
        print(f"    資金跌破 {th:>5}U → 下注降為 {bet:.0f}U  "
              f"（需連敗 ~{losses_to_drop:.0f} 次）")

    print(f"\n  即使全程以 MIN_BET={MIN_BET:.0f}U 下注，爆倉需 "
          f"{start_capital / MIN_BET:.0f} 連敗")


# ══════════════════════════════════════════════════════════════
#  蒙地卡羅：隨機跳過 50% 訊號，模擬無法全天候執行的影響
# ══════════════════════════════════════════════════════════════
def simulate_monte_carlo(df: pd.DataFrame,
                         start_capital: float = 600.0,
                         skip_rate: float = 0.50,
                         n_runs: int = 1000,
                         seed: int = 42):
    print(f"\n  {'='*64}")
    print(f"  蒙地卡羅模擬：跳過 {skip_rate*100:.0f}% 訊號 × {n_runs} 次")
    print(f"  起始本金 {start_capital:.0f}U")
    print(f"  {'='*64}")

    rng    = np.random.default_rng(seed)
    trades = df.sort_values('time').reset_index(drop=True)
    wins   = trades['win'].to_numpy()
    syms   = trades['sym'].to_numpy()
    n      = len(trades)

    final_caps     = []
    early_min_caps = []   # 資金首次達到 2000U 之前的最低點（早期風險）
    abs_min_caps   = []   # 全程最低點（絕對值）
    danger_hits    = []   # 跌破 250U
    milestone_hits = {m: 0 for m in MILESTONES}

    EARLY_THRESHOLD = 2000.0   # 超過此金額後才算「穩定期」

    for _ in range(n_runs):
        mask  = rng.random(n) >= skip_rate   # True = 執行這筆
        cap   = start_capital
        min_c = cap
        early_min = cap
        in_early  = True
        danger    = False
        reached   = set()

        for i in range(n):
            if not mask[i]:
                continue
            bet = _staged_bet(cap, syms[i])
            if wins[i]:
                cap += bet * PAYOUT
            else:
                cap -= bet
            if cap < min_c:
                min_c = cap
            if in_early:
                if cap < early_min:
                    early_min = cap
                if cap >= EARLY_THRESHOLD:
                    in_early = False
            if cap < 250:
                danger = True
            for m in MILESTONES:
                if m not in reached and cap >= m:
                    reached.add(m)

        final_caps.append(cap)
        early_min_caps.append(early_min)
        abs_min_caps.append(min_c)
        danger_hits.append(danger)
        for m in reached:
            milestone_hits[m] += 1

    final_caps     = np.array(final_caps)
    early_min_caps = np.array(early_min_caps)
    abs_min_caps   = np.array(abs_min_caps)

    p5, p25, p50, p75, p95 = np.percentile(final_caps, [5, 25, 50, 75, 95])

    print(f"\n  最終資金分佈（{n_runs} 次模擬）")
    print(f"    最悲觀  5%：{p5:>12,.0f}U")
    print(f"    25 百分位：{p25:>12,.0f}U")
    print(f"    中位數  50%：{p50:>12,.0f}U")
    print(f"    75 百分位：{p75:>12,.0f}U")
    print(f"    最樂觀 95%：{p95:>12,.0f}U")

    # 早期風險（資金未超過 2000U 前的最低點）
    em_p5  = np.percentile(early_min_caps, 5)
    em_p50 = np.percentile(early_min_caps, 50)
    print(f"\n  早期風險（資金達到 {EARLY_THRESHOLD:.0f}U 之前的最低點）")
    print(f"    中位數最低點：{em_p50:,.1f}U")
    print(f"    最壞  5%：    {em_p5:,.1f}U")

    danger_pct = np.mean(danger_hits) * 100
    below300   = np.mean(abs_min_caps < 300) * 100
    below200   = np.mean(abs_min_caps < 200) * 100
    print(f"\n  風險指標（全程最低點統計）")
    print(f"    最低資金中位數：{np.percentile(abs_min_caps, 50):,.1f}U")
    print(f"    最低資金  5%：  {np.percentile(abs_min_caps, 5):,.1f}U")
    print(f"    曾跌破 300U 機率：{below300:.1f}%")
    print(f"    曾跌破 250U 機率：{danger_pct:.1f}%")
    print(f"    曾跌破 200U 機率：{below200:.1f}%")

    # 里程碑達成率
    rows_m = []
    for m in MILESTONES:
        rate = milestone_hits[m] / n_runs * 100
        label = MILESTONE_LABELS.get(m, '')
        rows_m.append((f'{m:,}U', f'{rate:.1f}%', label))
    print_table(rows_m, ['里程碑', '達成率', '說明'], '里程碑達成率（跳過50%訊號）')

    # 對比基準（不跳過任何訊號）
    cap_base = start_capital
    peak_b   = cap_base
    for i in range(n):
        bet = _staged_bet(cap_base, syms[i])
        if wins[i]:
            cap_base += bet * PAYOUT
        else:
            cap_base -= bet
        peak_b = max(peak_b, cap_base)
    print(f"\n  對比：全程不跳過（100% 執行）最終資金 {cap_base:,.0f}U")
    print(f"  跳過 50% 後中位數：{p50:,.0f}U（約為全程的 {p50/cap_base*100:.1f}%）")


# ══════════════════════════════════════════════════════════════
#  主程式
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='trades_tiered_live.csv',
                        help='交易明細 CSV 路徑')
    parser.add_argument('--milestone_date', default=None,
                        help='里程碑模擬起始日期，格式 YYYY-MM-DD（預設從資料最早日）')
    parser.add_argument('--milestone_capital', type=float, default=100.0,
                        help='里程碑模擬起始本金（預設 100）')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"  找不到 {csv_path}，請先執行 event_contract_backtest.py")
        return

    df = pd.read_csv(csv_path, parse_dates=['time'])
    print(f"\n{'='*64}")
    print(f"  事件合約交易分析  |  {csv_path.name}  |  {len(df):,} 筆")
    print(f"  期間：{df['time'].min().date()} ~ {df['time'].max().date()}")
    print(f"{'='*64}")

    analyze_breakdowns(df)
    analyze_tier_cross(df)
    simulate_filter_impact(df)
    simulate_milestones(df, start_capital=args.milestone_capital,
                        start_date=args.milestone_date)
    simulate_stress_test(df, start_capital=600.0)
    simulate_monte_carlo(df, start_capital=600.0, skip_rate=0.50, n_runs=1000)
    plot_analysis(df, 'ml_trade_analysis.png')
    print()


if __name__ == '__main__':
    main()
