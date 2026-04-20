"""
event_contract_backtest.py
══════════════════════════
幣安事件合約訊號探索回測（30 分鐘合約，賠率 85%）

資料：ETH / BTC 15m 或 5m CSV
結算：偵測 bar T → bar T+N 收盤即為精確結算價（30 分鐘）

損益平衡勝率：1 / (1 + 0.85) ≈ 54.1%

用法：
    python event_contract_backtest.py            # 預設 BTC 15m
    python event_contract_backtest.py --sym BOTH
    python event_contract_backtest.py --sym BOTH --max_concurrent 5

"""

import argparse
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# ── 預設參數 ───────────────────────────────────────────────────
DEFAULT_PAYOUT   = 0.85   # 30 分鐘賠率
SETTLE_BARS      = 2      # 30 分鐘 = 2 × 15m bars
INITIAL_CAP      = 1000.0 # 模擬初始本金 1000 USDT
MAX_BET_ETH      = 125.0
MAX_BET_BTC      = 250.0
MIN_BET          = 5.0


# ══════════════════════════════════════════════════════════════
#  資料載入與指標計算
# ══════════════════════════════════════════════════════════════
def load_and_build(sym: str, tf: str = '15m', config_path="config.json",
                   start: str = None, end: str = None) -> pd.DataFrame:
    path = Path(__file__).parent / 'data' / f'{sym}USDT_{tf}.csv'
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.iloc[:-1].copy()

    # scale：以 15m 為基準，5m → scale=3，所有時間窗口等比放大
    bar_min = int(tf[:-1])
    scale   = 15 // bar_min   # 15m→1, 5m→3

    # 時間過濾：CLI 參數優先，其次 config，再次預設值
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        bt = cfg.get('backtest', {})
        cfg_start = bt.get('start_date', '2020-01-01')
        cfg_end   = bt.get('end_date',   '2099-01-01')
    except Exception:
        cfg_start, cfg_end = '2020-01-01', '2099-01-01'

    t_start = pd.Timestamp(start if start else cfg_start)
    t_end   = pd.Timestamp(end   if end   else cfg_end)
    df = df[(df.index >= t_start) & (df.index <= t_end)]

    c = df['close']
    h = df['high']
    l = df['low']

    # ── 附加指標 ───────────────────────────────────────────
    # RSI 已有，額外計算慢速版
    def _rsi(s, n):
        d = s.diff()
        g = d.clip(lower=0).rolling(n).mean()
        ls = (-d.clip(upper=0)).rolling(n).mean()
        return 100 - 100 / (1 + g / ls.replace(0, np.nan))

    df['RSI_7']  = _rsi(c, 7)
    df['RSI_14'] = df['RSI']          # 已有
    df['RSI_21'] = _rsi(c, 21)

    # MACD (12,26,9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']

    # Bollinger Bands (20, 2σ) — BB_Mid 已有
    bb_std = c.rolling(20).std(ddof=0)
    df['BB_upper'] = df['BB_Mid'] + 2 * bb_std
    df['BB_lower'] = df['BB_Mid'] - 2 * bb_std
    df['BB_pct']   = (c - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])  # 0~1

    # ATR 已有；Volatility ratio（當前 ATR / 20 棒均 ATR）
    # 均值窗口乘以 scale，保持覆蓋時間等長（20×15m = 60×5m = 5小時）
    df['ATR_ratio'] = df['ATR'] / df['ATR'].rolling(20 * scale).mean()

    # 動能：N 棒前收盤的漲跌幅（保持短期，不 scale）
    for n in [1, 2, 4, 8]:
        df[f'ret_{n}'] = c.pct_change(n)

    # EMA 差距（%）
    df['ema_gap'] = (c - df['EMA']) / df['EMA'] * 100

    # 成交量 z-score（窗口乘以 scale，覆蓋相同時間長度）
    df['vol_z'] = (df['volume'] - df['volume'].rolling(20 * scale).mean()) / \
                  df['volume'].rolling(20 * scale).std(ddof=0)

    # 高低位相對位置：邏輯 N 棒 × scale，使 15m/5m 覆蓋相同時間區間
    # hlpct_4 = 4×15m=1h / 12×5m=1h；hlpct_12 = 12×15m=3h / 36×5m=3h
    for n in [4, 8, 12]:
        rh = h.rolling(n * scale).max()
        rl = l.rolling(n * scale).min()
        df[f'hlpct_{n}'] = (c - rl) / (rh - rl).replace(0, np.nan)

    # ── Regime 判定（BBmid + EMA_200 雙斜率）────────────────
    # 斜率窗口乘以 scale（20×15m = 60×5m = 5小時）
    SLOPE_N = 20 * scale
    bbmid_slope = df['BB_Mid'].diff(SLOPE_N)
    ema_slope   = df['EMA'].diff(SLOPE_N)
    bull = (df['ADX'] >= 20) & (bbmid_slope > 0) & (ema_slope > 0)
    bear = (df['ADX'] >= 20) & (bbmid_slope < 0) & (ema_slope < 0)
    df['regime'] = 'ranging'
    df.loc[bull, 'regime'] = 'bull'
    df.loc[bear, 'regime'] = 'bear'

    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════════════
#  訊號定義（保留回測驗證有效的 10 個訊號）
#  每個訊號回傳 pd.Series：1=做多（賭漲）, -1=做空（賭跌）, 0=不開單
# ══════════════════════════════════════════════════════════════
def define_signals(df: pd.DataFrame) -> dict:
    sigs = {}

    pct12 = df['hlpct_12']
    _base = pd.Series(0, index=df.index)
    _base[pct12 < 0.10] = 1
    _base[pct12 > 0.90] = -1

    f1_mask = df['ATR_ratio'] < 1.20

    # F1：排除高波動（ATR_ratio ≥ 1.20）—— 實盤使用的基準訊號
    sigs['hlpct12_10_F1'] = _base.where(f1_mask, 0)

    # ── 超賣/超買確認（均值回歸：方向與極端位置一致才進場）────
    rsi = df['RSI_14']
    rsi_ok_35 = ((_base ==  1) & (rsi < 35)) | ((_base == -1) & (rsi > 65))
    rsi_ok_30 = ((_base ==  1) & (rsi < 30)) | ((_base == -1) & (rsi > 70))
    rsi_ok_25 = ((_base ==  1) & (rsi < 25)) | ((_base == -1) & (rsi > 75))
    rsi_ok_20 = ((_base ==  1) & (rsi < 20)) | ((_base == -1) & (rsi > 80))

    sigs['F1_RSI35'] = _base.where(f1_mask & rsi_ok_35, 0)
    sigs['F1_RSI30'] = _base.where(f1_mask & rsi_ok_30, 0)
    sigs['F1_RSI25'] = _base.where(f1_mask & rsi_ok_25, 0)
    sigs['F1_RSI20'] = _base.where(f1_mask & rsi_ok_20, 0)

    # ── MACD hist 動能確認（動能仍在反彈方向的對立面才進）──────
    # 做多：hist < 0（跌勢中），做空：hist > 0（漲勢中）
    mh = df['MACD_hist']
    macd_ok = ((_base ==  1) & (mh < 0)) | ((_base == -1) & (mh > 0))

    sigs['F1_MACD'] = _base.where(f1_mask & macd_ok, 0)

    # ── RSI + MACD 雙重確認 ───────────────────────────────────
    sigs['F1_RSI35_MACD'] = _base.where(f1_mask & rsi_ok_35 & macd_ok, 0)
    sigs['F1_RSI30_MACD'] = _base.where(f1_mask & rsi_ok_30 & macd_ok, 0)

    return sigs


# ══════════════════════════════════════════════════════════════
#  回測核心
# ══════════════════════════════════════════════════════════════
def backtest_signal(df: pd.DataFrame, sig: pd.Series,
                    payout: float, settle_bars: int,
                    max_bet: float,
                    fixed_bet: float = 0.0,
                    init_cap: float = 0.0,
                    sample_rate: float = 1.0,
                    rng: np.random.Generator = None,
                    use_cooldown: bool = False,
                    cd_losses: int = 2,
                    cd_hours: float = 3.0,
                    cd_skip_hours: float = 2.0,
                    max_concurrent: int = 0) -> dict:
    """
    fixed_bet > 0：固定下注模式
      init_cap > 0：設定初始本金，本金低於 MIN_BET 時停止（模擬停損保護）
    fixed_bet = 0：半 Kelly 動態下注
    sample_rate：實際下注比例（0~1），模擬無法全天候手動下單
    rng：隨機數生成器，None 時使用全部訊號（sample_rate 無效）
    """
    close  = df['close']
    settle = close.shift(-settle_bars)

    mask = (sig != 0) & settle.notna()
    # 隨機抽樣：保留 sample_rate 比例的訊號
    if rng is not None and sample_rate < 1.0:
        all_idx   = np.where(mask)[0]
        keep_n    = max(1, int(len(all_idx) * sample_rate))
        keep_idx  = rng.choice(all_idx, size=keep_n, replace=False)
        keep_idx.sort()
        new_mask  = np.zeros(len(mask), dtype=bool)
        new_mask[keep_idx] = True
        mask = pd.Series(new_mask, index=mask.index)
    if mask.sum() < 20:
        return None

    direction = sig[mask]
    entry     = close[mask]
    exit_p    = settle[mask]
    correct   = ((exit_p > entry) & (direction == 1)) | \
                ((exit_p < entry) & (direction == -1))
    correct   = correct & (exit_p != entry)

    n_trades_total = len(direction)
    b = payout
    use_fixed   = fixed_bet > 0
    use_stopcap = init_cap > 0   # 有給初始本金就啟用停損，不限模式

    # 計算全樣本勝率（供 Kelly 參考，不受停損影響）
    win_rate_full = correct.sum() / n_trades_total
    kelly      = max(0.0, (win_rate_full * b - (1 - win_rate_full)) / b)
    half_kelly = kelly / 2

    # 模擬資金曲線
    cap    = init_cap if use_stopcap else (fixed_bet if use_fixed else INITIAL_CAP)
    equity = [cap]
    bets_log  = []
    n_played  = 0      # 實際下注筆數（停損後不再下注）
    n_wins    = 0
    stopped   = False  # 是否已觸發停損

    # 冷卻機制狀態
    cd_loss_times  = []    # 連輸時間戳記
    cd_skip_next   = False
    cd_skip_expiry = None  # 跳過時效截止時間（None = 無限期，不啟用時效）

    # 同時持倉上限
    bar_dur      = df.index[1] - df.index[0]
    open_settles = []   # 各部位的結算時間戳

    for trade_time, w in correct.items():
        # 冷卻跳過：有時效限制時，過期自動解除
        if use_cooldown and cd_skip_next:
            if cd_skip_expiry is None or trade_time <= cd_skip_expiry:
                cd_skip_next   = False
                cd_skip_expiry = None
                equity.append(cap)
                continue
            else:
                # 超過時效，不跳過，直接解除冷卻
                cd_skip_next   = False
                cd_skip_expiry = None

        # 同時持倉上限：移除已結算部位，超限則跳過
        if max_concurrent > 0:
            open_settles = [t for t in open_settles if t > trade_time]
            if len(open_settles) >= max_concurrent:
                equity.append(cap)
                continue

        if cap < MIN_BET:          # 通用停損：本金不足最小下單量即停止
            stopped = True
            break
        if use_fixed:
            bet = min(fixed_bet, cap) if use_stopcap else fixed_bet
        else:
            bet = min(max_bet, cap * half_kelly)  # Kelly：按比例下注，不強制 MIN_BET
            bet = max(bet, MIN_BET)               # 但至少 MIN_BET（已確認 cap >= MIN_BET）
        bets_log.append(bet)
        cap += bet * payout if w else -bet
        equity.append(cap)
        n_played += 1
        if max_concurrent > 0:
            open_settles.append(trade_time + settle_bars * bar_dur)
        if w:
            n_wins += 1
            if use_cooldown:
                cd_loss_times = []   # 贏了，連輸計數重置
        elif use_cooldown:
            cd_loss_times.append(trade_time)
            cutoff = trade_time - pd.Timedelta(hours=cd_hours)
            cd_loss_times = [t for t in cd_loss_times if t > cutoff]
            if len(cd_loss_times) >= cd_losses:
                cd_skip_next   = True
                cd_skip_expiry = (trade_time + pd.Timedelta(hours=cd_skip_hours)
                                  if cd_skip_hours is not None else None)
                cd_loss_times  = []

    win_rate_played = n_wins / n_played if n_played > 0 else 0
    equity_s = pd.Series(equity)
    init_val = init_cap if use_stopcap else (fixed_bet if use_fixed else INITIAL_CAP)

    # MDD
    if use_fixed:
        running_max = equity_s.cummax()
        mdd     = (equity_s - running_max).min()
        mdd_pct = mdd / init_val * 100
    else:
        peak    = equity_s.cummax()
        mdd     = None
        mdd_pct = ((equity_s - peak) / peak.replace(0, np.nan)).min() * 100

    # Sharpe
    rets_s = pd.Series([bt * payout if w else -bt
                        for bt, w in zip(bets_log, correct[:n_played])])
    sharpe = (rets_s.mean() / rets_s.std() * np.sqrt(252)
              if len(rets_s) > 1 and rets_s.std() > 0 else 0)

    return {
        'n':             n_played,
        'n_total':       n_trades_total,
        'win_rate':      win_rate_played * 100,
        'win_rate_full': win_rate_full * 100,
        'kelly':         kelly * 100,
        'half_k':        half_kelly * 100,
        'final':         cap,
        'init':          init_val,
        'ret%':          (cap / init_val - 1) * 100 if init_val > 0 else 0,
        'mdd_abs':       mdd,
        'mdd%':          mdd_pct,
        'ev':            (win_rate_full * payout - (1 - win_rate_full)) * 100,
        'stopped':       stopped,
        'equity':        equity_s,
        'times':         df.index[mask],
    }


# ══════════════════════════════════════════════════════════════
#  BTC + ETH 合併回測（共用持倉槽）
# ══════════════════════════════════════════════════════════════
def backtest_combined(df_btc: pd.DataFrame, sig_btc: pd.Series,
                      df_eth: pd.DataFrame, sig_eth: pd.Series,
                      payout: float, settle_bars: int,
                      fixed_bet: float = 0.0,
                      init_cap: float = 0.0,
                      max_concurrent: int = 5) -> dict | None:
    """
    BTC 和 ETH 的訊號合併成單一時間軸，共用同一個持倉槽（max_concurrent）和本金。
    Kelly 各幣種獨立追蹤勝率，前 100 筆用固定 WR=0.55。
    """
    bar_dur = df_btc.index[1] - df_btc.index[0]

    def build_events(df, sig, sym, mb):
        close_s  = df['close']
        settle_s = close_s.shift(-settle_bars)
        events = []
        for ts, d in sig[sig != 0].items():
            if ts not in df.index:
                continue
            ep = close_s.loc[ts]
            sp = settle_s.loc[ts]
            if pd.isna(sp):
                continue
            win = (((sp > ep) and d == 1) or ((sp < ep) and d == -1)) and sp != ep
            events.append({'time': ts, 'sym': sym, 'win': win, 'max_bet': mb})
        return events

    events = build_events(df_btc, sig_btc, 'BTC', MAX_BET_BTC)
    events += build_events(df_eth, sig_eth, 'ETH', MAX_BET_ETH)
    events.sort(key=lambda x: x['time'])

    if not events:
        return None

    use_fixed   = fixed_bet > 0
    use_stopcap = init_cap > 0
    cap      = init_cap if use_stopcap else (fixed_bet if use_fixed else INITIAL_CAP)
    init_val = cap

    equity       = [cap]
    open_settles = []
    n_played = n_wins = 0
    stopped  = False
    bets_log = []
    win_hist = {'BTC': [], 'ETH': []}   # 各幣 rolling WR

    for ev in events:
        ts, sym, w, mb = ev['time'], ev['sym'], ev['win'], ev['max_bet']

        # 移除已結算部位
        if max_concurrent > 0:
            open_settles = [t for t in open_settles if t > ts]
            if len(open_settles) >= max_concurrent:
                continue

        if cap < MIN_BET:
            stopped = True
            break

        # 下注大小
        if use_fixed:
            bet = min(fixed_bet, mb)
            if use_stopcap:
                bet = min(bet, cap)
        else:
            hist = win_hist[sym]
            wr   = sum(hist[-100:]) / len(hist[-100:]) if len(hist) >= 100 else 0.55
            hk   = max(0.0, (wr * payout - (1 - wr)) / payout) / 2
            bet  = min(mb, cap * hk)
            bet  = max(bet, MIN_BET)

        bets_log.append(bet)
        cap += bet * payout if w else -bet
        equity.append(cap)
        n_played += 1
        win_hist[sym].append(1 if w else 0)
        if w:
            n_wins += 1
        if max_concurrent > 0:
            open_settles.append(ts + settle_bars * bar_dur)

    if n_played == 0:
        return None

    wr_final  = n_wins / n_played
    equity_s  = pd.Series(equity)
    peak      = equity_s.cummax()
    mdd_abs   = (equity_s - peak).min()
    mdd_pct   = ((equity_s - peak) / peak.replace(0, np.nan)).min() * 100
    avg_bet   = np.mean(bets_log) if bets_log else 0
    half_k    = max(0.0, (wr_final * payout - (1 - wr_final)) / payout) / 2

    return {
        'n':        n_played,
        'win_rate': wr_final * 100,
        'avg_bet':  avg_bet,
        'half_k':   half_k * 100,
        'final':    cap,
        'init':     init_val,
        'ret%':     (cap / init_val - 1) * 100 if init_val > 0 else 0,
        'mdd_abs':  mdd_abs,
        'mdd%':     mdd_pct,
        'ev':       (wr_final * payout - (1 - wr_final)) * 100,
        'stopped':  stopped,
        'equity':   equity_s,
    }


# ══════════════════════════════════════════════════════════════
#  分層下注合併回測（premium 加注 / regular 正常）
# ══════════════════════════════════════════════════════════════
def backtest_tiered_combined(df_btc, sig_f1_btc, sig_prem_btc,
                              df_eth, sig_f1_eth, sig_prem_eth,
                              payout, settle_bars,
                              fixed_bet=0.0, init_cap=0.0,
                              max_concurrent=5) -> dict:
    """
    Premium 信號（RSI30_MACD）用自己的滾動 WR 計算 Kelly（加注）。
    Regular 信號（F1 減去 premium）用各自的滾動 WR（正常注）。
    兩層前 100 筆各用固定 WR=0.55 預熱。
    """
    bar_dur = df_btc.index[1] - df_btc.index[0]

    def build_events(df, sig_f1, sig_prem, sym):
        close_s  = df['close']
        settle_s = close_s.shift(-settle_bars)
        events = []
        for ts, d in sig_f1[sig_f1 != 0].items():
            sp = settle_s.get(ts)
            if sp is None or pd.isna(sp):
                continue
            ep  = close_s.loc[ts]
            win = (((sp > ep) and d == 1) or ((sp < ep) and d == -1)) and sp != ep
            tier = 'prem' if sig_prem.get(ts, 0) != 0 else 'reg'
            events.append({'time': ts, 'sym': sym, 'win': win, 'tier': tier})
        return events

    events = build_events(df_btc, sig_f1_btc, sig_prem_btc, 'BTC')
    events += build_events(df_eth, sig_f1_eth, sig_prem_eth, 'ETH')
    events.sort(key=lambda x: x['time'])

    if not events:
        return None

    use_fixed   = fixed_bet > 0
    use_stopcap = init_cap > 0
    cap      = init_cap if use_stopcap else (fixed_bet if use_fixed else INITIAL_CAP)
    init_val = cap

    equity        = [cap]
    open_settles  = []
    played_events = []
    n_played = n_wins = 0
    stopped  = False
    bets_log = []
    win_hist = {'prem': [], 'reg': []}

    for ev in events:
        ts, win, tier = ev['time'], ev['win'], ev['tier']
        sym = ev['sym']
        mb  = MAX_BET_BTC if sym == 'BTC' else MAX_BET_ETH

        if max_concurrent > 0:
            open_settles = [t for t in open_settles if t > ts]
            if len(open_settles) >= max_concurrent:
                continue

        if cap < MIN_BET:
            stopped = True
            break

        if use_fixed:
            bet = min(fixed_bet, mb)
            if use_stopcap:
                bet = min(bet, cap)
        else:
            hist = win_hist[tier]
            wr   = sum(hist[-100:]) / len(hist[-100:]) if len(hist) >= 100 else 0.55
            hk   = max(0.0, (wr * payout - (1 - wr)) / payout) / 2
            bet  = min(mb, cap * hk)
            bet  = max(bet, MIN_BET)

        bets_log.append(bet)
        cap += bet * payout if win else -bet
        equity.append(cap)
        n_played += 1
        played_events.append(ev)
        win_hist[tier].append(1 if win else 0)
        if win:
            n_wins += 1
        if max_concurrent > 0:
            open_settles.append(ts + settle_bars * bar_dur)

    if n_played == 0:
        return None

    wr_final = n_wins / n_played
    equity_s = pd.Series(equity)
    peak     = equity_s.cummax()
    mdd_pct  = ((equity_s - peak) / peak.replace(0, np.nan)).min() * 100
    half_k   = max(0.0, (wr_final * payout - (1 - wr_final)) / payout) / 2

    n_prem = sum(1 for e in played_events if e['tier'] == 'prem')
    n_reg  = sum(1 for e in played_events if e['tier'] == 'reg')

    return {
        'n':       n_played,
        'n_prem':  n_prem,
        'n_reg':   n_reg,
        'win_rate': wr_final * 100,
        'half_k':  half_k * 100,
        'final':   cap,
        'init':    init_val,
        'ret%':    (cap / init_val - 1) * 100 if init_val > 0 else 0,
        'mdd%':    mdd_pct,
        'ev':      (wr_final * payout - (1 - wr_final)) * 100,
        'stopped': stopped,
        'equity':  equity_s,
    }


# ══════════════════════════════════════════════════════════════
#  主程式
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sym',     default='BTC',  help='ETH、BTC 或 BOTH（兩幣合併）')
    parser.add_argument('--tf',      default='5m',   help='K 棒時間框架：15m 或 5m（預設 5m）')
    parser.add_argument('--payout',  type=float, default=DEFAULT_PAYOUT)
    parser.add_argument('--settle',  type=int,   default=None,
                        help='結算 bar 數（預設依 tf 自動：15m=2, 5m=6）')
    parser.add_argument('--top',     type=int,   default=20,
                        help='顯示前 N 個訊號（依勝率排序）')
    parser.add_argument('--min_wr',  type=float, default=54.1,
                        help='顯示的最低勝率閾值')
    parser.add_argument('--min_n',     type=int,   default=50,
                        help='最少交易次數')
    parser.add_argument('--fixed_bet', type=float, default=0.0,
                        help='固定下注金額（USDT），0 = 使用半 Kelly')
    parser.add_argument('--init_cap',  type=float, default=0.0,
                        help='初始本金（USDT），搭配 --fixed_bet 使用；本金 < 最小下單量時停止')
    parser.add_argument('--start', default=None,
                        help='回測起始日期，格式 YYYY-MM-DD（覆蓋 config）')
    parser.add_argument('--end',   default=None,
                        help='回測結束日期，格式 YYYY-MM-DD（覆蓋 config）')
    parser.add_argument('--sample_rate', type=float, default=1.0,
                        help='隨機抽樣比例（0~1），模擬無法全天候手動下單，預設 1.0 = 全部')
    parser.add_argument('--trials',      type=int,   default=200,
                        help='隨機抽樣試驗次數（--sample_rate < 1 時生效），預設 200')
    parser.add_argument('--cooldown',    action='store_true',
                        help='啟用冷卻機制')
    parser.add_argument('--cd_losses',   type=int,   default=2,
                        help='冷卻觸發連輸次數（預設 2）')
    parser.add_argument('--cd_hours',      type=float, default=3.0,
                        help='冷卻觀察時間窗口，小時（預設 3.0）')
    parser.add_argument('--cd_skip_hours', type=float, default=2.0,
                        help='跳過時效上限，小時（預設 2.0；0 = 無限期）')
    parser.add_argument('--max_concurrent', type=int, default=5,
                        help='同時持倉上限（預設 5；0 = 無限制）')
    args = parser.parse_args()

    sym          = args.sym.upper()
    tf           = args.tf
    bar_min      = int(tf[:-1])
    tf_scale     = 15 // bar_min          # 15m→1, 5m→3
    payout       = args.payout
    settle_b     = args.settle if args.settle is not None else SETTLE_BARS * tf_scale
    fixed_bet    = args.fixed_bet
    init_cap     = args.init_cap
    sample_rate  = max(0.01, min(1.0, args.sample_rate))
    use_cooldown  = args.cooldown
    cd_losses     = args.cd_losses
    cd_hours      = args.cd_hours
    cd_skip_hours  = args.cd_skip_hours if args.cd_skip_hours > 0 else None
    max_concurrent = args.max_concurrent
    trials      = args.trials
    break_even  = 1 / (1 + payout) * 100

    if fixed_bet > 0:
        bet_mode = f"固定 {fixed_bet} USDT/筆"
        if init_cap > 0:
            bet_mode += f"，初始本金 {init_cap} USDT（本金 < {MIN_BET} USDT 停止）"
    else:
        bet_mode = "半 Kelly 動態"
    period_str = f"{args.start or 'config'} → {args.end or 'config'}"

    # ── BTC + ETH 合併模式 ────────────────────────────────────
    if sym == 'BOTH':
        print(f"\n{'='*60}")
        print(f"  幣安事件合約回測  |  BTC + ETH 合併  |  {tf}  |  賠率 {payout*100:.0f}%")
        print(f"  期間: {period_str}")
        print(f"  結算: {settle_b} bars × {tf} = {settle_b * bar_min} 分鐘")
        print(f"  損益平衡勝率: {break_even:.1f}%  |  同時持倉上限: {max_concurrent} 筆（BTC+ETH 共用）")
        print(f"  下注模式: {bet_mode}")
        print(f"{'='*60}\n")

        print("  載入 BTC 資料...", end='', flush=True)
        df_btc = load_and_build('BTC', tf=tf, start=args.start, end=args.end)
        print(f" {len(df_btc):,} 根")
        print("  載入 ETH 資料...", end='', flush=True)
        df_eth = load_and_build('ETH', tf=tf, start=args.start, end=args.end)
        print(f" {len(df_eth):,} 根")

        print("  掃描訊號...", end='', flush=True)
        sigs_btc = define_signals(df_btc)
        sigs_eth = define_signals(df_eth)
        common_names = [n for n in sigs_btc if n in sigs_eth]
        print(f" {len(common_names)} 個共同訊號")

        results = []
        for name in common_names:
            r = backtest_combined(df_btc, sigs_btc[name],
                                  df_eth, sigs_eth[name],
                                  payout, settle_b,
                                  fixed_bet=fixed_bet, init_cap=init_cap,
                                  max_concurrent=max_concurrent)
            if r and r['n'] >= args.min_n:
                r['name'] = name
                results.append(r)

        results.sort(key=lambda x: x['win_rate'], reverse=True)

        above = [r for r in results if r['win_rate'] >= args.min_wr]
        print(f"\n  勝率 ≥ {args.min_wr}% 的訊號共 {len(above)} 個（總掃描 {len(results)} 個）\n")

        top_n = results[:args.top]
        hdr = (f"  {'訊號名稱':<22} {'交易數':>7} {'勝率%':>7} {'期望值':>7} "
               f"{'Half-K%':>8} {'模擬終值':>12} {'淨損益':>12} {'MDD%':>8}")
        print(hdr)
        print(f"  {'─'*82}")
        for r in top_n:
            flag = '★' if r['win_rate'] >= args.min_wr else ' '
            net_pnl = r['final'] - r['init']
            print(f"{flag} {r['name']:<22} {r['n']:>7} {r['win_rate']:>6.1f}% "
                  f"{r['ev']:>+6.2f}% {r['half_k']:>7.1f}% "
                  f"{r['final']:>11,.1f} {net_pnl:>+11,.1f} {r['mdd%']:>+6.1f}%")
        print(f"  {'─'*82}")
        print(f"  （★ = 勝率 ≥ 損益平衡 {break_even:.1f}%，MDD = 相對峰值百分比）\n")

        # ── 分層下注：F1_RSI30_MACD 加注 + 其餘 F1 正常注 ──
        prem_name = 'F1_RSI30_MACD'
        if prem_name in sigs_btc and prem_name in sigs_eth:
            tr = backtest_tiered_combined(
                df_btc, sigs_btc['hlpct12_10_F1'], sigs_btc[prem_name],
                df_eth, sigs_eth['hlpct12_10_F1'], sigs_eth[prem_name],
                payout, settle_b,
                fixed_bet=fixed_bet, init_cap=init_cap,
                max_concurrent=max_concurrent)
            if tr:
                print(f"  {'─'*60}")
                print(f"  分層下注試驗：{prem_name} 加注 × 其餘 F1 正常注")
                print(f"  {'─'*60}")
                print(f"  總交易數：{tr['n']:,}  "
                      f"（Premium {tr['n_prem']:,} / Regular {tr['n_reg']:,}）")
                print(f"  合併勝率：{tr['win_rate']:.1f}%  "
                      f"期望值：{tr['ev']:+.2f}%  Half-K：{tr['half_k']:.1f}%")
                print(f"  模擬終值：{tr['final']:,.1f} USDT  "
                      f"淨損益：{tr['final']-tr['init']:+,.1f}  "
                      f"MDD：{tr['mdd%']:+.1f}%")
                print()
        return

    # ── 單一幣種模式（BTC 或 ETH）───────────────────────────
    max_bet     = MAX_BET_BTC if sym == 'BTC' else MAX_BET_ETH
    do_sampling = sample_rate < 1.0

    print(f"\n{'='*60}")
    print(f"  幣安事件合約回測  |  {sym}/USDT  |  {tf}  |  賠率 {payout*100:.0f}%")
    print(f"  期間: {period_str}")
    print(f"  結算: {settle_b} bars × {tf} = {settle_b * bar_min} 分鐘")
    print(f"  損益平衡勝率: {break_even:.1f}%")
    print(f"  下注模式: {bet_mode}")
    print(f"{'='*60}\n")

    print("  載入資料與計算指標...", end='', flush=True)

    df = load_and_build(sym, tf=tf, start=args.start, end=args.end)
    print(f" 完成  {len(df):,} 根 K 棒")

    print("  掃描訊號...", end='', flush=True)
    signals = define_signals(df)
    print(f" {len(signals)} 個訊號")

    # 固定下注時同樣套用幣種上限
    eff_fixed = min(fixed_bet, max_bet) if fixed_bet > 0 else 0.0
    if fixed_bet > 0 and eff_fixed != fixed_bet:
        print(f"  ⚠ 固定下注 {fixed_bet} USDT 超過 {sym} 上限，已自動截為 {eff_fixed} USDT")

    results = []
    for name, sig in signals.items():
        r = backtest_signal(df, sig, payout, settle_b, max_bet, eff_fixed, init_cap,
                            use_cooldown=use_cooldown,
                            cd_losses=cd_losses, cd_hours=cd_hours,
                            cd_skip_hours=cd_skip_hours,
                            max_concurrent=max_concurrent)
        if r and r['n'] >= args.min_n:
            r['name'] = name
            results.append(r)

    # 依勝率排序
    results.sort(key=lambda x: x['win_rate'], reverse=True)

    # ── 輸出表格 ─────────────────────────────────────────────
    above = [r for r in results if r['win_rate'] >= args.min_wr]
    show_stopcap = init_cap > 0
    print(f"\n  勝率 ≥ {args.min_wr}% 的訊號共 {len(above)} 個（總掃描 {len(results)} 個）\n")

    top_n = results[:args.top]
    W = 90 if show_stopcap else 82
    hdr = (f"  {'訊號名稱':<22} {'交易數':>7} {'勝率%':>7} {'期望值':>7} "
           f"{'Half-K%':>8} {'模擬終值':>12} {'淨損益':>12} "
           f"{'MDD(U)' if eff_fixed > 0 else 'MDD%':>8}")
    if show_stopcap:
        hdr += f"  {'停損':>4}"
    print(hdr)
    print(f"  {'─'*W}")
    for r in top_n:
        flag = '★' if r['win_rate'] >= args.min_wr else ' '
        mdd_str = f"{r['mdd_abs']:>+7.1f}U" if eff_fixed > 0 else f"{r['mdd%']:>+6.1f}%"
        net_pnl = r['final'] - r['init']
        row = (f"{flag} {r['name']:<22} {r['n']:>7} {r['win_rate']:>6.1f}% "
               f"{r['ev']:>+6.2f}% {r['half_k']:>7.1f}% "
               f"{r['final']:>11,.1f} {net_pnl:>+11,.1f} {mdd_str:>8}")
        if show_stopcap:
            row += f"  {'是' if r['stopped'] else '否':>4}"
        print(row)
    print(f"  {'─'*W}")
    print(f"  （★ = 勝率 ≥ 損益平衡 {break_even:.1f}%，MDD = {'最大絕對回撤 USDT' if eff_fixed > 0 else '相對峰值百分比'}）\n")

    # ── 隨機抽樣多次試驗 ────────────────────────────────────
    if do_sampling:
        # 只針對全量勝率 ≥ 損益平衡的訊號做抽樣分析
        sample_targets = [r for r in results if r['win_rate'] >= args.min_wr][:args.top]
        if not sample_targets:
            sample_targets = top_n[:5]

        print(f"\n  {'='*60}")
        print(f"  隨機抽樣測試  |  抽樣率 {sample_rate*100:.0f}%  |  試驗 {trials} 次")
        print(f"  模擬每次只下其中 {sample_rate*100:.0f}% 的訊號（隨機選取）")
        print(f"  {'='*60}")

        use_abs_mdd = eff_fixed > 0

        for base in sample_targets:
            sig_name = base['name']
            sig_ser  = signals[sig_name]

            finals, wrs, mdds, stopped_cnt = [], [], [], 0
            for seed in range(trials):
                rng = np.random.default_rng(seed)
                tr = backtest_signal(df, sig_ser, payout, settle_b, max_bet,
                                     eff_fixed, init_cap,
                                     sample_rate=sample_rate, rng=rng,
                                     use_cooldown=use_cooldown,
                                     cd_losses=cd_losses, cd_hours=cd_hours,
                                     cd_skip_hours=cd_skip_hours,
                                     max_concurrent=max_concurrent)
                if tr is None:
                    continue
                finals.append(tr['final'])
                wrs.append(tr['win_rate'])
                mdds.append(tr['mdd_abs'] if use_abs_mdd else tr['mdd%'])
                if tr['stopped']:
                    stopped_cnt += 1

            if not finals:
                continue

            finals = np.array(finals)
            wrs    = np.array(wrs)
            mdds   = np.array(mdds)
            pcts_f = np.percentile(finals, [0, 25, 50, 75, 100])
            pcts_w = np.percentile(wrs,    [0, 25, 50, 75, 100])
            pcts_m = np.percentile(mdds,   [0, 25, 50, 75, 100])

            init_val = base['init']
            print(f"\n  ▶ {sig_name}  （全量勝率 {base['win_rate']:.1f}%，"
                  f"初始 {init_val:.0f} USDT）")
            print(f"  {'':20} {'最差':>10} {'P25':>10} {'中位':>10} "
                  f"{'P75':>10} {'最佳':>10}")
            print(f"  {'─'*70}")

            # 終值
            row_f = f"  {'終值 (USDT)':<20}"
            for v in pcts_f:
                row_f += f" {v:>10,.1f}"
            print(row_f)

            # 淨損益
            row_p = f"  {'淨損益 (USDT)':<20}"
            for v in pcts_f:
                row_p += f" {v - init_val:>+10,.1f}"
            print(row_p)

            # 勝率
            row_w = f"  {'勝率 (%)':<20}"
            for v in pcts_w:
                row_w += f" {v:>9.1f}%"
            print(row_w)

            # MDD
            mdd_label = 'MDD (U)' if use_abs_mdd else 'MDD (%)'
            row_m = f"  {mdd_label:<20}"
            for v in pcts_m:
                if use_abs_mdd:
                    row_m += f" {v:>+10.1f}"
                else:
                    row_m += f" {v:>+9.1f}%"
            print(row_m)

            # 停損率
            stop_rate = stopped_cnt / len(finals) * 100
            profitable = (finals > init_val).sum() / len(finals) * 100
            print(f"  {'盈利機率':<20} {profitable:>9.1f}%  "
                  f"（停損率 {stop_rate:.1f}%，共 {len(finals)} 次試驗）")

        print()

    # ── 繪圖：前 5 個訊號的模擬複利曲線 ────────────────────
    plot_top = [r for r in top_n if r['win_rate'] >= break_even][:5]
    if not plot_top:
        plot_top = top_n[:5]

    if do_sampling:
        # 抽樣模式：畫分位數帶狀圖（針對最佳訊號）
        best = plot_top[0]
        sig_ser = signals[best['name']]
        all_equities = []
        for seed in range(min(trials, 100)):   # 最多 100 條曲線避免太慢
            rng = np.random.default_rng(seed)
            tr = backtest_signal(df, sig_ser, payout, settle_b, max_bet,
                                 eff_fixed, init_cap,
                                 sample_rate=sample_rate, rng=rng,
                                 use_cooldown=use_cooldown,
                                 cd_losses=cd_losses, cd_hours=cd_hours,
                                 cd_skip_hours=cd_skip_hours,
                                 max_concurrent=max_concurrent)
            if tr is not None:
                all_equities.append(tr['equity'].values)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f'{sym} 事件合約 – 隨機抽樣 {sample_rate*100:.0f}%  ({trials} 次試驗)',
                     fontsize=13)

        # 左：全量勝率橫條圖
        names_plot = [r['name'] for r in top_n]
        wrs_plot   = [r['win_rate'] for r in top_n]
        colors = ['#2ecc71' if w >= break_even else '#e74c3c' for w in wrs_plot]
        ax1.barh(names_plot[::-1], wrs_plot[::-1], color=colors[::-1])
        ax1.axvline(break_even, color='black', linestyle='--', linewidth=1,
                    label=f'損益平衡 {break_even:.1f}%')
        ax1.set_xlabel('勝率 %（全量）')
        ax1.set_title(f'訊號勝率 Top {args.top}')
        ax1.legend(fontsize=8)

        # 右：最佳訊號抽樣曲線分位帶
        if all_equities:
            max_len = max(len(e) for e in all_equities)
            mat = np.full((len(all_equities), max_len), np.nan)
            for i, eq in enumerate(all_equities):
                mat[i, :len(eq)] = eq

            p5  = np.nanpercentile(mat, 5,  axis=0)
            p25 = np.nanpercentile(mat, 25, axis=0)
            p50 = np.nanpercentile(mat, 50, axis=0)
            p75 = np.nanpercentile(mat, 75, axis=0)
            p95 = np.nanpercentile(mat, 95, axis=0)
            xs  = np.arange(max_len)

            ax2.fill_between(xs, p5,  p95, alpha=0.15, color='#3498db', label='P5–P95')
            ax2.fill_between(xs, p25, p75, alpha=0.30, color='#3498db', label='P25–P75')
            ax2.plot(xs, p50, color='#2980b9', linewidth=1.5, label='中位數')
            init_val = best['init']
            ax2.axhline(init_val, color='gray', linestyle=':', linewidth=0.8,
                        label=f'初始 {init_val:.0f} U')
            ax2.set_xlabel('交易筆數（抽樣後）')
            ax2.set_ylabel('資金 (USDT)')
            ax2.set_title(f"{best['name']}  抽樣 {sample_rate*100:.0f}%  資金分佈")
            ax2.legend(fontsize=8)
            if not eff_fixed:
                ax2.set_yscale('log')

        out = f'event_contract_{sym.lower()}_sample{int(sample_rate*100)}.png'
    else:
        fig = plt.figure(figsize=(14, 7))
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

        # 左：勝率 top 20 橫條圖
        ax1 = fig.add_subplot(gs[0])
        names = [r['name'] for r in top_n]
        wrs   = [r['win_rate'] for r in top_n]
        colors = ['#2ecc71' if w >= break_even else '#e74c3c' for w in wrs]
        ax1.barh(names[::-1], wrs[::-1], color=colors[::-1])
        ax1.axvline(break_even, color='black', linestyle='--', linewidth=1,
                    label=f'損益平衡 {break_even:.1f}%')
        ax1.set_xlabel('勝率 %')
        ax1.set_title(f'{sym} 事件合約 – 訊號勝率 Top {args.top}')
        ax1.legend(fontsize=8)

        # 右：前幾個有效訊號的複利曲線
        ax2 = fig.add_subplot(gs[1])
        for r in plot_top:
            ax2.plot(r['equity'].values,
                     label=f"{r['name']} ({r['win_rate']:.1f}%)")
        ax2.axhline(INITIAL_CAP, color='gray', linestyle=':', linewidth=0.8)
        ax2.set_xlabel('交易筆數')
        ax2.set_ylabel('資金 (USDT)')
        ax2.set_title(f'{sym} 半 Kelly 模擬複利（起始 {INITIAL_CAP} USDT）')
        ax2.legend(fontsize=7)
        ax2.set_yscale('log')

        out = f'event_contract_{sym.lower()}.png'

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  圖表已儲存: {out}")


if __name__ == '__main__':
    main()
