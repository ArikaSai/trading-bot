"""
event_signal_bot.py
═══════════════════
幣安事件合約「報明牌 + 模擬下單」Bot

訊號邏輯：hlpct12_10
  收盤價在最近 36 根 5m K 棒高低區間的底部 10%  → 做多（賭漲）
  收盤價在最近 36 根 5m K 棒高低區間的頂部 10%  → 做空（賭跌）

市場狀態分類：
  死水盤（ADX < 15）→ 跳過，不下注
  震盪盤（15 ≤ ADX < 25）→ tier=ranging
  趨勢盤（ADX ≥ 25）→ 依訊號方向 tier=trend_aligned / trend_counter

下注邏輯：階梯式固定注（Staged Betting）
  每 250U 升一個檔位，每檔增加 5U
  bet = max(5, floor(capital/250) × 5)
  ETH 上限 125U，BTC 上限 250U

執行方式：
    python event_signal_bot.py          # 持續運行
    python event_signal_bot.py --once   # 跑一個週期（測試用）
    python event_signal_bot.py --reset  # 重置模擬本金為 INITIAL_CAPITAL
"""

import argparse
import json
import time
import traceback
from datetime import datetime, timedelta, timezone

TZ_CST = timezone(timedelta(hours=8))   # UTC+8（台灣）
from pathlib import Path

import ccxt
import requests

# ══════════════════════════════════════════════════════════════
#  設定（只需修改這裡）
# ══════════════════════════════════════════════════════════════
DISCORD_WEBHOOK  = "https://discord.com/api/webhooks/1494868584835452929/MgdUkdX2nh4HE9_-mHtC_DrrRdEO7BlPb8wbYO6ydX7WwF3XUd_7j7UTGiZFxY1vaewe"   # ← 填入新頻道 webhook

INITIAL_CAPITAL  = 100.0   # 初始模擬本金（USDT）
MIN_BET          = 5.0     # 最小下單量（低於此值停止）
PAYOUT_10        = 0.80    # 10 分鐘合約賠率
PAYOUT_30        = 0.85    # 30 分鐘合約賠率
SETTLE_10_MIN    = 10
SETTLE_30_MIN    = 30

# ── 市場分類閾值 ───────────────────────────────────────────────
ADX_TREND_THRESH = 25      # ADX ≥ 25 → 趨勢盤
ADX_DEAD_THRESH  = 15      # ADX < 15 → 死水盤，跳過

# ── 趨勢方向 EMA ─────────────────────────────────────────────
EMA_FAST = 20   # EMA20 > EMA60 → 上升趨勢
EMA_SLOW = 60

# ── 階梯式固定注 ──────────────────────────────────────────────
STEP_UNIT = 250.0   # 每 250U 升一個下注檔位
BET_UNIT  = 5.0     # 每檔增加 5U

TIER_ZH = {
    'trend_aligned': '趨勢順勢',
    'trend_counter': '趨勢逆向',
    'ranging':       '震盪',
    'dead':          '死水',
}

SYMBOLS          = ['ETH/USDT', 'BTC/USDT']
TIMEFRAME        = '5m'
LOOKBACK_N       = 36      # 36×5m = 180min = 3小時（與回測 12×15m 等長）
THRESHOLD_LO     = 0.10
THRESHOLD_HI     = 0.90
FETCH_LIMIT      = 100     # ATR_14 + MA60 + 1 = 75，取 100 保留餘裕
ATR_RATIO_MAX    = 1.20    # 高波動過濾閾值

MAX_BET_CAP      = {'ETH/USDT': 125, 'BTC/USDT': 250}
STATE_FILE       = Path(__file__).parent / 'event_bot_state.json'


# ══════════════════════════════════════════════════════════════
#  狀態管理
# ══════════════════════════════════════════════════════════════
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return _default_state()


def _default_state() -> dict:
    return {
        'capital':         INITIAL_CAPITAL,
        'pending':         [],
        'total_w':         0,
        'total_l':         0,
        'history':         [],
        'bankrupt_warned': False,
    }


def save_state(state: dict):
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def reset_state():
    s = _default_state()
    save_state(s)
    print(f"[重置] 模擬本金已恢復為 {INITIAL_CAPITAL} USDT，歷史記錄清除。")
    return s


# ══════════════════════════════════════════════════════════════
#  Discord
# ══════════════════════════════════════════════════════════════
def send_discord(msg: str):
    if not DISCORD_WEBHOOK or DISCORD_WEBHOOK == "":
        print(f"[Discord 未設定]\n{msg}\n")
        return
    try:
        resp = requests.post(DISCORD_WEBHOOK,
                             json={"content": msg}, timeout=10)
        if resp.status_code not in (200, 204):
            print(f"[Discord 回應異常] {resp.status_code}")
    except Exception as e:
        print(f"[Discord 發送失敗] {e}")


# ══════════════════════════════════════════════════════════════
#  交易所
# ══════════════════════════════════════════════════════════════
def get_exchange():
    return ccxt.binance({'enableRateLimit': True})


def fetch_ohlcv(exchange, symbol: str) -> list:
    bars = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=FETCH_LIMIT + 1)
    return bars[:-1]


def fetch_price(exchange, symbol: str) -> float:
    return float(exchange.fetch_ticker(symbol)['last'])


# ══════════════════════════════════════════════════════════════
#  技術指標計算
# ══════════════════════════════════════════════════════════════
def calc_hlpct(bars: list) -> float | None:
    if len(bars) < LOOKBACK_N:
        return None
    window   = bars[-LOOKBACK_N:]
    rng_high = max(b[2] for b in window)
    rng_low  = min(b[3] for b in window)
    close    = bars[-1][4]
    if rng_high == rng_low:
        return 0.5
    return (close - rng_low) / (rng_high - rng_low)


ATR_N  = 14
ATR_MA = 60

def calc_atr_ratio(bars: list) -> float | None:
    need = ATR_N + ATR_MA + 1
    if len(bars) < need:
        return None
    trs = [max(bars[i][2] - bars[i][3],
               abs(bars[i][2] - bars[i-1][4]),
               abs(bars[i][3] - bars[i-1][4]))
           for i in range(1, len(bars))]
    atrs = [sum(trs[i - ATR_N + 1: i + 1]) / ATR_N
            for i in range(ATR_N - 1, len(trs))]
    if len(atrs) < ATR_MA:
        return None
    atr_ma = sum(atrs[-ATR_MA:]) / ATR_MA
    if atr_ma == 0:
        return None
    return atrs[-1] / atr_ma


RSI_N = 14

def calc_rsi(bars: list) -> float | None:
    if len(bars) < RSI_N + 1:
        return None
    closes = [b[4] for b in bars[-(RSI_N + 1):]]
    diffs  = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [d if d > 0 else 0.0 for d in diffs]
    losses = [-d if d < 0 else 0.0 for d in diffs]
    avg_g  = sum(gains) / RSI_N
    avg_l  = sum(losses) / RSI_N
    if avg_l == 0:
        return 100.0
    return 100 - 100 / (1 + avg_g / avg_l)


def calc_macd_hist(bars: list) -> float | None:
    if len(bars) < 35:
        return None
    closes = [b[4] for b in bars]

    def ema(data, n):
        k = 2 / (n + 1)
        e = data[0]
        for v in data[1:]:
            e = v * k + e * (1 - k)
        return e

    ema12 = ema(closes[-26:], 12)
    ema26 = ema(closes,       26)
    return ema12 - ema26


VOL_Z_N = 60

def calc_vol_z(bars: list) -> float | None:
    if len(bars) < VOL_Z_N:
        return None
    vols   = [b[5] for b in bars[-VOL_Z_N:]]
    mean_v = sum(vols) / VOL_Z_N
    std_v  = (sum((v - mean_v) ** 2 for v in vols) / VOL_Z_N) ** 0.5
    if std_v == 0:
        return None
    return (vols[-1] - mean_v) / std_v


BB_N = 20

def calc_bb_pct(bars: list) -> float | None:
    if len(bars) < BB_N:
        return None
    closes = [b[4] for b in bars[-BB_N:]]
    mean_c = sum(closes) / BB_N
    std_c  = (sum((c - mean_c) ** 2 for c in closes) / BB_N) ** 0.5
    if std_c == 0:
        return None
    bb_lower = mean_c - 2 * std_c
    bb_upper = mean_c + 2 * std_c
    return (closes[-1] - bb_lower) / (bb_upper - bb_lower)


ADX_N = 14

def calc_adx(bars: list) -> float | None:
    need = ADX_N * 2 + 2
    if len(bars) < need:
        return None
    trs, pdms, ndms = [], [], []
    for i in range(1, len(bars)):
        h, l, pc = bars[i][2], bars[i][3], bars[i-1][4]
        ph, pl   = bars[i-1][2], bars[i-1][3]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        up, dn = h - ph, pl - l
        pdms.append(up  if up  > dn and up  > 0 else 0.0)
        ndms.append(dn  if dn  > up and dn  > 0 else 0.0)

    def wilder(data, n):
        s = sum(data[:n])
        smoothed = [s]
        for v in data[n:]:
            s = s - s / n + v
            smoothed.append(s)
        return smoothed

    s_tr  = wilder(trs,  ADX_N)
    s_pdm = wilder(pdms, ADX_N)
    s_ndm = wilder(ndms, ADX_N)

    dxs = []
    for tr, pdm, ndm in zip(s_tr, s_pdm, s_ndm):
        if tr == 0:
            continue
        pdi = 100 * pdm / tr
        ndi = 100 * ndm / tr
        denom = pdi + ndi
        dxs.append(100 * abs(pdi - ndi) / denom if denom else 0.0)

    if len(dxs) < ADX_N:
        return None
    adx = sum(dxs[:ADX_N]) / ADX_N
    for dx in dxs[ADX_N:]:
        adx = (adx * (ADX_N - 1) + dx) / ADX_N
    return adx


def calc_ema(closes: list, span: int) -> float:
    k = 2 / (span + 1)
    e = closes[0]
    for v in closes[1:]:
        e = v * k + e * (1 - k)
    return e


# ══════════════════════════════════════════════════════════════
#  市場狀態與 Tier 分類
# ══════════════════════════════════════════════════════════════
def classify_market(adx: float | None) -> str:
    if adx is not None and adx < ADX_DEAD_THRESH:
        return 'dead'
    if adx is not None and adx >= ADX_TREND_THRESH:
        return 'trending'
    return 'ranging'


def calc_trend_dir(bars: list) -> str | None:
    closes = [b[4] for b in bars]
    if len(closes) < EMA_SLOW:
        return None
    ema20 = calc_ema(closes[-EMA_FAST * 3:], EMA_FAST)
    ema60 = calc_ema(closes[-EMA_SLOW * 2:], EMA_SLOW)
    return 'up' if ema20 > ema60 else 'down'


def classify_tier(market_state: str, direction: str, trend_dir: str | None) -> str:
    if market_state == 'dead':
        return 'dead'
    if market_state == 'ranging':
        return 'ranging'
    # trending
    if trend_dir is None:
        return 'trend_counter'
    is_long = direction == 'LONG'
    aligned = (is_long and trend_dir == 'up') or (not is_long and trend_dir == 'down')
    return 'trend_aligned' if aligned else 'trend_counter'


# ══════════════════════════════════════════════════════════════
#  訊號品質（顯示用）
# ══════════════════════════════════════════════════════════════
def signal_quality(direction: str, rsi: float | None,
                   macd_hist: float | None,
                   adx: float | None = None,
                   vol_z: float | None = None,
                   bb_pct: float | None = None) -> str:
    if rsi is None:
        return ''
    is_long  = direction == 'LONG'
    rsi_30ok = rsi < 30 if is_long else rsi > 70
    rsi_35ok = rsi < 35 if is_long else rsi > 65
    macd_ok  = (macd_hist is not None and
                ((is_long and macd_hist < 0) or (not is_long and macd_hist > 0)))
    adx_ok   = adx is not None and adx >= 25
    adx_top  = adx is not None and 25 <= adx < 35
    volz_ok  = vol_z is not None and vol_z < 1
    bb_ok    = (bb_pct is not None and
                ((is_long and bb_pct < 0.05) or (not is_long and bb_pct > 0.95)))

    if rsi_30ok and macd_ok and adx_top and volz_ok and bb_ok:
        return '💎 頂級（RSI30+MACD+ADX25~35+量縮+BB觸碰）'
    if rsi_30ok and macd_ok and adx_ok and volz_ok:
        return '🔥 高品質（RSI30+MACD+ADX≥25+量縮）'
    if rsi_30ok and macd_ok:
        return '⭐ 優質（RSI30+MACD）'
    if rsi_35ok and macd_ok:
        return '✅ 良好（RSI35+MACD）'
    if rsi_30ok:
        return '✅ 良好（RSI30）'
    return '⭕ 一般（僅 F1 極端位置）'


# ══════════════════════════════════════════════════════════════
#  階梯式固定下注
# ══════════════════════════════════════════════════════════════
def calc_staged_bet(capital: float, symbol: str) -> float:
    step = int(capital // STEP_UNIT)
    bet  = max(MIN_BET, step * BET_UNIT)
    return min(bet, MAX_BET_CAP.get(symbol, 125))


def check_signals(exchange) -> list[dict]:
    signals = []
    for sym in SYMBOLS:
        try:
            bars  = fetch_ohlcv(exchange, sym)
            if not bars:
                continue
            hlpct = calc_hlpct(bars)
            if hlpct is None:
                continue

            atr_ratio = calc_atr_ratio(bars)
            if atr_ratio is None or atr_ratio >= ATR_RATIO_MAX:
                if atr_ratio is not None:
                    print(f"[{now_str()}] {sym} 跳過（ATR_ratio={atr_ratio:.2f} ≥ {ATR_RATIO_MAX}）")
                continue

            close  = bars[-1][4]
            bar_ts = datetime.fromtimestamp(
                bars[-1][0] / 1000, tz=TZ_CST).isoformat()

            if hlpct <= THRESHOLD_LO:
                direction = 'LONG'
            elif hlpct >= THRESHOLD_HI:
                direction = 'SHORT'
            else:
                continue

            rsi       = calc_rsi(bars)
            macd_hist = calc_macd_hist(bars)
            adx       = calc_adx(bars)
            vol_z     = calc_vol_z(bars)
            bb_pct    = calc_bb_pct(bars)
            trend_dir = calc_trend_dir(bars)
            quality   = signal_quality(direction, rsi, macd_hist, adx, vol_z, bb_pct)

            market_state = classify_market(adx)
            tier         = classify_tier(market_state, direction, trend_dir)

            # 所有 tier 均使用 30min 合約（回測最佳化結果）
            payout    = PAYOUT_30
            settle_min = SETTLE_30_MIN

            signals.append({
                'symbol':       sym,
                'direction':    direction,
                'close':        close,
                'hlpct':        hlpct,
                'atr_ratio':    atr_ratio,
                'bar_ts':       bar_ts,
                'rsi':          rsi,
                'adx':          adx,
                'vol_z':        vol_z,
                'bb_pct':       bb_pct,
                'quality':      quality,
                'market_state': market_state,
                'trend_dir':    trend_dir,
                'tier':         tier,
                'payout':       payout,
                'settle_min':   settle_min,
            })
        except Exception as e:
            print(f"[{sym}] 抓取失敗: {e}")
    return signals


# ══════════════════════════════════════════════════════════════
#  結算
# ══════════════════════════════════════════════════════════════
def settle_pending(exchange, state: dict) -> bool:
    now       = datetime.now(TZ_CST)
    remaining = []
    settled   = False

    for p in state['pending']:
        settle_at = datetime.fromisoformat(p['settle_at'])
        if now < settle_at:
            remaining.append(p)
            continue

        try:
            settle_price = fetch_price(exchange, p['symbol'])
        except Exception as e:
            print(f"[結算] 抓價失敗 {p['symbol']}: {e}，延後處理")
            remaining.append(p)
            continue

        entry   = p['entry_price']
        bet     = p['bet']
        payout  = p.get('payout', PAYOUT_30)
        tier    = p.get('tier', 'ranging')
        win     = ((p['direction'] == 'LONG'  and settle_price > entry) or
                   (p['direction'] == 'SHORT' and settle_price < entry))

        pnl = bet * payout if win else -bet
        if win:
            state['capital'] += bet * (1 + payout)
            state['total_w'] += 1
        else:
            state['total_l'] += 1

        state['history'].append({
            'time':          now.strftime('%Y-%m-%d %H:%M'),
            'symbol':        p['symbol'],
            'direction':     p['direction'],
            'entry_price':   entry,
            'settle_price':  settle_price,
            'bet':           bet,
            'pnl':           round(pnl, 4),
            'capital_after': round(state['capital'], 4),
            'tier':          tier,
        })

        total  = state['total_w'] + state['total_l']
        wr_str = f"{state['total_w'] / total * 100:.1f}%" if total > 0 else "—"
        emoji  = '✅' if win else '❌'
        zh_dir = '做多' if p['direction'] == 'LONG' else '做空'
        tier_zh = TIER_ZH.get(tier, tier)

        msg = (
            f"{emoji}  **事件合約結算**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"幣種：{p['symbol']}　方向：{zh_dir}　盤勢：{tier_zh}\n"
            f"進場：${entry:,.4f}  →  結算：${settle_price:,.4f}\n"
            f"損益：**{pnl:+.2f} USDT**\n"
            f"模擬本金：**{state['capital']:.2f} USDT**\n"
            f"累計：{state['total_w']}W / {state['total_l']}L　勝率 {wr_str}"
        )
        send_discord(msg)
        print(f"[{now_str()}] {emoji} 結算 {p['symbol']} {p['direction']}"
              f"  進={entry:.4f} 出={settle_price:.4f}"
              f"  {pnl:+.2f}U  本金={state['capital']:.2f}  tier={tier}")
        settled = True

    state['pending'] = remaining
    return settled


# ══════════════════════════════════════════════════════════════
#  訊號通知
# ══════════════════════════════════════════════════════════════
def format_signal_msg(sig: dict, capital: float, bet: float,
                      total_w: int, total_l: int) -> str:
    sym        = sig['symbol']
    direction  = sig['direction']
    close      = sig['close']
    hlpct      = sig['hlpct']
    settle_min = sig['settle_min']
    payout     = sig['payout']
    tier       = sig['tier']
    market_state = sig['market_state']
    trend_dir  = sig.get('trend_dir')

    settle_dt  = datetime.now(TZ_CST) + timedelta(minutes=settle_min)
    emoji      = '📈' if direction == 'LONG' else '📉'
    zh_dir     = '做多（賭漲）' if direction == 'LONG' else '做空（賭跌）'
    pos_str    = '底部' if direction == 'LONG' else '頂部'
    max_b      = MAX_BET_CAP.get(sym, 125)
    total      = total_w + total_l
    wr_str     = f"{total_w / total * 100:.1f}%" if total > 0 else "—"
    tier_zh    = TIER_ZH.get(tier, tier)
    ms_zh      = {'dead': '死水盤', 'ranging': '震盪盤', 'trending': '趨勢盤'}.get(market_state, market_state)
    td_zh      = {'up': '上升', 'down': '下降'}.get(trend_dir, '不明') if trend_dir else '不明'

    atr_ratio  = sig.get('atr_ratio', 0)
    rsi        = sig.get('rsi')
    adx        = sig.get('adx')
    vol_z      = sig.get('vol_z')
    bb_pct     = sig.get('bb_pct')
    quality    = sig.get('quality', '')
    rsi_str    = f"{rsi:.1f}"    if rsi    is not None else "—"
    adx_str    = f"{adx:.1f}"    if adx    is not None else "—"
    volz_str   = f"{vol_z:.2f}"  if vol_z  is not None else "—"
    bb_str     = f"{bb_pct:.2f}" if bb_pct is not None else "—"

    return (
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{emoji}  **事件合約訊號**\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"幣種：{sym}\n"
        f"方向：{zh_dir}\n"
        f"當前收盤：${close:,.4f}\n"
        f"區間位置：{pos_str} {hlpct*100:.1f}%（近 {LOOKBACK_N} 棒）\n"
        f"RSI14：{rsi_str}　ADX：{adx_str}　vol_z：{volz_str}　BB_pct：{bb_str}　ATR_ratio：{atr_ratio:.2f}\n"
        f"信號品質：**{quality}**\n"
        f"盤勢：**{ms_zh}**（趨勢：{td_zh}）　Tier：**{tier_zh}**\n"
        f"合約類型：**{settle_min} 分鐘**（賠率 {int(payout*100)}%）\n"
        f"結算時間：{settle_dt.strftime('%H:%M:%S')}\n"
        f"模擬下注：**{bet:.1f} USDT**（檔位 {int(capital // STEP_UNIT)}，上限 {max_b} USDT）\n"
        f"模擬本金：{capital:.2f} USDT\n"
        f"累計勝率：{wr_str}（{total_w}W / {total_l}L）"
    )


# ══════════════════════════════════════════════════════════════
#  主迴圈
# ══════════════════════════════════════════════════════════════
BAR_SEC    = TIMEFRAME_SEC = 5 * 60
SIG_DELAY  = 5
SIG_WINDOW = 60
POLL_SEC       = 30
HEARTBEAT_MIN  = 60


def run_loop(exchange, once: bool):
    state = load_state()
    print(f"[{now_str()}] 事件合約 Signal Bot 啟動（模擬模式）")
    print(f"  本金：{state['capital']:.2f} USDT  "
          f"勝：{state['total_w']}  敗：{state['total_l']}")
    print(f"  監控：{', '.join(SYMBOLS)}  "
          f"訊號：hlpct12_10_F1  下注：階梯式固定注")
    step = int(state['capital'] // STEP_UNIT)
    cur_bet_eth = min(max(MIN_BET, step * BET_UNIT), MAX_BET_CAP['ETH/USDT'])
    cur_bet_btc = min(max(MIN_BET, step * BET_UNIT), MAX_BET_CAP['BTC/USDT'])
    print(f"  結算輪詢：每 {POLL_SEC}s  訊號偵測：收盤後 {SIG_DELAY}s")
    print(f"  當前下注：ETH {cur_bet_eth:.0f}U / BTC {cur_bet_btc:.0f}U（檔位 {step}）")
    print()

    sent_this_bar:  dict  = {}
    last_heartbeat: float = 0.0

    while True:
        try:
            now_ts  = time.time()
            bar_age = now_ts % BAR_SEC

            # 0. 心跳通知
            if HEARTBEAT_MIN > 0:
                now_ts_hb = time.time()
                if now_ts_hb - last_heartbeat >= HEARTBEAT_MIN * 60:
                    total  = state['total_w'] + state['total_l']
                    wr_str = f"{state['total_w']/total*100:.1f}%" if total else "—"
                    send_discord(
                        f"💓  **Bot 心跳**　{datetime.now(TZ_CST).strftime('%m/%d %H:%M')}\n"
                        f"本金：{state['capital']:.2f} USDT　"
                        f"待結算：{len(state['pending'])} 筆　"
                        f"累計勝率：{wr_str}（{state['total_w']}W/{state['total_l']}L）"
                    )
                    last_heartbeat = now_ts_hb

            # 1. 結算到期訊號
            if settle_pending(exchange, state):
                save_state(state)

            # 2. 本金不足，停止
            if state['capital'] < MIN_BET:
                if not state.get('bankrupt_warned'):
                    warn = (
                        f"🚨  **模擬本金不足，停止發訊**\n"
                        f"目前本金：{state['capital']:.2f} USDT\n"
                        f"最小下單：{MIN_BET} USDT\n"
                        f"如需繼續，請執行 `python event_signal_bot.py --reset`"
                    )
                    send_discord(warn)
                    state['bankrupt_warned'] = True
                    save_state(state)
                print(f"[{now_str()}] 🚨 本金 {state['capital']:.2f} U "
                      f"< {MIN_BET} U，停止。")
                break

            # 3. 訊號偵測
            in_sig_window = SIG_DELAY <= bar_age < SIG_DELAY + SIG_WINDOW
            if in_sig_window:
                new_sigs = check_signals(exchange)
                any_sent = False
                for sig in new_sigs:
                    key = (sig['symbol'], sig['bar_ts'])
                    if key in sent_this_bar:
                        continue

                    tier      = sig['tier']
                    payout    = sig['payout']
                    settle_min = sig['settle_min']

                    if tier == 'dead':
                        print(f"[{now_str()}] ⏭ 跳過 {sig['symbol']} {sig['direction']} "
                              f"（死水盤 ADX={sig['adx']:.1f}）")
                        sent_this_bar[key] = True
                        continue

                    bet = calc_staged_bet(state['capital'], sig['symbol'])
                    if bet < MIN_BET:
                        continue

                    settle_at = (datetime.now(TZ_CST)
                                 + timedelta(minutes=settle_min)).isoformat()

                    state['capital'] -= bet
                    state['pending'].append({
                        'symbol':      sig['symbol'],
                        'direction':   sig['direction'],
                        'entry_price': sig['close'],
                        'bet':         bet,
                        'settle_at':   settle_at,
                        'payout':      payout,
                        'tier':        tier,
                    })
                    save_state(state)
                    sent_this_bar[key] = True
                    any_sent = True

                    msg = format_signal_msg(
                        sig, state['capital'] + bet, bet,
                        state['total_w'], state['total_l'])
                    send_discord(msg)
                    print(f"[{now_str()}] 📢 {sig['symbol']} {sig['direction']}"
                          f"  close={sig['close']:,.4f}"
                          f"  tier={TIER_ZH[tier]}"
                          f"  bet={bet:.1f}U  本金={state['capital']:.2f}")

                if not any_sent:
                    print(f"[{now_str()}] 無訊號  "
                          f"本金={state['capital']:.2f}  "
                          f"待結算={len(state['pending'])} 筆")

        except Exception:
            traceback.print_exc()

        if once:
            break

        now_ts  = time.time()
        bar_age = now_ts % BAR_SEC

        if bar_age < SIG_DELAY:
            sleep_sec = SIG_DELAY - bar_age + 0.5
        elif state['pending']:
            time_to_next = BAR_SEC - bar_age + SIG_DELAY
            sleep_sec = min(POLL_SEC, time_to_next)
        else:
            sleep_sec = BAR_SEC - bar_age + SIG_DELAY

        time.sleep(max(sleep_sec, 1))


def now_str():
    return datetime.now(TZ_CST).strftime('%H:%M:%S')


# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once',  action='store_true',
                        help='只跑一個週期（測試用）')
    parser.add_argument('--reset', action='store_true',
                        help=f'重置模擬本金為 {INITIAL_CAPITAL} USDT')
    args = parser.parse_args()

    if args.reset:
        reset_state()
        return

    exchange = get_exchange()
    run_loop(exchange, once=args.once)


if __name__ == '__main__':
    main()
