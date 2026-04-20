"""
event_signal_bot.py
═══════════════════
幣安事件合約「報明牌 + 模擬下單」Bot

訊號邏輯：hlpct12_10
  收盤價在最近 12 根 15m K 棒高低區間的底部 10%  → 做多（賭漲）
  收盤價在最近 12 根 15m K 棒高低區間的頂部 10%  → 做空（賭跌）

資金模擬：
  - 半 Kelly 動態下注（前 100 筆用固定勝率 0.55，之後滾動 100 筆實際勝率）
  - 訊號發出後 30 分鐘自動結算，更新模擬本金
  - 本金 < MIN_BET 停止發訊並發 Discord 警告
  - 狀態持久化到 event_bot_state.json

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
PAYOUT           = 0.85    # 30 分鐘合約賠率

# ── 半 Kelly 下注參數 ──────────────────────────────────────────
KELLY_MIN_TRADES = 100     # 累積幾筆後才切換到滾動勝率
KELLY_FIXED_WR   = 0.55    # 初期保守固定勝率（略高於損益平衡 54.1%）
KELLY_ROLL_N     = 100     # 滾動視窗筆數

SYMBOLS          = ['ETH/USDT', 'BTC/USDT']
TIMEFRAME        = '5m'
LOOKBACK_N       = 36      # 36×5m = 180min = 3小時（與回測 12×15m 等長）
THRESHOLD_LO     = 0.10
THRESHOLD_HI     = 0.90
FETCH_LIMIT      = 100     # ATR_14 + MA60 + 1 = 75，取 100 保留餘裕
SETTLE_MIN       = 30      # 結算等待分鐘數（不變）

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
        'capital':       INITIAL_CAPITAL,
        'pending':       [],
        'total_w':       0,
        'total_l':       0,
        'history':       [],
        'bankrupt_warned': False, # 本金不足警告已送出，避免重複
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
    return ccxt.binance({'options': {'defaultType': 'future'}})


def fetch_ohlcv(exchange, symbol: str) -> list:
    bars = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=FETCH_LIMIT + 1)
    return bars[:-1]


def fetch_price(exchange, symbol: str) -> float:
    return float(exchange.fetch_ticker(symbol)['last'])


# ══════════════════════════════════════════════════════════════
#  訊號偵測
# ══════════════════════════════════════════════════════════════
def calc_hlpct(bars: list) -> float | None:
    if len(bars) < LOOKBACK_N:
        return None
    window   = bars[-LOOKBACK_N:]   # 含當前已收盤 K 棒，與回測 rolling(12) 一致
    rng_high = max(b[2] for b in window)
    rng_low  = min(b[3] for b in window)
    close    = bars[-1][4]
    if rng_high == rng_low:
        return 0.5
    return (close - rng_low) / (rng_high - rng_low)


ATR_N  = 14   # ATR 週期
ATR_MA = 60   # ATR 均值窗口：60×5m = 300min = 20×15m（與回測一致）

def calc_atr_ratio(bars: list) -> float | None:
    """ATR_ratio = 當前 ATR_14 / 近 20 棒 ATR_14 均值；>= 1.20 表示高波動"""
    need = ATR_N + ATR_MA + 1   # +1 是計算第一個 TR 需要前一根
    if len(bars) < need:
        return None

    # True Range
    trs = [max(bars[i][2] - bars[i][3],
               abs(bars[i][2] - bars[i-1][4]),
               abs(bars[i][3] - bars[i-1][4]))
           for i in range(1, len(bars))]

    # ATR_14（SMA of TR，與回測用法一致）
    atrs = [sum(trs[i - ATR_N + 1: i + 1]) / ATR_N
            for i in range(ATR_N - 1, len(trs))]

    if len(atrs) < ATR_MA:
        return None

    atr_ma = sum(atrs[-ATR_MA:]) / ATR_MA
    if atr_ma == 0:
        return None
    return atrs[-1] / atr_ma


ATR_RATIO_MAX = 1.20   # F1 過濾：高波動環境不進場

RSI_N = 14   # RSI 週期（與回測一致）

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
    """MACD(12,26,9) hist；需要至少 35 根 bar"""
    if len(bars) < 35:
        return None
    closes = [b[4] for b in bars]

    def ema(data, n):
        k = 2 / (n + 1)
        e = data[0]
        for v in data[1:]:
            e = v * k + e * (1 - k)
        return e

    ema12 = ema(closes[-26:], 12)   # 用最後 26 根算 EMA12（近似）
    ema26 = ema(closes,       26)
    macd  = ema12 - ema26
    # signal line：需要前 9 個 MACD 值，此處用單點近似，差異可接受
    return macd   # hist ≈ macd（signal 收斂後接近 0，此處取 macd 作方向判斷）


VOL_Z_N = 60   # 與回測 ATR_MA 一致（60×5m = 5 小時）

def calc_vol_z(bars: list) -> float | None:
    if len(bars) < VOL_Z_N:
        return None
    vols   = [b[5] for b in bars[-VOL_Z_N:]]
    mean_v = sum(vols) / VOL_Z_N
    std_v  = (sum((v - mean_v) ** 2 for v in vols) / VOL_Z_N) ** 0.5
    if std_v == 0:
        return None
    return (vols[-1] - mean_v) / std_v


ADX_N = 14   # ADX 週期（與回測一致）

def calc_adx(bars: list) -> float | None:
    need = ADX_N * 2 + 2
    if len(bars) < need:
        return None
    # True Range 與方向動量
    trs, pdms, ndms = [], [], []
    for i in range(1, len(bars)):
        h, l, pc = bars[i][2], bars[i][3], bars[i-1][4]
        ph, pl   = bars[i-1][2], bars[i-1][3]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        up, dn = h - ph, pl - l
        pdms.append(up  if up  > dn and up  > 0 else 0.0)
        ndms.append(dn  if dn  > up and dn  > 0 else 0.0)

    # Wilder 平滑（初始值用前 ADX_N 筆的總和）
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
    # ADX = Wilder 平滑的 DX
    adx_s = wilder(dxs, ADX_N)
    return adx_s[-1]


def signal_quality(direction: str, rsi: float | None,
                   macd_hist: float | None,
                   adx: float | None = None,
                   vol_z: float | None = None) -> str:
    """回傳信號品質標籤，供 Discord 訊息使用。"""
    if rsi is None:
        return ''
    is_long  = direction == 'LONG'
    rsi_30ok = rsi < 30 if is_long else rsi > 70
    rsi_35ok = rsi < 35 if is_long else rsi > 65
    macd_ok  = (macd_hist is not None and
                ((is_long and macd_hist < 0) or (not is_long and macd_hist > 0)))
    adx_ok   = adx  is not None and adx  >= 25
    volz_ok  = vol_z is not None and vol_z < 1

    if rsi_30ok and macd_ok and adx_ok and volz_ok:
        return '💎 頂級（RSI30 + MACD + ADX≥25 + 量縮）'
    if rsi_30ok and macd_ok:
        return '🔥 高品質（RSI30 + MACD 確認）'
    if rsi_35ok and macd_ok:
        return '⭐ 優質（RSI35 + MACD 確認）'
    if rsi_30ok:
        return '⭐ 優質（RSI30 確認）'
    if rsi_35ok:
        return '✅ 良好（RSI35 確認）'
    return ''


# ══════════════════════════════════════════════════════════════
#  Half-Kelly 下注計算
# ══════════════════════════════════════════════════════════════
def calc_kelly_bet(state: dict, symbol: str) -> tuple:
    """
    計算半 Kelly 下注額。
    - 前 KELLY_MIN_TRADES 筆：用固定保守勝率 KELLY_FIXED_WR
    - 之後：用最近 KELLY_ROLL_N 筆 live 交易的實際勝率
    回傳 (bet, half_kelly_pct, wr_used, wr_source)
    """
    history = state.get('history', [])
    total   = len(history)

    if total < KELLY_MIN_TRADES:
        wr         = KELLY_FIXED_WR
        wr_source  = f"固定 {wr:.0%}（尚累積 {total}/{KELLY_MIN_TRADES} 筆）"
    else:
        recent = history[-KELLY_ROLL_N:]
        wins   = sum(1 for h in recent if h['pnl'] > 0)
        wr     = wins / len(recent)
        wr_source = f"滾動 {KELLY_ROLL_N} 筆實際勝率 {wr:.1%}"

    kelly      = max(0.0, (wr * PAYOUT - (1 - wr)) / PAYOUT)
    half_kelly = kelly / 2

    cap   = state['capital']
    max_b = MAX_BET_CAP.get(symbol, 125)
    bet   = cap * half_kelly
    bet   = max(bet, MIN_BET)        # 至少 MIN_BET
    bet   = min(bet, max_b, cap)     # 不超過幣種上限與現有本金

    return bet, half_kelly * 100, wr, wr_source


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

            # F1 過濾：ATR_ratio >= 1.20 → 高波動，跳過
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
            quality   = signal_quality(direction, rsi, macd_hist, adx, vol_z)

            signals.append({
                'symbol':    sym,
                'direction': direction,
                'close':     close,
                'hlpct':     hlpct,
                'atr_ratio': atr_ratio,
                'bar_ts':    bar_ts,
                'rsi':       rsi,
                'adx':       adx,
                'vol_z':     vol_z,
                'quality':   quality,
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

        entry = p['entry_price']
        bet   = p['bet']
        win   = ((p['direction'] == 'LONG'  and settle_price > entry) or
                 (p['direction'] == 'SHORT' and settle_price < entry))

        # 事件合約結算：下單時已扣款，贏了退回 bet × (1 + payout)，輸了得 0
        pnl = bet * PAYOUT if win else -bet   # 記錄用淨損益（+0.85X 或 -X）
        if win:
            state['capital'] += bet * (1 + PAYOUT)  # 退回本金 + 獲利
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
        })

        total  = state['total_w'] + state['total_l']
        wr_str = f"{state['total_w'] / total * 100:.1f}%" if total > 0 else "—"
        emoji  = '✅' if win else '❌'
        zh_dir = '做多' if p['direction'] == 'LONG' else '做空'

        msg = (
            f"{emoji}  **事件合約結算**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"幣種：{p['symbol']}　方向：{zh_dir}\n"
            f"進場：${entry:,.4f}  →  結算：${settle_price:,.4f}\n"
            f"損益：**{pnl:+.2f} USDT**\n"
            f"模擬本金：**{state['capital']:.2f} USDT**\n"
            f"累計：{state['total_w']}W / {state['total_l']}L　勝率 {wr_str}"
        )
        send_discord(msg)
        print(f"[{now_str()}] {emoji} 結算 {p['symbol']} {p['direction']}"
              f"  進={entry:.4f} 出={settle_price:.4f}"
              f"  {pnl:+.2f}U  本金={state['capital']:.2f}")
        settled = True

    state['pending'] = remaining
    return settled


# ══════════════════════════════════════════════════════════════
#  訊號通知
# ══════════════════════════════════════════════════════════════
def format_signal_msg(sig: dict, capital: float, bet: float,
                      half_k_pct: float, wr_source: str,
                      total_w: int, total_l: int) -> str:
    sym       = sig['symbol']
    direction = sig['direction']
    close     = sig['close']
    hlpct     = sig['hlpct']
    settle_dt = datetime.now(TZ_CST) + timedelta(minutes=SETTLE_MIN)
    emoji     = '📈' if direction == 'LONG' else '📉'
    zh_dir    = '做多（賭漲）' if direction == 'LONG' else '做空（賭跌）'
    pos_str   = '底部' if direction == 'LONG' else '頂部'
    max_b     = MAX_BET_CAP.get(sym, 125)
    total     = total_w + total_l
    wr_str    = f"{total_w / total * 100:.1f}%" if total > 0 else "—"

    atr_ratio = sig.get('atr_ratio', 0)
    rsi       = sig.get('rsi')
    adx       = sig.get('adx')
    vol_z     = sig.get('vol_z')
    quality   = sig.get('quality', '')
    rsi_str   = f"{rsi:.1f}"   if rsi   is not None else "—"
    adx_str   = f"{adx:.1f}"   if adx   is not None else "—"
    volz_str  = f"{vol_z:.2f}" if vol_z is not None else "—"
    quality_line = f"信號品質：**{quality}**\n" if quality else ""
    return (
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{emoji}  **事件合約訊號**\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"幣種：{sym}\n"
        f"方向：{zh_dir}\n"
        f"當前收盤：${close:,.4f}\n"
        f"區間位置：{pos_str} {hlpct*100:.1f}%（近 {LOOKBACK_N} 棒）\n"
        f"RSI14：{rsi_str}　ADX：{adx_str}　vol_z：{volz_str}　ATR_ratio：{atr_ratio:.2f}\n"
        f"{quality_line}"
        f"合約類型：**{SETTLE_MIN} 分鐘**（賠率 {int(PAYOUT*100)}%）\n"
        f"結算時間：{settle_dt.strftime('%H:%M:%S')}\n"
        f"模擬下注：**{bet:.1f} USDT**（半Kelly {half_k_pct:.1f}%，上限 {max_b} USDT）\n"
        f"Kelly依據：{wr_source}\n"
        f"模擬本金：{capital:.2f} USDT\n"
        f"累計勝率：{wr_str}（{total_w}W / {total_l}L）"
    )


# ══════════════════════════════════════════════════════════════
#  主迴圈
# ══════════════════════════════════════════════════════════════
BAR_SEC    = TIMEFRAME_SEC = 5 * 60   # K 棒週期（秒）
SIG_DELAY  = 5    # K 棒收盤後等幾秒再抓（讓交易所資料穩定）
SIG_WINDOW = 60   # 收盤後幾秒內視為「剛收盤」（此窗口內才偵測訊號）
POLL_SEC   = 30   # 結算輪詢間隔（秒）—— 每 30 秒檢查一次待結算訂單


def run_loop(exchange, once: bool):
    state = load_state()
    print(f"[{now_str()}] 事件合約 Signal Bot 啟動（模擬模式）")
    print(f"  本金：{state['capital']:.2f} USDT  "
          f"勝：{state['total_w']}  敗：{state['total_l']}")
    print(f"  監控：{', '.join(SYMBOLS)}  "
          f"訊號：hlpct12_10_F1  下注：半Kelly動態")
    print(f"  結算輪詢：每 {POLL_SEC}s  訊號偵測：收盤後 {SIG_DELAY}s")
    print()

    sent_this_bar: dict = {}

    while True:
        try:
            now_ts  = time.time()
            bar_age = now_ts % BAR_SEC   # 距上根 K 棒收盤已過的秒數

            # 1. 結算到期訊號（每輪都跑，每 POLL_SEC 秒一次）
            if settle_pending(exchange, state):
                save_state(state)

            # 2. 本金不足，停止（只發一次警告）
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

            # 3. 訊號偵測：只在 K 棒收盤後 SIG_DELAY ~ SIG_DELAY+SIG_WINDOW 秒內執行
            in_sig_window = SIG_DELAY <= bar_age < SIG_DELAY + SIG_WINDOW
            if in_sig_window:
                new_sigs = check_signals(exchange)
                any_sent = False
                for sig in new_sigs:
                    key = (sig['symbol'], sig['bar_ts'])
                    if key in sent_this_bar:
                        continue

                    bet, half_k_pct, _, wr_source = calc_kelly_bet(
                        state, sig['symbol'])
                    if bet < MIN_BET:
                        continue   # 可用資金不足最小下單量，跳過
                    settle_at = (datetime.now(TZ_CST)
                                 + timedelta(minutes=SETTLE_MIN)).isoformat()

                    state['capital'] -= bet   # 下單即扣款（事件合約機制）
                    state['pending'].append({
                        'symbol':      sig['symbol'],
                        'direction':   sig['direction'],
                        'entry_price': sig['close'],
                        'bet':         bet,
                        'settle_at':   settle_at,
                    })
                    save_state(state)
                    sent_this_bar[key] = True
                    any_sent = True

                    msg = format_signal_msg(
                        sig, state['capital'], bet,
                        half_k_pct, wr_source,
                        state['total_w'], state['total_l'])
                    send_discord(msg)
                    print(f"[{now_str()}] 📢 {sig['symbol']} {sig['direction']}"
                          f"  close={sig['close']:,.4f}"
                          f"  hlpct={sig['hlpct']*100:.1f}%"
                          f"  本金={state['capital']:.2f}")

                if not any_sent:
                    print(f"[{now_str()}] 無訊號  "
                          f"本金={state['capital']:.2f}  "
                          f"待結算={len(state['pending'])} 筆")

        except Exception:
            traceback.print_exc()

        if once:
            break

        # 計算最優睡眠時間：
        #   有待結算單 → 每 POLL_SEC 秒輪詢，確保準時結算
        #   無待結算單 → 直接睡到下根 K 棒收盤後 SIG_DELAY 秒，不需頻繁喚醒
        now_ts  = time.time()
        bar_age = now_ts % BAR_SEC

        if bar_age < SIG_DELAY:
            # 剛收盤還沒到 5 秒：等到收盤後 5 秒再偵測
            sleep_sec = SIG_DELAY - bar_age + 0.5
        elif state['pending']:
            # 有待結算單：每 POLL_SEC 秒輪詢，但不超過下根 K 棒窗口
            time_to_next = BAR_SEC - bar_age + SIG_DELAY
            sleep_sec = min(POLL_SEC, time_to_next)
        else:
            # 無待結算單：直接睡到下根 K 棒收盤後 SIG_DELAY 秒
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
