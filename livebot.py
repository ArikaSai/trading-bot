

import ccxt
import pandas as pd
import numpy as np
import time
import requests
import json
import os
from datetime import datetime, timedelta, timezone
from strategy import CoreStrategy

# ADA Donchian 參數：在 __init__ 中從 config['ada_donchian'] 讀取


class LiveTradingBot:
    def __init__(self, config_path="config.json", state_path="order_state_dual.json"):
        self.state_path = state_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ 找不到設定檔：{config_path}")

        for _enc in ('utf-8-sig', 'utf-8', 'cp950', 'big5'):
            try:
                with open(config_path, 'r', encoding=_enc) as f:
                    self.config = json.load(f)
                break
            except (UnicodeDecodeError, ValueError):
                continue
        else:
            raise ValueError(f"❌ 無法解碼設定檔（嘗試過 utf-8/cp950/big5）：{config_path}")

        self.api_key            = self.config['api']['api_key']
        self.api_secret         = self.config['api']['api_secret']
        self.discord_url        = self.config['api']['discord_webhook_url']
        self.risk_per_trade     = self.config['risk']['risk_per_trade']
        self.max_pos_ratio      = self.config['risk']['max_pos_ratio']
        self.leverage           = self.config['risk'].get('leverage', 1.0)
        self.capital            = self.config['risk']['initial_capital']
        self.adx_threshold      = self.config['strategy']['adx_threshold']
        self.trailing_atr       = self.config['strategy']['trailing_atr']
        self.initial_sl_atr     = self.config['strategy']['initial_sl_atr']
        self.min_sl_pct         = self.config['strategy'].get('min_sl_pct', 0.001)
        self.max_trade_usdt_cap  = self.config['risk'].get('max_trade_usdt_cap', 200000.0)
        self.max_position_cap    = 40_000_000.0   # 幣安最大持倉名目價值（8× 槓桿下）
        self.max_consec_losses   = self.config['risk'].get('max_consec_losses', 3)  # SOL 專屬

        # SOL 自適應 TWAP 狀態（重啟後從 N=1 重新計數，不需持久化）
        self._sol_twap_active    = False
        self._sol_twap_remaining = 0
        self._sol_twap_size_each = 0.0
        self._sol_twap_direction = 0
        self.taker_fee_rate     = self.config['risk'].get('taker_fee_rate', 0.0005)
        self.mmr                = self.config['risk'].get('maintenance_margin_rate', 0.005)

        # ── ADA Donchian 參數（從 config 讀取）───────────────
        ada_cfg = self.config.get('ada_donchian', {})
        self.ADA_ENTRY_N       = ada_cfg.get('entry_n', 10)
        self.ADA_TRAIL_ATR     = ada_cfg.get('trail_atr', 3.0)
        self.ADA_ATR_SL_MULT   = ada_cfg.get('atr_sl_mult', 2.0)
        self.ADA_RISK_PCT      = ada_cfg.get('risk_pct', 0.15)
        self.ADA_LEVERAGE      = ada_cfg.get('leverage', 1)
        self.ADA_MAX_TRADE_CAP = ada_cfg.get('max_trade_cap', 200_000.0)
        self.ADA_MAX_CONSEC    = ada_cfg.get('max_consec_losses', 3)
        self.ADA_FEE           = self.taker_fee_rate

        # ── XRP Fib 參數（從 config 讀取）────────────────────
        xrp_cfg = self.config.get('xrp_fib', {})
        self.XRP_SWING_N       = xrp_cfg.get('swing_n', 20)
        self.XRP_FIB_LEVEL     = xrp_cfg.get('fib_level', 0.618)
        self.XRP_TRAIL_ATR     = xrp_cfg.get('trail_atr', 3.0)
        self.XRP_FIB_TOL       = xrp_cfg.get('fib_tol', 0.005)
        self.XRP_ATR_SL_MULT   = xrp_cfg.get('atr_sl_mult', 2.0)
        self.XRP_RISK_PCT      = xrp_cfg.get('risk_pct', 0.15)
        self.XRP_LEVERAGE      = xrp_cfg.get('leverage', 1)
        self.XRP_MAX_TRADE_CAP  = xrp_cfg.get('max_trade_cap', 200_000.0)
        self.XRP_MAX_CONSEC     = xrp_cfg.get('max_consec_losses', 3)
        self.XRP_LIMIT_MAX_HOURS = xrp_cfg.get('xrp_limit_max_hours', 0)  # 0 = 不逾時
        self.XRP_FEE            = self.taker_fee_rate

        # ── DOGE BB Squeeze 參數（從 config 讀取）────────────────────
        doge_cfg = self.config.get('doge_squeeze', {})
        self.DOGE_BB_PERIOD     = doge_cfg.get('bb_period', 20)
        self.DOGE_BB_STD        = doge_cfg.get('bb_std', 2.0)
        self.DOGE_KC_PERIOD     = doge_cfg.get('kc_period', 10)
        self.DOGE_KC_MULT       = doge_cfg.get('kc_mult', 1.25)
        self.DOGE_MOM_PERIOD    = doge_cfg.get('mom_period', 12)
        self.DOGE_TRAIL_ATR     = doge_cfg.get('trail_atr', 3.5)
        self.DOGE_ATR_SL_MULT   = doge_cfg.get('atr_sl_mult', 2.0)
        self.DOGE_LEVERAGE      = doge_cfg.get('leverage', 1)
        self.DOGE_MAX_TRADE_CAP = doge_cfg.get('max_trade_cap', 200_000.0)
        self.DOGE_MAX_CONSEC    = doge_cfg.get('max_consec_losses', 3)
        self.DOGE_FEE           = self.taker_fee_rate

        self.symbols    = {
            'SOL':  'SOL/USDT',
            'ADA':  ada_cfg.get('symbol', 'ADA/USDT'),
            'XRP':  xrp_cfg.get('symbol', 'XRP/USDT'),
            'DOGE': doge_cfg.get('symbol', 'DOGE/USDT'),
        }
        self.timeframes = {
            'SOL':  '15m',
            'ADA':  ada_cfg.get('timeframe', '1h'),
            'XRP':  xrp_cfg.get('timeframe', '1h'),
            'DOGE': doge_cfg.get('timeframe', '1h'),
        }
        self._close_time = {}   # 記錄各策略最近平倉時間，防止 API 快取觸發雲端接管

        # SOL 盤整縮緊：N=8, X=1.25, tight=1.0×
        from collections import deque as _deque
        self._consol_highs    = _deque(maxlen=8)
        self._consol_lows     = _deque(maxlen=8)
        self._consol_last_ts  = None   # 防止同一根 K 棒重複 append

        # ADA / XRP / DOGE 盤整縮緊：N=6, X=1.5, tight=0.5×；插針防護：3.0×ATR
        self._ada_consol_highs   = _deque(maxlen=6)
        self._ada_consol_lows    = _deque(maxlen=6)
        self._ada_consol_last_ts = None   # 防止同一根 K 棒重複 append（對齊 SOL）
        self._xrp_consol_highs   = _deque(maxlen=6)
        self._xrp_consol_lows    = _deque(maxlen=6)
        self._xrp_consol_last_ts = None
        self._doge_consol_highs  = _deque(maxlen=6)
        self._doge_consol_lows   = _deque(maxlen=6)
        self._doge_consol_last_ts = None

        # ADA TWAP 狀態
        self._ada_twap_active    = False
        self._ada_twap_remaining = 0
        self._ada_twap_size_each = 0.0
        self._ada_twap_direction = 0

        # DOGE TWAP 狀態
        self._doge_twap_active    = False
        self._doge_twap_remaining = 0
        self._doge_twap_size_each = 0.0
        self._doge_twap_direction = 0

        # XRP 限價單狀態（方案A：無確認，掛在 Fib 水平）
        self._xrp_limit_order_id  = None   # 交易所 order ID；模擬模式用 'pending'
        self._xrp_limit_price     = 0.0    # 掛單價格（Fib 水平）
        self._xrp_limit_direction = 0      # 1 做多 / -1 做空
        self._xrp_limit_size      = 0.0    # 掛單數量
        self._xrp_limit_placed_ts = 0.0    # 掛單時間戳（Unix 秒），用於逾時判斷

        self.live_trade    = self.config['system']['live_trade']
        self.check_interval = self.config['system']['check_interval']

        self.exchange = ccxt.binance({
            'apiKey':          self.api_key    if self.live_trade else '',
            'secret':          self.api_secret if self.live_trade else '',
            'enableRateLimit': True,
            'options':         {'defaultType': 'future'}
        })

        self.state = {
            'SOL':  self._get_default_state(),
            'ADA':  self._get_default_state(),
            'XRP':  self._get_default_state(),
            'DOGE': self._get_default_state(),
        }

        self.load_order_state()

        self.last_candle_time = {'SOL': None, 'ADA': None, 'XRP': None, 'DOGE': None}
        self.last_report_hour = -1   # 整點定時報告：記錄上次觸發的小時（台灣時間）
        self._last_status_print = -1  # 上次狀態列印的 5 分鐘時段編號

        print(f"🤖 四核心機器人就緒 | 模式: {'🔴 實盤' if self.live_trade else '🟢 模擬'}")
        print(f"   SOL 趨勢: {self.risk_per_trade*100}% 風險 | {self.leverage}x | 熔斷 {self.max_consec_losses} 次")
        print(f"   ADA Donchian: N={self.ADA_ENTRY_N} | Trail x{self.ADA_TRAIL_ATR} | {self.ADA_LEVERAGE}x | 熔斷 {self.ADA_MAX_CONSEC} 次")
        print(f"   XRP Fib: Swing={self.XRP_SWING_N} | Fib={self.XRP_FIB_LEVEL} | Trail x{self.XRP_TRAIL_ATR} | {self.XRP_LEVERAGE}x | 熔斷 {self.XRP_MAX_CONSEC} 次")
        print(f"   DOGE Squeeze: BB={self.DOGE_BB_PERIOD}/{self.DOGE_BB_STD} | KC={self.DOGE_KC_PERIOD}/{self.DOGE_KC_MULT} | Trail x{self.DOGE_TRAIL_ATR} | {self.DOGE_LEVERAGE}x | 熔斷 {self.DOGE_MAX_CONSEC} 次")

    # ── 狀態字典 ─────────────────────────────────────────────────────

    def _get_default_state(self) -> dict:
        return {
            "position":           0,
            "position_size":      0.0,
            "entry_price":        0.0,
            "stop_loss":          0.0,
            "trailing_stop":      0.0,
            "highest_price":      0.0,
            "lowest_price":       float('inf'),
            "liq_price":          0.0,
            "stop_order_id":      None,
            "be_activated":       False,     # SOL 保本是否已觸發
            "consecutive_losses": 0,
            "skip_next_trade":    False,     # SOL / ADA 熔斷
            "in_skip_zone":       False,     # SOL / ADA 熔斷
        }

    # FIX-1: 移除舊的無參數版本，唯一定義帶 strat_name 的正確版本
    def reset_position_state(self, strat_name: str):
        """重置持倉欄位，保留熔斷相關欄位不動"""
        s = self.state[strat_name]
        s["position"]      = 0
        s["position_size"] = 0.0
        s["entry_price"]   = 0.0
        s["stop_loss"]     = 0.0
        s["trailing_stop"] = 0.0
        s["highest_price"] = 0.0
        s["lowest_price"]  = float('inf')
        s["liq_price"]     = 0.0
        s["stop_order_id"] = None
        s["be_activated"]  = False
        self._close_time[strat_name] = time.time()   # 記錄平倉時刻
        if strat_name == 'SOL':
            self._sol_twap_active    = False
            self._sol_twap_remaining = 0
            self._consol_highs.clear()
            self._consol_lows.clear()
            self._consol_last_ts = None
        elif strat_name == 'ADA':
            self._ada_twap_active    = False
            self._ada_twap_remaining = 0
            self._ada_consol_highs.clear()
            self._ada_consol_lows.clear()
            self._ada_consol_last_ts = None
        elif strat_name == 'XRP':
            self._xrp_limit_order_id  = None
            self._xrp_consol_highs.clear()
            self._xrp_consol_lows.clear()
            self._xrp_consol_last_ts  = None
            self._xrp_limit_price     = 0.0
            self._xrp_limit_direction = 0
            self._xrp_limit_size      = 0.0
            self._xrp_limit_placed_ts = 0.0
        elif strat_name == 'DOGE':
            self._doge_twap_active    = False
            self._doge_twap_remaining = 0
            self._doge_consol_highs.clear()
            self._doge_consol_lows.clear()
            self._doge_consol_last_ts = None

    # ── 持久化 ───────────────────────────────────────────────────────

    def save_order_state(self):
        save_data = {}
        for strat, data in self.state.items():
            save_data[strat] = {
                "position":           data["position"],
                "position_size":      data["position_size"],
                "entry_price":        data["entry_price"],
                "stop_loss":          data["stop_loss"],
                "trailing_stop":      data["trailing_stop"],
                "highest_price":      data["highest_price"],
                "lowest_price":       data["lowest_price"],
                "liq_price":          data["liq_price"],
                "be_activated":       data["be_activated"],
                "stop_order_id":      data["stop_order_id"],
                "consecutive_losses": data["consecutive_losses"],
                "skip_next_trade":    data["skip_next_trade"],
                "in_skip_zone":       data["in_skip_zone"],
            }
        save_data['_xrp_limit'] = {
            'order_id':  self._xrp_limit_order_id,
            'price':     self._xrp_limit_price,
            'direction': self._xrp_limit_direction,
            'size':      self._xrp_limit_size,
            'placed_ts': self._xrp_limit_placed_ts,
        }
        with open(self.state_path, 'w') as f:
            json.dump(save_data, f, indent=2)

    def load_order_state(self):
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, 'r') as f:
                saved = json.load(f)
            for strat in self.symbols:
                if strat in saved:
                    # 只還原新版存在的欄位，忽略舊版多餘欄位
                    for k, v in saved[strat].items():
                        if k in self.state[strat]:
                            self.state[strat][k] = v
                    if self.state[strat].get('position', 0) != 0:
                        ep   = self.state[strat]['entry_price']
                        side = '多單' if self.state[strat]['position'] == 1 else '空單'
                        sl   = self.state[strat]['trailing_stop']
                        print(f"♻️  {strat} 持倉還原：{side} @ {ep:.4f} | 追蹤止損 {sl:.4f}")
            # 還原 XRP 限價單狀態
            if '_xrp_limit' in saved:
                lim = saved['_xrp_limit']
                self._xrp_limit_order_id  = lim.get('order_id')
                self._xrp_limit_price     = lim.get('price', 0.0)
                self._xrp_limit_direction = lim.get('direction', 0)
                self._xrp_limit_size      = lim.get('size', 0.0)
                self._xrp_limit_placed_ts = lim.get('placed_ts', 0.0)
                if self._xrp_limit_order_id and self.state['XRP'].get('position', 0) == 0:
                    side_str = '做多' if self._xrp_limit_direction == 1 else '做空'
                    print(f"♻️  XRP 限價掛單還原：{side_str} @ {self._xrp_limit_price:.4f} | size={self._xrp_limit_size:.2f}")
        except Exception as e:
            print(f"⚠️ 讀取存檔失敗: {e}")

    # ── Discord ──────────────────────────────────────────────────────

    def send_discord_msg(self, message: str):
        if not self.discord_url or "你的" in self.discord_url:
            return
        try:
            requests.post(self.discord_url, json={"content": message}, timeout=5)
        except Exception:
            pass

    # ── 報告（FIX-6：改用 self.state 字典）──────────────────────────

    def send_periodic_report(self):
        now_tw = datetime.now(timezone.utc) + timedelta(hours=8)
        if now_tw.hour == self.last_report_hour:
            return
        self.last_report_hour = now_tw.hour
        try:
            balance = float(self.exchange.fetch_balance()['total']['USDT']) if self.live_trade else 0.0
            sol_s   = self.state['SOL']

            # SOL 持倉：即時抓現價計算浮盈虧
            if sol_s['position'] != 0:
                try:
                    sol_price = float(self.exchange.fetch_ticker('SOL/USDT')['last']) if self.live_trade else sol_s['entry_price']
                except Exception:
                    sol_price = sol_s['entry_price']
                unr    = (sol_price - sol_s['entry_price']) * sol_s['position_size'] * sol_s['position']
                margin = (sol_s['entry_price'] * sol_s['position_size']) / self.leverage
                roe    = unr / margin * 100 if margin > 0 else 0
                pnl_e  = '🟢' if unr >= 0 else '🔴'
                side   = '多單' if sol_s['position'] == 1 else '空單'
                sol_cb = "🛡️ 保護中" if sol_s['skip_next_trade'] or sol_s['in_skip_zone'] else "🟢 正常"
                sol_block = (
                    f"🟡 SOL {side} | 進場: {sol_s['entry_price']:.2f} | 現價: {sol_price:.2f}\n"
                    f"   {pnl_e} 浮盈虧: {unr:+.2f} U ({roe:+.2f}%) | SL: {sol_s['trailing_stop']:.2f}\n"
                    f"   熔斷: {sol_cb} | 連損: {sol_s['consecutive_losses']}/{self.max_consec_losses}"
                )
            else:
                try:
                    sol_price = float(self.exchange.fetch_ticker('SOL/USDT')['last']) if self.live_trade else 0.0
                except Exception:
                    sol_price = 0.0
                cb_tag = " | 🛡️ 保護中" if sol_s['skip_next_trade'] or sol_s['in_skip_zone'] else ""
                sol_block = (
                    f"🟡 SOL 空倉 | 現價: {sol_price:.2f} | 👁️ 待機{cb_tag}\n"
                    f"   連損: {sol_s['consecutive_losses']}/{self.max_consec_losses}"
                )

            # ADA 持倉：即時抓現價計算浮盈虧
            ada_s = self.state['ADA']
            if ada_s['position'] != 0:
                try:
                    ada_price = float(self.exchange.fetch_ticker('ADA/USDT')['last']) if self.live_trade else ada_s['entry_price']
                except Exception:
                    ada_price = ada_s['entry_price']
                unr   = (ada_price - ada_s['entry_price']) * ada_s['position_size'] * ada_s['position']
                pnl_e = '🟢' if unr >= 0 else '🔴'
                side  = '多單' if ada_s['position'] == 1 else '空單'
                ada_cb = "🛡️ 保護中" if ada_s['skip_next_trade'] or ada_s['in_skip_zone'] else "🟢 正常"
                ada_block = (
                    f"🔵 ADA {side} | 進場: {ada_s['entry_price']:.4f} | 現價: {ada_price:.4f}\n"
                    f"   {pnl_e} 浮盈虧: {unr:+.2f} U | SL: {ada_s['trailing_stop']:.4f}\n"
                    f"   熔斷: {ada_cb} | 連損: {ada_s['consecutive_losses']}/{self.ADA_MAX_CONSEC}"
                )
            else:
                try:
                    ada_price = float(self.exchange.fetch_ticker('ADA/USDT')['last']) if self.live_trade else 0.0
                except Exception:
                    ada_price = 0.0
                cb_tag = " | 🛡️ 保護中" if ada_s['skip_next_trade'] or ada_s['in_skip_zone'] else ""
                ada_block = (
                    f"🔵 ADA 空倉 | 現價: {ada_price:.4f} | 👁️ 待機{cb_tag}\n"
                    f"   連損: {ada_s['consecutive_losses']}/{self.ADA_MAX_CONSEC}"
                )

            # XRP 持倉
            xrp_s = self.state['XRP']
            if xrp_s['position'] != 0:
                try:
                    xrp_price = float(self.exchange.fetch_ticker('XRP/USDT')['last']) if self.live_trade else xrp_s['entry_price']
                except Exception:
                    xrp_price = xrp_s['entry_price']
                unr   = (xrp_price - xrp_s['entry_price']) * xrp_s['position_size'] * xrp_s['position']
                pnl_e = '🟢' if unr >= 0 else '🔴'
                side  = '多單' if xrp_s['position'] == 1 else '空單'
                xrp_cb = "🛡️ 保護中" if xrp_s['skip_next_trade'] or xrp_s['in_skip_zone'] else "🟢 正常"
                xrp_block = (
                    f"🟣 XRP {side} | 進場: {xrp_s['entry_price']:.4f} | 現價: {xrp_price:.4f}\n"
                    f"   {pnl_e} 浮盈虧: {unr:+.2f} U | SL: {xrp_s['trailing_stop']:.4f}\n"
                    f"   熔斷: {xrp_cb} | 連損: {xrp_s['consecutive_losses']}/{self.XRP_MAX_CONSEC}"
                )
            else:
                try:
                    xrp_price = float(self.exchange.fetch_ticker('XRP/USDT')['last']) if self.live_trade else 0.0
                except Exception:
                    xrp_price = 0.0
                cb_tag = " | 🛡️ 保護中" if xrp_s['skip_next_trade'] or xrp_s['in_skip_zone'] else ""
                xrp_block = (
                    f"🟣 XRP 空倉 | 現價: {xrp_price:.4f} | 👁️ 待機{cb_tag}\n"
                    f"   連損: {xrp_s['consecutive_losses']}/{self.XRP_MAX_CONSEC}"
                )

            # DOGE 持倉
            doge_s = self.state['DOGE']
            if doge_s['position'] != 0:
                try:
                    doge_price = float(self.exchange.fetch_ticker('DOGE/USDT')['last']) if self.live_trade else doge_s['entry_price']
                except Exception:
                    doge_price = doge_s['entry_price']
                unr   = (doge_price - doge_s['entry_price']) * doge_s['position_size'] * doge_s['position']
                pnl_e = '🟢' if unr >= 0 else '🔴'
                side  = '多單' if doge_s['position'] == 1 else '空單'
                doge_cb = "🛡️ 保護中" if doge_s['skip_next_trade'] or doge_s['in_skip_zone'] else "🟢 正常"
                doge_block = (
                    f"🟤 DOGE {side} | 進場: {doge_s['entry_price']:.5f} | 現價: {doge_price:.5f}\n"
                    f"   {pnl_e} 浮盈虧: {unr:+.2f} U | SL: {doge_s['trailing_stop']:.5f}\n"
                    f"   熔斷: {doge_cb} | 連損: {doge_s['consecutive_losses']}/{self.DOGE_MAX_CONSEC}"
                )
            else:
                try:
                    doge_price = float(self.exchange.fetch_ticker('DOGE/USDT')['last']) if self.live_trade else 0.0
                except Exception:
                    doge_price = 0.0
                cb_tag = " | 🛡️ 保護中" if doge_s['skip_next_trade'] or doge_s['in_skip_zone'] else ""
                doge_block = (
                    f"🟤 DOGE 空倉 | 現價: {doge_price:.5f} | 👁️ 待機{cb_tag}\n"
                    f"   連損: {doge_s['consecutive_losses']}/{self.DOGE_MAX_CONSEC}"
                )

            msg = (
                f"⏰ **整點報告** | {now_tw.strftime('%m/%d %H:00')}\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"{sol_block}\n{ada_block}\n{xrp_block}\n{doge_block}\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"💰 餘額: {balance:.2f} USDT"
            )
            print(f"\n{msg}")
            self.send_discord_msg(msg)
        except Exception as e:
            print(f"⚠️ 定時報告失敗: {e}")

    def show_milestone_progress(self, pnl: float):
        try:
            balance    = float(self.exchange.fetch_balance()['total']['USDT'])
            milestones = [1000, 10000, 100000, 1000000]
            target     = next((m for m in milestones if balance < m), milestones[-1] * 10)
            progress   = min((balance / target) * 100, 100)
            bar        = "█" * int(20 * progress // 100) + "░" * (20 - int(20 * progress // 100))
            msg = (
                f"🎊 **獲利入袋！** 🎊\n"
                f"💵 本次入帳: +{pnl:.2f} USDT\n"
                f"💰 帳戶餘額: {balance:.2f} USDT\n"
                f"📈 [{bar}] {progress:.1f}%"
            )
            print(msg); self.send_discord_msg(msg)
        except Exception:
            pass

    # ── 交易所工具 ───────────────────────────────────────────────────

    def wipe_all_orders(self, raw_symbol: str):
        if not self.live_trade:
            return
        try:
            self.exchange.fapiPrivateDeleteAllOpenOrders({'symbol': raw_symbol})
            time.sleep(0.5)
        except Exception as e:
            print(f"⚠️ 清場: {e}")

    def execute_order(self, strat_name: str, side: str,
                      size=None, reason: str = "",
                      stop_price=None):
        """
        市價單 : size 必填，stop_price 為 None
        止損單 : stop_price 必填
        """
        symbol     = self.symbols[strat_name]
        raw_symbol = symbol.replace('/', '')
        state      = self.state[strat_name]

        if not self.live_trade:
            print(f"[模擬] {strat_name} {side.upper()} | {reason} | "
                  f"size={size} | stop={stop_price}")
            return None

        # ── 止損單 ──────────────────────────────────────────────
        if stop_price is not None:
            if state['stop_order_id']:
                if not getattr(self, f'_{strat_name}_sl_guard', False):
                    print(f"ℹ️ {strat_name} 已有硬性止損單，守衛中。")
                    setattr(self, f'_{strat_name}_sl_guard', True)
                return None
            try:
                params = {
                    'stopPrice':     self.exchange.price_to_precision(symbol, stop_price),
                    'closePosition': True,
                    'workingType':   'MARK_PRICE',
                }
                order = self.exchange.create_order(symbol, 'STOP_MARKET', side, None, None, params)
                state['stop_order_id'] = order['id']
                self.save_order_state()
                print(f"🛡️ {strat_name} SL 掛出 @ {round(stop_price, 4)}")
            except Exception as e:
                if "-4130" in str(e) or "existing" in str(e).lower():
                    state['stop_order_id'] = "exchange_existing"
                    self.save_order_state()
                else:
                    print(f"⚠️ {strat_name} SL 單失敗: {e}")
            return None

        # ── 市價單 ───────────────────────────────────────────────
        # 進場前先確認數量 >= 交易所最小單位（平倉單不做此檢查）
        is_close = state['position'] != 0   # 在 wipe 前定義，確保 except 區塊可引用
        if not is_close and size is not None:
            try:
                mkt_info  = self.exchange.market(symbol)
                min_amount = mkt_info.get('limits', {}).get('amount', {}).get('min') or 0
                if min_amount and float(size) < float(min_amount):
                    print(f"⚠️ {strat_name} 倉位量 {size:.6f} < 最小下單量 {min_amount}，放棄本次進場")
                    return None
            except Exception:
                pass
        self.wipe_all_orders(raw_symbol)
        setattr(self, f'_{strat_name}_sl_guard', False)
        try:
            size_str     = self.exchange.amount_to_precision(symbol, size)
            order_params = {'reduceOnly': True} if is_close else {}
            order         = self.exchange.create_order(symbol, 'market', side, size_str, None, order_params)

            close_summary = ""
            if state['position'] != 0:
                exit_price = float(order.get('average') or order.get('price') or 0)
                if exit_price == 0:
                    exit_price = float(self.exchange.fetch_ticker(symbol)['last'])
                gross        = (exit_price - state['entry_price']) * state['position_size'] * state['position']
                efee         = state['entry_price'] * state['position_size'] * self.taker_fee_rate
                xfee         = exit_price           * state['position_size'] * self.taker_fee_rate
                net          = gross - efee - xfee
                initial_risk = abs(state['entry_price'] - state['stop_loss']) * state['position_size']
                r_str        = f"{net / initial_risk:+.2f}R" if initial_risk > 0 else "N/A R"
                print(f"💹 {strat_name} 毛利: {gross:.2f} | 費: -{efee+xfee:.2f} | 淨: {net:.2f} U ({r_str})")
                close_summary = f" | {'✅' if net >= 0 else '❌'} {net:+.2f} U ({r_str})"
                if net > 0:
                    self.show_milestone_progress(net)
                self._update_loss_state(strat_name, net)
                self.reset_position_state(strat_name)
                self.save_order_state()

            msg = f"🎯 **{strat_name} 成交** | {side.upper()} | {reason}{close_summary}"
            print(msg); self.send_discord_msg(msg)
            return order
        except Exception as e:
            print(f"❌ {strat_name} 市價單失敗: {e}")
            if is_close:
                # 平倉單失敗（交易所已平 / reduceOnly 拒絕）→ 強制清本地狀態，終止無限循環
                self.reset_position_state(strat_name)
                self.save_order_state()
                print(f"⚠️ {strat_name} 平倉單失敗，已強制清倉位狀態")
            return None

    def _update_loss_state(self, strat_name: str, net_pnl: float):
        """SOL 和 ADA 各用自己的熔斷門檻，都用 skip_next_trade 機制"""
        state = self.state[strat_name]
        if net_pnl > 0:
            state['consecutive_losses'] = 0
        else:
            state['consecutive_losses'] += 1
            if strat_name == 'SOL':
                threshold = self.max_consec_losses
            elif strat_name == 'ADA':
                threshold = self.ADA_MAX_CONSEC
            elif strat_name == 'DOGE':
                threshold = self.DOGE_MAX_CONSEC
            else:
                threshold = self.XRP_MAX_CONSEC
            if state['consecutive_losses'] >= threshold:
                state['consecutive_losses'] = 0
                state['skip_next_trade'] = True
                msg = f"🚨 **{strat_name} 觸發連虧熔斷** | 連損 {threshold} 次"
                print(msg); self.send_discord_msg(msg)

    # ── SOL：追蹤止損出場（FIX-3：只給 SOL 用）────────────────────

    def monitor_exit_sol(self, cur):
        state    = self.state['SOL']
        cur_high = float(cur.high); cur_low  = float(cur.low)
        cur_atr   = float(cur.ATR);   cur_open  = float(cur.open)
        cur_close = float(cur.close)

        # ── 盤整縮緊（N=8, X=1.25× ATR, tight=1.0× ATR）──────────
        # deque 由主迴圈每根新 K 棒 append 一次，此處只讀取
        pos = state['position']
        if pos != 0 and len(self._consol_highs) == 8 and cur_atr > 0:
            _is_consol = (max(self._consol_highs) - min(self._consol_lows)) < 1.25 * cur_atr
            _eff_trail = 1.0 if _is_consol else self.trailing_atr
        else:
            _eff_trail = self.trailing_atr

        # ── 4.0× ATR 插針過濾：插針K棒改用收盤價更新追蹤參考點 ──
        is_spike = cur_atr > 0 and (cur_high - cur_low) > 4.0 * cur_atr
        ref_high = cur_close if (is_spike and pos == 1)  else cur_high
        ref_low  = cur_close if (is_spike and pos == -1) else cur_low

        state['trailing_stop'], state['highest_price'], state['lowest_price'] = \
            CoreStrategy.update_trailing_stop(
                pos, state['trailing_stop'],
                state['highest_price'], state['lowest_price'],
                ref_high, ref_low, cur_atr, _eff_trail
            )

        # ── 2.0R 保本（僅更新本地軟止損，不動交易所硬止損單）──
        if not state['be_activated'] and state['position'] != 0:
            sl_dist = abs(state['entry_price'] - state['stop_loss'])
            if state['position'] == 1 and cur_high >= state['entry_price'] + 2.0 * sl_dist:
                be_price = state['entry_price'] / ((1 - self.taker_fee_rate) * (1 - 0.001))
                state['trailing_stop'] = max(state['trailing_stop'], be_price)
                state['be_activated']  = True
            elif state['position'] == -1 and cur_low <= state['entry_price'] - 2.0 * sl_dist:
                be_price = state['entry_price'] / ((1 + self.taker_fee_rate) * (1 + 0.001))
                state['trailing_stop'] = min(state['trailing_stop'], be_price)
                state['be_activated']  = True

        closed, _, _, reason = CoreStrategy.check_exit(
            state['position'], cur_low, cur_high, cur_open,
            state['liq_price'], state['trailing_stop'], state['stop_loss'],
            state['entry_price'], state['position_size'], self.taker_fee_rate, 0.0
        )
        if closed:
            side = 'sell' if state['position'] == 1 else 'buy'
            print(f"🚨 SOL 本地 {reason} 觸發")
            self.execute_order('SOL', side, state['position_size'], reason=reason)

    # ── ADA：ATR trailing 出場 ──────────────────────────────────────

    def monitor_exit_ada(self, cur_high: float, cur_low: float,
                         cur_open: float, cur_atr: float):
        state = self.state['ADA']
        if state['position'] == 0:
            return
        pos = state['position']

        # 插針防護：3.0×ATR 插針改用收盤價更新追蹤參考點
        is_spike = cur_atr > 0 and (cur_high - cur_low) > 3.0 * cur_atr
        ref_high = cur_open if (is_spike and pos == 1)  else cur_high
        ref_low  = cur_open if (is_spike and pos == -1) else cur_low

        # 盤整縮緊：N=6, X=1.5, tight=0.5×ATR（deque 由主迴圈在新 K 棒時 append，此處只讀取）
        if len(self._ada_consol_highs) == 6 and cur_atr > 0:
            _is_consol = (max(self._ada_consol_highs) - min(self._ada_consol_lows)) < 1.5 * cur_atr
            _eff_trail = 0.5 if _is_consol else self.ADA_TRAIL_ATR
        else:
            _eff_trail = self.ADA_TRAIL_ATR

        if pos == 1:
            state['highest_price'] = max(state['highest_price'], ref_high)
            state['trailing_stop'] = max(
                state['trailing_stop'],
                state['highest_price'] - _eff_trail * cur_atr
            )
            if cur_low <= state['trailing_stop']:
                print(f"🚨 ADA 本地 Trailing 觸發")
                self.execute_order('ADA', 'sell', state['position_size'], reason="Trailing")
        elif pos == -1:
            state['lowest_price'] = min(state['lowest_price'], ref_low)
            state['trailing_stop'] = min(
                state['trailing_stop'],
                state['lowest_price'] + _eff_trail * cur_atr
            )
            if cur_high >= state['trailing_stop']:
                print(f"🚨 ADA 本地 Trailing 觸發")
                self.execute_order('ADA', 'buy', state['position_size'], reason="Trailing")

    # ── XRP：ATR trailing 出場 ─────────────────────────────────────────

    def monitor_exit_xrp(self, cur_high: float, cur_low: float,
                         cur_open: float, cur_atr: float):
        state = self.state['XRP']
        if state['position'] == 0:
            return
        pos = state['position']

        # 插針防護：3.0×ATR
        is_spike = cur_atr > 0 and (cur_high - cur_low) > 3.0 * cur_atr
        ref_high = cur_open if (is_spike and pos == 1)  else cur_high
        ref_low  = cur_open if (is_spike and pos == -1) else cur_low

        # 盤整縮緊：N=6, X=1.5, tight=0.5×ATR（deque 由主迴圈在新 K 棒時 append，此處只讀取）
        if len(self._xrp_consol_highs) == 6 and cur_atr > 0:
            _is_consol = (max(self._xrp_consol_highs) - min(self._xrp_consol_lows)) < 1.5 * cur_atr
            _eff_trail = 0.5 if _is_consol else self.XRP_TRAIL_ATR
        else:
            _eff_trail = self.XRP_TRAIL_ATR

        if pos == 1:
            state['highest_price'] = max(state['highest_price'], ref_high)
            state['trailing_stop'] = max(
                state['trailing_stop'],
                state['highest_price'] - _eff_trail * cur_atr
            )
            if cur_low <= state['trailing_stop']:
                print(f"🚨 XRP 本地 Trailing 觸發")
                self.execute_order('XRP', 'sell', state['position_size'], reason="Trailing")
        elif pos == -1:
            state['lowest_price'] = min(state['lowest_price'], ref_low)
            state['trailing_stop'] = min(
                state['trailing_stop'],
                state['lowest_price'] + _eff_trail * cur_atr
            )
            if cur_high >= state['trailing_stop']:
                print(f"🚨 XRP 本地 Trailing 觸發")
                self.execute_order('XRP', 'buy', state['position_size'], reason="Trailing")

    # ── DOGE：ATR trailing 出場 ────────────────────────────────────

    def monitor_exit_doge(self, cur_high: float, cur_low: float,
                          cur_open: float, cur_atr: float):
        state = self.state['DOGE']
        if state['position'] == 0:
            return
        pos = state['position']

        # 插針防護：3.0×ATR
        is_spike = cur_atr > 0 and (cur_high - cur_low) > 3.0 * cur_atr
        ref_high = cur_open if (is_spike and pos == 1)  else cur_high
        ref_low  = cur_open if (is_spike and pos == -1) else cur_low

        # 盤整縮緊：N=6, X=1.5, tight=0.5×ATR（deque 由主迴圈在新 K 棒時 append，此處只讀取）
        if len(self._doge_consol_highs) == 6 and cur_atr > 0:
            _is_consol = (max(self._doge_consol_highs) - min(self._doge_consol_lows)) < 1.5 * cur_atr
            _eff_trail = 0.5 if _is_consol else self.DOGE_TRAIL_ATR
        else:
            _eff_trail = self.DOGE_TRAIL_ATR

        if pos == 1:
            state['highest_price'] = max(state['highest_price'], ref_high)
            state['trailing_stop'] = max(
                state['trailing_stop'],
                state['highest_price'] - _eff_trail * cur_atr
            )
            if cur_low <= state['trailing_stop']:
                print(f"🚨 DOGE 本地 Trailing 觸發")
                self.execute_order('DOGE', 'sell', state['position_size'], reason="Trailing")
        elif pos == -1:
            state['lowest_price'] = min(state['lowest_price'], ref_low)
            state['trailing_stop'] = min(
                state['trailing_stop'],
                state['lowest_price'] + _eff_trail * cur_atr
            )
            if cur_high >= state['trailing_stop']:
                print(f"🚨 DOGE 本地 Trailing 觸發")
                self.execute_order('DOGE', 'buy', state['position_size'], reason="Trailing")

    # ── 同步持倉 ─────────────────────────────────────────────────────

    def sync_position(self):
        if not self.live_trade:
            return
        try:
            positions = self.exchange.fetch_positions(list(self.symbols.values()))
            for strat_name, symbol in self.symbols.items():
                state      = self.state[strat_name]
                active_pos = next(
                    (p for p in positions
                     if float(p.get('contracts', 0)) > 0
                     and (symbol in p['symbol'] or symbol.replace('/', '') in p['symbol'])),
                    None
                )
                if active_pos:
                    recently_closed = (time.time() - self._close_time.get(strat_name, 0)) < 15
                    if state['position'] == 0 and not recently_closed:
                        side                   = active_pos['side']
                        state['position']      = 1 if side == 'long' else -1
                        state['position_size'] = float(active_pos['contracts'])
                        state['entry_price']   = float(active_pos['entryPrice'])

                        tf    = self.timeframes[strat_name]
                        ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=500)
                        temp  = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
                        temp['ts'] = pd.to_datetime(temp['ts'], unit='ms')
                        temp.set_index('ts', inplace=True)
                        temp  = CoreStrategy.prepare_data(temp)

                        last_atr               = float(temp['ATR'].iloc[-1])
                        if strat_name == 'ADA':
                            sl_dist    = max(self.ADA_ATR_SL_MULT * last_atr, state['entry_price'] * self.min_sl_pct)
                            trail_dist = max(self.ADA_TRAIL_ATR * last_atr, state['entry_price'] * self.min_sl_pct)
                        elif strat_name == 'XRP':
                            sl_dist    = max(self.XRP_ATR_SL_MULT * last_atr, state['entry_price'] * self.min_sl_pct)
                            trail_dist = max(self.XRP_TRAIL_ATR * last_atr, state['entry_price'] * self.min_sl_pct)
                        elif strat_name == 'DOGE':
                            sl_dist    = max(self.DOGE_ATR_SL_MULT * last_atr, state['entry_price'] * self.min_sl_pct)
                            trail_dist = max(self.DOGE_TRAIL_ATR * last_atr, state['entry_price'] * self.min_sl_pct)
                        else:
                            sl_dist    = max(self.initial_sl_atr * last_atr, state['entry_price'] * self.min_sl_pct)
                            trail_dist = sl_dist
                        state['stop_loss']     = state['entry_price'] - sl_dist if state['position'] == 1 else state['entry_price'] + sl_dist
                        state['trailing_stop'] = state['entry_price'] - trail_dist if state['position'] == 1 else state['entry_price'] + trail_dist
                        state['highest_price'] = state['entry_price']
                        state['lowest_price']  = state['entry_price']
                        _lev = {'SOL': self.leverage, 'ADA': self.ADA_LEVERAGE, 'XRP': self.XRP_LEVERAGE, 'DOGE': self.DOGE_LEVERAGE}.get(strat_name, self.leverage)
                        state['liq_price']     = CoreStrategy.calc_liquidation_price(
                            state['entry_price'], state['position'], _lev, self.mmr)
                        print(f"🔄 {strat_name} 雲端接管 | {side.upper()} @ {state['entry_price']:.4f}")

                    if not state['stop_order_id']:
                        self.execute_order(strat_name,
                                           'sell' if state['position'] == 1 else 'buy',
                                           stop_price=state['stop_loss'])
                else:
                    if state['position'] != 0:
                        try:
                            exit_price = float(self.exchange.fetch_ticker(symbol)['last'])
                            gross        = (exit_price - state['entry_price']) * state['position_size'] * state['position']
                            efee         = state['entry_price'] * state['position_size'] * self.taker_fee_rate
                            xfee         = exit_price           * state['position_size'] * self.taker_fee_rate
                            net          = gross - efee - xfee
                            initial_risk = abs(state['entry_price'] - state['stop_loss']) * state['position_size']
                            r_str        = f"{net / initial_risk:+.2f}R" if initial_risk > 0 else "N/A R"
                            side_str     = "多單" if state['position'] == 1 else "空單"
                            print(f"🛑 {strat_name} [{side_str}] 交易所出場 | {exit_price:.4f} | 淨利: {net:.2f} U ({r_str})")
                            self.send_discord_msg(f"🛑 **{strat_name} 交易所出場** | {side_str} | {'✅' if net >= 0 else '❌'} {net:+.2f} U ({r_str})")
                            if net > 0:
                                self.show_milestone_progress(net)
                            self._update_loss_state(strat_name, net)
                        except Exception as e:
                            print(f"⚠️ {strat_name} 補算 PnL 失敗: {e}")
                        self.reset_position_state(strat_name)
                        self.save_order_state()
        except Exception as e:
            print(f"⚠️ sync_position 錯誤: {e}")

    def _print_dual_status(self, cl_sol, prev_sol, ada_dc_high=None,
                           ada_dc_low=None, ada_price=None):
        """每 5 分鐘印一次雙核心狀態，持倉/空倉各有不同資訊"""
        now_tw = datetime.now(timezone.utc) + timedelta(hours=8)
        sol_s  = self.state['SOL']
        ada_s  = self.state['ADA']

        # ── SOL 狀態行 ───────────────────────────────────────
        sol_price = float(cl_sol.close)
        if sol_s['position'] != 0:
            unr    = (sol_price - sol_s['entry_price']) * sol_s['position_size'] * sol_s['position']
            margin = (sol_s['entry_price'] * sol_s['position_size']) / self.leverage
            roe    = unr / margin * 100 if margin > 0 else 0
            pnl_icon = "🟢" if unr >= 0 else "🔴"
            side_str = "多單" if sol_s['position'] == 1 else "空單"
            sol_line = (f"🟡 SOL {side_str} | 現價: {sol_price:.2f} | "
                        f"{pnl_icon} {unr:+.2f}U ({roe:+.2f}%) | SL {sol_s['trailing_stop']:.2f}")
        else:
            cur_close = float(prev_sol.close)
            cur_ema   = float(prev_sol.EMA)
            cur_bb    = float(prev_sol.BB_Mid)
            cur_rsi   = float(prev_sol.RSI)
            cur_adx   = float(prev_sol.ADX) if hasattr(prev_sol, 'ADX') else 100.0
            if cur_close > cur_ema:
                adx_ok  = "✅" if cur_adx > self.adx_threshold else "❌"
                ema_ok  = "✅" if cur_close > cur_ema          else "❌"
                bb_ok   = "✅" if cur_close > cur_bb           else "❌"
                rsi_ok  = "✅" if cur_rsi < 58                 else "❌"
                sol_line = (f"🟡 SOL 空倉(偏多) | 現價: {sol_price:.2f} | "
                            f"ADX:{adx_ok} EMA200:{ema_ok} BB_Mid:{bb_ok} RSI(<58):{rsi_ok}")
            else:
                adx_ok  = "✅" if cur_adx > self.adx_threshold else "❌"
                ema_ok  = "✅" if cur_close < cur_ema           else "❌"
                bb_ok   = "✅" if cur_close < cur_bb            else "❌"
                rsi_ok  = "✅" if cur_rsi > 42                  else "❌"
                sol_line = (f"🟡 SOL 空倉(偏空) | 現價: {sol_price:.2f} | "
                            f"ADX:{adx_ok} EMA200:{ema_ok} BB_Mid:{bb_ok} RSI(>42):{rsi_ok}")

        # ── ADA 狀態行 ───────────────────────────────────────
        if ada_price is None:
            try:
                ada_price = float(self.exchange.fetch_ticker('ADA/USDT')['last']) if self.live_trade else 0.0
            except Exception:
                ada_price = 0.0

        if ada_s['position'] != 0:
            unr      = (ada_price - ada_s['entry_price']) * ada_s['position_size'] * ada_s['position']
            pnl_icon = "🟢" if unr >= 0 else "🔴"
            side_str = "多單" if ada_s['position'] == 1 else "空單"
            ada_line = (f"🔵 ADA {side_str} | 現價: {ada_price:.4f} | "
                        f"{pnl_icon} {unr:+.2f}U | SL {ada_s['trailing_stop']:.4f}")
        else:
            cb_tag = " 🛡️熔斷" if ada_s['skip_next_trade'] or ada_s['in_skip_zone'] else ""
            if ada_dc_high is not None and ada_dc_low is not None:
                ada_line = (f"🔵 ADA 空倉{cb_tag} | 現價: {ada_price:.4f} | "
                            f"DC高: {ada_dc_high:.4f} DC低: {ada_dc_low:.4f}")
            else:
                ada_line = f"🔵 ADA 空倉{cb_tag} | 現價: {ada_price:.4f} | 👁️ 待機"

        print(f"[{now_tw.strftime('%H:%M:%S')}] ── 雙核心狀態 ──────────────────────")
        print(f"  {sol_line}")
        print(f"  {ada_line}")

    # ── 統一狀態列印（每 5 分鐘，兩個幣一起印）──────────────────────

    def _print_status(self, cl_sol, ada_price: float = None,
                      ada_dc_high: float = None, ada_dc_low: float = None,
                      xrp_price: float = None, doge_price: float = None):
        """每 5 分鐘統一列印 SOL + ADA + XRP + DOGE 四核心狀態"""
        now_tw = datetime.now(timezone.utc) + timedelta(hours=8)
        ts_str = now_tw.strftime('%H:%M:%S')
        sol_s  = self.state['SOL']
        ada_s  = self.state['ADA']

        print("─" * 60)

        # ── SOL ──────────────────────────────────────────────────
        sol_price = float(cl_sol.close)
        if sol_s['position'] != 0:
            unr    = (sol_price - sol_s['entry_price']) * sol_s['position_size'] * sol_s['position']
            margin = (sol_s['entry_price'] * sol_s['position_size']) / self.leverage
            roe    = unr / margin * 100 if margin > 0 else 0
            pnl_e  = '🟢' if unr >= 0 else '🔴'
            side   = '多單' if sol_s['position'] == 1 else '空單'
            print(f"🟡 SOL [{ts_str}] {side} | 現價: {sol_price:.2f} | "
                  f"{pnl_e} {unr:+.2f} U ({roe:+.2f}%) | SL {sol_s['trailing_stop']:.2f}")
        else:
            cur_close = sol_price
            cur_ema   = float(cl_sol.EMA)
            cur_bb    = float(cl_sol.BB_Mid)
            cur_rsi   = float(cl_sol.RSI)
            cur_adx   = float(cl_sol.ADX) if hasattr(cl_sol, 'ADX') else 100.0

            if cur_close >= cur_ema:
                adx_ok = '✅' if cur_adx > self.adx_threshold else '❌'
                ema_ok = '✅' if cur_close > cur_ema   else '❌'
                bb_ok  = '✅' if cur_close > cur_bb    else '❌'
                rsi_ok = '✅' if cur_rsi < 58          else '❌'
                dir_label = '多方條件'
            else:
                adx_ok = '✅' if cur_adx > self.adx_threshold else '❌'
                ema_ok = '✅' if cur_close < cur_ema   else '❌'
                bb_ok  = '✅' if cur_close < cur_bb    else '❌'
                rsi_ok = '✅' if cur_rsi > 42          else '❌'
                dir_label = '空方條件'

            cb_tag = ' 🛡️熔斷' if sol_s['skip_next_trade'] or sol_s['in_skip_zone'] else ''
            print(f"🟡 SOL [{ts_str}] 空倉{cb_tag} | 現價: {sol_price:.2f} | "
                  f"[{dir_label}] ADX{adx_ok} EMA{ema_ok} BB{bb_ok} RSI{rsi_ok} "
                  f"(ADX:{cur_adx:.1f} RSI:{cur_rsi:.1f})")

        # ── ADA ──────────────────────────────────────────────────
        if ada_price is None:
            try:
                ada_price = float(self.exchange.fetch_ticker('ADA/USDT')['last']) if self.live_trade else 0.0
            except Exception:
                ada_price = 0.0

        if ada_s['position'] != 0:
            unr   = (ada_price - ada_s['entry_price']) * ada_s['position_size'] * ada_s['position']
            pnl_e = '🟢' if unr >= 0 else '🔴'
            side  = '多單' if ada_s['position'] == 1 else '空單'
            print(f"🔵 ADA [{ts_str}] {side} | 現價: {ada_price:.4f} | "
                  f"{pnl_e} {unr:+.2f} U | SL {ada_s['trailing_stop']:.4f}")
        else:
            cb_tag = ' 🛡️熔斷' if ada_s['skip_next_trade'] or ada_s['in_skip_zone'] else ''
            if ada_dc_high is not None and ada_dc_low is not None:
                print(f"🔵 ADA [{ts_str}] 空倉{cb_tag} | 現價: {ada_price:.4f} | "
                      f"DC高: {ada_dc_high:.4f} DC低: {ada_dc_low:.4f}")
            else:
                print(f"🔵 ADA [{ts_str}] 空倉{cb_tag} | 現價: {ada_price:.4f} | 👁️ 待機")

        # ── XRP ──────────────────────────────────────────────────
        xrp_s = self.state['XRP']
        if xrp_price is None:
            try:
                xrp_price = float(self.exchange.fetch_ticker('XRP/USDT')['last']) if self.live_trade else 0.0
            except Exception:
                xrp_price = 0.0

        if xrp_s['position'] != 0:
            unr   = (xrp_price - xrp_s['entry_price']) * xrp_s['position_size'] * xrp_s['position']
            pnl_e = '🟢' if unr >= 0 else '🔴'
            side  = '多單' if xrp_s['position'] == 1 else '空單'
            print(f"🟣 XRP [{ts_str}] {side} | 現價: {xrp_price:.4f} | "
                  f"{pnl_e} {unr:+.2f} U | SL {xrp_s['trailing_stop']:.4f}")
        else:
            cb_tag = ' 🛡️熔斷' if xrp_s['skip_next_trade'] or xrp_s['in_skip_zone'] else ''
            print(f"🟣 XRP [{ts_str}] 空倉{cb_tag} | 現價: {xrp_price:.4f} | 👁️ 待機")

        # ── DOGE ──────────────────────────────────────────────────
        doge_s = self.state['DOGE']
        if doge_price is None:
            try:
                doge_price = float(self.exchange.fetch_ticker('DOGE/USDT')['last']) if self.live_trade else 0.0
            except Exception:
                doge_price = 0.0

        if doge_s['position'] != 0:
            unr   = (doge_price - doge_s['entry_price']) * doge_s['position_size'] * doge_s['position']
            pnl_e = '🟢' if unr >= 0 else '🔴'
            side  = '多單' if doge_s['position'] == 1 else '空單'
            print(f"🟤 DOGE [{ts_str}] {side} | 現價: {doge_price:.5f} | "
                  f"{pnl_e} {unr:+.2f} U | SL {doge_s['trailing_stop']:.5f}")
        else:
            cb_tag = ' 🛡️熔斷' if doge_s['skip_next_trade'] or doge_s['in_skip_zone'] else ''
            print(f"🟤 DOGE [{ts_str}] 空倉{cb_tag} | 現價: {doge_price:.5f} | 👁️ 待機")

        print("─" * 60)

    # ── 主循環 ───────────────────────────────────────────────────────

    def _set_leverage_all(self):
        """啟動時對所有交易對強制設定槓桿，防止交易所預設值與 config 不符"""
        if not self.live_trade:
            return
        lev_map = {
            'SOL':  int(self.leverage),
            'ADA':  int(self.ADA_LEVERAGE),
            'XRP':  int(self.XRP_LEVERAGE),
            'DOGE': int(self.DOGE_LEVERAGE),
        }
        for strat_name, symbol in self.symbols.items():
            lev = lev_map.get(strat_name, 1)
            try:
                self.exchange.set_leverage(lev, symbol)
                print(f"  ✅ {strat_name} ({symbol}) 槓桿設為 {lev}x")
            except Exception as e:
                print(f"  ⚠️ {strat_name} 槓桿設定失敗: {e}")

    def run(self):
        print(f"⏳ 四核心輪詢啟動，每 {self.check_interval}s 一次")
        self._set_leverage_all()
        while True:
            try:
                self.sync_position()
                self.send_periodic_report()

                # ══ SOL 趨勢策略 ══════════════════════════════════
                ohlcv  = self.exchange.fetch_ohlcv('SOL/USDT', '15m', limit=500)
                df_sol = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                df_sol.set_index(pd.to_datetime(df_sol['timestamp'], unit='ms'), inplace=True)
                df_sol = CoreStrategy.prepare_data(df_sol)
                cl     = df_sol.iloc[-1]
                prev   = df_sol.iloc[-2]
                sol_s  = self.state['SOL']

                # 盤整縮緊 deque：每根新 K 棒只 append 一次已收盤的 prev（對齊回測）
                if cl.name != self._consol_last_ts:
                    self._consol_highs.append(float(prev.high))
                    self._consol_lows.append(float(prev.low))
                    self._consol_last_ts = cl.name

                sol_had_position = sol_s['position'] != 0   # 記錄此輪開始時是否有倉
                if sol_s['position'] != 0:
                    self.monitor_exit_sol(cl)
                    if sol_s['position'] == 0:
                        # 剛出場：標記此 K 棒已處理，防止同根 K 棒 5s 後再次偵測信號
                        self.last_candle_time['SOL'] = cl.name

                # 自適應 TWAP：後續子單（新 K 棒 + 持倉中 + 尚有剩餘子單）
                if (self.last_candle_time['SOL'] != cl.name and
                        sol_s['position'] != 0 and
                        self._sol_twap_active and self._sol_twap_remaining > 0 and
                        sol_s['position'] == self._sol_twap_direction):
                    self.last_candle_time['SOL'] = cl.name   # 標記此 K 棒已處理，防止重複觸發
                    if self.live_trade:
                        bal_data = self.exchange.fetch_balance()
                        free_bal = float(bal_data['free']['USDT'])
                    else:
                        free_bal = self.capital
                    max_by_margin = (free_bal * self.leverage) / float(cl.open)
                    sub_size = min(self._sol_twap_size_each, max_by_margin)
                    if sub_size > 0:
                        side = 'buy' if sol_s['position'] == 1 else 'sell'
                        if self.live_trade:
                            try:
                                sub_size_str = self.exchange.amount_to_precision('SOL/USDT', sub_size)
                                order = self.exchange.create_order('SOL/USDT', 'market', side, sub_size_str)
                                sub_ep = float(order.get('average') or cl.open)
                            except Exception as e:
                                print(f"⚠️ SOL TWAP 子單失敗: {e}")
                                sub_ep = None
                        else:
                            print(f"[模擬] SOL TWAP 追加 | {side.upper()} | size={sub_size:.4f} | 剩餘 {self._sol_twap_remaining}")
                            sub_ep = float(cl.open)
                        if sub_ep is not None:
                            total_sz = sol_s['position_size'] + sub_size
                            sol_s['entry_price']   = (sol_s['entry_price'] * sol_s['position_size'] + sub_ep * sub_size) / total_sz
                            sol_s['position_size'] = total_sz
                            sol_s['liq_price']     = CoreStrategy.calc_liquidation_price(
                                sol_s['entry_price'], sol_s['position'], self.leverage, self.mmr)
                            self._sol_twap_remaining -= 1
                            if self._sol_twap_remaining == 0:
                                self._sol_twap_active = False
                            self.send_discord_msg(
                                f"📦 SOL TWAP 子單 | 均價 {sol_s['entry_price']:.2f} | "
                                f"總量 {sol_s['position_size']:.4f} | 剩餘 {self._sol_twap_remaining} 筆")

                # 對齊回測邏輯：出場那根 K 棒不進場，下一根才允許（1 根間隔）
                # 首次啟動（None）時只記錄時間戳，不觸發信號，避免重啟後用前根K棒的陳舊信號進場
                if self.last_candle_time['SOL'] is None:
                    self.last_candle_time['SOL'] = cl.name
                elif self.last_candle_time['SOL'] != cl.name and sol_s['position'] == 0 and not sol_had_position:
                    self.last_candle_time['SOL'] = cl.name
                    now_tw = datetime.now(timezone.utc) + timedelta(hours=8)

                    l_cond, s_cond, _ = CoreStrategy.check_signals(prev, self.adx_threshold)

                    if sol_s['skip_next_trade']:
                        if l_cond or s_cond:
                            sol_s['skip_next_trade'] = False
                            sol_s['in_skip_zone']    = True
                            self.save_order_state()
                            self.send_discord_msg("🛡️ SOL 熔斷保護觸發，進入免疫區間")
                    elif sol_s['in_skip_zone']:
                        if not l_cond and not s_cond:
                            sol_s['in_skip_zone'] = False
                            self.save_order_state()
                            print("🔓 SOL 熔斷解除")
                    elif (l_cond or s_cond) and (time.time() - self._close_time.get('SOL', 0)) >= 60:
                        # 🔧 抓取總餘額(算風險)與可用餘額(算極限倉位)
                        if self.live_trade:
                            bal_data  = self.exchange.fetch_balance()
                            total_bal = float(bal_data['total']['USDT'])
                            free_bal  = float(bal_data['free']['USDT'])
                        else:
                            total_bal = free_bal = self.capital

                        sl_dist = max(self.initial_sl_atr * float(prev.ATR), float(cl.open) * self.min_sl_pct)

                        # 自適應 TWAP：計算 N，決定首筆下單量
                        _N = min(max(1, int(total_bal / self.max_trade_usdt_cap)),
                                 int(self.max_position_cap / self.max_trade_usdt_cap))
                        if _N == 1:
                            # 可用保證金 × 40%（free_bal 已扣除其他幣種鎖倉保證金）
                            size = (free_bal * 0.40 * self.leverage) / float(cl.open)
                        else:
                            size = self.max_trade_usdt_cap / float(cl.open)

                        if l_cond:
                            order = self.execute_order('SOL', 'buy', size, reason="SOL 做多進場")
                            if order is None:
                                print("⚠️ SOL 做多市價單失敗，放棄本次進場")
                            else:
                                ep    = float(order.get('average') or cl.open)
                                sol_s['position']      = 1
                                sol_s['position_size'] = size
                                sol_s['entry_price']   = ep
                                sol_s['stop_loss']     = sol_s['trailing_stop'] = ep - sl_dist
                                sol_s['highest_price'] = ep
                                sol_s['liq_price']     = CoreStrategy.calc_liquidation_price(ep, 1, self.leverage, self.mmr)
                                self.execute_order('SOL', 'sell', stop_price=sol_s['stop_loss'])
                        else:
                            order = self.execute_order('SOL', 'sell', size, reason="SOL 做空進場")
                            if order is None:
                                print("⚠️ SOL 做空市價單失敗，放棄本次進場")
                            else:
                                ep    = float(order.get('average') or cl.open)
                                sol_s['position']      = -1
                                sol_s['position_size'] = size
                                sol_s['entry_price']   = ep
                                sol_s['stop_loss']     = sol_s['trailing_stop'] = ep + sl_dist
                                sol_s['lowest_price']  = ep
                                sol_s['liq_price']     = CoreStrategy.calc_liquidation_price(ep, -1, self.leverage, self.mmr)
                                self.execute_order('SOL', 'buy', stop_price=sol_s['stop_loss'])

                        if sol_s['position'] != 0:
                            # 設定 TWAP 狀態（首筆已進，剩餘 N-1 筆待追加）
                            self._sol_twap_direction = sol_s['position']
                            self._sol_twap_size_each = (self.max_trade_usdt_cap / sol_s['entry_price']) if _N > 1 else size
                            self._sol_twap_remaining = _N - 1
                            self._sol_twap_active    = self._sol_twap_remaining > 0
                            self.save_order_state()
                            time.sleep(1)   # 等待交易所餘額刷新，避免下一策略抓到幽靈餘額


                # ══ ADA Donchian 突破策略 ══════════════════════════
                ada_s = self.state['ADA']

                try:
                    ohlcv_ada = self.exchange.fetch_ohlcv('ADA/USDT', '1h', limit=500)
                    df_ada = pd.DataFrame(ohlcv_ada, columns=['timestamp','open','high','low','close','volume'])
                    df_ada.set_index(pd.to_datetime(df_ada['timestamp'], unit='ms'), inplace=True)
                    df_ada = CoreStrategy.prepare_data(df_ada)
                except Exception as e:
                    print(f"⚠️ ADA 資料抓取失敗: {e}")
                    time.sleep(self.check_interval)
                    continue

                cl_ada   = df_ada.iloc[-1]
                prev_ada = df_ada.iloc[-2]
                ada_close = float(cl_ada.close)
                ada_atr   = float(cl_ada.ATR)

                # Donchian 通道（prev_ada 前 N 根，不含 prev_ada 自身）
                ada_highs = df_ada['high'].values
                ada_lows  = df_ada['low'].values
                prev_idx  = len(df_ada) - 2   # prev_ada 的位置
                if prev_idx >= self.ADA_ENTRY_N:
                    dc_high = float(np.max(ada_highs[prev_idx - self.ADA_ENTRY_N:prev_idx]))
                    dc_low  = float(np.min(ada_lows[prev_idx - self.ADA_ENTRY_N:prev_idx]))
                else:
                    dc_high = dc_low = ada_close

                # ── 盤整縮緊 deque：每根新 K 棒只 append 一次已收盤的 prev（對齊回測）
                if ada_s['position'] != 0 and cl_ada.name != self._ada_consol_last_ts:
                    self._ada_consol_highs.append(float(prev_ada.high))
                    self._ada_consol_lows.append(float(prev_ada.low))
                    self._ada_consol_last_ts = cl_ada.name

                # ── 持倉中：監控出場 + TWAP 追加 ─────────────────
                ada_had_position = ada_s['position'] != 0
                if ada_s['position'] != 0:
                    self.monitor_exit_ada(float(cl_ada.high), float(cl_ada.low),
                                          float(cl_ada.open), ada_atr)
                    if ada_s['position'] == 0:
                        self.last_candle_time['ADA'] = cl_ada.name

                # ADA TWAP 後續子單（新 K 棒 + 持倉中 + 尚有剩餘子單）
                if (self.last_candle_time['ADA'] != cl_ada.name and
                        ada_s['position'] != 0 and
                        self._ada_twap_active and self._ada_twap_remaining > 0 and
                        ada_s['position'] == self._ada_twap_direction):
                    self.last_candle_time['ADA'] = cl_ada.name
                    if self.live_trade:
                        bal_data = self.exchange.fetch_balance()
                        free_bal = float(bal_data['free']['USDT'])
                    else:
                        free_bal = self.capital
                    max_by_margin = (free_bal * self.ADA_LEVERAGE) / float(cl_ada.open)
                    sub_size = min(self._ada_twap_size_each, max_by_margin)
                    if sub_size > 0:
                        side = 'buy' if ada_s['position'] == 1 else 'sell'
                        if self.live_trade:
                            try:
                                sub_size_str = self.exchange.amount_to_precision('ADA/USDT', sub_size)
                                order = self.exchange.create_order('ADA/USDT', 'market', side, sub_size_str)
                                sub_ep = float(order.get('average') or cl_ada.open)
                            except Exception as e:
                                print(f"⚠️ ADA TWAP 子單失敗: {e}")
                                sub_ep = None
                        else:
                            print(f"[模擬] ADA TWAP 追加 | {side.upper()} | size={sub_size:.2f} | 剩餘 {self._ada_twap_remaining}")
                            sub_ep = float(cl_ada.open)
                        if sub_ep is not None:
                            total_sz = ada_s['position_size'] + sub_size
                            ada_s['entry_price']   = (ada_s['entry_price'] * ada_s['position_size'] + sub_ep * sub_size) / total_sz
                            ada_s['position_size'] = total_sz
                            ada_s['liq_price']     = CoreStrategy.calc_liquidation_price(
                                ada_s['entry_price'], ada_s['position'], self.ADA_LEVERAGE, self.mmr)
                            self._ada_twap_remaining -= 1
                            if self._ada_twap_remaining == 0:
                                self._ada_twap_active = False
                            self.send_discord_msg(
                                f"📦 ADA TWAP 子單 | 均價 {ada_s['entry_price']:.4f} | "
                                f"總量 {ada_s['position_size']:.2f} | 剩餘 {self._ada_twap_remaining} 筆")

                # ── 空倉入場：Donchian 突破 ──────────────────────
                if self.last_candle_time['ADA'] is None:
                    self.last_candle_time['ADA'] = cl_ada.name
                elif self.last_candle_time['ADA'] != cl_ada.name and ada_s['position'] == 0 and not ada_had_position:
                    self.last_candle_time['ADA'] = cl_ada.name

                    # 用已收盤 K 棒（prev_ada）的收盤價判斷突破
                    prev_close = float(prev_ada.close)
                    direction = 0
                    if prev_close > dc_high:
                        direction = 1
                    elif prev_close < dc_low:
                        direction = -1

                    # 熔斷邏輯
                    if ada_s['skip_next_trade']:
                        if direction != 0:
                            ada_s['skip_next_trade'] = False
                            ada_s['in_skip_zone']    = True
                            self.save_order_state()
                            self.send_discord_msg("🛡️ ADA 熔斷保護觸發，進入免疫區間")
                    elif ada_s['in_skip_zone']:
                        if direction == 0:
                            ada_s['in_skip_zone'] = False
                            self.save_order_state()
                            print("🔓 ADA 熔斷解除")
                    elif direction != 0 and (time.time() - self._close_time.get('ADA', 0)) >= 60:
                        prev_atr = float(prev_ada.ATR)
                        risk_per_unit = self.ADA_ATR_SL_MULT * prev_atr
                        if risk_per_unit > 0:
                            if self.live_trade:
                                bal_data  = self.exchange.fetch_balance()
                                total_bal = float(bal_data['total']['USDT'])
                                free_bal  = float(bal_data['free']['USDT'])
                            else:
                                total_bal = free_bal = self.capital

                            # 自適應 TWAP：計算 N
                            _N = max(1, int(total_bal * self.ADA_LEVERAGE / self.ADA_MAX_TRADE_CAP))
                            if _N == 1:
                                # 可用保證金 × 40%（free_bal 已扣除其他幣種鎖倉保證金）
                                size = (free_bal * 0.40 * self.ADA_LEVERAGE) / float(cl_ada.open)
                            else:
                                size = self.ADA_MAX_TRADE_CAP / float(cl_ada.open)

                            sl_dist    = max(self.ADA_ATR_SL_MULT * prev_atr, float(cl_ada.open) * self.min_sl_pct)
                            trail_dist = max(self.ADA_TRAIL_ATR * prev_atr, float(cl_ada.open) * self.min_sl_pct)

                            if size > 0:
                                now_tw = datetime.now(timezone.utc) + timedelta(hours=8)
                                print(f"[{now_tw.strftime('%H:%M:%S')}] 🟢 ADA Donchian 訊號 | "
                                      f"{'做多' if direction == 1 else '做空'} | DC高: {dc_high:.4f} DC低: {dc_low:.4f}")

                                if direction == 1:
                                    order = self.execute_order('ADA', 'buy', size, reason="ADA Donchian 做多")
                                    if order is None:
                                        print("⚠️ ADA 做多市價單失敗，放棄本次進場")
                                    else:
                                        ep = float(order.get('average') or cl_ada.open)
                                        ada_s['position']      = 1
                                        ada_s['position_size'] = size
                                        ada_s['entry_price']   = ep
                                        ada_s['stop_loss']     = ep - sl_dist       # 2.0× ATR（硬止損 + 倉位計算）
                                        ada_s['trailing_stop'] = ep - trail_dist     # 3.0× ATR（對齊回測）
                                        ada_s['highest_price'] = ep
                                        ada_s['liq_price']     = CoreStrategy.calc_liquidation_price(ep, 1, self.ADA_LEVERAGE, self.mmr)
                                        self.execute_order('ADA', 'sell', stop_price=ada_s['stop_loss'])
                                else:
                                    order = self.execute_order('ADA', 'sell', size, reason="ADA Donchian 做空")
                                    if order is None:
                                        print("⚠️ ADA 做空市價單失敗，放棄本次進場")
                                    else:
                                        ep = float(order.get('average') or cl_ada.open)
                                        ada_s['position']      = -1
                                        ada_s['position_size'] = size
                                        ada_s['entry_price']   = ep
                                        ada_s['stop_loss']     = ep + sl_dist       # 2.0× ATR（硬止損 + 倉位計算）
                                        ada_s['trailing_stop'] = ep + trail_dist     # 3.0× ATR（對齊回測）
                                        ada_s['lowest_price']  = ep
                                        ada_s['liq_price']     = CoreStrategy.calc_liquidation_price(ep, -1, self.ADA_LEVERAGE, self.mmr)
                                        self.execute_order('ADA', 'buy', stop_price=ada_s['stop_loss'])

                                if ada_s['position'] != 0:
                                    self._ada_twap_direction = ada_s['position']
                                    self._ada_twap_size_each = (self.ADA_MAX_TRADE_CAP / ada_s['entry_price']) if _N > 1 else size
                                    self._ada_twap_remaining = _N - 1
                                    self._ada_twap_active    = self._ada_twap_remaining > 0
                                    self.save_order_state()
                                    time.sleep(1)   # 等待交易所餘額刷新，避免下一策略抓到幽靈餘額

                                    msg = (
                                        f"🚀 **ADA Donchian 進場** | {'做多' if direction==1 else '做空'}\n"
                                        f"   進場: {ada_s['entry_price']:.4f} | SL: {ada_s['stop_loss']:.4f}\n"
                                        f"   size: {ada_s['position_size']:.2f} | TWAP剩餘: {self._ada_twap_remaining}"
                                    )
                                    self.send_discord_msg(msg)

                # ══ XRP 斐波策略 ══════════════════════════════════
                xrp_s = self.state['XRP']

                try:
                    ohlcv_xrp = self.exchange.fetch_ohlcv('XRP/USDT', '1h', limit=500)
                    df_xrp = pd.DataFrame(ohlcv_xrp, columns=['timestamp','open','high','low','close','volume'])
                    df_xrp.set_index(pd.to_datetime(df_xrp['timestamp'], unit='ms'), inplace=True)
                    df_xrp = CoreStrategy.prepare_data(df_xrp)
                except Exception as e:
                    print(f"⚠️ XRP 資料抓取失敗: {e}")
                    time.sleep(self.check_interval)
                    continue

                cl_xrp   = df_xrp.iloc[-1]
                prev_xrp = df_xrp.iloc[-2]
                xrp_close = float(cl_xrp.close)
                xrp_atr   = float(cl_xrp.ATR)
                xrp_ema   = float(prev_xrp.EMA)

                # Swing 高低點（prev_xrp 前 N 根，不含 prev_xrp 自身）
                xrp_highs = df_xrp['high'].values
                xrp_lows  = df_xrp['low'].values
                xrp_closes_arr = df_xrp['close'].values
                prev_xrp_idx = len(df_xrp) - 2
                if prev_xrp_idx >= self.XRP_SWING_N:
                    _win_xh    = xrp_highs[prev_xrp_idx - self.XRP_SWING_N:prev_xrp_idx]
                    _win_xl    = xrp_lows[prev_xrp_idx - self.XRP_SWING_N:prev_xrp_idx]
                    swing_high = float(np.max(_win_xh))
                    swing_low  = float(np.min(_win_xl))
                    _xhi_idx   = int(np.argmax(_win_xh))
                    _xlo_idx   = int(np.argmin(_win_xl))
                else:
                    swing_high = swing_low = xrp_close
                    _xhi_idx = _xlo_idx = 0

                # ── 盤整縮緊 deque：每根新 K 棒只 append 一次已收盤的 prev（對齊回測）
                if xrp_s['position'] != 0 and cl_xrp.name != self._xrp_consol_last_ts:
                    self._xrp_consol_highs.append(float(prev_xrp.high))
                    self._xrp_consol_lows.append(float(prev_xrp.low))
                    self._xrp_consol_last_ts = cl_xrp.name

                # ── 持倉中：監控出場 + TWAP 追加 ─────────────────
                xrp_had_position = xrp_s['position'] != 0
                if xrp_s['position'] != 0:
                    self.monitor_exit_xrp(float(cl_xrp.high), float(cl_xrp.low),
                                          float(cl_xrp.open), xrp_atr)
                    if xrp_s['position'] == 0:
                        self.last_candle_time['XRP'] = cl_xrp.name

                # ── 限價單成交確認（每根新 K 棒檢查一次）────────────────
                if (self.last_candle_time['XRP'] != cl_xrp.name and
                        xrp_s['position'] == 0 and
                        self._xrp_limit_order_id is not None):
                    filled_price = None

                    if self.live_trade:
                        # 向交易所查詢訂單狀態
                        try:
                            ord_info = self.exchange.fetch_order(self._xrp_limit_order_id, 'XRP/USDT')
                            if ord_info['status'] == 'closed':
                                filled_price = float(ord_info.get('average') or ord_info.get('price') or self._xrp_limit_price)
                        except Exception as e:
                            print(f"⚠️ XRP 限價單查詢失敗: {e}")
                            # 訂單不存在（已被成交或外部取消）→ 清除掛單狀態，避免無限重試
                            if any(k in str(e) for k in ['-2013', 'does not exist', 'order not found', 'Order does not exist']):
                                print("⚠️ XRP 限價單不存在，清除掛單狀態")
                                self._xrp_limit_order_id  = None
                                self._xrp_limit_price     = 0.0
                                self._xrp_limit_direction = 0
                                self._xrp_limit_size      = 0.0
                                self._xrp_limit_placed_ts = 0.0
                                self.save_order_state()
                    else:
                        # 模擬：用前一根 K 棒的 L/H 判斷是否成交
                        p_L = float(prev_xrp.low)
                        p_H = float(prev_xrp.high)
                        if self._xrp_limit_direction == 1 and p_L <= self._xrp_limit_price:
                            filled_price = self._xrp_limit_price
                        elif self._xrp_limit_direction == -1 and p_H >= self._xrp_limit_price:
                            filled_price = self._xrp_limit_price

                    if filled_price is not None:
                        # 限價單成交 → 建立倉位
                        ep = filled_price
                        prev_atr   = float(prev_xrp.ATR)
                        sl_dist    = max(self.XRP_ATR_SL_MULT * prev_atr, ep * self.min_sl_pct)
                        trail_dist = max(self.XRP_TRAIL_ATR  * prev_atr, ep * self.min_sl_pct)
                        xrp_s['position']      = self._xrp_limit_direction
                        xrp_s['position_size'] = self._xrp_limit_size
                        xrp_s['entry_price']   = ep
                        if self._xrp_limit_direction == 1:
                            xrp_s['stop_loss']     = ep - sl_dist
                            xrp_s['trailing_stop'] = ep - trail_dist
                            xrp_s['highest_price'] = ep
                        else:
                            xrp_s['stop_loss']     = ep + sl_dist
                            xrp_s['trailing_stop'] = ep + trail_dist
                            xrp_s['lowest_price']  = ep
                        xrp_s['liq_price'] = CoreStrategy.calc_liquidation_price(ep, self._xrp_limit_direction, self.XRP_LEVERAGE, self.mmr)
                        # 掛硬性止損
                        sl_side = 'sell' if self._xrp_limit_direction == 1 else 'buy'
                        self.execute_order('XRP', sl_side, stop_price=xrp_s['stop_loss'])
                        # 清除掛單狀態
                        self._xrp_limit_order_id  = None
                        self._xrp_limit_price     = 0.0
                        self._xrp_limit_direction = 0
                        self._xrp_limit_size      = 0.0
                        self._xrp_limit_placed_ts = 0.0
                        self.save_order_state()
                        now_tw = datetime.now(timezone.utc) + timedelta(hours=8)
                        msg = (
                            f"✅ **XRP 限價單成交** | {'做多' if xrp_s['position']==1 else '做空'}\n"
                            f"   成交: {ep:.4f} | SL: {xrp_s['stop_loss']:.4f}\n"
                            f"   size: {xrp_s['position_size']:.2f}"
                        )
                        print(f"[{now_tw.strftime('%H:%M:%S')}] {msg}")
                        self.send_discord_msg(msg)

                    # ── 逾時自動取消（未成交時檢查）────────────────────
                    elif (self.XRP_LIMIT_MAX_HOURS > 0 and
                            self._xrp_limit_placed_ts > 0 and
                            self._xrp_limit_order_id is not None and
                            time.time() - self._xrp_limit_placed_ts > self.XRP_LIMIT_MAX_HOURS * 3600):
                        if self.live_trade and self._xrp_limit_order_id != 'pending':
                            try:
                                self.exchange.cancel_order(self._xrp_limit_order_id, 'XRP/USDT')
                            except Exception:
                                pass
                        self._xrp_limit_order_id  = None
                        self._xrp_limit_price     = 0.0
                        self._xrp_limit_direction = 0
                        self._xrp_limit_size      = 0.0
                        self._xrp_limit_placed_ts = 0.0
                        self.save_order_state()
                        now_tw = datetime.now(timezone.utc) + timedelta(hours=8)
                        print(f"[{now_tw.strftime('%H:%M:%S')}] ⏰ XRP 限價單逾時 {self.XRP_LIMIT_MAX_HOURS}h，自動取消")
                        self.send_discord_msg(f"⏰ XRP 限價單逾時 {self.XRP_LIMIT_MAX_HOURS}h，自動取消")

                # ── 空倉掛限價單：Fib 水平（方案A：無確認）────────────
                if self.last_candle_time['XRP'] is None:
                    self.last_candle_time['XRP'] = cl_xrp.name
                elif self.last_candle_time['XRP'] != cl_xrp.name and xrp_s['position'] == 0 and not xrp_had_position:
                    self.last_candle_time['XRP'] = cl_xrp.name

                    swing_range = swing_high - swing_low
                    direction   = 0
                    fib_price   = 0.0

                    p_C = float(prev_xrp.close)
                    if swing_range > 0 and xrp_atr > 0:
                        if p_C > xrp_ema and _xhi_idx > _xlo_idx:   # 上升趨勢
                            fib_price = swing_high - self.XRP_FIB_LEVEL * swing_range
                            direction = 1
                        elif p_C < xrp_ema and _xlo_idx > _xhi_idx:  # 下降趨勢
                            fib_price = swing_low + self.XRP_FIB_LEVEL * swing_range
                            direction = -1

                    # 熔斷邏輯
                    if xrp_s['skip_next_trade']:
                        if direction != 0:
                            xrp_s['skip_next_trade'] = False
                            xrp_s['in_skip_zone']    = True
                            self.save_order_state()
                            self.send_discord_msg("🛡️ XRP 熔斷保護觸發，進入免疫區間")
                        direction = 0  # 熔斷期間不下單
                    elif xrp_s['in_skip_zone']:
                        if direction == 0:
                            xrp_s['in_skip_zone'] = False
                            self.save_order_state()
                            print("🔓 XRP 熔斷解除")
                        direction = 0  # 免疫區間不下單

                    # 平倉後 60 秒內不重新下單
                    if (time.time() - self._close_time.get('XRP', 0)) < 60:
                        direction = 0

                    # ── 取消方向相反或無效的舊掛單 ─────────────────────
                    if self._xrp_limit_order_id is not None:
                        if direction == 0 or direction != self._xrp_limit_direction:
                            # 趨勢反轉或消失 → 取消掛單
                            if self.live_trade:
                                try:
                                    self.exchange.cancel_order(self._xrp_limit_order_id, 'XRP/USDT')
                                    print(f"🗑️ XRP 限價單取消（趨勢改變）")
                                except Exception as e:
                                    print(f"⚠️ XRP 限價單取消失敗: {e}")
                            else:
                                print(f"[模擬] XRP 限價單取消（趨勢改變）")
                            self._xrp_limit_order_id  = None
                            self._xrp_limit_price     = 0.0
                            self._xrp_limit_direction = 0
                            self._xrp_limit_size      = 0.0
                            self._xrp_limit_placed_ts = 0.0
                            self.save_order_state()
                        elif abs(fib_price - self._xrp_limit_price) / self._xrp_limit_price > self.XRP_FIB_TOL * 2:
                            # Fib 水平偏移超過容差 → 取消後重新掛
                            if self.live_trade:
                                try:
                                    self.exchange.cancel_order(self._xrp_limit_order_id, 'XRP/USDT')
                                except Exception:
                                    pass
                            # 清除舊掛單狀態並持久化，避免重啟後帶著失效 ID
                            self._xrp_limit_order_id  = None
                            self._xrp_limit_price     = 0.0
                            self._xrp_limit_direction = 0
                            self._xrp_limit_size      = 0.0
                            self._xrp_limit_placed_ts = 0.0
                            self.save_order_state()

                    # ── 掛新限價單（無掛單且方向有效）──────────────────
                    if direction != 0 and self._xrp_limit_order_id is None and fib_price > 0:
                        prev_atr      = float(prev_xrp.ATR)
                        risk_per_unit = self.XRP_ATR_SL_MULT * prev_atr
                        if risk_per_unit > 0:
                            if self.live_trade:
                                bal_data  = self.exchange.fetch_balance()
                                total_bal = float(bal_data['total']['USDT'])
                                free_bal  = float(bal_data['free']['USDT'])
                            else:
                                total_bal = free_bal = self.capital

                            # 可用保證金 × 40%（free_bal 已扣除其他幣種鎖倉保證金）
                            size = (free_bal * 0.40 * self.XRP_LEVERAGE) / fib_price

                            if size > 0:
                                side = 'buy' if direction == 1 else 'sell'
                                now_tw = datetime.now(timezone.utc) + timedelta(hours=8)

                                if self.live_trade:
                                    try:
                                        size_str  = self.exchange.amount_to_precision('XRP/USDT', size)
                                        price_str = self.exchange.price_to_precision('XRP/USDT', fib_price)
                                        order = self.exchange.create_order('XRP/USDT', 'limit', side, size_str, price_str)
                                        self._xrp_limit_order_id = order['id']
                                    except Exception as e:
                                        print(f"⚠️ XRP 限價單掛出失敗: {e}")
                                        self._xrp_limit_order_id = None
                                else:
                                    self._xrp_limit_order_id = 'pending'

                                if self._xrp_limit_order_id is not None:
                                    self._xrp_limit_price     = fib_price
                                    self._xrp_limit_direction = direction
                                    self._xrp_limit_size      = size
                                    self._xrp_limit_placed_ts = time.time()
                                    self.save_order_state()
                                    print(f"[{now_tw.strftime('%H:%M:%S')}] 📋 XRP 限價單掛出 | "
                                          f"{'做多' if direction==1 else '做空'} @ {fib_price:.4f} | "
                                          f"size={size:.2f} | Swing: {swing_high:.4f}/{swing_low:.4f}")
                                    self.send_discord_msg(
                                        f"📋 **XRP 限價掛單** | {'做多' if direction==1 else '做空'}\n"
                                        f"   掛單: {fib_price:.4f} | size: {size:.2f}\n"
                                        f"   Swing: {swing_high:.4f} / {swing_low:.4f}"
                                    )

                # ══ DOGE BB Squeeze 策略 ══════════════════════════
                doge_s = self.state['DOGE']

                try:
                    ohlcv_doge = self.exchange.fetch_ohlcv('DOGE/USDT', '1h', limit=500)
                    df_doge = pd.DataFrame(ohlcv_doge, columns=['timestamp','open','high','low','close','volume'])
                    df_doge.set_index(pd.to_datetime(df_doge['timestamp'], unit='ms'), inplace=True)
                    df_doge = CoreStrategy.prepare_data(df_doge)
                except Exception as e:
                    print(f"⚠️ DOGE 資料抓取失敗: {e}")
                    time.sleep(self.check_interval)
                    continue

                # ── 計算 BB / KC / Momentum / Squeeze ──────────────
                doge_close_arr = df_doge['close'].values

                # BB（布林帶）
                bb_ser = pd.Series(doge_close_arr)
                bb_mid = bb_ser.rolling(self.DOGE_BB_PERIOD).mean()
                bb_std = bb_ser.rolling(self.DOGE_BB_PERIOD).std(ddof=0)
                bb_up  = bb_mid + self.DOGE_BB_STD * bb_std
                bb_lo  = bb_mid - self.DOGE_BB_STD * bb_std

                # KC（Keltner Channel）：用 CoreStrategy.prepare_data 計算的 ATR
                doge_atr_arr = df_doge['ATR'].values
                kc_mid_ser   = bb_ser.ewm(span=self.DOGE_KC_PERIOD, adjust=False).mean()
                kc_mid = kc_mid_ser.values
                kc_up  = kc_mid + self.DOGE_KC_MULT * doge_atr_arr
                kc_lo  = kc_mid - self.DOGE_KC_MULT * doge_atr_arr

                # Momentum
                mom_arr = bb_ser.diff(self.DOGE_MOM_PERIOD).values

                # Squeeze：BB 在 KC 內 = squeeze=1；釋放 = squeeze=0
                sq_arr = np.where(
                    (bb_up.values <= kc_up) & (bb_lo.values >= kc_lo), 1, 0
                )

                cl_doge   = df_doge.iloc[-1]
                prev_doge = df_doge.iloc[-2]
                doge_close = float(cl_doge.close)
                doge_atr   = float(cl_doge.ATR)
                di         = len(df_doge) - 1   # current bar index

                # ── 盤整縮緊 deque：每根新 K 棒只 append 一次已收盤的 prev（對齊回測）
                if doge_s['position'] != 0 and cl_doge.name != self._doge_consol_last_ts:
                    self._doge_consol_highs.append(float(prev_doge.high))
                    self._doge_consol_lows.append(float(prev_doge.low))
                    self._doge_consol_last_ts = cl_doge.name

                # ── 持倉中：監控出場 ──────────────────────────────
                doge_had_position = doge_s['position'] != 0
                if doge_s['position'] != 0:
                    self.monitor_exit_doge(float(cl_doge.high), float(cl_doge.low),
                                           float(cl_doge.open), doge_atr)
                    if doge_s['position'] == 0:
                        self.last_candle_time['DOGE'] = cl_doge.name

                # ── DOGE TWAP 後續子單（新 K 棒 + 持倉中 + 尚有剩餘子單）─
                if (self.last_candle_time['DOGE'] != cl_doge.name and
                        doge_s['position'] != 0 and
                        self._doge_twap_active and self._doge_twap_remaining > 0 and
                        doge_s['position'] == self._doge_twap_direction):
                    self.last_candle_time['DOGE'] = cl_doge.name
                    if self.live_trade:
                        bal_data = self.exchange.fetch_balance()
                        free_bal = float(bal_data['free']['USDT'])
                    else:
                        free_bal = self.capital
                    max_by_margin = (free_bal * self.DOGE_LEVERAGE) / float(cl_doge.open)
                    sub_size = min(self._doge_twap_size_each, max_by_margin)
                    if sub_size > 0:
                        side = 'buy' if doge_s['position'] == 1 else 'sell'
                        if self.live_trade:
                            try:
                                sub_size_str = self.exchange.amount_to_precision('DOGE/USDT', sub_size)
                                order = self.exchange.create_order('DOGE/USDT', 'market', side, sub_size_str)
                                sub_ep = float(order.get('average') or cl_doge.open)
                            except Exception as e:
                                print(f"⚠️ DOGE TWAP 子單失敗: {e}")
                                sub_ep = None
                        else:
                            print(f"[模擬] DOGE TWAP 追加 | {side.upper()} | size={sub_size:.0f} | 剩餘 {self._doge_twap_remaining}")
                            sub_ep = float(cl_doge.open)
                        if sub_ep is not None:
                            total_sz = doge_s['position_size'] + sub_size
                            doge_s['entry_price']   = (doge_s['entry_price'] * doge_s['position_size'] + sub_ep * sub_size) / total_sz
                            doge_s['position_size'] = total_sz
                            doge_s['liq_price']     = CoreStrategy.calc_liquidation_price(
                                doge_s['entry_price'], doge_s['position'], self.DOGE_LEVERAGE, self.mmr)
                            self._doge_twap_remaining -= 1
                            if self._doge_twap_remaining == 0:
                                self._doge_twap_active = False
                            self.send_discord_msg(
                                f"📦 DOGE TWAP 子單 | 均價 {doge_s['entry_price']:.5f} | "
                                f"總量 {doge_s['position_size']:.0f} | 剩餘 {self._doge_twap_remaining} 筆")

                # ── 空倉入場：BB Squeeze 釋放訊號 ───────────────────
                if self.last_candle_time['DOGE'] is None:
                    self.last_candle_time['DOGE'] = cl_doge.name
                elif self.last_candle_time['DOGE'] != cl_doge.name and doge_s['position'] == 0 and not doge_had_position:
                    self.last_candle_time['DOGE'] = cl_doge.name

                    # 用已收盤 K 棒（prev_doge，即 di-1）偵測 squeeze fire
                    # fire = 前根壓縮（sq=1）、本根釋放（sq=0）
                    pi = di - 1   # prev_doge 的 index
                    is_fire = (pi >= 1 and
                               sq_arr[pi - 1] == 1 and sq_arr[pi] == 0 and
                               not np.isnan(doge_atr_arr[pi]) and doge_atr_arr[pi] > 0)

                    direction = 0
                    if is_fire:
                        direction = 1 if mom_arr[pi] > 0 else -1

                    # 熔斷邏輯
                    if doge_s['skip_next_trade']:
                        if direction != 0:
                            doge_s['skip_next_trade'] = False
                            doge_s['in_skip_zone']    = True
                            self.save_order_state()
                            self.send_discord_msg("🛡️ DOGE 熔斷保護觸發，進入免疫區間")
                        direction = 0
                    elif doge_s['in_skip_zone']:
                        if direction == 0:
                            doge_s['in_skip_zone'] = False
                            self.save_order_state()
                            print("🔓 DOGE 熔斷解除")
                        direction = 0

                    # 平倉後 60 秒內不重新下單
                    if (time.time() - self._close_time.get('DOGE', 0)) < 60:
                        direction = 0

                    if direction != 0:
                        d_atr_prev = float(doge_atr_arr[pi])   # 前根 ATR（無前視）
                        risk_per_unit = self.DOGE_ATR_SL_MULT * d_atr_prev
                        if risk_per_unit > 0:
                            if self.live_trade:
                                bal_data  = self.exchange.fetch_balance()
                                total_bal = float(bal_data['total']['USDT'])
                                free_bal  = float(bal_data['free']['USDT'])
                            else:
                                total_bal = free_bal = self.capital

                            # 自適應 TWAP：計算 N
                            _N = max(1, int(total_bal * self.DOGE_LEVERAGE / self.DOGE_MAX_TRADE_CAP))
                            if _N == 1:
                                # 可用保證金 × 40%（free_bal 已扣除其他幣種鎖倉保證金）
                                size = (free_bal * 0.40 * self.DOGE_LEVERAGE) / float(cl_doge.open)
                            else:
                                size = self.DOGE_MAX_TRADE_CAP / float(cl_doge.open)

                            sl_dist    = max(self.DOGE_ATR_SL_MULT * d_atr_prev, float(cl_doge.open) * self.min_sl_pct)
                            trail_dist = max(self.DOGE_TRAIL_ATR * d_atr_prev, float(cl_doge.open) * self.min_sl_pct)

                            if size > 0:
                                now_tw = datetime.now(timezone.utc) + timedelta(hours=8)
                                print(f"[{now_tw.strftime('%H:%M:%S')}] 🟤 DOGE Squeeze 訊號 | "
                                      f"{'做多' if direction == 1 else '做空'} | Mom: {mom_arr[pi]:.6f}")

                                if direction == 1:
                                    order = self.execute_order('DOGE', 'buy', size, reason="DOGE Squeeze 做多")
                                    if order is None:
                                        print("⚠️ DOGE 做多市價單失敗，放棄本次進場")
                                    else:
                                        ep = float(order.get('average') or cl_doge.open)
                                        doge_s['position']      = 1
                                        doge_s['position_size'] = size
                                        doge_s['entry_price']   = ep
                                        doge_s['stop_loss']     = ep - sl_dist
                                        doge_s['trailing_stop'] = ep - trail_dist
                                        doge_s['highest_price'] = ep
                                        doge_s['liq_price']     = CoreStrategy.calc_liquidation_price(ep, 1, self.DOGE_LEVERAGE, self.mmr)
                                        self.execute_order('DOGE', 'sell', stop_price=doge_s['stop_loss'])
                                else:
                                    order = self.execute_order('DOGE', 'sell', size, reason="DOGE Squeeze 做空")
                                    if order is None:
                                        print("⚠️ DOGE 做空市價單失敗，放棄本次進場")
                                    else:
                                        ep = float(order.get('average') or cl_doge.open)
                                        doge_s['position']      = -1
                                        doge_s['position_size'] = size
                                        doge_s['entry_price']   = ep
                                        doge_s['stop_loss']     = ep + sl_dist
                                        doge_s['trailing_stop'] = ep + trail_dist
                                        doge_s['lowest_price']  = ep
                                        doge_s['liq_price']     = CoreStrategy.calc_liquidation_price(ep, -1, self.DOGE_LEVERAGE, self.mmr)
                                        self.execute_order('DOGE', 'buy', stop_price=doge_s['stop_loss'])

                                if doge_s['position'] != 0:
                                    self._doge_twap_direction = doge_s['position']
                                    self._doge_twap_size_each = (self.DOGE_MAX_TRADE_CAP / doge_s['entry_price']) if _N > 1 else size
                                    self._doge_twap_remaining = _N - 1
                                    self._doge_twap_active    = self._doge_twap_remaining > 0
                                    self.save_order_state()
                                    time.sleep(1)   # 等待交易所餘額刷新（保持一致性）

                                    msg = (
                                        f"🚀 **DOGE Squeeze 進場** | {'做多' if direction==1 else '做空'}\n"
                                        f"   進場: {doge_s['entry_price']:.5f} | SL: {doge_s['stop_loss']:.5f}\n"
                                        f"   size: {doge_s['position_size']:.0f} | TWAP剩餘: {self._doge_twap_remaining}"
                                    )
                                    self.send_discord_msg(msg)

                # 狀態列印：每逢整 5 分鐘（:00, :05, :10 ...）印一次
                now_tw = datetime.now(timezone.utc) + timedelta(hours=8)
                _cur_slot = now_tw.hour * 60 + now_tw.minute - now_tw.minute % 5
                if _cur_slot != self._last_status_print:
                    self._last_status_print = _cur_slot
                    self._print_status(cl, ada_price=ada_close,
                                       ada_dc_high=dc_high, ada_dc_low=dc_low,
                                       xrp_price=xrp_close,
                                       doge_price=doge_close)

            except Exception as e:
                import traceback
                error_log = f"‼️ **主循環發生嚴重異常** ‼️\n錯誤內容: {e}"
                print(error_log)
                # 主動推送到 Discord
                try:
                    self.send_discord_msg(error_log)
                except:
                    pass # 如果是斷網導致的，這裡也會失敗，所以加個 try 避免崩潰
                traceback.print_exc()

            time.sleep(self.check_interval)


if __name__ == "__main__":
    bot = LiveTradingBot()
    bot.run()