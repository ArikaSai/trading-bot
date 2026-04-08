

import ccxt
import pandas as pd
import numpy as np
import time
import requests
import json
import os
from datetime import datetime, timedelta, timezone
from strategy import CoreStrategy

# ══════════════════════════════════════════════════════════════
#  ADA Donchian 突破策略固定參數
# ══════════════════════════════════════════════════════════════
ADA_ENTRY_N       = 10       # Donchian 通道長度（前 N 根 K 棒）
ADA_TRAIL_ATR     = 3.0      # ATR trailing 倍數
ADA_ATR_SL_MULT   = 2.0      # 倉位計算用 SL 倍數
ADA_RISK_PCT      = 0.15     # 每筆風險佔可用資金比例
ADA_MAX_TRADE_CAP = 200_000.0  # 單筆最大名目價值（TWAP 拆單門檻）
ADA_MAX_CONSEC    = 3        # ADA 熔斷門檻（連損次數）
ADA_FEE           = 0.0005   # taker 手續費


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

        self.symbols    = {'SOL': 'SOL/USDT', 'ADA': 'ADA/USDT'}
        self.timeframes = {'SOL': '15m',       'ADA': '1h'}
        self._close_time = {}   # 記錄各策略最近平倉時間，防止 API 快取觸發雲端接管

        # SOL 盤整縮緊：N=8, X=1.25, tight=1.0×
        from collections import deque as _deque
        self._consol_highs    = _deque(maxlen=8)
        self._consol_lows     = _deque(maxlen=8)
        self._consol_last_ts  = None   # 防止同一根 K 棒重複 append

        # ADA TWAP 狀態
        self._ada_twap_active    = False
        self._ada_twap_remaining = 0
        self._ada_twap_size_each = 0.0
        self._ada_twap_direction = 0

        self.live_trade    = self.config['system']['live_trade']
        self.check_interval = self.config['system']['check_interval']

        self.exchange = ccxt.binance({
            'apiKey':          self.api_key    if self.live_trade else '',
            'secret':          self.api_secret if self.live_trade else '',
            'enableRateLimit': True,
            'options':         {'defaultType': 'future'}
        })

        self.state = {
            'SOL': self._get_default_state(),
            'ADA': self._get_default_state()
        }

        self.load_order_state()

        self.last_candle_time = {'SOL': None, 'ADA': None}
        self.last_report_hour = -1   # 整點定時報告：記錄上次觸發的小時（台灣時間）
        self._last_status_print = -1  # 上次狀態列印的 5 分鐘時段編號

        print(f"🤖 雙核心機器人就緒 | 模式: {'🔴 實盤' if self.live_trade else '🟢 模擬'}")
        print(f"   SOL 趨勢: {self.risk_per_trade*100}% 風險 | {self.leverage}x | 熔斷 {self.max_consec_losses} 次")
        print(f"   ADA Donchian: N={ADA_ENTRY_N} | Trail ×{ADA_TRAIL_ATR} | 熔斷 {ADA_MAX_CONSEC} 次")

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
                    f"   熔斷: {ada_cb} | 連損: {ada_s['consecutive_losses']}/{ADA_MAX_CONSEC}"
                )
            else:
                try:
                    ada_price = float(self.exchange.fetch_ticker('ADA/USDT')['last']) if self.live_trade else 0.0
                except Exception:
                    ada_price = 0.0
                cb_tag = " | 🛡️ 保護中" if ada_s['skip_next_trade'] or ada_s['in_skip_zone'] else ""
                ada_block = (
                    f"🔵 ADA 空倉 | 現價: {ada_price:.4f} | 👁️ 待機{cb_tag}\n"
                    f"   連損: {ada_s['consecutive_losses']}/{ADA_MAX_CONSEC}"
                )
            msg = (
                f"⏰ **整點報告** | {now_tw.strftime('%m/%d %H:00')}\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"{sol_block}\n{ada_block}\n"
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
        is_close = state['position'] != 0   # 在 wipe 前定義，確保 except 區塊可引用
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
            threshold = self.max_consec_losses if strat_name == 'SOL' else ADA_MAX_CONSEC
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

        # ATR trailing stop 更新
        if pos == 1:
            state['highest_price'] = max(state['highest_price'], cur_high)
            state['trailing_stop'] = max(
                state['trailing_stop'],
                state['highest_price'] - ADA_TRAIL_ATR * cur_atr
            )
            if cur_low <= state['trailing_stop']:
                side = 'sell'
                print(f"🚨 ADA 本地 Trailing 觸發")
                self.execute_order('ADA', side, state['position_size'], reason="Trailing")
        elif pos == -1:
            state['lowest_price'] = min(state['lowest_price'], cur_low)
            state['trailing_stop'] = min(
                state['trailing_stop'],
                state['lowest_price'] + ADA_TRAIL_ATR * cur_atr
            )
            if cur_high >= state['trailing_stop']:
                side = 'buy'
                print(f"🚨 ADA 本地 Trailing 觸發")
                self.execute_order('ADA', side, state['position_size'], reason="Trailing")

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
                            sl_dist    = max(ADA_ATR_SL_MULT * last_atr, state['entry_price'] * self.min_sl_pct)
                            trail_dist = max(ADA_TRAIL_ATR * last_atr, state['entry_price'] * self.min_sl_pct)
                        else:
                            sl_dist    = max(self.initial_sl_atr * last_atr, state['entry_price'] * self.min_sl_pct)
                            trail_dist = sl_dist
                        state['stop_loss']     = state['entry_price'] - sl_dist if state['position'] == 1 else state['entry_price'] + sl_dist
                        state['trailing_stop'] = state['entry_price'] - trail_dist if state['position'] == 1 else state['entry_price'] + trail_dist
                        state['highest_price'] = state['entry_price']
                        state['lowest_price']  = state['entry_price']
                        state['liq_price']     = CoreStrategy.calc_liquidation_price(
                            state['entry_price'], state['position'], self.leverage, self.mmr)
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
                      ada_dc_high: float = None, ada_dc_low: float = None):
        """每 5 分鐘統一列印 SOL + ADA 雙核心狀態"""
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

        print("─" * 60)

    # ── 主循環 ───────────────────────────────────────────────────────

    def run(self):
        print(f"⏳ 雙核心輪詢啟動，每 {self.check_interval}s 一次")
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

                    l_cond, s_cond, diag = CoreStrategy.check_signals(prev, self.adx_threshold)

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
                            size, _ = CoreStrategy.calculate_position_size(
                                total_bal, self.risk_per_trade, sl_dist, float(cl.open),
                                self.max_pos_ratio, self.leverage, self.max_trade_usdt_cap
                            )
                        else:
                            size = self.max_trade_usdt_cap / float(cl.open)

                        # 🛡️ 用「剩餘可用保證金」去壓制最大下單量
                        max_size_by_free_margin = (free_bal * self.leverage) / float(cl.open)
                        size = min(size, max_size_by_free_margin)

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
                if prev_idx >= ADA_ENTRY_N:
                    dc_high = float(np.max(ada_highs[prev_idx - ADA_ENTRY_N:prev_idx]))
                    dc_low  = float(np.min(ada_lows[prev_idx - ADA_ENTRY_N:prev_idx]))
                else:
                    dc_high = dc_low = ada_close

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
                    max_by_margin = (free_bal * self.leverage) / float(cl_ada.open)
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
                                ada_s['entry_price'], ada_s['position'], self.leverage, self.mmr)
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
                        risk_per_unit = ADA_ATR_SL_MULT * prev_atr
                        if risk_per_unit > 0:
                            if self.live_trade:
                                bal_data  = self.exchange.fetch_balance()
                                total_bal = float(bal_data['total']['USDT'])
                                free_bal  = float(bal_data['free']['USDT'])
                            else:
                                total_bal = free_bal = self.capital

                            # 自適應 TWAP：計算 N
                            _N = max(1, int(total_bal * self.leverage / ADA_MAX_TRADE_CAP))
                            if _N == 1:
                                size = min(
                                    (total_bal * ADA_RISK_PCT) / risk_per_unit,
                                    (total_bal * self.leverage) / float(cl_ada.open)
                                )
                            else:
                                size = ADA_MAX_TRADE_CAP / float(cl_ada.open)

                            max_size_by_free_margin = (free_bal * self.leverage) / float(cl_ada.open)
                            size = min(size, max_size_by_free_margin)

                            sl_dist    = max(ADA_ATR_SL_MULT * prev_atr, float(cl_ada.open) * self.min_sl_pct)
                            trail_dist = max(ADA_TRAIL_ATR * prev_atr, float(cl_ada.open) * self.min_sl_pct)

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
                                        ada_s['liq_price']     = CoreStrategy.calc_liquidation_price(ep, 1, self.leverage, self.mmr)
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
                                        ada_s['liq_price']     = CoreStrategy.calc_liquidation_price(ep, -1, self.leverage, self.mmr)
                                        self.execute_order('ADA', 'buy', stop_price=ada_s['stop_loss'])

                                if ada_s['position'] != 0:
                                    self._ada_twap_direction = ada_s['position']
                                    self._ada_twap_size_each = (ADA_MAX_TRADE_CAP / ada_s['entry_price']) if _N > 1 else size
                                    self._ada_twap_remaining = _N - 1
                                    self._ada_twap_active    = self._ada_twap_remaining > 0
                                    self.save_order_state()

                                    msg = (
                                        f"🚀 **ADA Donchian 進場** | {'做多' if direction==1 else '做空'}\n"
                                        f"   進場: {ada_s['entry_price']:.4f} | SL: {ada_s['stop_loss']:.4f}\n"
                                        f"   size: {ada_s['position_size']:.2f} | TWAP剩餘: {self._ada_twap_remaining}"
                                    )
                                    self.send_discord_msg(msg)

                # 狀態列印：每逢整 5 分鐘（:00, :05, :10 ...）印一次
                now_tw = datetime.now(timezone.utc) + timedelta(hours=8)
                _cur_slot = now_tw.hour * 60 + now_tw.minute - now_tw.minute % 5
                if _cur_slot != self._last_status_print:
                    self._last_status_print = _cur_slot
                    self._print_status(cl, ada_price=ada_close,
                                       ada_dc_high=dc_high, ada_dc_low=dc_low)

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