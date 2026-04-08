import pandas as pd
import pandas_ta as ta


class CoreStrategy:

    # ── 資料準備 ────────────────────────────────────────────────
    @staticmethod
    def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ATR'] = df.ta.atr(length=14)
        df['RSI'] = df.ta.rsi(length=14)
        ema = df.ta.ema(length=200)
        df['EMA'] = ema.squeeze() if hasattr(ema, 'squeeze') else ema

        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            df['ADX'] = df.ta.adx(length=14)['ADX_14']
        else:
            df['ADX'] = 100.0

        bb = df.ta.bbands(length=20, std=2)
        df['BB_Mid'] = bb.iloc[:, 1]

        df.dropna(inplace=True)
        return df

    # ── 進場信號 ─────────────────────────────────────────────────
    @staticmethod
    def check_signals(row, adx_threshold: float) -> tuple[bool, bool, str]:
        cur_close  = row.close  if hasattr(row, 'close')  else row['close']
        cur_ema    = row.EMA    if hasattr(row, 'EMA')    else row['EMA']
        cur_bb_mid = row.BB_Mid if hasattr(row, 'BB_Mid') else row['BB_Mid']
        cur_rsi    = row.RSI    if hasattr(row, 'RSI')    else row['RSI']
        cur_adx    = row.ADX    if hasattr(row, 'ADX')    else row['ADX']

        l_adx_ok = cur_adx > adx_threshold
        l_ema_ok = cur_close > cur_ema
        l_bb_ok  = cur_close > cur_bb_mid
        l_rsi_ok = cur_rsi < 58
        l_cond   = l_adx_ok and l_ema_ok and l_bb_ok and l_rsi_ok

        s_adx_ok = cur_adx > adx_threshold
        s_ema_ok = cur_close < cur_ema
        s_bb_ok  = cur_close < cur_bb_mid
        s_rsi_ok = cur_rsi > 42
        s_cond   = s_adx_ok and s_ema_ok and s_bb_ok and s_rsi_ok

        msg = ""
        if not l_cond and not s_cond:
            if cur_close > cur_ema:
                msg = (f"🔍 [未達做多門檻] ADX:{'✅' if l_adx_ok else '❌'} | "
                       f"EMA200:{'✅' if l_ema_ok else '❌'} | "
                       f"BB_Mid:{'✅' if l_bb_ok else '❌'} | RSI(<58):{'✅' if l_rsi_ok else '❌'}")
            else:
                msg = (f"🔍 [未達做空門檻] ADX:{'✅' if s_adx_ok else '❌'} | "
                       f"EMA200:{'✅' if s_ema_ok else '❌'} | "
                       f"BB_Mid:{'✅' if s_bb_ok else '❌'} | RSI(>42):{'✅' if s_rsi_ok else '❌'}")

        return l_cond, s_cond, msg

    # ── 倉位計算 ─────────────────────────────────────────────────
    @staticmethod
    def calculate_position_size(capital: float, risk_ratio: float, sl_dist: float,
                                 entry_price: float, max_pos_ratio: float,
                                 leverage: float, max_trade_usdt: float) -> tuple[float, float]:
        if sl_dist <= 0:
            return 0.0, 0.0
        size = min(
            (capital * risk_ratio) / sl_dist,
            (capital * max_pos_ratio * leverage) / entry_price,
            max_trade_usdt / entry_price
        )
        return size, size * sl_dist

    # ── 清算價計算 ───────────────────────────────────────────────
    @staticmethod
    def calc_liquidation_price(entry_price: float, position: int,
                                leverage: float, mmr: float) -> float:
        """
        Binance 永續合約清算價估算。
        position: 1=多單, -1=空單
        mmr: 維持保證金率（例如 0.004）
        """
        if position == 1:
            return entry_price * (1 - 1 / leverage + mmr)
        else:
            return entry_price * (1 + 1 / leverage - mmr)

    # ── 追蹤止損更新 ─────────────────────────────────────────────
    @staticmethod
    def update_trailing_stop(position: int,
                              trailing_stop: float,
                              highest_price: float,
                              lowest_price:  float,
                              cur_high:      float,
                              cur_low:       float,
                              cur_atr:       float,
                              trailing_atr:  float) -> tuple[float, float, float]:
        """
        更新追蹤止損價、最高/最低價。
        回傳 (trailing_stop, highest_price, lowest_price)
        """
        if position == 1:
            highest_price = max(highest_price, cur_high)
            trailing_stop = max(trailing_stop, highest_price - trailing_atr * cur_atr)
        elif position == -1:
            lowest_price  = min(lowest_price, cur_low)
            trailing_stop = min(trailing_stop, lowest_price + trailing_atr * cur_atr)
        return trailing_stop, highest_price, lowest_price

    # ── 出場判斷 ─────────────────────────────────────────────────
    @staticmethod
    def check_exit(position:      int,
                   cur_low:       float,
                   cur_high:      float,
                   cur_open:      float,
                   liq_price:     float,
                   trailing_stop: float,
                   stop_loss:     float,
                   entry_price:   float,
                   position_size: float,
                   fee_rate:      float,
                   slippage_pct:  float) -> tuple[bool, float, float, str]:
        """
        判斷當根K線是否觸發出場。
        回傳 (trade_closed, exit_price, closed_pnl, exit_reason)
        """
        if position == 1:
            if cur_low <= liq_price:
                ep  = liq_price * (1 - slippage_pct)
                pnl = (ep - entry_price) * position_size - position_size * ep * fee_rate
                return True, ep, pnl, '💀 Liquidation'
            elif cur_low <= trailing_stop or cur_low <= stop_loss:
                is_tr = cur_low <= trailing_stop
                ep    = max(trailing_stop if is_tr else stop_loss, cur_open) * (1 - slippage_pct)
                pnl   = (ep - entry_price) * position_size - position_size * ep * fee_rate
                return True, ep, pnl, 'Trailing' if is_tr else 'Stop_Loss'

        elif position == -1:
            if cur_high >= liq_price:
                ep  = liq_price * (1 + slippage_pct)
                pnl = (entry_price - ep) * position_size - position_size * ep * fee_rate
                return True, ep, pnl, '💀 Liquidation'
            elif cur_high >= trailing_stop or cur_high >= stop_loss:
                is_tr = cur_high >= trailing_stop
                ep    = min(trailing_stop if is_tr else stop_loss, cur_open) * (1 + slippage_pct)
                pnl   = (entry_price - ep) * position_size - position_size * ep * fee_rate
                return True, ep, pnl, 'Trailing' if is_tr else 'Stop_Loss'

        return False, 0.0, 0.0, ""

    # ── 動態風險（保留備用）──────────────────────────────────────
    @staticmethod
    def get_dynamic_risk(current_equity: float, equity_ma: float,
                          normal_risk: float, reduced_risk: float) -> float:
        return normal_risk if current_equity >= equity_ma else reduced_risk