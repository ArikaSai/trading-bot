import ccxt
import pandas as pd
import time
from strategy import CoreStrategy

SYMBOL     = 'ADA/USDT'
TIMEFRAME  = '1h'
START_DATE = '2020-01-01'
END_DATE   = '2026-06-30'
FILENAME   = f"data/{SYMBOL.replace('/', '')}_{TIMEFRAME}.csv"


def fetch_binance_data(symbol: str = SYMBOL,
                       timeframe: str = TIMEFRAME,
                       start_date: str = START_DATE,
                       end_date:   str = END_DATE) -> pd.DataFrame:
    """從幣安下載歷史K線（原始OHLCV，不含指標）。"""
    print(f"📥 開始下載 {symbol} {timeframe} 資料（{start_date} ~ {end_date}）...")
    exchange       = ccxt.binance({'enableRateLimit': True})
    since          = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_timestamp  = int(pd.Timestamp(end_date).timestamp() * 1000)
    all_ohlcv      = []

    while since < end_timestamp:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since        = ohlcv[-1][0] + 1
            current_date = pd.to_datetime(ohlcv[-1][0], unit='ms').strftime('%Y-%m-%d')
            print(f"⏳ 已下載至: {current_date}", end='\r')
            time.sleep(0.1)
        except Exception as e:
            print(f"\n⚠️ 下載錯誤: {e}，5 秒後重試...")
            time.sleep(5)

    print(f"\n✅ 下載完成，共 {len(all_ohlcv)} 筆原始資料")

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.ffill(inplace=True)
    return df


if __name__ == "__main__":
    # 1. 下載原始 OHLCV
    df_raw = fetch_binance_data()

    # 2. 計算指標（單一來源：CoreStrategy.prepare_data）
    print("⚙️  計算技術指標...")
    df = CoreStrategy.prepare_data(df_raw)
    print(f"✅ 指標計算完成，有效資料 {len(df)} 筆")

    # 3. 儲存
    df.to_csv(FILENAME, float_format='%.4f')
    print(f"💾 已儲存至 {FILENAME}")
    print("\n📊 資料預覽（最後 5 筆）：")
    print(df.tail())